import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.datasets.ham10000 import (
    HAM10000Dataset,
    build_manifest,
    class_weights_for_loss,
    get_transforms,
    make_sampler,
    split_manifest,
)
from src.models.xai_skinnet import XAISkinNet
from src.utils import AverageMeter, ensure_dir, get_device, load_config, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)

    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    all_preds, all_labels = [], []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(loader, leave=False)
        for images, labels, metadata in pbar:
            images, labels = images.to(device), labels.to(device)
            metadata = metadata.to(device)

            if is_train:
                optimizer.zero_grad()

            use_amp = scaler is not None
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images, metadata=metadata)
                loss = criterion(logits, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            preds = logits.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            acc = (preds == labels).float().mean().item()

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            pbar.set_postfix(loss=loss_meter.avg, acc=acc_meter.avg)

    macro_f1 = f1_score(
        torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(),
        average="macro", zero_division=0,
    )
    return loss_meter.avg, acc_meter.avg, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=None,
                        help="Override train.output_dir (e.g. a Google Drive folder so "
                             "checkpoints survive Colab disconnects).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any last.pt and start from epoch 1.")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed precision on CUDA (less GPU memory, faster).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    class_names = cfg["data"]["class_names"]
    num_classes = cfg["data"]["num_classes"]

    df = build_manifest(args.data_root, class_names)
    train_df, val_df, test_df = split_manifest(
        df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed=cfg["seed"]
    )
    print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_ds = HAM10000Dataset(train_df, get_transforms(cfg, train=True))
    val_ds = HAM10000Dataset(val_df, get_transforms(cfg, train=False))

    # Class imbalance: balancing with BOTH an oversampling sampler AND an
    # inverse-frequency weighted loss double-corrects (the model then
    # over-predicts rare classes, e.g. NV -> MEL, which hurts accuracy and
    # precision). Default is one mechanism only; override with train.imbalance
    # = "sampler" | "loss" | "both".
    imbalance = cfg["train"].get("imbalance", "sampler")
    use_sampler = imbalance in ("sampler", "both")
    use_loss_w = imbalance in ("loss", "both")
    print(f"Imbalance handling: {imbalance}")
    sampler = make_sampler(train_df, num_classes) if use_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        shuffle=not use_sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    model = XAISkinNet(cfg).to(device)

    class_weights = class_weights_for_loss(train_df, num_classes).to(device) if use_loss_w else None
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg["train"].get("label_smoothing", 0.0)
    )

    # Differential learning rates: pretrained backbones get a smaller LR than the
    # freshly initialised fusion head, which avoids wrecking pretrained features
    # when they unfreeze (the big accuracy dip right after the warm-up).
    backbone_mult = cfg["train"].get("backbone_lr_mult", 1.0)
    backbone_ids = {id(p) for p in list(model.cnn.backbone.parameters())
                    + list(model.vit.backbone.parameters())}
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": cfg["train"]["lr"] * backbone_mult},
            {"params": head_params, "lr": cfg["train"]["lr"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )
    select_metric = cfg["train"].get("select_metric", "macro_f1")  # or "val_acc"
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    run_dir = ensure_dir(Path(cfg["train"]["output_dir"]) / cfg["run_name"])
    history = []
    best_val_acc = 0.0
    epochs_without_improvement = 0
    start_epoch = 1

    freeze_epochs = cfg["train"]["freeze_epochs"]
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    # Save splits up front so an interrupted run still has them.
    test_df.to_csv(run_dir / "test_split.csv", index=False)
    val_df.to_csv(run_dir / "val_split.csv", index=False)

    last_path = run_dir / "last.pt"
    if last_path.exists() and not args.no_resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        history = ck["history"]
        best_val_acc = ck["best_val_acc"]
        epochs_without_improvement = ck["epochs_without_improvement"]
        start_epoch = ck["epoch"] + 1
        print(f"Resuming from epoch {ck['epoch']} (best val_acc so far {best_val_acc:.4f})")
        # Re-apply the freeze state for the resumed epoch BEFORE optimizer state loads,
        # so the optimizer's param groups match what was saved.
        if start_epoch <= freeze_epochs:
            model.freeze_for_warmup()
        else:
            model.unfreeze_all()
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        if scaler is not None and ck.get("scaler_state"):
            scaler.load_state_dict(ck["scaler_state"])

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        if epoch == 1:
            model.freeze_for_warmup()
            print(f"Epoch {epoch}: backbones frozen (warm-up, {freeze_epochs} epochs)")
        elif epoch == freeze_epochs + 1:
            model.unfreeze_all()
            print(f"Epoch {epoch}: all backbones unfrozen, full fine-tuning begins")

        t0 = time.time()
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device, optimizer=None)
        score = val_f1 if select_metric == "macro_f1" else val_acc
        scheduler.step()
        dt = time.time() - t0

        print(
            f"[{epoch:03d}/{cfg['train']['epochs']}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({dt:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )

        stop = False
        if score > best_val_acc:
            best_val_acc = score
            epochs_without_improvement = 0
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg, "val_acc": val_acc,
                 "val_f1": val_f1, "epoch": epoch},
                run_dir / "best.pt",
            )
            print(f"  -> new best {select_metric}={score:.4f}, checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{cfg['train']['early_stop_patience']} epochs)")
                stop = True

        # Full resumable state every epoch (write-then-rename so a disconnect
        # mid-write can't corrupt the previous good checkpoint).
        tmp = run_dir / "last.pt.tmp"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "history": history,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch": epoch,
            },
            tmp,
        )
        tmp.replace(last_path)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        if stop:
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    test_df.to_csv(run_dir / "test_split.csv", index=False)
    # Saved alongside test_split.csv so src/calibrate.py can fit temperature
    # scaling on a held-out split the model never trained/early-stopped on
    # (using the test split for calibration would leak into the numbers
    # reported from src/evaluate.py).
    val_df.to_csv(run_dir / "val_split.csv", index=False)
    print(f"Training done. Best val_acc={best_val_acc:.4f}. Checkpoint + logs in {run_dir}")


if __name__ == "__main__":
    main()
