import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
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


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(loader, leave=False)
        for images, labels, metadata in pbar:
            images, labels = images.to(device), labels.to(device)
            metadata = metadata.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images, metadata=metadata)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            pbar.set_postfix(loss=loss_meter.avg, acc=acc_meter.avg)

    return loss_meter.avg, acc_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
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

    sampler = make_sampler(train_df, num_classes)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
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

    class_weights = class_weights_for_loss(train_df, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    run_dir = ensure_dir(Path(cfg["train"]["output_dir"]) / cfg["run_name"])
    history = []
    best_val_acc = 0.0
    epochs_without_improvement = 0

    freeze_epochs = cfg["train"]["freeze_epochs"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if epoch == 1:
            model.freeze_for_warmup()
            print(f"Epoch {epoch}: backbones frozen (warm-up, {freeze_epochs} epochs)")
        elif epoch == freeze_epochs + 1:
            model.unfreeze_all()
            print(f"Epoch {epoch}: all backbones unfrozen, full fine-tuning begins")

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        dt = time.time() - t0

        print(
            f"[{epoch:03d}/{cfg['train']['epochs']}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({dt:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg, "val_acc": val_acc, "epoch": epoch},
                run_dir / "best.pt",
            )
            print(f"  -> new best val_acc={val_acc:.4f}, checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{cfg['train']['early_stop_patience']} epochs)")
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
