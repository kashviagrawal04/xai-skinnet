"""
quick_train.py - CPU-friendly training on a 200-image subset of HAM10000.
"""
import argparse, json, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ham10000 import (
    HAM10000Dataset, build_manifest, class_weights_for_loss,
    get_transforms, make_sampler, split_manifest,
)
from src.models.xai_skinnet import XAISkinNet
from src.utils import AverageMeter, ensure_dir, get_device, load_config, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, leave=False)
        for images, labels, metadata in pbar:
            images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)
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
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")
    return loss_meter.avg, acc_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    class_names = cfg["data"]["class_names"]
    num_classes  = cfg["data"]["num_classes"]

    df = build_manifest(args.data_root, class_names)
    print(f"Full dataset: {len(df)} rows")

    per_class = max(2, args.max_samples // num_classes)
    df = (df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_class), random_state=42))
            .reset_index(drop=True))
    print(f"Subsampled to: {len(df)} rows ({per_class} per class max)")

    train_df, val_df, test_df = split_manifest(
        df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed=cfg["seed"]
    )
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_ds = HAM10000Dataset(train_df, get_transforms(cfg, train=True))
    val_ds   = HAM10000Dataset(val_df,   get_transforms(cfg, train=False))

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              sampler=make_sampler(train_df, num_classes),
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=False)

    model = XAISkinNet(cfg).to(device)
    class_weights = class_weights_for_loss(train_df, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    run_dir = ensure_dir(Path(cfg["train"]["output_dir"]) / cfg["run_name"])
    history, best_val_acc = [], 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if epoch == 1:
            model.freeze_for_warmup()
            print(f"Epoch {epoch}: backbones frozen (warm-up)")
        elif epoch == cfg["train"]["freeze_epochs"] + 1:
            model.unfreeze_all()
            print(f"Epoch {epoch}: all layers unfrozen")

        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        print(f"[{epoch:02d}/{cfg['train']['epochs']}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} ({time.time()-t0:.1f}s)")
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=vl_loss, val_acc=vl_acc))

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "val_acc": vl_acc, "epoch": epoch},
                       run_dir / "best.pt")
            print(f"  -> new best val_acc={vl_acc:.4f}, checkpoint saved")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    test_df.to_csv(run_dir / "test_split.csv", index=False)
    val_df.to_csv(run_dir / "val_split.csv",   index=False)
    print(f"\nDone! Best val_acc={best_val_acc:.4f}. Checkpoint at: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
