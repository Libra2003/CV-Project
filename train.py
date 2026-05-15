import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from paths import LOG_ROOT
from ho3d_dataset import HO3DDataset
from checkpointing import save_training_state, load_training_state
from gdrive_log_sync import start_log_sync
from mano_wrapper import MANOLayer
from losses import HandReconstructionLoss
from student_model import MobileMaskHand


CHECKPOINT_EVERY_N_STEPS = 500
GRAD_CLIP_MAX_NORM = 1.0
WEIGHT_DECAY = 1e-4


def move_batch_to_device(batch, device):
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def build_dataloader(split, batch_size, num_workers):
    dataset = HO3DDataset(split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        pin_memory=True,
        drop_last=(split == "train"),
    )


def log_loss_breakdown(writer, prefix, breakdown, step):
    for key, value in breakdown.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def evaluate(model, loss_fn, loader, device):
    model.eval()
    accumulated = {"pose": 0.0, "shape": 0.0, "vertex": 0.0, "joint": 0.0, "total": 0.0}
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            batch = move_batch_to_device(batch, device)
            predictions = model(batch["edge_map"])
            _, breakdown = loss_fn(predictions, batch)
            for key in accumulated:
                accumulated[key] += breakdown[key]
            num_batches += 1
    return {key: value / max(num_batches, 1) for key, value in accumulated.items()}


def train_one_epoch(model, loss_fn, loader, optimizer, scaler, writer, device, global_step,
                    run_name, best_val_loss, current_epoch, scheduler):
    model.train()
    progress = tqdm(loader, desc=f"epoch {current_epoch}")
    for batch in progress:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            predictions = model(batch["edge_map"])
            loss, breakdown = loss_fn(predictions, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        log_loss_breakdown(writer, "train", breakdown, global_step)
        progress.set_postfix(total=breakdown["total"], joint=breakdown["joint"], step=global_step)

        if global_step % CHECKPOINT_EVERY_N_STEPS == 0:
            save_training_state(run_name, model, optimizer, scheduler, scaler,
                                current_epoch, global_step, best_val_loss, is_best=False)

    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-epochs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = build_dataloader("train", args.batch_size, args.num_workers)
    val_loader = build_dataloader("val", args.batch_size, args.num_workers)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = MobileMaskHand().to(device)
    mano_layer = MANOLayer().to(device)
    loss_fn = HandReconstructionLoss(mano_layer).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    resume_state = load_training_state(args.run_name, model, optimizer, scheduler, scaler, device)
    start_epoch = resume_state["epoch"]
    global_step = resume_state["global_step"]
    best_val_loss = resume_state["best_val_loss"]
    print(f"Resuming from epoch {start_epoch}, step {global_step}, best_val_loss {best_val_loss:.4f}")

    log_observer, log_uploader = start_log_sync(args.run_name)
    writer = SummaryWriter(log_dir=str(LOG_ROOT / args.run_name))

    try:
        for epoch in range(start_epoch, args.num_epochs):
            global_step = train_one_epoch(
                model, loss_fn, train_loader, optimizer, scaler, writer, device,
                global_step, args.run_name, best_val_loss, epoch, scheduler,
            )
            scheduler.step()

            val_breakdown = evaluate(model, loss_fn, val_loader, device)
            log_loss_breakdown(writer, "val", val_breakdown, global_step)
            print(f"Epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in val_breakdown.items()))

            val_loss = val_breakdown["total"]
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            save_training_state(args.run_name, model, optimizer, scheduler, scaler,
                                epoch + 1, global_step, best_val_loss, is_best=is_best)
    finally:
        writer.close()
        log_observer.stop()
        log_observer.join()
        log_uploader.shutdown()


if __name__ == "__main__":
    main()
