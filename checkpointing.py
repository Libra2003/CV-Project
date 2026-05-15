import torch

from paths import CHECKPOINT_ROOT


CHECKPOINT_FILENAME = "latest.pt"
BEST_CHECKPOINT_FILENAME = "best.pt"


def get_run_checkpoint_dir(run_name):
    run_dir = CHECKPOINT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def atomic_save(state, target_path):
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(state, temp_path)
    temp_path.replace(target_path)


def save_training_state(run_name, model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, is_best):
    run_dir = get_run_checkpoint_dir(run_name)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    atomic_save(state, run_dir / CHECKPOINT_FILENAME)
    if is_best:
        atomic_save(state, run_dir / BEST_CHECKPOINT_FILENAME)


def load_training_state(run_name, model, optimizer, scheduler, scaler, device):
    checkpoint_path = get_run_checkpoint_dir(run_name) / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return {"epoch": 0, "global_step": 0, "best_val_loss": float("inf")}

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    return {
        "epoch": state["epoch"],
        "global_step": state["global_step"],
        "best_val_loss": state["best_val_loss"],
    }
