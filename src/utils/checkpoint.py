from pathlib import Path
import torch
from loguru import logger

def save_checkpoint(model, optimizer, epoch, metrics, exp_id: str, is_best: bool = False):
    """
    Save model checkpoint to experiments/{exp_id}/checkpoints/
    """
    ckpt_dir = Path("experiments") / exp_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }
    
    import os
    
    ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
    temp_ckpt_path = ckpt_path.with_suffix(".tmp")
    torch.save(ckpt, temp_ckpt_path)
    os.replace(temp_ckpt_path, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")
    
    if is_best:
        best_path = ckpt_dir / "best_model.pt"
        temp_best_path = best_path.with_suffix(".tmp")
        torch.save(ckpt, temp_best_path)
        os.replace(temp_best_path, best_path)
        logger.success(f"Saved new best model checkpoint: {best_path}")

def load_checkpoint(model, optimizer=None, checkpoint_path: str = None):
    """
    Load model weights and optional optimizer state from a checkpoint.
    """
    if not checkpoint_path:
        logger.warning("No checkpoint path provided. Starting from scratch.")
        return 0, {}
        
    path = Path(checkpoint_path)
    if not path.exists():
        logger.error(f"Checkpoint not found at: {path}")
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
        
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Successfully loaded model weights from: {path}")
    
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Successfully loaded optimizer state.")
        
    epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    return epoch, metrics
