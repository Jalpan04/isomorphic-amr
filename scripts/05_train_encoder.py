import torch
print("VERY START: CUDA available:", torch.cuda.is_available())
print("Starting train_encoder.py...")
print("Loading standard libraries...")
import sys
import yaml
from pathlib import Path
from torch.utils.data import ConcatDataset

print("Loading internal packages...")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import GraphDataset
from src.models.full_model import UnsupervisedAMRModel
from src.training.optimizer import get_optimizer_and_scheduler
from src.training.trainer import UnsupervisedAMRTrainer
from src.utils.reproducibility import set_seed
from src.utils.logger import setup_logger
from loguru import logger
print("All libraries loaded successfully. Starting execution...")

def main():
    # Load configuration
    config_path = Path("config/base_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    exp_id = config["experiment"]["id"]
    setup_logger(exp_id)
    logger.info(f"Starting experiment: {exp_id}")
    
    # 1. Set seed for reproducibility
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() and config["experiment"]["device"] == "cuda" else "cpu"
    logger.info(f"Using compute device: {device}")
    
    # 2. Load Datasets
    logger.info("Loading preprocessed graph datasets...")
    
    # English AMR dataset
    amr_dir = Path(config["paths"]["processed_amr"]) / "train"
    if not amr_dir.exists() or not list(amr_dir.glob("*.pt")):
        logger.error(f"English AMR preprocessed graphs not found at: {amr_dir}. Run scripts/02_preprocess_amr.py and 04_extract_embeddings.py first.")
        return
    amr_dataset = GraphDataset(str(amr_dir))
    logger.info(f"Loaded {len(amr_dataset)} English AMR graphs.")
    
    # Multilingual target dependency datasets
    tgt_datasets = []
    for lang in config["eval"]["languages"]:
        dep_dir = Path(config["paths"]["processed_dep"]) / lang
        if not dep_dir.exists() or not list(dep_dir.glob("*.pt")):
            logger.warning(f"Dependency graphs for {lang} not found at: {dep_dir}. Skipping.")
            continue
        tgt_datasets.append(GraphDataset(str(dep_dir)))
        logger.info(f"Loaded {len(tgt_datasets[-1])} dependency graphs for language: {lang}")
        
    if not tgt_datasets:
        logger.error("No target dependency datasets found. Run scripts/03_preprocess_dep.py and 04_extract_embeddings.py first.")
        return
        
    # Concatenate all target datasets
    target_dataset = ConcatDataset(tgt_datasets)
    logger.info(f"Combined target datasets: total size = {len(target_dataset)} graphs.")
    
    # 3. Initialize Model
    logger.info("Initializing GAT encoder model...")
    model = UnsupervisedAMRModel(
        in_channels=config["encoder"].get("in_channels", 768),
        hidden_channels=config["encoder"].get("hidden_channels", 256),
        out_channels=config["encoder"].get("out_channels", 256),
        heads=config["encoder"].get("heads", 4),
        dropout=config["encoder"].get("dropout", 0.1),
        num_layers=config["encoder"].get("num_layers", 2),
        conv_type=config["encoder"].get("conv_type", "gat")
    ).to(device)
    
    # 4. Initialize Optimizer & Scheduler
    # Estimate total steps
    num_batches = min(len(amr_dataset), len(target_dataset)) // config["train"]["batch_size"]
    total_steps = num_batches * config["train"]["epochs"]
    logger.info(f"Total training steps calculated: {total_steps} (over {config['train']['epochs']} epochs)")
    
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        lr=float(config["train"]["lr"]),
        weight_decay=config["train"]["weight_decay"],
        num_warmup_steps=config["train"]["warmup_steps"],
        num_training_steps=total_steps
    )
    
    # 5. Initialize Trainer & Fit
    trainer = UnsupervisedAMRTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=config["train"],
        alignment_config=config["alignment"],
        exp_id=exp_id,
        device=device
    )
    
    trainer.fit(
        en_train_dataset=amr_dataset,
        tgt_train_dataset=target_dataset
    )
    
    logger.success("Model encoder training script finished successfully.")

if __name__ == "__main__":
    main()
