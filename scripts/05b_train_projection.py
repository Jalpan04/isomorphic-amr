import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# Clear PyTorch VRAM cache at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("VERY START: CUDA available:", torch.cuda.is_available())
print("Starting train_projection.py...")
print("Loading standard libraries...")
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

print("Loading internal packages...")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import GraphDataset
from src.models.full_model import UnsupervisedAMRModel
from src.models.amr_decoder import AMRBARTDecoderWrapper
from src.utils.logger import setup_logger
from loguru import logger

from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from transformers.modeling_outputs import BaseModelOutput

def main():
    # Load configuration
    config_path = Path("config/base_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    exp_id = config["experiment"]["id"]
    setup_logger(exp_id)
    logger.info(f"Training projection layer for experiment: {exp_id}")
    
    device = "cuda" if torch.cuda.is_available() and config["experiment"]["device"] == "cuda" else "cpu"
    logger.info(f"Using compute device: {device}")
    
    if device == "cuda":
        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        total_gb = total_mem / (1024 ** 3)
        logger.info(f"GPU Memory: Free = {free_gb:.2f} GB, Total = {total_gb:.2f} GB")
        if free_gb < 4.5:
            logger.warning(
                f"Available VRAM is very low ({free_gb:.2f} GB). "
                "Close GPU-heavy applications (Chrome, Edge, Discord, games) to avoid slow memory paging."
            )
    
    # 1. Load GAT Encoder
    logger.info("Initializing GAT encoder model...")
    gat_model = UnsupervisedAMRModel(
        in_channels=config["encoder"].get("in_channels", 768),
        hidden_channels=config["encoder"].get("hidden_channels", 256),
        out_channels=config["encoder"].get("out_channels", 256),
        heads=config["encoder"].get("heads", 4),
        dropout=config["encoder"].get("dropout", 0.1),
        num_layers=config["encoder"].get("num_layers", 2),
        conv_type=config["encoder"].get("conv_type", "gat")
    )
    
    ckpt_path = Path("experiments") / exp_id / "checkpoints" / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        gat_model.load_state_dict(ckpt["model_state_dict"])
        logger.success(f"Successfully loaded best GAT encoder checkpoint from {ckpt_path}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}. GAT encoder is randomly initialized.")
        
    gat_model = gat_model.to(device)
    gat_model.eval()  # GAT encoder is frozen
    for param in gat_model.parameters():
        param.requires_grad = False
        
    # 2. Load AMRBART Decoder Wrapper
    logger.info("Initializing AMRBART decoder...")
    decoder = AMRBARTDecoderWrapper(
        model_name="xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2",
        gat_out_channels=config["encoder"].get("out_channels", 256)
    )
    
    decoder.model.eval()  # BART is frozen
    for param in decoder.model.parameters():
        param.requires_grad = False
        
    decoder.emb_projection.train()  # Only projection layer is trained
    for param in decoder.emb_projection.parameters():
        param.requires_grad = True
        
    # 3. Load Datasets
    logger.info("Loading preprocessed English AMR graph datasets...")
    train_dir = Path(config["paths"]["processed_amr"]) / "train"
    dev_dir = Path(config["paths"]["processed_amr"]) / "dev"
    
    train_dataset = GraphDataset(str(train_dir))
    dev_dataset = GraphDataset(str(dev_dir))
    logger.info(f"Loaded {len(train_dataset)} train graphs and {len(dev_dataset)} dev graphs.")
    
    proj_config = config.get("projection_train", {})
    epochs = proj_config.get("epochs", 5)
    batch_size = proj_config.get("batch_size", 8)
    lr = proj_config.get("lr", 1e-3)
    cache_size = proj_config.get("cache_size", 53635)
    max_steps_per_epoch = proj_config.get("max_steps_per_epoch", None)
    
    # Cache graphs in RAM to bypass slow disk reads on Windows HDD
    logger.info(f"Caching first {cache_size} training graphs into RAM for zero-disk-I/O speed...")
    cached_train = []
    import penman
    for i in tqdm(range(min(cache_size, len(train_dataset))), desc="Caching Train"):
        data = train_dataset[i]
        try:
            g = penman.decode(data.metadata["penman_str"])
            g.metadata = {}
            data.metadata["penman_str"] = penman.encode(g, indent=None).replace("\n", " ").strip()
        except Exception:
            pass
        cached_train.append(data)
        
    logger.info("Caching dev graphs into RAM...")
    cached_dev = []
    for i in range(len(dev_dataset)):
        data = dev_dataset[i]
        try:
            g = penman.decode(data.metadata["penman_str"])
            g.metadata = {}
            data.metadata["penman_str"] = penman.encode(g, indent=None).replace("\n", " ").strip()
        except Exception:
            pass
        cached_dev.append(data)
        
    # 4. Data Loaders
    train_loader = DataLoader(cached_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(cached_dev, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # 5. Optimizer & Mixed Precision Scaler
    optimizer = torch.optim.AdamW(decoder.emb_projection.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    
    best_loss = float("inf")
    start_epoch = 1
    checkpoint_dir = Path("experiments") / exp_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    resume_path = checkpoint_dir / "projection_resume.pt"
    if resume_path.exists():
        logger.info(f"Found resume checkpoint at {resume_path}. Loading...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        decoder.emb_projection.load_state_dict(checkpoint["projection_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        logger.success(f"Resumed from epoch {checkpoint['epoch']} (best loss: {best_loss:.4f})")
    
    logger.info(f"Starting projection training for epochs {start_epoch} to {epochs}...")
    for epoch in range(start_epoch, epochs + 1):
        decoder.emb_projection.train()
        total_loss = 0.0
        step = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass GAT encoder
                with torch.no_grad():
                    _, node_emb = gat_model(batch.x, batch.edge_index, batch.batch)
                    
                # Convert to dense batch
                dense_emb, mask = to_dense_batch(node_emb, batch.batch)
                
                # Extract gold Penman strings directly from metadata (avoids slow to_data_list() slicing)
                penman_strs = batch.metadata["penman_str"]
                
                # Tokenize targets
                target_encodings = decoder.tokenizer(
                    penman_strs,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)
                
                labels = target_encodings["input_ids"]
                # Replace pad token ID with -100 to ignore in loss
                labels[labels == decoder.tokenizer.pad_token_id] = -100
                
                attention_mask = mask.long()
                
                # Run BART forward and backward pass with automatic mixed precision (AMP)
                # This drastically speeds up execution and halves VRAM usage on GPU
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    # Project GAT node embeddings to BART hidden dimensions
                    projected_emb = decoder.emb_projection(dense_emb)
                    encoder_outputs = BaseModelOutput(last_hidden_state=projected_emb)
                    
                    outputs = decoder.model(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                # Scale loss and run backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                step += 1
                
                pbar.set_postfix(loss=loss.item())
                
                if step % 200 == 0:
                    logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                    
                if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                    break
                    
        avg_train_loss = total_loss / step
        logger.success(f"Epoch {epoch} complete. Average train loss: {avg_train_loss:.4f}")
        
        # Validation
        decoder.emb_projection.eval()
        val_loss = 0.0
        val_step = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                _, node_emb = gat_model(batch.x, batch.edge_index, batch.batch)
                dense_emb, mask = to_dense_batch(node_emb, batch.batch)
                
                penman_strs = batch.metadata["penman_str"]
                
                target_encodings = decoder.tokenizer(
                    penman_strs,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)
                
                labels = target_encodings["input_ids"]
                labels[labels == decoder.tokenizer.pad_token_id] = -100
                attention_mask = mask.long()
                
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    projected_emb = decoder.emb_projection(dense_emb)
                    encoder_outputs = BaseModelOutput(last_hidden_state=projected_emb)
                    
                    outputs = decoder.model(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                val_loss += outputs.loss.item()
                val_step += 1
                
        avg_val_loss = val_loss / val_step
        logger.info(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f}")
        
        # Print a sample generation to check readability
        with torch.no_grad():
            sample_batch = next(iter(dev_loader)).to(device)
            _, sample_node_emb = gat_model(sample_batch.x, sample_batch.edge_index, sample_batch.batch)
            sample_dense, sample_mask = to_dense_batch(sample_node_emb, sample_batch.batch)
            
            # Pass the 256-dim GAT dense representations directly (generate_amr handles projection internally)
            sample_out = decoder.generate_amr(sample_dense[:1], max_length=128)
            logger.info("Sample generation:")
            logger.info(sample_out[0])
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = checkpoint_dir / "projection.pt"
            torch.save(decoder.emb_projection.state_dict(), save_path)
            logger.success(f"New best validation loss! Saved projection state dict to {save_path}")

        # Save resume checkpoint
        checkpoint = {
            "epoch": epoch,
            "projection_state_dict": decoder.emb_projection.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_loss": best_loss
        }
        torch.save(checkpoint, resume_path)
        logger.info(f"Saved resume checkpoint for epoch {epoch} to {resume_path}")

    # Delete resume checkpoint upon successful completion
    if resume_path.exists():
        try:
            resume_path.unlink()
            logger.info("Cleared temporary resume checkpoint.")
        except Exception:
            pass

    logger.success("Projection training finished successfully!")

if __name__ == "__main__":
    main()
