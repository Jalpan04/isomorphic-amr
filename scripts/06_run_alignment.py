import torch
print("VERY START: CUDA available:", torch.cuda.is_available())
print("Starting run_alignment.py...")
print("Loading standard libraries...")
import sys
import yaml
import json
from pathlib import Path

print("Loading heavy ML libraries (numpy, tqdm)...")
import numpy as np
from tqdm import tqdm

print("Loading internal packages...")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import GraphDataset
from src.models.full_model import UnsupervisedAMRModel
from src.models.gw_alignment import fused_gromov_wasserstein_align, project_embeddings
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed
from loguru import logger
print("All libraries loaded successfully. Starting execution...")

def main():
    config_path = Path("config/base_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    exp_id = config["experiment"]["id"]
    setup_logger(exp_id)
    logger.info(f"Running Alignment Projection for experiment: {exp_id}")
    
    set_seed(config["experiment"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() and config["experiment"]["device"] == "cuda" else "cpu"
    
    # 1. Initialize and Load Model Checkpoint
    logger.info("Loading trained encoder model...")
    model = UnsupervisedAMRModel(
        in_channels=config["encoder"].get("in_channels", 768),
        hidden_channels=config["encoder"].get("hidden_channels", 256),
        out_channels=config["encoder"].get("out_channels", 256),
        heads=config["encoder"].get("heads", 4),
        dropout=config["encoder"].get("dropout", 0.1),
        num_layers=config["encoder"].get("num_layers", 2),
        conv_type=config["encoder"].get("conv_type", "gat")
    )
    
    # Check if a checkpoint exists
    ckpt_path = Path("experiments") / exp_id / "checkpoints" / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.success(f"Successfully loaded best model checkpoint from {ckpt_path}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}. Running with random model initialization.")
        
    model = model.to(device)
    model.eval()
    
    # 2. Load English AMR graphs (candidates for alignment)
    # We load the 'dev' split of English AMR to serve as target templates
    amr_dir = Path(config["paths"]["processed_amr"]) / "dev"
    if not amr_dir.exists() or not list(amr_dir.glob("*.pt")):
        logger.error(f"Dev split of English AMR graphs not found at: {amr_dir}")
        return
        
    amr_dataset = GraphDataset(str(amr_dir))
    logger.info(f"Loaded {len(amr_dataset)} candidate English AMR graphs.")
    
    # 3. Encode English AMR graphs
    logger.info("Encoding English AMR graphs...")
    en_graph_embs = []
    en_node_embs = []
    
    with torch.no_grad():
        for i in range(len(amr_dataset)):
            data = amr_dataset[i].to(device)
            # Forward pass
            graph_emb, node_emb = model(data.x, data.edge_index)
            en_graph_embs.append(graph_emb.squeeze(0).cpu())
            en_node_embs.append(node_emb.cpu())
            
    # Stack English graph embeddings: (N_en, out_channels)
    en_graph_embs = torch.stack(en_graph_embs)
    
    # 4. Perform alignment for each target language
    alpha = config["alignment"].get("alpha", 0.5)
    metric = config["alignment"].get("metric", "cosine")
    loss_fun = config["alignment"].get("loss_fun", "square_loss")
    
    for lang in config["eval"]["languages"]:
        logger.info(f"Running alignment for language: {lang}...")
        
        # Load target dependency graphs
        dep_dir = Path(config["paths"]["processed_dep"]) / lang
        if not dep_dir.exists() or not list(dep_dir.glob("*.pt")):
            logger.warning(f"Dependency graphs for {lang} not found. Skipping.")
            continue
            
        dep_dataset = GraphDataset(str(dep_dir))
        # We can align a subset (e.g. 100-300 graphs) for validation or the full set
        limit = min(300, len(dep_dataset))
        logger.info(f"Aligning the first {limit} graphs for {lang}")
        
        projected_emb_list = []
        matched_en_indices = []
        
        with torch.no_grad():
            for i in tqdm(range(limit), desc=f"Aligning {lang}"):
                tgt_data = dep_dataset[i].to(device)
                tgt_graph_emb, tgt_node_emb = model(tgt_data.x, tgt_data.edge_index)
                tgt_graph_emb = tgt_graph_emb.squeeze(0).cpu()
                tgt_node_emb = tgt_node_emb.cpu()
                
                # Step A: Find the closest English AMR graph in the latent space (Isomorphic retrieval)
                # Compute distance between target graph emb and all English graph embs
                if metric == "cosine":
                    # Cosine distance
                    norms_en = en_graph_embs.norm(dim=1) + 1e-8
                    norm_tgt = tgt_graph_emb.norm() + 1e-8
                    sim = (en_graph_embs @ tgt_graph_emb) / (norms_en * norm_tgt)
                    dists = 1.0 - sim
                else:
                    # Euclidean distance
                    dists = torch.norm(en_graph_embs - tgt_graph_emb, dim=1)
                    
                closest_idx = torch.argmin(dists).item()
                matched_en_indices.append(closest_idx)
                
                # Get matched English AMR graph node representations
                matched_data = amr_dataset[closest_idx].to(device)
                _, en_node_emb = model(matched_data.x, matched_data.edge_index)
                en_node_emb = en_node_emb.cpu()
                
                # Step B: Align and Project Target Embeddings to matched English node features
                alignment_method = config["alignment"].get("method", "fgw")
                
                if alignment_method == "procrustes":
                    # Procrustes alignment (ablation)
                    # Solve Procrustes between tgt_node_emb and matched en_node_emb
                    n_tgt = tgt_node_emb.shape[0]
                    n_en = en_node_emb.shape[0]
                    n_min = min(n_tgt, n_en)
                    
                    X = tgt_node_emb[:n_min].numpy()
                    Y = en_node_emb[:n_min].numpy()
                    
                    # Procrustes: find R to map X to Y
                    u, s, vt = np.linalg.svd(X.T @ Y)
                    R = torch.tensor(u @ vt, dtype=torch.float32)
                    
                    # Project target embeddings
                    projected = tgt_node_emb @ R
                    if n_tgt < n_en:
                        # Pad with zero vectors to match English graph size
                        padding = torch.zeros((n_en - n_tgt, projected.shape[1]), dtype=torch.float32)
                        projected = torch.cat([projected, padding], dim=0)
                    elif n_tgt > n_en:
                        projected = projected[:n_en]
                    projected_node_emb = projected
                    
                elif alignment_method == "random":
                    # Random orthogonal projection baseline
                    d_dim = tgt_node_emb.shape[1]
                    # Generate random orthogonal matrix R
                    Q, _ = torch.linalg.qr(torch.randn(d_dim, d_dim))
                    projected = tgt_node_emb @ Q
                    
                    # Match shape of English graph
                    n_tgt = tgt_node_emb.shape[0]
                    n_en = en_node_emb.shape[0]
                    if n_tgt < n_en:
                        padding = torch.zeros((n_en - n_tgt, d_dim), dtype=torch.float32)
                        projected = torch.cat([projected, padding], dim=0)
                    elif n_tgt > n_en:
                        projected = projected[:n_en]
                    projected_node_emb = projected
                    
                else:
                    # Fused Gromov-Wasserstein (default)
                    T, _ = fused_gromov_wasserstein_align(
                        source_emb=en_node_emb,
                        target_emb=tgt_node_emb,
                        source_features=matched_data.x.cpu(),
                        target_features=tgt_data.x.cpu(),
                        alpha=alpha,
                        metric=metric,
                        loss_fun=loss_fun,
                        log=False
                    )
                    projected_node_emb = project_embeddings(tgt_node_emb, T)
                
                projected_emb_list.append(projected_node_emb)
                
        # Save projected embeddings and metadata
        out_dir = Path("experiments") / exp_id / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save projected embeddings as a list of tensors
        projected_file = out_dir / f"projected_emb_{lang}.pt"
        torch.save(projected_emb_list, projected_file)
        
        # Save matched index metadata
        meta_file = out_dir / f"matched_metadata_{lang}.json"
        metadata = {
            "matched_en_indices": matched_en_indices,
            "metric": metric,
            "alpha": alpha
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.success(f"Saved {len(projected_emb_list)} projected embeddings for {lang} to {projected_file}")

if __name__ == "__main__":
    main()
