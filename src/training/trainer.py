import torch
from torch_geometric.loader import DataLoader
from loguru import logger
import numpy as np
from pathlib import Path

from src.models.gw_alignment import gromov_wasserstein_align
from src.training.losses import (
    compute_pytorch_distance_matrix,
    gromov_wasserstein_loss,
    graph_reconstruction_loss
)
from src.utils.checkpoint import save_checkpoint

class UnsupervisedAMRTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 train_config: dict,
                 alignment_config: dict,
                 exp_id: str,
                 device: str = "cuda"):
        """
        Trainer class for joint GAE reconstruction and unsupervised Gromov-Wasserstein alignment.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.batch_size = train_config.get("batch_size", 32)
        self.epochs = train_config.get("epochs", 10)
        self.recon_weight = train_config.get("reconstruction_weight", 1.0)
        self.gw_weight = train_config.get("gw_weight", 1.0)
        
        self.gw_metric = alignment_config.get("metric", "cosine")
        self.gw_loss_fun = alignment_config.get("loss_fun", "square_loss")
        
        self.exp_id = exp_id
        self.device = device
        self.best_loss = float("inf")

    def train_epoch(self, epoch: int, en_loader: DataLoader, tgt_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_gw = 0.0
        
        # Zip loaders. If they are different lengths, zip will truncate or repeat.
        # We will loop through both datasets.
        step = 0
        for en_batch, tgt_batch in zip(en_loader, tgt_loader):
            en_batch = en_batch.to(self.device)
            tgt_batch = tgt_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 1. Forward pass for English AMR graphs
            en_graph_emb, en_node_emb = self.model(
                en_batch.x, en_batch.edge_index, en_batch.batch
            )
            
            # 2. Forward pass for target dependency graphs
            tgt_graph_emb, tgt_node_emb = self.model(
                tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch
            )
            
            # 3. Calculate GAE reconstruction loss
            recon_en = graph_reconstruction_loss(
                en_node_emb, en_batch.edge_index, en_batch.num_nodes
            )
            recon_tgt = graph_reconstruction_loss(
                tgt_node_emb, tgt_batch.edge_index, tgt_batch.num_nodes
            )
            recon_loss = recon_en + recon_tgt
            
            # 4. Calculate Gromov-Wasserstein Loss (Unsupervised alignment of graph embeddings in the batch)
            # Ensure equal batch dimensions for calculating alignment coupling
            min_batch_size = min(en_graph_emb.size(0), tgt_graph_emb.size(0))
            if min_batch_size > 1:
                en_emb_subset = en_graph_emb[:min_batch_size]
                tgt_emb_subset = tgt_graph_emb[:min_batch_size]
                
                # Solve standard OT coupling on CPU (constant during optimization step)
                T, _ = gromov_wasserstein_align(
                    en_emb_subset.detach(),
                    tgt_emb_subset.detach(),
                    metric=self.gw_metric,
                    loss_fun=self.gw_loss_fun,
                    log=False
                )
                
                # Compute differentiable distance matrices in PyTorch
                C_s = compute_pytorch_distance_matrix(en_emb_subset, metric=self.gw_metric)
                C_t = compute_pytorch_distance_matrix(tgt_emb_subset, metric=self.gw_metric)
                
                # Compute GW loss
                gw_loss = gromov_wasserstein_loss(C_s, C_t, T)
            else:
                gw_loss = torch.tensor(0.0, device=self.device)
            
            # 5. Combined loss
            loss = self.recon_weight * recon_loss + self.gw_weight * gw_loss
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_gw += gw_loss.item()
            step += 1
            
            if step % 20 == 0:
                logger.info(
                    f"Epoch {epoch:02d} | Step {step:03d} | Loss: {loss.item():.4f} "
                    f"(Recon: {recon_loss.item():.4f}, GW: {gw_loss.item():.4f})"
                )
                
        avg_loss = total_loss / max(step, 1)
        avg_recon = total_recon / max(step, 1)
        avg_gw = total_gw / max(step, 1)
        
        logger.success(
            f"Epoch {epoch:02d} Complete. Avg Loss: {avg_loss:.4f} "
            f"(Recon: {avg_recon:.4f}, GW: {avg_gw:.4f})"
        )
        return avg_loss

    def fit(self, en_train_dataset, tgt_train_dataset, en_val_dataset=None, tgt_val_dataset=None):
        """
        Run the complete training pipeline.
        """
        en_loader = DataLoader(en_train_dataset, batch_size=self.batch_size, shuffle=True)
        tgt_loader = DataLoader(tgt_train_dataset, batch_size=self.batch_size, shuffle=True)
        
        logger.info(f"Starting training for {self.epochs} epochs.")
        logger.info(f"Batches: English={len(en_loader)}, Target={len(tgt_loader)}")
        
        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch(epoch, en_loader, tgt_loader)
            
            # Save checkpoint
            is_best = loss < self.best_loss
            if is_best:
                self.best_loss = loss
                
            metrics = {
                "loss": loss,
                "best_loss": self.best_loss
            }
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics,
                exp_id=self.exp_id,
                is_best=is_best
            )
            
        logger.success("Training run finished successfully.")
