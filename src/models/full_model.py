import torch
import torch.nn as nn
from src.models.graph_encoder import AMRGraphEncoder

class UnsupervisedAMRModel(nn.Module):
    def __init__(self,
                 in_channels: int = 768,
                 hidden_channels: int = 256,
                 out_channels: int = 256,
                 heads: int = 4,
                 dropout: float = 0.1,
                 num_layers: int = 2,
                 conv_type: str = "gat"):
        """
        Aggregate model wrapping the shared-weight GAT/GCN encoder.
        During training, we optimize the weights of this GAT/GCN encoder.
        """
        super().__init__()
        self.encoder = AMRGraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            num_layers=num_layers,
            conv_type=conv_type
        )

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x:          (total_nodes, 768) node feature matrix
            edge_index: (2, total_edges)
            batch:      (total_nodes,) batch assignment vector
        Returns:
            graph_emb:  (batch_size, out_channels) graph level representations
            node_emb:   (total_nodes, out_channels) node level representations
        """
        return self.encoder(x, edge_index, batch)
