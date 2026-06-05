import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

class AMRGraphEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 768,
                 hidden_channels: int = 256,
                 out_channels: int = 256,
                 heads: int = 4,
                 dropout: float = 0.1,
                 num_layers: int = 2,
                 conv_type: str = "gat"):
        """
        Graph Neural Network encoder to map PyG graph node embeddings
        to a fixed-size latent graph vector representation. Supports GAT and GCN layers.
        """
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.conv_type = conv_type

        # Projection layer to map input XLM-R embeddings to hidden dimension
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GNN layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            layer_heads = 1 if is_last else heads
            layer_out = out_channels if is_last else (hidden_channels // heads)
            
            if conv_type == "gcn":
                conv = GCNConv(
                    in_channels=hidden_channels,
                    out_channels=out_channels if is_last else hidden_channels
                )
            else:
                conv = GATConv(
                    in_channels=hidden_channels,
                    out_channels=layer_out,
                    heads=layer_heads,
                    dropout=dropout
                )
            self.gat_layers.append(conv)
            
            norm_dim = out_channels if is_last else hidden_channels
            self.norms.append(nn.LayerNorm(norm_dim))

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x:          (num_nodes, 768) node features
            edge_index: (2, num_edges) adjacency list
            batch:      (num_nodes,) graph assignment batch vector
        Returns:
            graph_emb:  (batch_size, out_channels)
            node_emb:   (num_nodes, out_channels)
        """
        # 1. Project input to hidden channels
        h = self.input_proj(x)

        # 2. Forward through GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h_next = gat(h, edge_index)
            h = F.elu(h_next)
            h = norm(h)
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Keep a reference to the final node embeddings
        node_embeddings = h

        # 3. Global pooling to produce graph-level embeddings
        if batch is None:
            # If batch is not provided, assume single graph batch
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        graph_embeddings = global_mean_pool(h, batch)
        
        return graph_embeddings, node_embeddings
