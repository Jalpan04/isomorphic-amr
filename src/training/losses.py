import torch
import torch.nn.functional as F
import numpy as np

def compute_pytorch_distance_matrix(x: torch.Tensor, y: torch.Tensor = None, metric: str = "cosine") -> torch.Tensor:
    """
    Compute pairwise distance matrix between x and y in PyTorch (differentiable).
    Args:
        x: (n, d) tensor
        y: (m, d) tensor, if None y is set to x
        metric: "cosine" or "euclidean"
    Returns:
        C: (n, m) tensor of distances
    """
    if metric == "cosine":
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        if y is None:
            sim = torch.mm(x_norm, x_norm.t())
        else:
            y_norm = y / (y.norm(dim=-1, keepdim=True) + 1e-8)
            sim = torch.mm(x_norm, y_norm.t())
        C = 1.0 - sim
    else:
        if y is None:
            C = torch.cdist(x, x, p=2)
        else:
            C = torch.cdist(x, y, p=2)
            
    # Normalize
    C_max = C.max()
    if C_max > 1e-8:
        C = C / C_max
        
    return C

def gromov_wasserstein_loss(C_s: torch.Tensor, C_t: torch.Tensor, T: np.ndarray) -> torch.Tensor:
    """
    Computes the differentiable Gromov-Wasserstein loss in PyTorch,
    using the coupling plan T optimized by the POT solver (treated as constant).
    Formula: trace(T C_t T^T C_s) expanded.
    """
    device = C_s.device
    T_tensor = torch.tensor(T, dtype=torch.float32, device=device)
    
    # Marginals
    p = T_tensor.sum(dim=1)
    q = T_tensor.sum(dim=0)
    
    # Constant terms
    const_s = torch.sum((C_s ** 2) * torch.outer(p, p))
    const_t = torch.sum((C_t ** 2) * torch.outer(q, q))
    
    # Cross term: trace(T C_t T^T C_s)
    cross_term = torch.matmul(T_tensor, C_t)
    cross_term = torch.matmul(cross_term, T_tensor.t())
    cross_term = torch.sum(cross_term * C_s)
    
    loss = const_s + const_t - 2.0 * cross_term
    return torch.clamp(loss, min=0.0)

def graph_reconstruction_loss(node_emb: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, neg_ratio: float = 1.0) -> torch.Tensor:
    """
    Graph Autoencoder (GAE) reconstruction loss.
    Attempts to reconstruct the adjacency matrix of a graph from node embeddings.
    Uses negative sampling to balance sparse link prediction.
    """
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=node_emb.device, requires_grad=True)

    # Positive samples: actual edges
    src, dst = edge_index
    pos_scores = torch.sum(node_emb[src] * node_emb[dst], dim=-1)
    pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
    
    # Negative samples: randomly sampled non-edges
    num_neg = int(edge_index.size(1) * neg_ratio)
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=node_emb.device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=node_emb.device)
    
    neg_scores = torch.sum(node_emb[neg_src] * node_emb[neg_dst], dim=-1)
    neg_loss = -torch.log(1.0 - torch.sigmoid(neg_scores) + 1e-8).mean()
    
    return pos_loss + neg_loss
