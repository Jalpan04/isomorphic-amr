import torch
import numpy as np
import ot   # Python Optimal Transport
from loguru import logger

def compute_distance_matrix(embeddings: torch.Tensor, metric: str = "cosine") -> np.ndarray:
    """
    Compute pairwise distance matrix for a set of embeddings.
    Args:
        embeddings: (N, d) tensor
        metric: "cosine" or "euclidean"
    Returns:
        C: (N, N) numpy array of pairwise distances, normalized to [0, 1]
    """
    emb = embeddings.float().cpu().numpy()
    if len(emb) == 0:
        return np.zeros((0, 0), dtype=np.float64)
        
    if metric == "cosine":
        # Normalize and compute dot product
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb_n = emb / norms
        sim = emb_n @ emb_n.T
        C = 1.0 - sim   # cosine distance
    else:
        # Euclidean distance
        diff = emb[:, None, :] - emb[None, :, :]
        C = np.sqrt((diff ** 2).sum(-1))

    # Normalize to [0, 1]
    C_max = C.max()
    if C_max > 1e-8:
        C = C / C_max
    else:
        C = np.zeros_like(C)
        
    return C.astype(np.float64)

def gromov_wasserstein_align(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    metric: str = "cosine",
    loss_fun: str = "square_loss",
    log: bool = True
) -> tuple[np.ndarray, dict]:
    """
    Compute Gromov-Wasserstein alignment between two embedding sets.
    Args:
        source_emb: (n, d) embeddings
        target_emb: (m, d) embeddings
        metric:     distance metric for intra-space distances ("cosine" or "euclidean")
        loss_fun:   GW loss function ("square_loss" or "kl_loss")
        log:        whether to return log info
    Returns:
        T:    (n, m) soft transport plan coupling matrix
        info: log dictionary
    """
    n = source_emb.shape[0]
    m = target_emb.shape[0]
    
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64), {}

    # 1. Compute intra-space distance matrices
    logger.debug(f"Computing distance matrices: source={n}, target={m}")
    C_s = compute_distance_matrix(source_emb, metric)
    C_t = compute_distance_matrix(target_emb, metric)

    # 2. Assign uniform distributions over source and target
    p = ot.unif(n)
    q = ot.unif(m)

    # 3. Solve Gromov-Wasserstein
    logger.debug("Running Gromov-Wasserstein optimization...")
    try:
        T, log_dict = ot.gromov.gromov_wasserstein(
            C_s, C_t, p, q,
            loss_fun=loss_fun,
            log=log,
            verbose=False
        )
        gw_loss = log_dict.get("gw_dist", float("nan"))
        logger.debug(f"GW alignment complete. loss: {gw_loss:.6f}")
    except Exception as e:
        logger.error(f"GW optimization failed: {e}")
        T = np.outer(p, q) # uniform fallback
        log_dict = {"error": str(e)}

    return T, log_dict

def fused_gromov_wasserstein_align(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    alpha: float = 0.5,
    metric: str = "cosine",
    loss_fun: str = "square_loss",
    log: bool = True
) -> tuple[np.ndarray, dict]:
    """
    Compute Fused Gromov-Wasserstein alignment which combines:
    1. Node feature distances (e.g. cross-lingual XLM-R feature matrix M)
    2. Intra-graph metric distances (C_s, C_t)
    """
    n = source_emb.shape[0]
    m = target_emb.shape[0]
    
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64), {}
        
    # 1. Compute intra-space metric distance matrices
    C_s = compute_distance_matrix(source_emb, metric)
    C_t = compute_distance_matrix(target_emb, metric)
    
    # 2. Compute inter-space feature distance matrix (M)
    sf = source_features.float().cpu().numpy()
    tf = target_features.float().cpu().numpy()
    
    if metric == "cosine":
        sf_norm = sf / (np.linalg.norm(sf, axis=1, keepdims=True) + 1e-8)
        tf_norm = tf / (np.linalg.norm(tf, axis=1, keepdims=True) + 1e-8)
        M = 1.0 - (sf_norm @ tf_norm.T)
    else:
        diff = sf[:, None, :] - tf[None, :, :]
        M = np.sqrt((diff ** 2).sum(-1))
        
    M_max = M.max()
    if M_max > 1e-8:
        M = M / M_max
    else:
        M = np.zeros_like(M)

    # 3. Distributions
    p = ot.unif(n)
    q = ot.unif(m)

    # 4. Solve Fused Gromov-Wasserstein
    try:
        T, log_dict = ot.gromov.fused_gromov_wasserstein(
            M, C_s, C_t, p, q,
            loss_fun=loss_fun,
            alpha=alpha,
            log=log,
            verbose=False
        )
        fgw_loss = log_dict.get("fgw_dist", float("nan"))
        logger.debug(f"FGW alignment complete. loss: {fgw_loss:.6f}")
    except Exception as e:
        logger.error(f"FGW optimization failed: {e}")
        T = np.outer(p, q) # uniform fallback
        log_dict = {"error": str(e)}
        
    return T, log_dict

def project_embeddings(
    target_emb: torch.Tensor,
    T: np.ndarray
) -> torch.Tensor:
    """
    Project target embeddings into source space using the transport plan T.
    Here T has shape (n_source, m_target).
    To map target to source, we use the normalized transport plan.
    Specifically, T[i, j] indicates how much source node i maps to target node j.
    We normalize across source nodes so that target nodes reconstruct source states.
    
    Returns:
        projected: (n_source, d) projected embeddings in the target space.
    """
    T_tensor = torch.tensor(T, dtype=torch.float32)
    # Row normalize to get soft projection coefficients (n_source, m_target)
    T_norm = T_tensor / (T_tensor.sum(dim=1, keepdim=True) + 1e-8)
    # Project: (n_source, m_target) @ (m_target, d) -> (n_source, d)
    projected = T_norm @ target_emb.float().cpu()
    return projected
