import networkx as nx
import numpy as np
import scipy.linalg as la
from torch_geometric.data import Data

def pyg_to_networkx(pyg_data: Data) -> nx.Graph:
    """
    Convert a PyTorch Geometric Data graph to a NetworkX Graph.
    """
    edge_index = pyg_data.edge_index.cpu().numpy()
    num_nodes = pyg_data.num_nodes
    
    # Create undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        G.add_edge(u, v)
        
    return G

def compute_spectral_distance(G1: nx.Graph, G2: nx.Graph, max_eigenvalues: int = 10) -> float:
    """
    Compute the spectral distance between the normalized Laplacian eigenvalues of two graphs.
    Smaller spectral distance indicates higher topological isomorphism.
    """
    # Get normalized Laplacian eigenvalues
    L1 = nx.normalized_laplacian_matrix(G1).toarray() if G1.number_of_nodes() > 0 else np.zeros((0, 0))
    L2 = nx.normalized_laplacian_matrix(G2).toarray() if G2.number_of_nodes() > 0 else np.zeros((0, 0))
    
    evals1 = la.eigvalsh(L1) if L1.size > 0 else np.zeros(0)
    evals2 = la.eigvalsh(L2) if L2.size > 0 else np.zeros(0)
    
    # Pad to max_eigenvalues with zeros if necessary
    evals1_pad = np.zeros(max_eigenvalues)
    evals2_pad = np.zeros(max_eigenvalues)
    
    len1 = min(len(evals1), max_eigenvalues)
    len2 = min(len(evals2), max_eigenvalues)
    
    if len1 > 0:
        evals1_pad[:len1] = np.sort(evals1)[:len1]
    if len2 > 0:
        evals2_pad[:len2] = np.sort(evals2)[:len2]
        
    # Euclidean distance between normalized spectra
    spectral_dist = np.linalg.norm(evals1_pad - evals2_pad)
    return float(spectral_dist)

def compute_weisfeiler_lehman_match(G1: nx.Graph, G2: nx.Graph, iterations: int = 3) -> float:
    """
    Check if Weisfeiler-Lehman hashes match.
    Returns 1.0 if identical structural coloring, 0.0 otherwise.
    """
    try:
        h1 = nx.weisfeiler_lehman_graph_hash(G1, iterations=iterations)
        h2 = nx.weisfeiler_lehman_graph_hash(G2, iterations=iterations)
        return 1.0 if h1 == h2 else 0.0
    except Exception:
        return 0.0

def compute_graph_isomorphism_metrics(pyg_data1: Data, pyg_data2: Data) -> dict:
    """
    Compute multiple graph isomorphism metrics between two PyG graphs.
    """
    G1 = pyg_to_networkx(pyg_data1)
    G2 = pyg_to_networkx(pyg_data2)
    
    spectral_dist = compute_spectral_distance(G1, G2)
    wl_match = compute_weisfeiler_lehman_match(G1, G2)
    
    # Compare basic graph properties
    density_diff = abs(nx.density(G1) - nx.density(G2))
    node_diff = abs(G1.number_of_nodes() - G2.number_of_nodes())
    edge_diff = abs(G1.number_of_edges() - G2.number_of_edges())
    
    return {
        "spectral_distance": spectral_dist,
        "wl_isomorphic_match": wl_match,
        "density_difference": density_diff,
        "node_count_difference": node_diff,
        "edge_count_difference": edge_diff
    }
