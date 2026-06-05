# Research Preparation: Unsupervised Cross-Lingual Semantic Parsing via Isomorphic Graph Projection

This document establishes the roadmap, mathematical framework, directory structure, and environment setup for conducting experiments on Unsupervised Cross-Lingual Semantic Parsing.

---

## 1. Research Topic Overview

**Title**: Unsupervised Cross-Lingual Semantic Parsing via Isomorphic Graph Projection
**Core Objective**: Map Abstract Meaning Representation (AMR) structures across languages without parallel translation corpora by aligning latent graph isometries in a shared cross-lingual embedding space.

### Mathematical Formulation
Let $G_X = (V_X, E_X, W_X)$ be the semantic/syntactic graph constructed from a sentence in the source language $X$ (e.g., Spanish, German), and let $G_Y = (V_Y, E_Y, W_Y)$ be the target AMR graph in the universal semantic space (typically English-centric AMR concepts).

We represent the node attributes (embeddings) of the source graph as $H_X \in \mathbb{R}^{|V_X| \times d}$ and the target AMR graph concepts as $H_Y \in \mathbb{R}^{|V_Y| \times d}$, both residing in a shared cross-lingual embedding space (e.g., XLM-RoBERTa, mBERT).

We define intra-graph structural distance matrices $D_X \in \mathbb{R}^{|V_X| \times |V_X|}$ and $D_Y \in \mathbb{R}^{|V_Y| \times |V_Y|}$ using shortest path distances, heat kernels, or adjacency matrices.

Our goal is to find an optimal transport alignment matrix (coupling) $C \in \mathbb{R}^{|V_X| \times |V_Y|}$ that minimizes the Fused Gromov-Wasserstein (FGW) distance:

$$FGW(D_X, D_Y, H_X, H_Y) = \inf_{C \in \Pi(\mu, \nu)} \left( (1-\alpha) \sum_{i,j} d(H_X[i], H_Y[j]) C_{i,j} + \alpha \sum_{i,j,k,l} L(D_X[i,k], D_Y[j,l]) C_{i,j} C_{k,l} \right)$$

where:
* $\Pi(\mu, \nu)$ is the set of all coupling matrices matching the marginal distributions $\mu$ and $\nu$.
* $d(H_X[i], H_Y[j])$ is the distance between the node embeddings of node $i$ in $G_X$ and node $j$ in $G_Y$.
* $L(D_X[i,k], D_Y[j,l])$ is a loss function (e.g., squared difference) comparing the pairwise topological distances within each graph.
* $\alpha \in [0, 1]$ is a trade-off parameter balancing semantic similarity and structural alignment.

Once $C$ is optimized in an unsupervised manner, we use it to project source graphs onto target AMR structures.

---

## 2. Directory Layout

To maintain a clean and structured codebase, the repository is organized as follows:

```text
nlp_research/
│
├── agent.md                        <-- This research roadmap
├── requirements.txt                <-- Python dependencies
│
├── data/                           <-- Dataset storage
│   ├── raw/                        <-- Original AMR corpus (LDC format, Europarl, etc.)
│   └── processed/                  <-- Preprocessed graphs, vocabulary, and embeddings
│
├── src/                            <-- Source code
│   ├── __init__.py
│   ├── data_loader.py              <-- Loading and parsing AMR graphs (Penman parsing)
│   ├── graph_builder.py            <-- Building dependency graphs from source sentences
│   ├── embedder.py                 <-- Generating shared cross-lingual node embeddings
│   ├── alignment.py                <-- Fused Gromov-Wasserstein & Procrustes alignment algorithms
│   ├── projection.py               <-- Projecting source graphs to AMR structures
│   └── evaluation.py               <-- Smatch score and graph isometry metrics
│
├── scripts/                        <-- Command-line run scripts
│   ├── preprocess.py               <-- Preprocess data and generate graph metrics
│   ├── run_alignment.py            <-- Run graph alignment experiments
│   └── evaluate.py                 <-- Calculate evaluation metrics on validation/test sets
│
└── notebooks/                      <-- Jupyter Notebooks for exploratory data analysis
    └── exploratory_analysis.ipynb
```

---

## 3. Recommended Pipeline Phases

### Phase 1: Environment & Dependency Setup
* Set up a virtual environment (venv/conda).
* Install dependencies listed in `requirements.txt`.
* Download necessary spacy language models (e.g., `es_core_news_md` for Spanish, `de_core_news_md` for German) and pretrained cross-lingual transformer models.

### Phase 2: Data & Graph Preprocessing
* Load target English AMR graphs using `penman`.
* Load source sentences and build source syntactic dependency graphs using `spacy` or dependency parsers.
* Compute pairwise distance matrices (shortest-path/diffusion) for both source dependency graphs and target AMR graphs.
* Extract node representations using a cross-lingual encoder (e.g., XLM-R).

### Phase 3: Unsupervised Graph Alignment (Isomorphism Matching)
* Implement Fused Gromov-Wasserstein (FGW) optimization using the `POT` (Python Optimal Transport) library.
* Experiment with unsupervised metric-space projection mappings (e.g., Orthogonal Procrustes on the node embeddings).
* Define objective functions to optimize the projection map without parallel sentences.

### Phase 4: Structural Projection & Graph Generation
* Use the learned mapping matrix $C$ to project the source nodes to AMR concepts.
* Map edge connections and relation labels.
* Use `penman` to format and generate the predicted AMR graphs.

### Phase 5: Evaluation & Validation
* Evaluate projection quality using the `Smatch` metric.
* Compute graph isomorphism similarity metrics.
* Establish baseline comparisons:
  1. Translate-then-parse (machine translation + English AMR parser).
  2. Direct cross-lingual zero-shot parsing.

---

## 4. Next Steps

1. **Verify Python Environment**: Prepare to run `pip install -r requirements.txt` once environment details are established.
2. **Review incoming document**: Awaiting the user's detailed specification sheet to refine alignment objectives, target languages, and specific dataset paths.
