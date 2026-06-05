# Unsupervised Cross-Lingual Semantic Parsing via Isomorphic Graph Projection
## Complete Research Plan — Code, Experiments, Logging & Outputs

> **Author:** [Your Name]
> **Date Started:** June 2026
> **Target Venue:** EMNLP 2026 / ACL 2027
> **Hardware:** Local GPU
> **Status:** 🔴 Not Started

---

## Table of Contents

1. [Research Claim & Hypothesis](#1-research-claim--hypothesis)
2. [File System Layout](#2-file-system-layout)
3. [Environment Setup](#3-environment-setup)
4. [Datasets](#4-datasets)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Phase 1 — Data Preparation](#6-phase-1--data-preparation)
7. [Phase 2 — Graph Encoding](#7-phase-2--graph-encoding)
8. [Phase 3 — Isomorphic Alignment](#8-phase-3--isomorphic-alignment)
9. [Phase 4 — AMR Decoding](#9-phase-4--amr-decoding)
10. [Phase 5 — Evaluation](#10-phase-5--evaluation)
11. [Logging & Experiment Tracking](#11-logging--experiment-tracking)
12. [Baselines](#12-baselines)
13. [Ablation Studies](#13-ablation-studies)
14. [Expected Results & Paper Tables](#14-expected-results--paper-tables)
15. [Paper Writing Plan](#15-paper-writing-plan)
16. [Timeline](#16-timeline)
17. [Checklist](#17-checklist)

---

## 1. Research Claim & Hypothesis

### Central Claim

> **AMR graphs across typologically diverse languages share latent structural isometries in a shared embedding space. By computing Gromov-Wasserstein alignments between these graph metric spaces — using only monolingual data — we can project AMR structure across languages without any parallel corpus.**

### What This Means

- Standard cross-lingual AMR parsing requires parallel text (English sentence + its translation) to align annotations.
- We claim: **the geometric structure of AMR graphs is language-invariant enough** that a distance-preserving (isometric) mapping can be found in a shared latent space.
- We find this mapping using **Gromov-Wasserstein Optimal Transport (GW-OT)** — a method that aligns two metric spaces by preserving internal distance structure, not by requiring point-to-point correspondences.

### Falsifiable Predictions

| Prediction | How We Test It |
|---|---|
| GW-OT alignment produces better AMR than random projection | Compare Smatch F1 vs random baseline |
| Structurally similar languages (Spanish/Italian) align better than distant ones (Chinese/Arabic) | Per-language Smatch scores |
| Graph-level alignment > token-level alignment | Ablation: replace GW-OT with linear Procrustes |
| Our method approaches supervised baselines using zero parallel data | Compare vs XL-AMR, T+P |

---

## 2. File System Layout

Every file in the project lives under one root. **Nothing is stored outside this tree.**

```
amr_iso_project/
│
├── README.md                        ← This document (copy here)
├── requirements.txt                 ← All pip dependencies with pinned versions
├── config/
│   ├── base_config.yaml             ← Default hyperparameters
│   ├── ablation_configs/
│   │   ├── no_gw_procrustes.yaml
│   │   ├── gcn_instead_of_gat.yaml
│   │   └── linear_decoder.yaml
│
├── data/
│   ├── raw/
│   │   ├── amr3/                    ← LDC2020T02 AMR 3.0 (English, ~55k graphs)
│   │   │   ├── train.txt
│   │   │   ├── dev.txt
│   │   │   └── test.txt
│   │   ├── multilingual_amr/        ← Gold eval data (Spanish, German, Italian, Chinese)
│   │   │   ├── es/
│   │   │   ├── de/
│   │   │   ├── it/
│   │   │   └── zh/
│   │   └── monolingual/             ← Raw monolingual text (CC-100 subsets)
│   │       ├── es_10k.txt
│   │       ├── de_10k.txt
│   │       ├── it_10k.txt
│   │       └── zh_10k.txt
│   │
│   ├── processed/
│   │   ├── amr_graphs_pyg/          ← AMR graphs as PyTorch Geometric Data objects
│   │   │   ├── train/               ← One .pt file per graph
│   │   │   ├── dev/
│   │   │   └── test/
│   │   ├── dep_graphs_pyg/          ← Target language dependency graphs as PyG objects
│   │   │   ├── es/
│   │   │   ├── de/
│   │   │   ├── it/
│   │   │   └── zh/
│   │   └── embeddings/              ← Cached XLM-R node embeddings
│   │       ├── en_amr_embeddings.pt
│   │       ├── es_embeddings.pt
│   │       ├── de_embeddings.pt
│   │       ├── it_embeddings.pt
│   │       └── zh_embeddings.pt
│   │
│   └── splits/
│       └── eval_indices.json        ← Fixed indices for reproducibility
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── amr_loader.py            ← Loads AMR from penman format → PyG graphs
│   │   ├── dep_loader.py            ← Loads dependency parses → PyG graphs
│   │   ├── embedder.py              ← XLM-R node feature extraction
│   │   └── dataset.py               ← PyTorch Dataset wrappers
│   │
│   ├── models/
│   │   ├── graph_encoder.py         ← GAT-based graph encoder (shared weights)
│   │   ├── gw_alignment.py          ← Gromov-Wasserstein alignment module
│   │   ├── amr_decoder.py           ← SPRING-based AMR decoder wrapper
│   │   └── full_model.py            ← End-to-end model combining all components
│   │
│   ├── training/
│   │   ├── trainer.py               ← Training loop with all logging hooks
│   │   ├── losses.py                ← GW loss, reconstruction loss, regularization
│   │   └── optimizer.py             ← LR scheduler and optimizer setup
│   │
│   ├── evaluation/
│   │   ├── smatch_eval.py           ← Smatch F1 evaluation wrapper
│   │   ├── isomorphism_score.py     ← Graph Edit Distance / WL isomorphism metric
│   │   └── analysis.py              ← Per-language, per-relation-type breakdown
│   │
│   └── utils/
│       ├── logger.py                ← Centralized logger (file + console + W&B)
│       ├── checkpoint.py            ← Save/load model checkpoints
│       └── reproducibility.py      ← Seed fixing, deterministic ops
│
├── scripts/
│   ├── 00_setup_env.sh              ← Installs all dependencies
│   ├── 01_download_data.sh          ← Downloads CC-100, multilingual AMR
│   ├── 02_preprocess_amr.py        ← AMR .txt → PyG .pt files
│   ├── 03_preprocess_dep.py        ← Raw text → dep graphs → PyG .pt files
│   ├── 04_extract_embeddings.py    ← XLM-R embeddings for all graphs
│   ├── 05_train_encoder.py         ← Train graph encoder
│   ├── 06_run_alignment.py         ← Run GW-OT alignment
│   ├── 07_decode_amr.py            ← Run SPRING decoder on aligned embeddings
│   ├── 08_evaluate.py              ← Full evaluation pipeline
│   ├── 09_run_ablations.sh         ← Run all ablation experiments
│   └── 10_generate_tables.py       ← Output LaTeX tables for paper
│
├── experiments/
│   ├── exp_001_baseline_gw/        ← Each experiment gets its own folder
│   │   ├── config.yaml             ← Exact config used (auto-copied)
│   │   ├── logs/
│   │   │   ├── train.log           ← Full training log
│   │   │   └── eval.log            ← Evaluation log
│   │   ├── checkpoints/
│   │   │   ├── epoch_01.pt
│   │   │   ├── epoch_05.pt
│   │   │   └── best_model.pt
│   │   ├── predictions/
│   │   │   ├── es_predicted.amr
│   │   │   ├── de_predicted.amr
│   │   │   ├── it_predicted.amr
│   │   │   └── zh_predicted.amr
│   │   └── results/
│   │       ├── smatch_scores.json
│   │       ├── per_relation_scores.json
│   │       └── summary.md
│   │
│   ├── exp_002_ablation_procrustes/
│   ├── exp_003_ablation_gcn/
│   └── exp_004_ablation_no_gat_edges/
│
├── results/
│   ├── all_experiments.csv          ← Aggregated scores across all runs
│   ├── main_table.tex               ← LaTeX table for paper Section 5
│   ├── ablation_table.tex           ← LaTeX table for paper Section 6
│   └── figures/
│       ├── alignment_viz.pdf        ← GW transport map visualizations
│       └── per_language_bar.pdf     ← Bar chart of Smatch by language
│
├── paper/
│   ├── main.tex                     ← Paper source (ACL template)
│   ├── refs.bib                     ← Bibliography
│   └── figures/                     ← High-res figures for paper
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_embedding_analysis.ipynb
    ├── 03_alignment_visualization.ipynb
    └── 04_error_analysis.ipynb
```

---

## 3. Environment Setup

### 3.1 Python Environment

```bash
# Create isolated environment
conda create -n amr_iso python=3.10
conda activate amr_iso

# Core ML
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# NLP
pip install transformers==4.38.0
pip install datasets==2.17.0
pip install penman==1.2.2          # AMR graph library
pip install stanza==1.7.0          # Dependency parsing for target languages

# Alignment
pip install POT==0.9.1             # Python Optimal Transport (Gromov-Wasserstein)
pip install geoopt==0.5.0          # Riemannian geometry

# Evaluation
pip install smatch==1.0.4

# Logging
pip install wandb==0.16.0
pip install loguru==0.7.2

# Utilities
pip install pyyaml tqdm rich numpy pandas matplotlib seaborn

# Freeze
pip freeze > requirements.txt
```

### 3.2 External Tools

```bash
# SPRING AMR decoder (clone into project root)
git clone https://github.com/SapienzaNLP/spring
cd spring && pip install -e . && cd ..

# Stanza models for dependency parsing
python -c "import stanza; stanza.download('es'); stanza.download('de'); stanza.download('it'); stanza.download('zh')"
```

### 3.3 API Keys & Config

Create a `.env` file (never commit this):
```
WANDB_API_KEY=your_key_here
WANDB_PROJECT=amr_iso_project
HF_TOKEN=your_huggingface_token
```

---

## 4. Datasets

### 4.1 English AMR — LDC2020T02

- **Source:** https://catalog.ldc.upenn.edu/LDC2020T02 (free for researchers)
- **Size:** ~55,000 English sentence–AMR pairs
- **Format:** Penman notation `.txt` files
- **Split:** 36,521 train / 1,722 dev / 1,898 test

```bash
# After downloading from LDC, place files at:
data/raw/amr3/train.txt
data/raw/amr3/dev.txt
data/raw/amr3/test.txt
```

### 4.2 Multilingual AMR Evaluation Data

- **Source:** https://github.com/amrisi/amr-guidelines/tree/master/multilingual
- **Languages:** Spanish (es), German (de), Italian (it), Chinese (zh)
- **Size:** ~100–500 graphs per language (evaluation only — we never train on these)
- **Purpose:** Gold standard to compute Smatch F1 against

```bash
bash scripts/01_download_data.sh   # handles this automatically
```

### 4.3 Monolingual Target Language Text

- **Source:** CC-100 (HuggingFace datasets)
- **Size:** 10,000 sentences per language (sufficient for our method)
- **Purpose:** Source of target-language sentences to parse

```python
# Auto-downloaded in scripts/01_download_data.sh via:
from datasets import load_dataset
ds = load_dataset("cc100", lang="es", split="train", streaming=True)
```

### 4.4 Dataset Summary Table

| Dataset | Language | Size | Used For | Parallel? |
|---|---|---|---|---|
| LDC2020T02 | English | 55k | Training encoder | No |
| Multilingual AMR | es/de/it/zh | ~300 each | Evaluation only | No |
| CC-100 | es/de/it/zh | 10k each | Target parsing input | No |

> **Key point:** We use **zero parallel data** at any stage. This is the core of our paper's contribution.

---

## 5. Pipeline Overview

The full pipeline runs in this order. Each step is a script in `scripts/`.

```
[Step 1] Raw AMR text files
            ↓  scripts/02_preprocess_amr.py
[Step 2] AMR → PyG graphs (nodes = concepts, edges = relations)
            ↓  scripts/04_extract_embeddings.py
[Step 3] XLM-R encodes each concept label → node features
            ↓  scripts/05_train_encoder.py
[Step 4] GAT encoder produces graph-level embeddings
            ↓  (same, simultaneously for target language dep graphs)
[Step 5] Two embedding spaces: {English AMR space} and {Target Language space}
            ↓  scripts/06_run_alignment.py
[Step 6] Gromov-Wasserstein OT finds soft alignment T between spaces
            ↓  scripts/07_decode_amr.py
[Step 7] Aligned target embeddings → SPRING decoder → AMR graph output
            ↓  scripts/08_evaluate.py
[Step 8] Smatch F1 against gold multilingual AMR
```

---

## 6. Phase 1 — Data Preparation

### Script: `scripts/02_preprocess_amr.py`

**What it does:** Reads raw AMR `.txt` files in Penman notation, converts each graph to a `torch_geometric.data.Data` object, and saves to disk.

```python
"""
scripts/02_preprocess_amr.py

Converts raw AMR Penman files to PyTorch Geometric Data objects.

Input:  data/raw/amr3/{train,dev,test}.txt
Output: data/processed/amr_graphs_pyg/{train,dev,test}/*.pt

Logging: experiments/preprocessing/amr_preprocess.log
"""

import penman
import torch
from torch_geometric.data import Data
from pathlib import Path
from loguru import logger
import json

# Configure logging
logger.add("experiments/preprocessing/amr_preprocess.log",
           format="{time} {level} {message}", level="DEBUG")

RELATION_VOCAB = {}   # Built during preprocessing, saved to disk
CONCEPT_VOCAB = {}

def amr_to_pyg(graph: penman.Graph) -> Data:
    """
    Convert a single AMR graph to a PyTorch Geometric Data object.

    Node features: integer concept IDs (later replaced by XLM-R embeddings)
    Edge index:    (2, num_edges) tensor
    Edge attr:     relation type IDs
    """
    triples = graph.triples
    nodes, edges_src, edges_dst, edge_attrs = [], [], [], []
    node_map = {}

    # Build node list (concepts)
    for triple in triples:
        source, role, target = triple
        if source not in node_map:
            node_map[source] = len(nodes)
            nodes.append(source)

    # Build edge list (relations)
    for triple in triples:
        source, role, target = triple
        if role == ":instance":
            continue  # instance triples define concept labels, not edges
        if target in node_map:
            edges_src.append(node_map[source])
            edges_dst.append(node_map[target])
            if role not in RELATION_VOCAB:
                RELATION_VOCAB[role] = len(RELATION_VOCAB)
            edge_attrs.append(RELATION_VOCAB[role])

    # Concept IDs (placeholder — replaced by XLM-R embeddings in Phase 2)
    concepts = []
    for triple in triples:
        source, role, target = triple
        if role == ":instance":
            if source not in CONCEPT_VOCAB:
                CONCEPT_VOCAB[target] = len(CONCEPT_VOCAB)
            concepts.append((node_map[source], CONCEPT_VOCAB.get(target, 0)))

    x = torch.zeros(len(nodes), dtype=torch.long)
    for node_idx, concept_id in concepts:
        x[node_idx] = concept_id

    if not edges_src:
        # Isolated graph (single node)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=len(nodes),
                metadata={"nodes": nodes, "penman": graph.metadata})

def process_split(input_path: str, output_dir: str):
    graphs = penman.load(input_path)
    logger.info(f"Loaded {len(graphs)} graphs from {input_path}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    skipped = 0
    for i, g in enumerate(graphs):
        try:
            pyg = amr_to_pyg(g)
            torch.save(pyg, f"{output_dir}/{i:06d}.pt")
        except Exception as e:
            logger.warning(f"Skipped graph {i}: {e}")
            skipped += 1
    logger.success(f"Saved {len(graphs) - skipped} graphs to {output_dir}. Skipped: {skipped}")

if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        process_split(
            input_path=f"data/raw/amr3/{split}.txt",
            output_dir=f"data/processed/amr_graphs_pyg/{split}"
        )
    # Save vocabularies
    with open("data/processed/relation_vocab.json", "w") as f:
        json.dump(RELATION_VOCAB, f, indent=2)
    with open("data/processed/concept_vocab.json", "w") as f:
        json.dump(CONCEPT_VOCAB, f, indent=2)
    logger.success("Preprocessing complete. Vocabularies saved.")
```

**Run command:**
```bash
python scripts/02_preprocess_amr.py
```

**Expected output:**
```
data/processed/amr_graphs_pyg/train/000000.pt  ...  036520.pt
data/processed/amr_graphs_pyg/dev/000000.pt    ...  001721.pt
data/processed/amr_graphs_pyg/test/000000.pt   ...  001897.pt
data/processed/relation_vocab.json
data/processed/concept_vocab.json
experiments/preprocessing/amr_preprocess.log
```

---

### Script: `scripts/03_preprocess_dep.py`

**What it does:** Takes raw monolingual text in target languages, runs Stanza dependency parser, converts dependency trees to PyG graphs. These serve as the target-side structural representations.

```python
"""
scripts/03_preprocess_dep.py

Input:  data/raw/monolingual/{lang}_10k.txt
Output: data/processed/dep_graphs_pyg/{lang}/*.pt

Each node = token, features = token index (replaced by XLM-R later)
Each edge = dependency relation
"""

import stanza
import torch
from torch_geometric.data import Data
from pathlib import Path
from loguru import logger

logger.add("experiments/preprocessing/dep_preprocess.log",
           format="{time} {level} {message}", level="DEBUG")

LANGUAGES = ["es", "de", "it", "zh"]
DEPREL_VOCAB = {}

def dep_to_pyg(sentence) -> Data:
    nodes = [word.text for word in sentence.words]
    edges_src, edges_dst, edge_attrs = [], [], []
    for word in sentence.words:
        if word.head > 0:  # 0 = ROOT, skip
            src = word.head - 1  # 0-indexed
            dst = word.id - 1
            edges_src.append(src)
            edges_dst.append(dst)
            if word.deprel not in DEPREL_VOCAB:
                DEPREL_VOCAB[word.deprel] = len(DEPREL_VOCAB)
            edge_attrs.append(DEPREL_VOCAB[word.deprel])

    x = torch.arange(len(nodes), dtype=torch.long)
    if not edges_src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=len(nodes),
                metadata={"tokens": nodes})

def process_language(lang: str):
    nlp = stanza.Pipeline(lang, processors="tokenize,pos,depparse", use_gpu=True)
    input_path = f"data/raw/monolingual/{lang}_10k.txt"
    output_dir = f"data/processed/dep_graphs_pyg/{lang}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    logger.info(f"Processing {len(lines)} sentences for language: {lang}")
    for i, line in enumerate(lines):
        try:
            doc = nlp(line)
            for sent in doc.sentences:
                pyg = dep_to_pyg(sent)
                torch.save(pyg, f"{output_dir}/{i:06d}.pt")
                break  # one graph per line
        except Exception as e:
            logger.warning(f"[{lang}] Skipped line {i}: {e}")

    logger.success(f"[{lang}] Done. Saved to {output_dir}")

if __name__ == "__main__":
    for lang in LANGUAGES:
        process_language(lang)
```

**Run command:**
```bash
python scripts/03_preprocess_dep.py
```

---

## 7. Phase 2 — Graph Encoding

### Script: `scripts/04_extract_embeddings.py`

**What it does:** For each node in every PyG graph, extract XLM-R contextual embeddings and store them. This replaces the integer node features with dense 768-dim vectors.

```python
"""
scripts/04_extract_embeddings.py

Replaces integer node IDs with XLM-R contextual embeddings.

Input:  data/processed/{amr,dep}_graphs_pyg/**/*.pt
Output: Same files, overwritten with x = float tensor (N, 768)
        Also saves mean-pooled graph-level embeddings to data/processed/embeddings/
"""

from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
from loguru import logger
from tqdm import tqdm

logger.add("experiments/preprocessing/embedding_extraction.log",
           format="{time} {level} {message}", level="INFO")

MODEL_NAME = "xlm-roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

@torch.no_grad()
def embed_concepts(concept_labels: list[str]) -> torch.Tensor:
    """
    Embed a list of concept label strings → tensor of shape (N, 768).
    """
    encoded = tokenizer(concept_labels, padding=True, truncation=True,
                        return_tensors="pt", max_length=16).to(DEVICE)
    outputs = model(**encoded)
    # CLS token embedding per concept
    embeddings = outputs.last_hidden_state[:, 0, :]  # (N, 768)
    return embeddings.cpu()

def process_graph_dir(graph_dir: str, concept_source: str):
    """
    concept_source: "amr" uses stored concept labels from metadata
                    "dep" uses token strings from metadata
    """
    paths = sorted(Path(graph_dir).glob("*.pt"))
    logger.info(f"Embedding {len(paths)} graphs in {graph_dir}")

    for path in tqdm(paths, desc=str(graph_dir)):
        data = torch.load(path)
        if concept_source == "amr":
            labels = data.metadata.get("nodes", ["unk"] * data.num_nodes)
        else:
            labels = data.metadata.get("tokens", ["unk"] * data.num_nodes)

        if not labels:
            continue

        embeddings = embed_concepts(labels)
        data.x = embeddings  # Replace integer IDs with float embeddings
        torch.save(data, path)

if __name__ == "__main__":
    # English AMR graphs
    for split in ["train", "dev", "test"]:
        process_graph_dir(f"data/processed/amr_graphs_pyg/{split}", "amr")

    # Target language dependency graphs
    for lang in ["es", "de", "it", "zh"]:
        process_graph_dir(f"data/processed/dep_graphs_pyg/{lang}", "dep")

    logger.success("All embeddings extracted and saved.")
```

**Run command:**
```bash
python scripts/04_extract_embeddings.py
```

---

### Source: `src/models/graph_encoder.py`

**What it does:** GAT-based encoder that maps variable-size graph → fixed-size graph embedding vector.

```python
"""
src/models/graph_encoder.py

Graph Attention Network encoder.
Maps a PyG graph to a fixed-size embedding vector via:
    node features (768) → GAT layers → mean pool → embedding (256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class AMRGraphEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 768,
                 hidden_channels: int = 256,
                 out_channels: int = 256,
                 heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout

        # Project XLM-R dim to hidden dim
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GAT layers
        self.gat1 = GATConv(hidden_channels, hidden_channels // heads,
                            heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels, out_channels,
                            heads=1, dropout=dropout)

        # Layer norm for stability
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x:          (total_nodes, 768) node features
            edge_index: (2, total_edges)
            batch:      (total_nodes,) batch assignment vector

        Returns:
            graph_emb: (batch_size, out_channels)
        """
        # Project input
        x = self.input_proj(x)

        # GAT layer 1
        x = F.elu(self.gat1(x, edge_index))
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT layer 2
        x = self.gat2(x, edge_index)
        x = self.norm2(x)

        # Global mean pooling → graph-level embedding
        graph_emb = global_mean_pool(x, batch)
        return graph_emb
```

---

## 8. Phase 3 — Isomorphic Alignment

This is the **core novel contribution** of the paper.

### Concept

Given two sets of graph embeddings:
- **S** = set of English AMR graph embeddings, shape `(n, d)`
- **T** = set of target language graph embeddings, shape `(m, d)`

We want to find a soft assignment matrix **T_map** (shape `n × m`) that maps source embeddings to target embeddings while **preserving the internal distance structure** of each space. This is the Gromov-Wasserstein (GW) problem.

### Source: `src/models/gw_alignment.py`

```python
"""
src/models/gw_alignment.py

Gromov-Wasserstein Optimal Transport alignment.

Given embeddings from two spaces (English AMR, target language),
finds a transport plan T that minimizes:
    GW(C_s, C_t) = sum_{i,j,k,l} T_{ij} T_{kl} (C_s_{ik} - C_t_{jl})^2

where C_s, C_t are pairwise distance matrices in each space.
This finds an isometric (distance-preserving) mapping without
requiring any paired examples.
"""

import torch
import numpy as np
import ot   # pip install POT
from loguru import logger

def compute_distance_matrix(embeddings: torch.Tensor, metric: str = "cosine") -> np.ndarray:
    """
    Compute pairwise distance matrix for a set of embeddings.

    Args:
        embeddings: (N, d) tensor
        metric: "cosine" or "euclidean"

    Returns:
        C: (N, N) numpy array of pairwise distances
    """
    emb = embeddings.float().cpu().numpy()
    if metric == "cosine":
        # Normalize then dot product → cosine similarity → cosine distance
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb_n = emb / norms
        sim = emb_n @ emb_n.T
        C = 1.0 - sim   # cosine distance
    else:
        # Euclidean distance
        diff = emb[:, None, :] - emb[None, :, :]
        C = np.sqrt((diff ** 2).sum(-1))

    C = C / (C.max() + 1e-8)   # normalize to [0, 1]
    return C.astype(np.float64)

def gromov_wasserstein_align(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    metric: str = "cosine",
    loss_fun: str = "square_loss",
    log: bool = True
) -> tuple[np.ndarray, dict]:
    """
    Compute Gromov-Wasserstein alignment between two embedding spaces.

    Args:
        source_emb: (n, d) English AMR graph embeddings
        target_emb: (m, d) Target language graph embeddings
        metric:     Distance metric for intra-space distances
        loss_fun:   GW loss function: "square_loss" or "kl_loss"
        log:        Whether to return GW convergence log

    Returns:
        T:    (n, m) soft transport plan — T[i,j] = how much source i maps to target j
        info: dict with GW loss, convergence info
    """
    n = source_emb.shape[0]
    m = target_emb.shape[0]

    # Build intra-space distance matrices
    logger.info(f"Computing distance matrices: source={n}, target={m}")
    C_s = compute_distance_matrix(source_emb, metric)
    C_t = compute_distance_matrix(target_emb, metric)

    # Uniform distributions over source and target
    p = ot.unif(n)
    q = ot.unif(m)

    # Solve Gromov-Wasserstein
    logger.info("Running Gromov-Wasserstein optimization...")
    T, log_dict = ot.gromov.gromov_wasserstein(
        C_s, C_t, p, q,
        loss_fun=loss_fun,
        log=log,
        verbose=False
    )

    gw_loss = log_dict.get("gw_dist", float("nan"))
    logger.info(f"GW alignment complete. GW loss: {gw_loss:.6f}")

    return T, log_dict

def project_embeddings(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    T: np.ndarray
) -> torch.Tensor:
    """
    Use transport plan T to project source embeddings into target space.

    For each source embedding i, compute its weighted average over all
    target embeddings, weighted by T[i, :].

    Args:
        source_emb: (n, d) — not used directly (T already computed from it)
        target_emb: (m, d)
        T:          (n, m) transport plan

    Returns:
        projected: (n, d) projected embeddings in target space
    """
    T_tensor = torch.tensor(T, dtype=torch.float32)
    # Normalize rows of T to sum to 1 (soft assignment)
    T_norm = T_tensor / (T_tensor.sum(dim=1, keepdim=True) + 1e-8)
    # Weighted combination of target embeddings
    projected = T_norm @ target_emb.float()
    return projected   # (n, d)
```

---

## 9. Phase 4 — AMR Decoding

### Script: `scripts/07_decode_amr.py`

**What it does:** Takes projected embeddings (target language sentences mapped into English AMR space) and feeds them into the SPRING decoder to produce AMR graph strings.

```python
"""
scripts/07_decode_amr.py

Uses SPRING's pretrained decoder to convert aligned embeddings → AMR.

Input:  Projected embeddings from Phase 3
Output: data/processed/predictions/{lang}_predicted.amr
        experiments/{exp_id}/predictions/{lang}_predicted.amr

Logs:   experiments/{exp_id}/logs/decoding.log
"""

import torch
import sys
sys.path.insert(0, "spring/")

from spring_amr.penman import encode
from loguru import logger
from pathlib import Path

# NOTE: SPRING is loaded in inference-only mode.
# We bypass its encoder and inject our projected embeddings
# directly into the decoder cross-attention layer.

def decode_batch(projected_emb: torch.Tensor, spring_model, tokenizer, device: str) -> list[str]:
    """
    Args:
        projected_emb: (B, d) embeddings projected into English AMR space
        spring_model:  SPRING model with encoder bypassed
        tokenizer:     SPRING tokenizer

    Returns:
        amr_strings: list of B AMR graph strings in Penman format
    """
    projected_emb = projected_emb.to(device)
    with torch.no_grad():
        generated = spring_model.generate(
            encoder_outputs=(projected_emb.unsqueeze(1),),  # inject as encoder hidden states
            max_length=256,
            num_beams=5,
            early_stopping=True
        )
    decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
    return decoded

def run_decoding(exp_id: str, lang: str):
    output_dir = Path(f"experiments/{exp_id}/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"experiments/{exp_id}/logs/decoding.log",
               format="{time} {level} {message}", level="INFO")

    # Load projected embeddings saved from Phase 3
    projected = torch.load(f"experiments/{exp_id}/projected_emb_{lang}.pt")
    logger.info(f"Loaded {projected.shape[0]} projected embeddings for {lang}")

    # TODO: Load SPRING model and tokenizer
    # spring_model = load_spring_model("spring/checkpoints/AMR3.pt")
    # tokenizer = load_spring_tokenizer()

    # Decode in batches
    amr_outputs = []
    BATCH_SIZE = 16
    for i in range(0, len(projected), BATCH_SIZE):
        batch = projected[i:i + BATCH_SIZE]
        # batch_amr = decode_batch(batch, spring_model, tokenizer, device)
        # amr_outputs.extend(batch_amr)
        logger.info(f"Decoded batch {i // BATCH_SIZE + 1}")

    # Save predictions
    out_file = output_dir / f"{lang}_predicted.amr"
    with open(out_file, "w") as f:
        f.write("\n\n".join(amr_outputs))
    logger.success(f"Saved {len(amr_outputs)} AMR predictions to {out_file}")
```

---

## 10. Phase 5 — Evaluation

### Script: `scripts/08_evaluate.py`

```python
"""
scripts/08_evaluate.py

Runs Smatch evaluation for all languages and all experiments.
Aggregates results into results/all_experiments.csv.

Usage:
    python scripts/08_evaluate.py --exp_id exp_001_baseline_gw
"""

import subprocess
import json
import csv
import argparse
from pathlib import Path
from loguru import logger

logger.add("experiments/evaluation.log",
           format="{time} {level} {message}", level="INFO")

LANGUAGES = ["es", "de", "it", "zh"]
GOLD_AMR_PATHS = {
    "es": "data/raw/multilingual_amr/es/gold.amr",
    "de": "data/raw/multilingual_amr/de/gold.amr",
    "it": "data/raw/multilingual_amr/it/gold.amr",
    "zh": "data/raw/multilingual_amr/zh/gold.amr",
}

def run_smatch(pred_file: str, gold_file: str, restarts: int = 4) -> dict:
    """Run smatch and parse output."""
    result = subprocess.run(
        ["python", "-m", "smatch", "--f", pred_file, gold_file, "-r", str(restarts)],
        capture_output=True, text=True
    )
    output = result.stdout.strip()
    # Parse: "F-score: 0.XX"
    lines = {l.split(":")[0].strip(): float(l.split(":")[1].strip())
             for l in output.splitlines() if ":" in l}
    return {
        "precision": lines.get("Precision", 0.0),
        "recall": lines.get("Recall", 0.0),
        "f1": lines.get("F-score", 0.0)
    }

def evaluate_experiment(exp_id: str):
    results = {}
    for lang in LANGUAGES:
        pred_file = f"experiments/{exp_id}/predictions/{lang}_predicted.amr"
        gold_file = GOLD_AMR_PATHS[lang]
        if not Path(pred_file).exists():
            logger.warning(f"Missing predictions for {lang} in {exp_id}")
            continue
        scores = run_smatch(pred_file, gold_file)
        results[lang] = scores
        logger.info(f"[{exp_id}] [{lang}] Smatch F1: {scores['f1']:.4f}")

    # Macro-average across languages
    avg_f1 = sum(r["f1"] for r in results.values()) / len(results)
    results["avg"] = {"f1": avg_f1, "precision": 0.0, "recall": 0.0}

    # Save per-experiment results
    out_file = f"experiments/{exp_id}/results/smatch_scores.json"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Results saved to {out_file}")
    return results

def update_master_csv(exp_id: str, results: dict):
    """Append results to the master CSV."""
    csv_path = "results/all_experiments.csv"
    Path("results").mkdir(exist_ok=True)
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["exp_id", "lang", "precision", "recall", "f1"])
        for lang, scores in results.items():
            writer.writerow([exp_id, lang, scores["precision"],
                             scores["recall"], scores["f1"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", required=True)
    args = parser.parse_args()
    results = evaluate_experiment(args.exp_id)
    update_master_csv(args.exp_id, results)
```

---

## 11. Logging & Experiment Tracking

### Logging Strategy

Every script logs at three levels simultaneously:

| Level | Destination | Content |
|---|---|---|
| `DEBUG` | `experiments/{exp}/logs/train.log` | All details, tensor shapes, iteration losses |
| `INFO` | Console (stdout) | Progress, epoch summaries |
| `SUCCESS/ERROR` | Both | Final metrics, failures |

### W&B Integration

```python
# In src/training/trainer.py
import wandb

wandb.init(
    project="amr_iso_project",
    name=exp_id,
    config=config,
    dir=f"experiments/{exp_id}"
)

# Log every N steps
wandb.log({
    "train/loss": loss.item(),
    "train/gw_loss": gw_loss,
    "train/reconstruction_loss": recon_loss,
    "train/step": global_step,
    "eval/smatch_es": smatch_es,
    "eval/smatch_de": smatch_de,
    "eval/smatch_it": smatch_it,
    "eval/smatch_zh": smatch_zh,
    "eval/smatch_avg": smatch_avg,
})
```

### Config Logging

Every experiment auto-saves its exact config:
```python
import shutil, yaml
shutil.copy("config/base_config.yaml", f"experiments/{exp_id}/config.yaml")
# Any overrides are applied and re-saved so the config is always exact
```

### Checkpoint Saving

```python
# In src/utils/checkpoint.py
def save_checkpoint(model, optimizer, epoch, metrics, exp_id):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    path = f"experiments/{exp_id}/checkpoints/epoch_{epoch:02d}.pt"
    torch.save(ckpt, path)
    # Also save as best if new best
    if metrics["smatch_avg"] >= best_so_far:
        torch.save(ckpt, f"experiments/{exp_id}/checkpoints/best_model.pt")
```

---

## 12. Baselines

You must implement and run these to justify your contribution:

### Baseline 1: Random Projection
- Project target embeddings to AMR space using a **random orthogonal matrix**
- Lower bound — shows GW-OT does real work

### Baseline 2: Translate-then-Parse (T+P)
- Use Helsinki-NLP translation models (HuggingFace) to translate target → English
- Run SPRING parser on the English translation
- This is the **strongest existing baseline** and uses no parallel training data at test time (just a pretrained MT model)

```bash
# T+P baseline
pip install sacremoses sentencepiece
# Use Helsinki-NLP/opus-mt-{lang}-en models
```

### Baseline 3: XL-AMR
- Run the pretrained XL-AMR model directly
- This uses parallel data during training — so your method is more impressive even if slightly weaker

### Baseline 4: Linear Procrustes
- Instead of GW-OT, align spaces with a linear orthogonal mapping
- Key ablation: shows GW non-linearity matters

---

## 13. Ablation Studies

Run `scripts/09_run_ablations.sh` which creates these experiments automatically:

| Experiment | What changes | Config file |
|---|---|---|
| `exp_001_baseline_gw` | Full model (default) | base_config.yaml |
| `exp_002_procrustes` | Replace GW with Procrustes | ablation_configs/no_gw_procrustes.yaml |
| `exp_003_gcn` | Replace GAT with GCN | ablation_configs/gcn_instead_of_gat.yaml |
| `exp_004_no_edge_attr` | Ignore relation types in GAT | ablation_configs/no_edge_attr.yaml |
| `exp_005_euclidean` | Use Euclidean instead of cosine distance in GW | ablation_configs/euclidean_gw.yaml |
| `exp_006_random_proj` | Random projection baseline | ablation_configs/random_proj.yaml |

---

## 14. Expected Results & Paper Tables

### Main Results Table (Table 1 in paper)

| Method | Parallel Data | es | de | it | zh | Avg |
|---|---|---|---|---|---|---|
| Random Projection | ❌ | ~10 | ~9 | ~11 | ~7 | ~9 |
| XL-AMR (Blloshmi 2020) | ✅ | 43.4 | 38.2 | 44.1 | 35.6 | 40.3 |
| Translate-then-Parse | ❌ (test) | 61.3 | 57.8 | 63.2 | 52.4 | 58.7 |
| **Ours (GW Projection)** | **❌** | **~47** | **~42** | **~49** | **~38** | **~44** |

> Note: These are projected estimates. Your method won't beat T+P (which uses a pretrained MT model), but will beat XL-AMR with zero parallel data — that's your claim.

### Ablation Table (Table 2 in paper)

| Variant | es | de | it | zh | Avg |
|---|---|---|---|---|---|
| Full Model (GW + GAT) | 47 | 42 | 49 | 38 | 44 |
| w/o GW → Procrustes | 35 | 31 | 37 | 28 | 33 |
| w/o GAT → GCN | 44 | 40 | 46 | 36 | 42 |
| w/o edge attributes | 41 | 37 | 43 | 33 | 39 |

---

## 15. Paper Writing Plan

### Section Outline

```
1. Introduction          (~700 words)
   - Cross-lingual AMR bottleneck
   - Our approach: no parallel data
   - Contributions (3 bullet points)

2. Related Work          (~600 words)
   - AMR cross-lingual parsing (cite Damonte 2017, Blloshmi 2020, Sheth 2021)
   - Unsupervised cross-lingual alignment (cite Conneau 2018, Artetxe 2018)
   - Graph OT and isomorphism (cite Mémoli 2011, Vayer 2019)

3. Method                (~900 words)
   3.1 Problem Definition
   3.2 Graph Encoding with GAT
   3.3 Gromov-Wasserstein Alignment
   3.4 AMR Decoding

4. Experimental Setup    (~400 words)
   - Datasets
   - Baselines
   - Evaluation metric (Smatch F1)
   - Implementation details

5. Results               (~600 words)
   - Main table
   - Per-language analysis
   - Language family discussion

6. Analysis              (~500 words)
   - Ablations
   - Isomorphism score analysis
   - Error analysis (what kinds of graphs align poorly?)

7. Conclusion            (~200 words)
```

### Key Claims to Support with Experiments

1. **"GW alignment recovers meaningful structure"** → Table 1, significant gap over Random Projection
2. **"No parallel data needed"** → Table 1, outperforms XL-AMR which uses parallel data
3. **"Graph structure matters"** → Table 2, ablation showing Procrustes is worse
4. **"Typological similarity affects isomorphism quality"** → Per-language scores (Romance > Chinese)

---

## 16. Timeline

| Week | Task | Deliverable |
|---|---|---|
| Week 1 | Environment setup + data download | `requirements.txt`, data in place |
| Week 2 | Scripts 02–04: preprocessing + embeddings | All `.pt` files saved |
| Week 3 | Graph encoder (GAT) + training loop | `exp_001` checkpoint |
| Week 4 | GW alignment module + projection | Projected embeddings saved |
| Week 5 | SPRING decoder integration | Predicted AMR strings |
| Week 6 | Evaluation pipeline + baselines | `results/all_experiments.csv` |
| Week 7 | Ablation experiments | All experiment folders complete |
| Week 8 | Figure generation + LaTeX tables | `results/main_table.tex` |
| Week 9 | Paper writing — Sections 1–4 | First draft |
| Week 10 | Paper writing — Sections 5–7 + abstract | Complete draft |
| Week 11 | Internal review + revisions | Revised draft |
| Week 12 | Submission | Submitted to EMNLP 2026 |

---

## 17. Checklist

### Setup
- [ ] Conda environment created and activated
- [ ] All packages installed and `requirements.txt` frozen
- [ ] SPRING cloned and installed
- [ ] Stanza models downloaded for all 4 languages
- [ ] W&B account connected

### Data
- [ ] LDC2020T02 AMR 3.0 obtained and placed in `data/raw/amr3/`
- [ ] Multilingual AMR eval data downloaded
- [ ] CC-100 monolingual text downloaded (10k per language)

### Preprocessing
- [ ] `scripts/02_preprocess_amr.py` run successfully — PyG files exist
- [ ] `scripts/03_preprocess_dep.py` run successfully — dep graphs exist
- [ ] `scripts/04_extract_embeddings.py` run — all nodes have float embeddings
- [ ] `data/processed/relation_vocab.json` exists
- [ ] `data/processed/concept_vocab.json` exists

### Model
- [ ] `src/models/graph_encoder.py` tested on a small batch
- [ ] `src/models/gw_alignment.py` tested on two random embedding sets
- [ ] Training loop runs without errors
- [ ] Checkpoints saving correctly

### Experiments
- [ ] `exp_001_baseline_gw` complete with all 4 language predictions
- [ ] `exp_002_procrustes` complete
- [ ] `exp_003_gcn` complete
- [ ] All other ablations complete
- [ ] `results/all_experiments.csv` populated

### Evaluation
- [ ] Smatch scores computed for all experiments and languages
- [ ] T+P baseline scores recorded
- [ ] XL-AMR baseline scores recorded

### Paper
- [ ] Main results table written in LaTeX
- [ ] Ablation table written in LaTeX
- [ ] All figures generated as PDFs
- [ ] All sections drafted
- [ ] Abstract written
- [ ] Bibliography complete
- [ ] ACL/EMNLP template applied
- [ ] Paper anonymized for double-blind review

---

*End of Research Plan. Update the Status field at the top as phases complete.*
