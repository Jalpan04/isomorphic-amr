import os
import torch
print("VERY START: CUDA available:", torch.cuda.is_available())
print("VERY START: os.environ CUDA:", {k: v for k, v in os.environ.items() if "CUDA" in k or "GPU" in k})

print("Starting extract_embeddings.py...")
print("Loading standard libraries...")
import sys
from pathlib import Path

print("Loading heavy ML libraries (tqdm, torch)...")
from tqdm import tqdm
import torch

print("Loading internal packages...")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.embedder import XLMREmbedder
from src.utils.logger import setup_logger
from loguru import logger
print("All libraries loaded successfully. Starting execution...")

BATCH_SIZE = 64  # Number of graphs to process per GPU forward pass

def process_directory(embedder: XLMREmbedder, directory_path: Path, node_type: str):
    """
    Load PyG graphs from a directory, extract XLM-R embeddings for concepts/tokens,
    overwrite data.x with the dense embeddings, and resave the file.

    node_type: "amr" (use metadata['concepts']) or "dep" (use metadata['tokens'])

    Resume logic: if data.x is already float32 with dim 768, it is already embedded - skip it.
    Batching: collect tokens from BATCH_SIZE graphs, run one GPU forward pass, write back.
    """
    if not directory_path.exists():
        logger.warning(f"Directory {directory_path} does not exist. Skipping.")
        return

    paths = sorted(list(directory_path.glob("*.pt")))

    # Split paths into chunks of BATCH_SIZE
    chunks = [paths[i:i + BATCH_SIZE] for i in range(0, len(paths), BATCH_SIZE)]
    logger.info(f"Processing {len(paths)} graphs in {directory_path} ({len(chunks)} batches of {BATCH_SIZE})...")

    skipped_resume = 0
    processed = 0

    with tqdm(total=len(paths), desc=directory_path.name, unit="graph") as pbar:
        for chunk in chunks:
            # Load all graphs in this chunk
            loaded = []
            for path in chunk:
                try:
                    data = torch.load(path, map_location="cpu", weights_only=False)
                    loaded.append((path, data))
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
                    loaded.append((path, None))

            # Filter out already-embedded graphs (resume logic)
            to_embed = []   # (chunk_index, path, data, labels)
            for path, data in loaded:
                if data is None:
                    pbar.update(1)
                    continue
                # Check if already embedded: x is float32 with 768 features
                if (data.x is not None and
                        data.x.dtype == torch.float32 and
                        data.x.ndim == 2 and
                        data.x.shape[1] == 768):
                    skipped_resume += 1
                    pbar.update(1)
                    continue

                if node_type == "amr":
                    labels = data.metadata.get("concepts", [])
                else:
                    labels = data.metadata.get("tokens", [])

                if not labels:
                    pbar.update(1)
                    continue

                to_embed.append((path, data, labels))

            if not to_embed:
                continue

            # Flatten all labels from this batch into one big list
            flat_labels = []
            label_counts = []
            for _, _, labels in to_embed:
                flat_labels.extend(labels)
                label_counts.append(len(labels))

            # Single GPU forward pass for all nodes in this batch
            try:
                all_embeddings = embedder.embed_texts(flat_labels)
            except Exception as e:
                logger.error(f"Embedding failed for batch: {e}")
                pbar.update(len(to_embed))
                continue

            # Split embeddings back per graph and save
            offset = 0
            for (path, data, labels), count in zip(to_embed, label_counts):
                data.x = all_embeddings[offset: offset + count].clone()
                offset += count
                try:
                    temp_path = path.with_suffix(".tmp")
                    torch.save(data, temp_path)
                    os.replace(temp_path, path)
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to save {path}: {e}")
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception:
                            pass
                pbar.update(1)

    logger.success(f"{directory_path.name}: {processed} embedded, {skipped_resume} already done (resumed).")

def main():
    setup_logger("preprocessing")
    logger.info("Starting XLM-R dense node embedding extraction...")
    
    # Initialize embedder
    embedder = XLMREmbedder()
    
    # Skip English AMR graphs (already embedded)
    # for split in ["train", "dev", "test"]:
    #     process_directory(
    #         embedder=embedder,
    #         directory_path=Path(f"data/processed/amr_graphs_pyg/{split}"),
    #         node_type="amr"
    #     )
        
    # Process multilingual target dependency graphs
    for lang in ["es", "de", "it", "zh"]:
        process_directory(
            embedder=embedder,
            directory_path=Path(f"data/processed/dep_graphs_pyg/{lang}"),
            node_type="dep"
        )
        
    logger.success("XLM-R node embedding extraction complete.")

if __name__ == "__main__":
    main()
