print("Starting preprocess_amr.py...")
print("Loading standard libraries...")
import json
from pathlib import Path
import sys

print("Loading heavy ML libraries (datasets, penman, torch)...")
from datasets import load_dataset
import penman
import torch

print("Loading internal packages...")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.amr_loader import amr_to_pyg, parse_sentence_from_instruction
from src.utils.logger import setup_logger
from loguru import logger
print("All libraries loaded successfully. Starting execution...")

RELATION_VOCAB = {"unk": 0}
CONCEPT_VOCAB = {"unk": 0}

def main():
    setup_logger("preprocessing")
    logger.info("Starting English AMR Preprocessing using HF hoshuhan/amr-3-parsed...")

    # Load dataset from Hugging Face
    try:
        dataset = load_dataset("hoshuhan/amr-3-parsed")
    except Exception as e:
        logger.critical(f"Failed to load dataset: {e}")
        return

    # Check splits in the Hugging Face dataset
    # If the dataset only contains a single split (e.g., 'train'), we split it manually
    logger.info(f"Loaded dataset keys: {list(dataset.keys())}")
    
    if "train" in dataset and len(dataset.keys()) == 1:
        # Manual split following standard AMR 3.0 sizes
        full_data = list(dataset["train"])
        total_len = len(full_data)
        logger.info(f"Total rows in dataset: {total_len}")
        
        train_end = 36521
        dev_end = train_end + 1722
        
        splits = {
            "train": full_data[:train_end],
            "dev": full_data[train_end:dev_end],
            "test": full_data[dev_end:]
        }
    else:
        # Use existing splits if present (or fallback to manual split)
        splits = {}
        for split_name in ["train", "dev", "test"]:
            if split_name in dataset:
                splits[split_name] = list(dataset[split_name])
            elif split_name == "validation" in dataset:
                splits["dev"] = list(dataset["validation"])
                
        # If any split is missing, partition the train split
        if "train" not in splits or not splits["train"]:
            logger.critical("No train split found in dataset.")
            return
            
        if "dev" not in splits or not splits["dev"]:
            train_data = splits["train"]
            splits["train"] = train_data[:-2000]
            splits["dev"] = train_data[-2000:-1000]
            splits["test"] = train_data[-1000:]

    # Directory for output
    out_root = Path("data/processed/amr_graphs_pyg")
    
    for split_name, examples in splits.items():
        logger.info(f"Processing split '{split_name}' with {len(examples)} examples...")
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        skipped = 0
        for i, example in enumerate(examples):
            try:
                # hoshuhan/amr-3-parsed formatting:
                # Conversations turn 0 = user (with sentence), turn 1 = assistant (with Penman)
                convs = example["conversations"]
                user_msg = convs[0]["content"]
                assistant_msg = convs[1]["content"]
                
                sentence = parse_sentence_from_instruction(user_msg)
                amr_str = assistant_msg.strip()
                
                # Parse Penman graph
                graph = penman.decode(amr_str)
                
                # Inject sentence into metadata
                graph.metadata["snt"] = sentence
                
                # Convert to PyG Graph
                pyg_data = amr_to_pyg(graph, RELATION_VOCAB, CONCEPT_VOCAB)
                
                # Save Data object
                import os
                final_path = split_dir / f"{i:06d}.pt"
                temp_path = final_path.with_suffix(".tmp")
                torch.save(pyg_data, temp_path)
                os.replace(temp_path, final_path)
            except Exception as e:
                skipped += 1
                if skipped < 5:
                    logger.debug(f"Skipped example {i} in {split_name}: {e}")
                    
        logger.success(f"Split '{split_name}' complete. Saved {len(examples) - skipped} graphs. Skipped {skipped}.")

    # Save vocabularies
    vocab_dir = Path("data/processed")
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    with open(vocab_dir / "relation_vocab.json", "w") as f:
        json.dump(RELATION_VOCAB, f, indent=2)
    with open(vocab_dir / "concept_vocab.json", "w") as f:
        json.dump(CONCEPT_VOCAB, f, indent=2)
        
    logger.success(f"Vocabularies saved: relation={len(RELATION_VOCAB)}, concept={len(CONCEPT_VOCAB)}")

if __name__ == "__main__":
    main()
