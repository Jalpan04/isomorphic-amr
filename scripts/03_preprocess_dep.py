print("Starting preprocess_dep.py...")
print("Loading standard libraries...")
import json
import sys
from pathlib import Path

print("Loading heavy ML libraries (torch)...")
import torch
from tqdm import tqdm

print("Loading internal packages...")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dep_loader import DependencyGraphBuilder
from src.utils.logger import setup_logger
from loguru import logger
print("All libraries loaded successfully. Starting execution...")

LANGUAGES = ["es", "de", "it", "zh"]
DEPREL_VOCAB = {"unk": 0}

def main():
    setup_logger("preprocessing")
    logger.info("Starting Target Languages Dependency Preprocessing...")
    
    # Process each language
    for lang in LANGUAGES:
        logger.info(f"Processing language: {lang}")
        input_file = Path(f"data/raw/monolingual/{lang}_10k.txt")
        if not input_file.exists():
            logger.error(f"Monolingual file not found at: {input_file}. Run scripts/01_download_data.py first.")
            continue
            
        output_dir = Path(f"data/processed/dep_graphs_pyg/{lang}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sentences
        with open(input_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Loaded {len(sentences)} sentences for {lang}")
        
        # Initialize Stanza parser
        try:
            builder = DependencyGraphBuilder(lang)
        except Exception as e:
            logger.error(f"Failed to initialize Stanza builder for {lang}: {e}")
            continue
            
        # Resume: count already-saved graphs and skip those sentences
        existing = sorted(output_dir.glob("*.pt"))
        resume_from = len(existing)
        if resume_from > 0:
            logger.info(f"[{lang}] Resuming from sentence {resume_from} ({resume_from} graphs already saved).")
            saved_count = resume_from
        else:
            saved_count = 0

        skipped = 0
        remaining = sentences[resume_from:]

        with tqdm(remaining, desc=f"[{lang}]", unit="sent", initial=resume_from, total=len(sentences)) as pbar:
            for i, sentence in enumerate(remaining, start=resume_from):
                try:
                    # Build PyG graph
                    pyg_data = builder.sentence_to_pyg(sentence, DEPREL_VOCAB)

                    # Save graph structure
                    import os
                    final_path = output_dir / f"{saved_count:06d}.pt"
                    temp_path = final_path.with_suffix(".tmp")
                    torch.save(pyg_data, temp_path)
                    os.replace(temp_path, final_path)
                    saved_count += 1
                except Exception as e:
                    skipped += 1
                    if skipped < 5:
                        logger.debug(f"[{lang}] Skipped sentence {i}: {e}")

                pbar.update(1)
                pbar.set_postfix(saved=saved_count, skipped=skipped)

        logger.success(f"[{lang}] Processing complete. Saved {saved_count} graphs. Skipped {skipped}.")
        
    # Save dependency relations vocabulary
    vocab_dir = Path("data/processed")
    vocab_dir.mkdir(parents=True, exist_ok=True)
    with open(vocab_dir / "deprel_vocab.json", "w") as f:
        json.dump(DEPREL_VOCAB, f, indent=2)
        
    logger.success(f"Dependency relation vocabulary saved: count={len(DEPREL_VOCAB)}")

if __name__ == "__main__":
    main()
