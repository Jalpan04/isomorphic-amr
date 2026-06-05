import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dep_loader import DependencyGraphBuilder

def main():
    print("Starting preprocessed dependency graph repair utility...")
    
    # 1. Load the vocabulary mapping for relation labels
    vocab_path = Path("data/processed/deprel_vocab.json")
    if vocab_path.exists():
        with open(vocab_path, "r") as f:
            deprel_vocab = json.load(f)
        print(f"Loaded relation vocabulary with {len(deprel_vocab)} entries.")
    else:
        deprel_vocab = {"unk": 0}
        print("Relation vocabulary not found. Starting with default.")

    # 2. Check each target language directory
    languages = ["es", "de", "it", "zh"]
    
    for lang in languages:
        output_dir = Path(f"data/processed/dep_graphs_pyg/{lang}")
        if not output_dir.exists():
            continue
            
        # Scan for files
        paths = sorted(list(output_dir.glob("*.pt")))
        if not paths:
            continue
            
        corrupted_indices = []
        
        # Test loading each file
        print(f"Scanning {lang} files for corruption...")
        for path in paths:
            try:
                torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                # Get the index from file name (e.g., "005205.pt" -> 5205)
                idx = int(path.stem)
                corrupted_indices.append((idx, path))
                
        if not corrupted_indices:
            print(f"No corruption detected in {lang}.")
            continue
            
        print(f"Found {len(corrupted_indices)} corrupted files in {lang}. Initializing Stanza to rebuild them...")
        
        # Load the original monolingual sentences
        input_file = Path(f"data/raw/monolingual/{lang}_10k.txt")
        if not input_file.exists():
            print(f"Error: Original monolingual file not found at {input_file}. Cannot repair.")
            continue
            
        with open(input_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
            
        # Initialize Stanza parser
        try:
            builder = DependencyGraphBuilder(lang)
        except Exception as e:
            print(f"Failed to initialize Stanza parser for {lang}: {e}")
            continue
            
        # Rebuild each corrupted file
        rebuilt_count = 0
        for idx, path in tqdm(corrupted_indices, desc=f"Repairing {lang}"):
            if idx >= len(sentences):
                print(f"Error: Index {idx} out of range for sentences list (len={len(sentences)}).")
                continue
                
            sentence = sentences[idx]
            try:
                # Re-parse sentence
                pyg_data = builder.sentence_to_pyg(sentence, deprel_vocab)
                
                # Save correct file
                torch.save(pyg_data, path)
                rebuilt_count += 1
            except Exception as e:
                print(f"Failed to repair index {idx}: {e}")
                
        print(f"Successfully repaired {rebuilt_count}/{len(corrupted_indices)} files for {lang}.")
        
    # Re-save the relation vocabulary just in case new relations were seen (unlikely but safe)
    vocab_dir = Path("data/processed")
    with open(vocab_dir / "deprel_vocab.json", "w") as f:
        json.dump(deprel_vocab, f, indent=2)
        
    print("Dependency graph repair completed successfully.")

if __name__ == "__main__":
    main()
