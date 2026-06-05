import os
import torch
from pathlib import Path
from tqdm import tqdm

def shrink_files(directory: Path):
    if not directory.exists():
        print(f"Directory {directory} does not exist. Skipping.")
        return
        
    paths = list(directory.glob("*.pt"))
    if not paths:
        print(f"No files found in {directory}.")
        return

    print(f"Scanning {len(paths)} files in {directory}...")
    shrunk_count = 0
    error_count = 0

    for path in tqdm(paths, desc=directory.name):
        try:
            # Try to load. If it's corrupted, we'll catch it.
            data = torch.load(path, map_location="cpu", weights_only=False)
            if data is not None and hasattr(data, "x") and data.x is not None:
                # Check if it's large and needs shrinking
                # We can check file size. If file size > 500 KB, it's definitely bloated
                file_size = os.path.getsize(path)
                if file_size > 200 * 1024:  # > 200 KB
                    data.x = data.x.clone()
                    # Write to a temp file first, then rename, to avoid write errors
                    temp_path = path.with_suffix(".tmp")
                    torch.save(data, temp_path)
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        os.replace(temp_path, path)
                        shrunk_count += 1
                    else:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise RuntimeError("Temp file write resulted in 0 bytes or failed.")
        except Exception as e:
            error_count += 1
            print(f"\nError processing {path}: {e}")

    print(f"Completed {directory.name}: shrunk {shrunk_count} bloated files. Errors encountered: {error_count}")

def main():
    print("Starting disk space reclamation utility...")
    
    # Check CWD
    print(f"Current Working Directory: {os.getcwd()}")
    
    # We will process Spanish, German, Italian, Chinese and English datasets
    target_dirs = [
        Path("data/processed/dep_graphs_pyg/es"),
        Path("data/processed/dep_graphs_pyg/de"),
        Path("data/processed/dep_graphs_pyg/it"),
        Path("data/processed/dep_graphs_pyg/zh"),
        Path("data/processed/amr_graphs_pyg/train"),
        Path("data/processed/amr_graphs_pyg/dev"),
        Path("data/processed/amr_graphs_pyg/test"),
    ]
    
    for d in target_dirs:
        shrink_files(d)
        
    print("Disk space reclamation completed.")

if __name__ == "__main__":
    main()
