import os
import urllib.request
from pathlib import Path
from tqdm import tqdm

FILES = [
    "config.json",
    "pytorch_model.bin",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json"
]

REPO_URL = "https://huggingface.co/xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2/resolve/main/"
DEST_DIR = Path("data/raw/amrbart_v2")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, desc, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # We download to a temp file first and rename it to be safe
    temp_dest = dest.with_suffix(".tmp")
    
    print(f"Downloading {desc}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=temp_dest, reporthook=t.update_to)
        if temp_dest.exists():
            if dest.exists():
                os.remove(dest)
            os.rename(temp_dest, dest)
            print(f"Successfully downloaded {desc} to {dest}")
    except Exception as e:
        if temp_dest.exists():
            os.remove(temp_dest)
        print(f"Failed to download {desc}: {e}")
        raise e

def main():
    print("Initializing local AMRBART v2 download...")
    for filename in FILES:
        url = REPO_URL + filename
        dest = DEST_DIR / filename
        
        if dest.exists() and dest.stat().st_size > 0:
            print(f"File {filename} already exists locally. Skipping.")
            continue
            
        download_file(url, filename, dest)
        
    print("All AMRBART v2 files downloaded successfully!")

if __name__ == "__main__":
    main()
