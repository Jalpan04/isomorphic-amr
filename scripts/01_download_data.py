print("Starting download script...")
print("Loading standard libraries...")
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

print("Loading Stanza...")
import stanza

print("Loading Hugging Face datasets...")
from datasets import load_dataset

print("Loading Transformers & PyTorch (this might take a few seconds)...")
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
print("All libraries loaded successfully. Starting execution...")

def setup_directories():
    print("Setting up directory structure...")
    dirs = [
        "data/raw/monolingual",
        "data/raw/multilingual_amr",
        "data/processed/amr_graphs_pyg",
        "data/processed/dep_graphs_pyg",
        "data/processed/embeddings",
        "config/ablation_configs",
        "experiments",
        "results/figures",
        "paper/figures",
        "notebooks"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("Directory structure initialized.")

def download_multilingual_gold_amr():
    print("Downloading gold multilingual AMR datasets from amr-guidelines repository...")
    zip_path = Path("data/raw/amr_guidelines.zip")
    extract_dir = Path("data/raw/temp_guidelines_extract")
    
    if zip_path.exists():
        os.remove(zip_path)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
        
    url = "https://github.com/amrisi/amr-guidelines/archive/refs/heads/master.zip"
    
    try:
        # Download ZIP file to avoid Windows .git file permission issues
        print(f"Downloading ZIP from: {url}")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        # Find the extracted folder (usually named amr-guidelines-master)
        extracted_folders = list(extract_dir.glob("amr-guidelines-*"))
        if extracted_folders:
            root_folder = extracted_folders[0]
            src_multilingual = root_folder / "multilingual"
            dest_multilingual = Path("data/raw/multilingual_amr")
            
            if src_multilingual.exists():
                if dest_multilingual.exists():
                    shutil.rmtree(dest_multilingual)
                shutil.copytree(src_multilingual, dest_multilingual)
                print("Successfully copied gold multilingual AMR datasets.")
            else:
                print("Warning: 'multilingual' directory was not found in the cloned repository.")
                print("LDC multilingual gold standards are restricted. You will need to place your gold AMR files (e.g. gold.amr)")
                print("for Spanish, German, Italian, or Chinese manually under data/raw/multilingual_amr/{lang}/ prior to evaluation.")
                
                # Create stub folders for target languages to allow manual placement
                for lang in ["es", "de", "it", "zh"]:
                    (dest_multilingual / lang).mkdir(parents=True, exist_ok=True)
        else:
            print("Error: Could not locate extracted folder root.")
            
    except Exception as e:
        print(f"Failed to fetch gold multilingual AMR repository: {e}")
        print("Please place your gold AMR evaluation graphs under data/raw/multilingual_amr/{lang}/ manually.")
    finally:
        # Clean up temporary downloads
        if zip_path.exists():
            os.remove(zip_path)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

def download_stanza_models():
    print("Downloading Stanza dependency parser models...")
    for lang in ["es", "de", "it", "zh"]:
        print(f"Downloading Stanza model for: {lang}")
        try:
            stanza.download(lang, processors="tokenize,pos,lemma,depparse")
        except Exception as e:
            print(f"Failed to download Stanza model for {lang}: {e}")
    print("Stanza models downloaded.")

def cache_huggingface_models():
    print("Pre-downloading and caching Hugging Face models...")
    # Cache XLM-R
    print("Caching xlm-roberta-base...")
    try:
        AutoTokenizer.from_pretrained("xlm-roberta-base")
        AutoModel.from_pretrained("xlm-roberta-base")
    except Exception as e:
        print(f"Error caching XLM-R: {e}")
    
    # Cache AMRBART
    print("Caching AMRBART-large-finetuned-AMR3.0-AMRParsing...")
    try:
        AutoTokenizer.from_pretrained("xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing")
        AutoModelForSeq2SeqLM.from_pretrained("xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing")
    except Exception as e:
        print(f"Error caching AMRBART: {e}")
    print("Hugging Face models cached.")

def download_monolingual_texts():
    print("Loading CC-100 target language subsets (10,000 sentences each)...")
    languages = {
        "es": "es",
        "de": "de",
        "it": "it",
        "zh": "zh-Hans"
    }
    
    for lang_code, lang_name in languages.items():
        output_file = Path(f"data/raw/monolingual/{lang_code}_10k.txt")
        if output_file.exists():
            print(f"Monolingual file for {lang_code} already exists. Skipping.")
            continue
            
        print(f"Fetching CC-100 for: {lang_code}...")
        try:
            # Load from pre-sharded parquet dataset on Hugging Face (does not run scripts)
            dataset = load_dataset("xu-song/cc100-samples", name=lang_name, split="train")
            
            sentences = []
            for row in dataset:
                text = row["text"].strip()
                if len(text) > 10 and not text.isspace():
                    sentences.append(text)
            
            # Save to text file
            with open(output_file, "w", encoding="utf-8") as f:
                for sent in sentences:
                    clean_sent = " ".join(sent.splitlines())
                    f.write(clean_sent + "\n")
            print(f"Saved {len(sentences)} sentences to {output_file}")
            
        except Exception as e:
            print(f"Error fetching CC-100 for {lang_code}: {e}")

if __name__ == "__main__":
    setup_directories()
    download_multilingual_gold_amr()
    download_stanza_models()
    cache_huggingface_models()
    download_monolingual_texts()
    print("Data download and environmental preparation complete.")
