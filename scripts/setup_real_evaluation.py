import os
import re
import urllib.request
import tarfile
import json
from pathlib import Path
from loguru import logger

# Constants
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
TAR_PATH = Path("data/raw/amazon-massive-dataset-1.1.tar.gz")
EXTRACT_DIR = Path("data/raw/massive_nlu_1.1")
MASSIVE_AMR_JSONL = Path("data/raw/massive_amr/data/massive_amr.jsonl")

LANGUAGES = {
    "es": "es-ES",
    "de": "de-DE",
    "it": "it-IT",
    "zh": "zh-CN"
}

def download_and_extract():
    if not TAR_PATH.exists():
        logger.info(f"Downloading MASSIVE NLU dataset from {MASSIVE_URL}...")
        TAR_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MASSIVE_URL, TAR_PATH)
        logger.success("Download complete.")
        
    if not EXTRACT_DIR.exists():
        logger.info(f"Extracting MASSIVE NLU dataset to {EXTRACT_DIR}...")
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)
        logger.success("Extraction complete.")

def create_named_entity(name):
    named_entity = '/ name '
    name = name.split()
    thisIdx = 1
    for n in name:
        if n.lower() not in ['the', 'this']:
            n = n.replace("'s", "")
            token = f':op{thisIdx} "{n.strip()}" '
            named_entity += token
            thisIdx += 1
    named_entity = named_entity.strip()
    named_entity += ')'
    return named_entity

def get_name_from_annotation(utt):
    entities = []
    entity_found = re.findall(r'\[(.*?)\]', utt)
    if entity_found:
        for ef in entity_found:
            if ':' not in ef:
                continue
            ef_split = ef.split(':')[1]
            ef_split = create_named_entity(ef_split.strip())
            ef_type = ef.split(':')[0].strip()
            entities.append((ef_type, ef_split))
    if not entities:
        entities = [(0, 'NONE')]
    return entities

def process_amr_named_entities(raw_amr, en_utt, tgt_utt_annotated):
    en_entities = get_name_from_annotation(en_utt)
    tgt_entities = get_name_from_annotation(tgt_utt_annotated)
    
    en_entities = sorted(en_entities)
    tgt_entities = sorted(tgt_entities)
    
    temp_raw_amr = raw_amr
    for en_entity, tgt_entity in zip(en_entities, tgt_entities):
        if en_entity[0] == tgt_entity[0] and en_entity[1] != 'NONE' and tgt_entity[1] != 'NONE':
            temp_raw_amr = temp_raw_amr.replace(en_entity[1], tgt_entity[1])
            
    return temp_raw_amr

def main():
    logger.info("Setting up real evaluation dataset using MASSIVE-AMR...")
    
    # 1. Download and extract original MASSIVE NLU dataset
    download_and_extract()
    
    # 2. Load English AMR annotations
    if not MASSIVE_AMR_JSONL.exists():
        logger.critical(f"MASSIVE-AMR file not found at {MASSIVE_AMR_JSONL}. Run git clone first.")
        return
        
    logger.info("Loading English AMR graphs...")
    en_amr_dict = {}
    with open(MASSIVE_AMR_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            en_amr_dict[item["id"]] = item
            
    logger.success(f"Loaded {len(en_amr_dict)} English AMR graphs.")
    
    # Locate extracted data folder (usually data/raw/massive_nlu_1.1/1.1/data)
    extracted_data_dir = EXTRACT_DIR / "1.1" / "data"
    if not extracted_data_dir.exists():
        # Fallback check
        extracted_data_dir = list(EXTRACT_DIR.glob("**/data"))[0]
        
    logger.info(f"Using NLU data folder: {extracted_data_dir}")
    
    for lang, locale in LANGUAGES.items():
        logger.info(f"Processing target language: {lang} ({locale})...")
        locale_file = extracted_data_dir / f"{locale}.jsonl"
        if not locale_file.exists():
            logger.error(f"Locale file not found at {locale_file}")
            continue
            
        sentences = []
        amr_graphs = []
        
        # Load translations
        with open(locale_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Only keep QA scenario examples that are in the test partition
                if item["scenario"] == "qa" and item["partition"] == "test":
                    example_id = item["id"]
                    if example_id in en_amr_dict:
                        en_item = en_amr_dict[example_id]
                        raw_amr = en_item["raw_amr"]
                        en_utt = en_item["annot_utt"]
                        tgt_utt_annotated = item["annot_utt"]
                        tgt_utt_plain = item["utt"]
                        
                        # Apply entity replacement to localise English names in graph
                        localised_amr = process_amr_named_entities(raw_amr, en_utt, tgt_utt_annotated)
                        
                        # Validate Penman bracket count
                        if localised_amr.count("(") == localised_amr.count(")"):
                            # Collapse AMR into single line representation for Smatch evaluation formatting
                            # and replace newlines to keep it clean
                            clean_amr = " ".join([line.strip() for line in localised_amr.splitlines()])
                            
                            sentences.append(tgt_utt_plain)
                            amr_graphs.append(clean_amr)
                            
        logger.success(f"Extracted {len(sentences)} parallel test examples for {lang}")
        
        # 3. Write target sentences to monolingual file (overwriting CC-100)
        mono_path = Path("data/raw/monolingual") / f"{lang}_10k.txt"
        mono_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mono_path, "w", encoding="utf-8") as f:
            for sent in sentences:
                f.write(sent + "\n")
        logger.info(f"Saved real sentences to {mono_path}")
        
        # 4. Write gold AMR graphs to multilingual gold file
        gold_dir = Path("data/raw/multilingual_amr") / lang
        gold_dir.mkdir(parents=True, exist_ok=True)
        gold_path = gold_dir / f"{lang}_gold.amr"
        
        # Clean up any existing mock gold file
        mock_gold_path = gold_dir / f"{lang}_gold_mock.amr"
        if mock_gold_path.exists():
            os.remove(mock_gold_path)
            
        with open(gold_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(amr_graphs))
        logger.info(f"Saved real gold AMRs to {gold_path}")
        
    logger.success("Setup of real evaluation dataset complete.")

if __name__ == "__main__":
    main()
