import shutil
from pathlib import Path
from loguru import logger

LANGUAGES = ["es", "de", "it", "zh"]
EXP_ID = "exp_001_baseline_gw"

def main():
    print("Setting up mock gold files for evaluation testing...")
    
    pred_dir = Path("experiments") / EXP_ID / "predictions"
    gold_root = Path("data/raw/multilingual_amr")
    
    if not pred_dir.exists():
        logger.error(f"Predictions directory not found: {pred_dir}. Run scripts/07_decode_amr.py first.")
        return
        
    for lang in LANGUAGES:
        pred_file = pred_dir / f"{lang}_predicted.amr"
        if not pred_file.exists():
            logger.warning(f"Predicted file not found: {pred_file}")
            continue
            
        lang_gold_dir = gold_root / lang
        lang_gold_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if there are already files in the gold directory (don't overwrite real gold files)
        real_files = [
            f for f in list(lang_gold_dir.glob("*.amr")) + list(lang_gold_dir.glob("*.txt"))
            if not f.name.endswith("_mock.amr") and not f.name.endswith("_mock.txt")
        ]
        if real_files:
            logger.info(f"Gold folder for {lang} already contains real gold files: {[f.name for f in real_files]}. Skipping copying mock.")
            continue
            
        dest_file = lang_gold_dir / f"{lang}_gold_mock.amr"
        shutil.copy(pred_file, dest_file)
        logger.success(f"Copied {pred_file.name} to {dest_file}")
        
    print("Mock gold setup complete. You can now run evaluation/ablations.")

if __name__ == "__main__":
    main()
