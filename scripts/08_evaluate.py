print("Starting evaluate.py...")
print("Loading standard libraries...")
import argparse
import sys
import csv
import json
import yaml
from pathlib import Path

print("Loading internal packages...")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.smatch_eval import run_smatch_cli
from src.evaluation.analysis import compile_performance_summary
from src.utils.logger import setup_logger
from loguru import logger
print("All libraries loaded successfully. Starting execution...")

LANGUAGES = ["es", "de", "it", "zh"]

def find_gold_file(lang: str) -> Path:
    """
    Locates the gold AMR file in the raw multilingual directory for a language.
    Searches for any .amr or .txt files.
    """
    lang_dir = Path("data/raw/multilingual_amr") / lang
    if not lang_dir.exists():
        # Fallback check directly in multilingual folder
        lang_dir = Path("data/raw/multilingual_amr")
        if not lang_dir.exists():
            return None
            
    files = list(lang_dir.glob(f"**/*{lang}*.amr")) + \
            list(lang_dir.glob(f"**/*{lang}*.txt")) + \
            list(lang_dir.glob(f"**/*gold*.amr")) + \
            list(lang_dir.glob(f"**/*gold*.txt"))
            
    if not files:
        # Generic fallback: look for any .txt or .amr file in lang subfolder
        subfolder = Path("data/raw/multilingual_amr") / lang
        if subfolder.exists():
            files = list(subfolder.glob("*.txt")) + list(subfolder.glob("*.amr"))
            
    if files:
        return files[0]
        
    return None

def update_master_csv(exp_id: str, results: dict):
    """
    Append results to the master CSV file.
    """
    csv_path = Path("results/all_experiments.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["exp_id", "lang", "precision", "recall", "f1"])
            
        for lang, scores in results.items():
            if lang == "avg":
                writer.writerow([exp_id, "avg_macro", "", "", scores["f1"]])
            else:
                writer.writerow([exp_id, lang, scores["precision"], scores["recall"], scores["f1"]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default=None, help="Experiment ID to evaluate. Defaults to base_config setting.")
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path("config/base_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    exp_id = args.exp_id if args.exp_id else config["experiment"]["id"]
    setup_logger(exp_id)
    logger.info(f"Running Smatch evaluation for experiment: {exp_id}")
    
    results = {}
    restarts = config["eval"].get("smatch_restarts", 4)
    
    for lang in LANGUAGES:
        pred_file = Path("experiments") / exp_id / "predictions" / f"{lang}_predicted.amr"
        if not pred_file.exists():
            logger.warning(f"No predicted file found for {lang} at: {pred_file}. Skipping.")
            continue
            
        gold_file = find_gold_file(lang)
        if not gold_file:
            logger.error(f"Could not locate gold reference file for language: {lang}")
            continue
            
        logger.info(f"Evaluating Smatch: Pred={pred_file.name} vs Gold={gold_file.name}")
        scores = run_smatch_cli(str(pred_file), str(gold_file), restarts=restarts)
        results[lang] = scores
        logger.success(f"[{exp_id}] [{lang.upper()}] Precision: {scores['precision']:.4f} | Recall: {scores['recall']:.4f} | Smatch F1: {scores['f1']:.4f}")

    if not results:
        logger.error("No predictions could be evaluated. Ensure you have run scripts/06_run_alignment.py and scripts/07_decode_amr.py.")
        return
        
    # Calculate macro-average
    avg_f1 = sum(scores["f1"] for scores in results.values()) / len(results)
    results["avg"] = {"f1": avg_f1}
    
    # Save individual experiment results
    out_file = Path("experiments") / exp_id / "results" / "smatch_scores.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.success(f"Macro-Average Smatch F1: {avg_f1:.4f}")
    
    # Write to master csv and generate summary markdown
    update_master_csv(exp_id, results)
    compile_performance_summary(exp_id, results)

if __name__ == "__main__":
    main()
