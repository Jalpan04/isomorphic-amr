import subprocess
from pathlib import Path
from loguru import logger

def run_smatch_cli(pred_file: str, gold_file: str, restarts: int = 4) -> dict:
    """
    Run the official smatch CLI command via subprocess and parse the output metrics.
    
    Args:
        pred_file: Path to the generated/predicted AMR Penman file.
        gold_file: Path to the reference/gold AMR Penman file.
        restarts: Number of random restarts for the Hill-Climbing alignment solver.
    Returns:
        metrics: Dictionary containing precision, recall, and f1 score.
    """
    pred_path = Path(pred_file)
    gold_path = Path(gold_file)
    
    if not pred_path.exists():
        logger.error(f"Predicted AMR file not found: {pred_file}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    if not gold_path.exists():
        logger.error(f"Gold AMR file not found: {gold_file}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Execute Smatch CLI via subprocess using Python module call
    cmd = [
        "python", "-m", "smatch",
        "--f", str(pred_path),
        str(gold_path),
        "-r", str(restarts)
    ]
    
    try:
        logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        # Parse output fields (e.g. "Precision: 0.45", "Recall: 0.42", "F-score: 0.43")
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        
        for line in output.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower()
                try:
                    val_float = float(val.strip())
                    if "precision" in key:
                        precision = val_float
                    elif "recall" in key:
                        recall = val_float
                    elif "f-score" in key or "f1" in key:
                        f1 = val_float
                except ValueError:
                    continue
                    
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    except Exception as e:
        logger.error(f"Smatch execution failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
