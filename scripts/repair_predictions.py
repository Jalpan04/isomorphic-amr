import sys
from pathlib import Path
from loguru import logger
import penman

LANGUAGES = ["es", "de", "it", "zh"]
EXP_ID = "exp_006_random_proj"

def validate_and_format_amr(amr_str: str) -> str:
    """
    Validates the generated AMR string. If valid Penman format AND valid Smatch format,
    returns it as a single-line representation. Otherwise, falls back to a dummy empty AMR.
    """
    clean_str = amr_str.strip()
    try:
        g = penman.decode(clean_str)
        one_line = penman.encode(g).replace("\n", " ").strip()
        if one_line.startswith("(") and one_line.endswith(")"):
            import smatch
            res = smatch.amr.AMR.parse_AMR_line(one_line)
            if res is not None:
                return one_line
    except Exception:
        pass
    return "(a / amr-empty)"

def main():
    print("Starting predicted AMR layout repair utility...")
    pred_dir = Path("experiments") / EXP_ID / "predictions"
    
    if not pred_dir.exists():
        logger.error(f"Predictions directory not found: {pred_dir}")
        return
        
    for lang in LANGUAGES:
        pred_file = pred_dir / f"{lang}_predicted.amr"
        if not pred_file.exists():
            logger.warning(f"Predicted file not found: {pred_file}")
            continue
            
        logger.info(f"Repairing layout and syntax for {lang} predicted AMRs...")
        
        # Read the raw predicted content
        with open(pred_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split by double newline to get individual graph blocks
        raw_blocks = content.split("\n\n")
        
        repaired_blocks = []
        for i, block in enumerate(raw_blocks):
            if not block.strip():
                continue
            cleaned = validate_and_format_amr(block)
            repaired_blocks.append(cleaned)
            
        # Ensure we write exactly 288 graphs
        if len(repaired_blocks) != 288:
            logger.warning(f"Expected 288 graphs, but found {len(repaired_blocks)} blocks in {pred_file.name}. Adjusting...")
            while len(repaired_blocks) < 288:
                repaired_blocks.append("(a / amr-empty)")
            repaired_blocks = repaired_blocks[:288]
            
        # Write back to predicted AMR file
        with open(pred_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(repaired_blocks))
            
        logger.success(f"Successfully repaired and formatted {pred_file.name}")
        
    print("All predicted AMR files have been successfully repaired and reformatted.")

if __name__ == "__main__":
    main()
