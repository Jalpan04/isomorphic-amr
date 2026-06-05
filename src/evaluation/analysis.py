import json
from pathlib import Path
from loguru import logger
import numpy as np

def compile_performance_summary(exp_id: str, results_by_lang: dict) -> str:
    """
    Generate a formatted markdown report analyzing the performance across all target languages.
    """
    report = []
    report.append(f"# Experiment Performance Analysis: {exp_id}\n")
    
    # Table Header
    report.append("| Language | Precision | Recall | Smatch F1 |")
    report.append("|---|---|---|---|")
    
    f1_list = []
    for lang, metrics in results_by_lang.items():
        if lang == "avg":
            continue
        p = metrics.get("precision", 0.0)
        r = metrics.get("recall", 0.0)
        f1 = metrics.get("f1", 0.0)
        f1_list.append(f1)
        report.append(f"| {lang.upper()} | {p:.4f} | {r:.4f} | {f1:.4f} |")
        
    # Add Average row
    avg_f1 = np.mean(f1_list) if f1_list else 0.0
    report.append(f"| **AVERAGE** | - | - | **{avg_f1:.4f}** |\n")
    
    # Analyze alignment properties (e.g. nearest neighbor mappings)
    report.append("## Alignment Analysis")
    for lang in results_by_lang.keys():
        if lang == "avg":
            continue
        meta_file = Path("experiments") / exp_id / "predictions" / f"matched_metadata_{lang}.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
            unique_matches = len(set(meta.get("matched_en_indices", [])))
            total_matches = len(meta.get("matched_en_indices", []))
            report.append(
                f"* **{lang.upper()}**: Mapped {total_matches} sentences onto {unique_matches} unique English AMR graph structures "
                f"(Structure Reuse Ratio: {unique_matches / max(total_matches, 1):.2%})."
            )
            
    summary_text = "\n".join(report)
    
    # Save to disk
    out_file = Path("experiments") / exp_id / "results" / "summary.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary_text)
        
    logger.info(f"Saved experiment results summary to: {out_file}")
    return summary_text
