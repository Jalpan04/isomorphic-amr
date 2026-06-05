print("Starting generate_tables.py...")
print("Loading standard libraries...")
import csv
from pathlib import Path
from loguru import logger
print("All libraries loaded successfully. Starting execution...")

def format_latex_float(val) -> str:
    try:
        return f"{float(val) * 100:.1f}"  # Show as percentage (e.g. 0.4567 -> 45.7)
    except (ValueError, TypeError):
        return str(val)

def main():
    csv_path = Path("results/all_experiments.csv")
    if not csv_path.exists():
        logger.error(f"Results file not found at: {csv_path}. Run experiments first.")
        return
        
    # Read all CSV rows
    # Columns: exp_id, lang, precision, recall, f1
    scores = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = row["exp_id"]
            lang = row["lang"]
            f1 = row["f1"]
            if exp_id not in scores:
                scores[exp_id] = {}
            scores[exp_id][lang] = f1

    # 1. Generate Main Results Table (Table 1)
    # Literature baselines (Hardcoded for comparative purposes)
    xl_amr = {"es": "43.4", "de": "38.2", "it": "44.1", "zh": "35.6", "avg_macro": "40.3"}
    ttp = {"es": "61.3", "de": "57.8", "it": "63.2", "zh": "52.4", "avg_macro": "58.7"}
    
    # Retrieve ours from CSV
    ours = scores.get("exp_001_baseline_gw", {})
    random_proj = scores.get("exp_006_random_proj", {})
    
    table1 = []
    table1.append(r"\begin{table*}[t]")
    table1.append(r"\centering")
    table1.append(r"\begin{tabular}{lcccccc}")
    table1.append(r"\hline")
    table1.append(r"\textbf{Method} & \textbf{Parallel Data} & \textbf{es} & \textbf{de} & \textbf{it} & \textbf{zh} & \textbf{Avg} \\")
    table1.append(r"\hline")
    
    # Random Projection Row
    rand_es = format_latex_float(random_proj.get("es", "0.0"))
    rand_de = format_latex_float(random_proj.get("de", "0.0"))
    rand_it = format_latex_float(random_proj.get("it", "0.0"))
    rand_zh = format_latex_float(random_proj.get("zh", "0.0"))
    rand_avg = format_latex_float(random_proj.get("avg_macro", "0.0"))
    table1.append(f"Random Projection & No & {rand_es} & {rand_de} & {rand_it} & {rand_zh} & {rand_avg} \\\\")
    
    # XL-AMR Row
    table1.append(f"XL-AMR (Blloshmi et al., 2020) & Yes & {xl_amr['es']} & {xl_amr['de']} & {xl_amr['it']} & {xl_amr['zh']} & {xl_amr['avg_macro']} \\\\")
    
    # Translate-then-Parse Row
    table1.append(f"Translate-then-Parse (T+P) & No (Test-only MT) & {ttp['es']} & {ttp['de']} & {ttp['it']} & {ttp['zh']} & {ttp['avg_macro']} \\\\")
    
    # Ours Row
    ours_es = format_latex_float(ours.get("es", "0.0"))
    ours_de = format_latex_float(ours.get("de", "0.0"))
    ours_it = format_latex_float(ours.get("it", "0.0"))
    ours_zh = format_latex_float(ours.get("zh", "0.0"))
    ours_avg = format_latex_float(ours.get("avg_macro", "0.0"))
    table1.append(f"\\textbf{{Ours (GW Projection)}} & \\textbf{{No}} & \\textbf{{{ours_es}}} & \\textbf{{{ours_de}}} & \\textbf{{{ours_it}}} & \\textbf{{{ours_zh}}} & \\textbf{{{ours_avg}}} \\\\")
    
    table1.append(r"\hline")
    table1.append(r"\end{tabular}")
    table1.append(r"\caption{Main results comparing Smatch F1 scores across languages under unsupervised transfer.}")
    table1.append(r"\label{tab:main_results}")
    table1.append(r"\end{table*}")
    
    latex_t1 = "\n".join(table1)
    
    # Write Table 1
    t1_path = Path("results/main_table.tex")
    t1_path.parent.mkdir(parents=True, exist_ok=True)
    with open(t1_path, "w", encoding="utf-8") as f:
        f.write(latex_t1)
    logger.success(f"Generated main LaTeX results table: {t1_path}")

    # 2. Generate Ablation Table (Table 2)
    procrustes = scores.get("exp_002_procrustes", {})
    gcn = scores.get("exp_003_gcn", {})
    euclidean = scores.get("exp_005_euclidean", {})
    
    table2 = []
    table2.append(r"\begin{table}[t]")
    table2.append(r"\centering")
    table2.append(r"\begin{tabular}{lccccc}")
    table2.append(r"\hline")
    table2.append(r"\textbf{Variant} & \textbf{es} & \textbf{de} & \textbf{it} & \textbf{zh} & \textbf{Avg} \\")
    table2.append(r"\hline")
    
    # Full Model
    table2.append(f"Full Model (GW + GAT) & {ours_es} & {ours_de} & {ours_it} & {ours_zh} & {ours_avg} \\\\")
    
    # Procrustes
    proc_es = format_latex_float(procrustes.get("es", "0.0"))
    proc_de = format_latex_float(procrustes.get("de", "0.0"))
    proc_it = format_latex_float(procrustes.get("it", "0.0"))
    proc_zh = format_latex_float(procrustes.get("zh", "0.0"))
    proc_avg = format_latex_float(procrustes.get("avg_macro", "0.0"))
    table2.append(f"w/o GW $\\rightarrow$ Procrustes & {proc_es} & {proc_de} & {proc_it} & {proc_zh} & {proc_avg} \\\\")
    
    # GCN
    gcn_es = format_latex_float(gcn.get("es", "0.0"))
    gcn_de = format_latex_float(gcn.get("de", "0.0"))
    gcn_it = format_latex_float(gcn.get("it", "0.0"))
    gcn_zh = format_latex_float(gcn.get("zh", "0.0"))
    gcn_avg = format_latex_float(gcn.get("avg_macro", "0.0"))
    table2.append(f"w/o GAT $\\rightarrow$ GCN & {gcn_es} & {gcn_de} & {gcn_it} & {gcn_zh} & {gcn_avg} \\\\")
    
    # Euclidean
    eucl_es = format_latex_float(euclidean.get("es", "0.0"))
    eucl_de = format_latex_float(euclidean.get("de", "0.0"))
    eucl_it = format_latex_float(euclidean.get("it", "0.0"))
    eucl_zh = format_latex_float(euclidean.get("zh", "0.0"))
    eucl_avg = format_latex_float(euclidean.get("avg_macro", "0.0"))
    table2.append(f"Euclidean Dist in GW & {eucl_es} & {eucl_de} & {eucl_it} & {eucl_zh} & {eucl_avg} \\\\")
    
    table2.append(r"\hline")
    table2.append(r"\end{tabular}")
    table2.append(r"\caption{Ablation studies examining alternative encoder designs and distance metrics.}")
    table2.append(r"\label{tab:ablation}")
    table2.append(r"\end{table}")
    
    latex_t2 = "\n".join(table2)
    
    # Write Table 2
    t2_path = Path("results/ablation_table.tex")
    with open(t2_path, "w", encoding="utf-8") as f:
        f.write(latex_t2)
    logger.success(f"Generated ablation LaTeX table: {t2_path}")

if __name__ == "__main__":
    main()
