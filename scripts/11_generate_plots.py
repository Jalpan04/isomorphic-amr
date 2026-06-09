import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

# Set beautiful styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 15
})

LANGUAGES = ["es", "de", "it", "zh"]
LANG_NAMES = {
    "es": "Spanish (es)",
    "de": "German (de)",
    "it": "Italian (it)",
    "zh": "Chinese (zh)"
}

# Rich, curated HSL-derived palette colors for each language
COLORS = {
    "es": "#3b82f6",  # Sleek blue
    "de": "#10b981",  # Emerald green
    "it": "#f59e0b",  # Amber orange
    "zh": "#8b5cf6"   # Indigo purple
}

def main():
    print("Starting generate_plots.py...")
    logger.info("Initializing plot generation for representation collapse...")
    
    exp_id = "exp_001_baseline_gw"
    pred_dir = Path("experiments") / exp_id / "predictions"
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files exist
    valid_langs = []
    data_by_lang = {}
    
    for lang in LANGUAGES:
        meta_file = pred_dir / f"matched_metadata_{lang}.json"
        if not meta_file.exists():
            logger.warning(f"Metadata file not found: {meta_file}. Skipping {lang}.")
            continue
            
        with open(meta_file, "r") as f:
            meta = json.load(f)
            
        indices = meta.get("matched_en_indices", [])
        if not indices:
            logger.warning(f"No indices found in {meta_file}. Skipping {lang}.")
            continue
            
        data_by_lang[lang] = indices
        valid_langs.append(lang)
        
    if not valid_langs:
        logger.error("No valid alignment metadata files found. Make sure you have run the baseline experiment.")
        return
        
    logger.info(f"Generating histograms for languages: {valid_langs}")
    
    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()
    
    for idx, lang in enumerate(LANGUAGES):
        ax = axes[idx]
        if lang in data_by_lang:
            indices = data_by_lang[lang]
            
            # Plot the histogram/density showing matched graph IDs
            sns.histplot(
                indices,
                bins=50,
                color=COLORS[lang],
                ax=ax,
                kde=False,
                edgecolor="white",
                alpha=0.85
            )
            
            # Calculate unique structures count
            unique_count = len(set(indices))
            total_count = len(indices)
            reuse_ratio = (unique_count / total_count) * 100
            
            ax.set_title(f"{LANG_NAMES[lang]} (Unique templates: {unique_count}/{total_count}, {reuse_ratio:.1f}%)")
            ax.set_ylabel("Match Count")
            ax.set_xlabel("English AMR Graph Candidate Index")
        else:
            ax.text(0.5, 0.5, f"No Data Available for {LANG_NAMES[lang]}", 
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(LANG_NAMES[lang])
            
    fig.suptitle("Latent Projection Collapse: Distribution of Retracted Template Indices\n"
                 "(Spikes show structural collapse where multiple target utterances map to identical English templates)", 
                 y=0.98, weight="bold")
    
    plt.tight_layout()
    out_file = fig_dir / "latent_space_collapse.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.success(f"Successfully generated and saved collapse visualization to: {out_file}")

if __name__ == "__main__":
    main()
