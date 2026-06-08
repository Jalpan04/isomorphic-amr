print("Starting run_ablations.py...")
print("Loading standard libraries...")
import os
import shutil
import subprocess
import yaml
import sys
import argparse
from pathlib import Path

# Check CUDA availability and Python version compatibility
try:
    import torch
    if not torch.cuda.is_available():
        print("=" * 80)
        print("WARNING: CUDA (GPU) is NOT available in this Python environment!")
        print("Please run the script using the parent virtual environment's interpreter:")
        print(r'  & "d:\python projects\.venv\Scripts\python.exe" scripts/09_run_ablations.py')
        print("=" * 80)
    else:
        print("CUDA (GPU) is available! All steps will run on GPU.")
except ImportError:
    print("WARNING: torch is not installed in the current environment.")

if sys.version_info >= (3, 14):
    print("=" * 80)
    print(f"WARNING: You are running Python {sys.version_block if hasattr(sys, 'version_block') else sys.version}. Python >= 3.14 has known compatibility issues with third-party libraries (e.g. pathlib errors in transformers/torch_geometric).")
    print("Please use the parent virtual environment's interpreter which runs a stable Python version with CUDA support:")
    print(r'  & "d:\python projects\.venv\Scripts\python.exe" scripts/09_run_ablations.py')
    print("=" * 80)

print("Setting up logging...")
from loguru import logger

# Configure loguru logging for ablations runner
logger.remove()
logger.add(
    sys.stdout,
    format="<yellow>{time:YYYY-MM-DD HH:mm:ss}</yellow> | <level>{level:7}</level> | <bold>{message}</bold>",
    level="INFO"
)
print("All libraries loaded successfully. Starting execution...")

# Define ablation experiments
ABLATIONS = {
    "exp_001_baseline_gw": {
        "alignment.method": "fgw",
        "alignment.metric": "cosine",
        "encoder.conv_type": "gat"
    },
    "exp_002_procrustes": {
        "alignment.method": "procrustes",
        "alignment.metric": "cosine",
        "encoder.conv_type": "gat"
    },
    "exp_003_gcn": {
        "alignment.method": "fgw",
        "alignment.metric": "cosine",
        "encoder.conv_type": "gcn"
    },
    "exp_005_euclidean": {
        "alignment.method": "fgw",
        "alignment.metric": "euclidean",
        "encoder.conv_type": "gat"
    },
    "exp_006_random_proj": {
        "alignment.method": "random",
        "alignment.metric": "cosine",
        "encoder.conv_type": "gat"
    }
}

def update_nested_dict(d: dict, dotted_key: str, value):
    """
    Helper to update a dictionary using dotted key syntax (e.g. 'alignment.method').
    """
    keys = dotted_key.split(".")
    curr = d
    for k in keys[:-1]:
        if k not in curr:
            curr[k] = {}
        curr = curr[k]
    curr[keys[-1]] = value

def run_script(script_name: str, args: list = None):
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    logger.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default=None, choices=list(ABLATIONS.keys()), help="Run only a specific experiment ID.")
    args = parser.parse_args()

    base_config_path = Path("config/base_config.yaml")
    if not base_config_path.exists():
        logger.error(f"Base config not found at: {base_config_path}")
        return
        
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    # Run each ablation experiment
    for exp_id, overrides in ABLATIONS.items():
        # If running a specific experiment, skip all others
        if args.exp_id and exp_id != args.exp_id:
            continue
            
        # Check if already completed
        results_file = Path("experiments") / exp_id / "results" / "smatch_scores.json"
        if results_file.exists():
            logger.info(f"Ablation experiment {exp_id} is already completed. Skipping.")
            continue
            
        logger.info(f"==================================================")
        logger.info(f"STARTING ABLATION STUDY: {exp_id}")
        logger.info(f"==================================================")
        
        # 1. Create modified configuration dict
        exp_config = yaml.safe_load(yaml.dump(base_config)) # deep copy
        exp_config["experiment"]["id"] = exp_id
        
        for key, val in overrides.items():
            update_nested_dict(exp_config, key, val)
            
        # 2. Write configuration to experiments directory
        exp_dir = Path("experiments") / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_config_file = exp_dir / "config.yaml"
        
        with open(exp_config_file, "w") as f:
            yaml.dump(exp_config, f, default_flow_style=False)
            
        # 3. Swap current base_config.yaml with experiment-specific config
        shutil.copy(exp_config_file, base_config_path)
        
        # 4. Sequentially run training, alignment, decoding, and evaluation
        try:
            # Determine if we can reuse checkpoints from exp_001_baseline_gw
            # We can reuse if it uses the same GAT encoder backbone and cosine metric
            can_reuse = (
                overrides.get("encoder.conv_type", "gat") == "gat" and
                overrides.get("alignment.metric", "cosine") == "cosine"
            )
            
            baseline_model_ckpt = Path("experiments/exp_001_baseline_gw/checkpoints/best_model.pt")
            baseline_proj_ckpt = Path("experiments/exp_001_baseline_gw/checkpoints/projection.pt")
            
            checkpoints_dir = exp_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Train Graph Encoder or Copy Checkpoint
            if can_reuse and baseline_model_ckpt.exists():
                logger.info(f"[{exp_id}] Reusing GAT encoder checkpoint from baseline...")
                shutil.copy(baseline_model_ckpt, checkpoints_dir / "best_model.pt")
            else:
                logger.info(f"[{exp_id}] Step 1/5: Training graph encoder...")
                run_script("scripts/05_train_encoder.py")
            
            # Step 2: Train Projection Layer or Copy Checkpoint
            if can_reuse and baseline_proj_ckpt.exists():
                logger.info(f"[{exp_id}] Reusing projection layer checkpoint from baseline...")
                shutil.copy(baseline_proj_ckpt, checkpoints_dir / "projection.pt")
            else:
                logger.info(f"[{exp_id}] Step 2/5: Training projection layer...")
                run_script("scripts/05b_train_projection.py")
            
            # Step 3: Run Fused Gromov-Wasserstein Alignment and projection mapping
            logger.info(f"[{exp_id}] Step 3/5: Aligning embedding spaces...")
            run_script("scripts/06_run_alignment.py")
            
            # Step 4: Run AMR decoding using AMRBART
            logger.info(f"[{exp_id}] Step 4/5: Decoding Penman AMRs...")
            run_script("scripts/07_decode_amr.py")
            
            # Step 5: Run Smatch evaluation
            logger.info(f"[{exp_id}] Step 5/5: Evaluating Smatch scores...")
            run_script("scripts/08_evaluate.py", args=["--exp_id", exp_id])
            
            logger.success(f"Ablation experiment {exp_id} completed successfully!")
            
        except Exception as e:
            logger.error(f"Error executing ablation experiment {exp_id}: {e}")
            
    # Restore default baseline config
    logger.info("Restoring baseline configuration...")
    baseline_config_file = Path("experiments") / "exp_001_baseline_gw" / "config.yaml"
    if baseline_config_file.exists():
        shutil.copy(baseline_config_file, base_config_path)
        
    logger.success("All ablation experiments completed.")

if __name__ == "__main__":
    main()
