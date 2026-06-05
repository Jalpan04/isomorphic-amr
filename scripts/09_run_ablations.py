print("Starting run_ablations.py...")
print("Loading standard libraries...")
import os
import shutil
import subprocess
import yaml
import sys
from pathlib import Path

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
    base_config_path = Path("config/base_config.yaml")
    if not base_config_path.exists():
        logger.error(f"Base config not found at: {base_config_path}")
        return
        
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    # Run each ablation experiment
    for exp_id, overrides in ABLATIONS.items():
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
            # Step 1: Train Graph Encoder
            logger.info(f"[{exp_id}] Step 1/4: Training graph encoder...")
            run_script("scripts/05_train_encoder.py")
            
            # Step 2: Run Fused Gromov-Wasserstein Alignment and projection mapping
            logger.info(f"[{exp_id}] Step 2/4: Aligning embedding spaces...")
            run_script("scripts/06_run_alignment.py")
            
            # Step 3: Run AMR decoding using AMRBART
            logger.info(f"[{exp_id}] Step 3/4: Decoding Penman AMRs...")
            run_script("scripts/07_decode_amr.py")
            
            # Step 4: Run Smatch evaluation
            logger.info(f"[{exp_id}] Step 4/4: Evaluating Smatch scores...")
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
