import torch
print("VERY START: CUDA available:", torch.cuda.is_available())
print("Starting decode_amr.py...")
print("Loading standard libraries...")
import sys
import yaml
from pathlib import Path

print("Loading heavy ML libraries (tqdm)...")
from tqdm import tqdm

print("Loading internal packages...")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.amr_decoder import AMRBARTDecoderWrapper
from src.utils.logger import setup_logger
from loguru import logger
print("All libraries loaded successfully. Starting execution...")

def main():
    config_path = Path("config/base_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    exp_id = config["experiment"]["id"]
    setup_logger(exp_id)
    logger.info(f"Running AMR Decoding for experiment: {exp_id}")
    
    device = "cuda" if torch.cuda.is_available() and config["experiment"]["device"] == "cuda" else "cpu"
    
    # 1. Initialize Decoder Wrapper
    # Default to loading AMRBART from Hugging Face
    try:
        decoder = AMRBARTDecoderWrapper(
            model_name="xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing",
            gat_out_channels=config["encoder"].get("out_channels", 256)
        )
    except Exception as e:
        logger.critical(f"Failed to initialize AMRBART decoder: {e}")
        return
        
    # 2. Decode for each language
    for lang in config["eval"]["languages"]:
        logger.info(f"Decoding projected embeddings for language: {lang}...")
        
        projected_file = Path("experiments") / exp_id / "predictions" / f"projected_emb_{lang}.pt"
        if not projected_file.exists():
            logger.warning(f"Projected embeddings file not found at: {projected_file}. Skipping.")
            continue
            
        # Load list of projected tensors (each of shape: num_nodes x out_channels)
        projected_embs = torch.load(projected_file, map_location="cpu", weights_only=False)
        logger.info(f"Loaded {len(projected_embs)} graphs to decode.")
        
        predicted_amrs = []
        
        # Decode one by one to avoid padding and handle variable size nodes easily
        for i, emb in enumerate(tqdm(projected_embs, desc=f"Decoding {lang}")):
            # shape: (1, num_nodes, out_channels)
            emb_batch = emb.unsqueeze(0).to(device)
            try:
                # Generate AMR string
                outputs = decoder.generate_amr(emb_batch, max_length=256, num_beams=5)
                # Save first generation
                if outputs:
                    predicted_amrs.append(outputs[0])
                else:
                    predicted_amrs.append("# ::empty\n(a / amr-empty)")
            except Exception as e:
                logger.error(f"Error decoding graph {i}: {e}")
                predicted_amrs.append("# ::error\n(e / amr-error)")
                
        # Save results to predicted.amr file
        out_file = Path("experiments") / exp_id / "predictions" / f"{lang}_predicted.amr"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Standard Penman block separation: double newline
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(predicted_amrs))
            
        logger.success(f"Saved predicted AMRs to {out_file}")

if __name__ == "__main__":
    main()
