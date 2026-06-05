import sys
from pathlib import Path
from loguru import logger

def setup_logger(exp_id: str = "default_run"):
    """
    Configure loguru logger.
    Logs DEBUG messages to a file under experiments/{exp_id}/logs/train.log
    and INFO messages to stdout.
    """
    # Remove default handler
    logger.remove()
    
    # Configure console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    # Configure file handler
    log_dir = Path("experiments") / exp_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:7} | {file}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB"
    )
    
    logger.info(f"Logger initialized. File log: {log_file}")
