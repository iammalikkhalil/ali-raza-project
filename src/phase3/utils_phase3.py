from pathlib import Path
import yaml
import logging

def load_config(path="configs/phase3_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase3")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    return logger
