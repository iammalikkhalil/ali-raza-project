from src.phase3.utils_phase3 import load_config, setup_logging
from src.phase3.build_segments import make_segments
from src.phase3.train_model import train_phase3
from src.phase3.predict_matrix import build_matrices

def main():
    cfg = load_config()
    logger = setup_logging(cfg["log_file"])

    make_segments(cfg, logger)
    train_phase3(cfg, logger)
    build_matrices(cfg, logger)

if __name__ == "__main__":
    main()
