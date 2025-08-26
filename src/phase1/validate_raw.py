from pathlib import Path
from typing import List
from src.phase1.utils_data import load_config, setup_logging

def validate_raw() -> List[Path]:
    config = load_config()
    logger = setup_logging(config["logging"]["log_file"], config["logging"]["log_level"])

    raw_dir = Path(config["raw_data_dir"])
    min_size = int(config["min_file_size_kb"]) * 1024
    max_size = int(config["max_file_size_mb"]) * 1024 * 1024
    required_ext = str(config["required_file_extension"]).lower()

    valid_files: List[Path] = []
    skipped_big = 0

    logger.info(f"Starting raw data validation in {raw_dir}")

    if not raw_dir.exists():
        logger.error(f"Raw data dir not found: {raw_dir}")
        return valid_files

    for user_dir in sorted(raw_dir.iterdir()):
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.exists():
            continue
        for f in traj_dir.iterdir():
            if f.suffix.lower() != required_ext:
                continue
            size = f.stat().st_size
            if size < min_size:
                continue
            if size > max_size:
                skipped_big += 1
                logger.warning(f"Skipping (too large per config): {f} size={size/1048576:.2f} MB")
                continue
            valid_files.append(f)

    logger.info(f"Validation complete. {len(valid_files)} valid files found. Skipped large files: {skipped_big}")
    return valid_files