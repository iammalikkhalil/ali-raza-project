from src.phase1.validate_raw import validate_raw
from src.phase1.generate_metadata import build_phase1_artifacts

def main():
    valid_files = validate_raw()
    build_phase1_artifacts(valid_files)

if __name__ == "__main__":
    main()