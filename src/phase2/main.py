from src.phase2.utils_phase2 import load_config, setup_logging
from src.phase2.sample_phase1 import make_p2_sample
# from src.phase2.clean_and_trips import make_clean_and_trips
from src.phase2.clean_and_trips import clean_and_segment_in_batches
from src.phase2.cluster_stops import make_bus_stops
from src.phase2.assign_stops import assign_points_to_stops
from src.phase2.generate_report import build_p2_report

def main():
    cfg = load_config()
    logger = setup_logging(cfg["log_file"])


    # 1) Sample down to ≤1M (bbox + optional morning window)
    make_p2_sample(cfg, logger)

    # 2) Clean + trip segmentation (+ long trips)
    # make_clean_and_trips(cfg, logger)
    clean_and_segment_in_batches(cfg, logger)

    # 3) Cluster stops (KMeans scan 200–300) + depot inferred automatically
    make_bus_stops(cfg, logger)

    # 4) Assign points to nearest stop within radius
    assign_points_to_stops(cfg, logger)

    # 5) Quality report + figures
    build_p2_report(cfg, logger)

if __name__ == "__main__":
    main()