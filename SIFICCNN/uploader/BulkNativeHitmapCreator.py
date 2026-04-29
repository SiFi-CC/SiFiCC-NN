#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- LOGGING ---
def _init_logger():
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

logger = _init_logger()

# --- CONFIG ---
RESULTS_ROOT = Path("/home/philippe/RWTHscrath1clement/SiFiCCNN/results")
OUT_BASE = Path("/home/philippe/RWTHscrath1clement/SM/paper2025")
CREATOR = "NativeHitmapCreator.py"
PYTHON = "python"
REDUCED_STATISTICS = 0.01  # fraction of events to process
DRY_RUN = False  # set True to just print commands
# Number of parallel workers; can be overridden via env var WORKERS
try:
    MAX_WORKERS = int(os.getenv("WORKERS", "0"))
    if MAX_WORKERS <= 0:
        # Sensible default: min(available CPUs, 8)
        import os as _os

        MAX_WORKERS = min((_os.cpu_count() or 2), 8)
except Exception:
    MAX_WORKERS = 1


def _process_single_job(job_dir_str: str, class_name: str) -> tuple[str, str]:
    """
    Process a single job directory by invoking the NativeHitmapCreator.

    Returns a tuple (status, job_dir_str) where status is one of
    'processed', 'skipped_missing', 'failed'.
    """
    job_dir = Path(job_dir_str)
    job_name = job_dir.name

    energy_npy = job_dir / f"{job_name}_regE_pred.npy"
    position_npy = job_dir / f"{job_name}_ClassXZ_pred.npy"

    if not energy_npy.exists() or not position_npy.exists():
        logger.warning(
            "Skipping %s: missing npy files (energy=%s exists=%s, position=%s exists=%s)",
            job_dir,
            energy_npy,
            energy_npy.exists(),
            position_npy,
            position_npy.exists(),
        )
        return ("skipped_missing", job_dir_str)

    out_dir = OUT_BASE / class_name / "hitmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON,
        CREATOR,
        "--output_prefix",
        str(out_dir),
        "--energy_npy",
        str(energy_npy),
        "--position_npy",
        str(position_npy),
        "--reduced_statistics",
        str(REDUCED_STATISTICS),
    ]

    logger.info("Prepared command for %s: %s", job_dir, " ".join(cmd))
    if DRY_RUN:
        logger.info("DRY_RUN=True: not executing command for %s", job_dir)
        return ("processed", job_dir_str)

    try:
        subprocess.run(cmd, check=True)
        logger.info("Completed: %s", job_dir)
        return ("processed", job_dir_str)
    except subprocess.CalledProcessError as e:
        logger.error("Command failed for %s with return code %s", job_dir, e.returncode, exc_info=True)
        return ("failed", job_dir_str)
    except Exception:
        logger.exception("Unexpected error while processing %s", job_dir)
        return ("failed", job_dir_str)

# -------------------------------
logger.info("Starting Bulk Native Hitmap creation")
logger.info("Config: RESULTS_ROOT=%s | OUT_BASE=%s | CREATOR=%s | PYTHON=%s | DRY_RUN=%s",
            RESULTS_ROOT, OUT_BASE, CREATOR, PYTHON, DRY_RUN)

processed = 0
skipped_missing = 0
failed = 0

# Scan and collect all jobs first
jobs: list[tuple[str, str]] = []  # (job_dir_str, class_name)
for class_dir in sorted(RESULTS_ROOT.glob("CCBestClassdo40")):
    logger.debug("Scanning class directory: %s", class_dir)

    # Collect subdirectories matching desired patterns
    patterns = [
        "run*_sifi",
        "OptimisedGeometry_CodedMaskHIT_Spot*_1e10_protons_MK",
    ]
    for pattern in patterns:
        matched = list(class_dir.glob(pattern))
        logger.debug("Pattern '%s' matched %d entries in %s", pattern, len(matched), class_dir)
        for job_dir in matched:
            print(f"found class_dir: {class_dir}, job_dir: {job_dir}")
            jobs.append((str(job_dir), class_dir.name))

logger.info("Discovered %d jobs across class directories. Using up to %d workers.", len(jobs), MAX_WORKERS)

if jobs:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_process_single_job, job_dir_str, class_name) for job_dir_str, class_name in jobs]
        for fut in as_completed(futures):
            try:
                status, job_dir_str = fut.result()
                if status == "processed":
                    processed += 1
                elif status == "skipped_missing":
                    skipped_missing += 1
                elif status == "failed":
                    failed += 1
                else:
                    logger.warning("Unknown status '%s' for job %s", status, job_dir_str)
            except Exception:
                failed += 1
                logger.exception("Worker raised an exception")
else:
    logger.info("No jobs found. Nothing to do.")

logger.info("Summary: processed=%d | skipped_missing=%d | failed=%d", processed, skipped_missing, failed)
