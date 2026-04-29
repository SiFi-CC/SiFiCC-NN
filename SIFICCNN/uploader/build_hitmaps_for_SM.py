#!/usr/bin/env python3
"""
Build AE hitmaps and energy deposition maps from prediction .npy files using
NativeHitmapCreator.generate_hitmaps, and save them in the exact filenames
expected by generateSM.py:

    hitmaps/bin_{bin_id}_hitmaps.root

Notes
-----
- Only ROOT files are produced (no npy/npz/png). We do this by setting
    save_extras=False in NativeHitmapCreator.
- We do NOT use run IDs; we write per-bin outputs and rename the ROOT file to
    reflect the bin number directly inside the hitmaps folder.

Usage
-----
python SIFICCNN/uploader/build_hitmaps_for_SM.py \
  --pred-dir /path/to/preds \
  --out-dir  /net/scratch_g4rt1/clement/SM/paper2025/CCBestClassdo40/systemMatrix \
  [--limit 200]

The script will create an 'hitmaps' subfolder in out-dir and populate it with
NPY files for thresholds [0, 500, 1000, 1500].
"""
import argparse
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple, Optional

# Ensure we can import sibling module when executed directly
CURRENT_DIR = Path(__file__).resolve().parent

try:
    from NativeHitmapCreator import generate_hitmaps, THRESHOLDS  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Failed to import NativeHitmapCreator: {e}\nMake sure you run this script from the repository or add SIFICCNN/uploader to PYTHONPATH.")


def init_logger() -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("build_hitmaps_for_SM")


BIN_REGEX = re.compile(r"bin_?(\d+)\.npy$")


def find_bin_id(path: Path) -> int | None:
    m = BIN_REGEX.search(path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def scan_predictions(pred_dir: Path) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    """Return mappings bin_id -> energy_path and bin_id -> position_path.

    Accepts both patterns 'bin_123.npy' and 'bin123.npy'.
    Energy files are detected by 'regE' in name; position by 'pos' or 'clas'.
    """
    energy_by_bin: Dict[int, Path] = {}
    pos_by_bin: Dict[int, Path] = {}

    for p in pred_dir.glob("*.npy"):
        bid = find_bin_id(p)
        if bid is None:
            continue
        name_lower = p.name.lower()
        if "rege" in name_lower:
            energy_by_bin[bid] = p
        elif "pos" in name_lower or "clas" in name_lower:
            pos_by_bin[bid] = p

    return energy_by_bin, pos_by_bin


def _process_one_bin(
    bid: int,
    energy_path: Path,
    pos_path: Path,
    out_dir: Path,
    dead_fibers_txt: Optional[Path],
    exclude_dead: bool,
    overwrite: bool,
) -> Tuple[int, bool, bool, bool, str]:
    """Worker that generates AE maps for a single bin and copies NPYs to targets.

    Returns: (bid, success, skipped, had_generate_error, message)
    """
    try:
        hitmaps_dir = out_dir / "hitmaps"
        hitmaps_dir.mkdir(parents=True, exist_ok=True)

        # Final ROOT output path we will write/rename to (skip if exists and no overwrite)
        final_root = hitmaps_dir / f"bin_{bid:03d}_hitmaps.root"
        if (not overwrite) and final_root.exists():
            return (bid, True, True, False, "already-exists")

        # Use a short-lived per-bin working directory inside hitmaps to avoid clashes
        bin_work_dir = hitmaps_dir / f".bin_{bid:03d}_work"
        bin_work_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = bin_work_dir

        # Run generation (may raise during ROOT writing)
        had_generate_error = False
        try:
            generate_hitmaps(
                pred_energy_path=str(energy_path),
                pred_pos_path=str(pos_path),
                output_prefix=str(out_prefix),
                e_threshold=120000,
                label_path=None,
                dead_fibers_txt=str(dead_fibers_txt) if dead_fibers_txt else None,
                exclude_dead_fibers=bool(exclude_dead and dead_fibers_txt),
                save_extras=False,
            )
        except Exception:
            had_generate_error = True

        # Move/rename the produced ROOT file to the final bin-based name
        produced_roots = list(out_prefix.glob("*_hitmaps.root"))
        if not produced_roots:
            return (bid, False, False, had_generate_error, "missing-produced-root")
        # If multiple matches, pick the first deterministically (sorted)
        produced_roots.sort()
        src_root = produced_roots[0]
        shutil.move(str(src_root), str(final_root))

        # Cleanup work directory
        try:
            for p in bin_work_dir.glob("*"):
                p.unlink(missing_ok=True)
            bin_work_dir.rmdir()
        except Exception:
            pass

        return (bid, True, False, had_generate_error, "ok")
    except Exception as e:
        return (bid, False, False, False, f"exception: {e}")


def main():
    logger = init_logger()

    ap = argparse.ArgumentParser(description="Generate AE hitmaps/energy maps for generateSM.py")
    ap.add_argument("--pred-dir", required=True, type=Path, help="Directory containing pos/energy prediction .npy files")
    ap.add_argument("--out-dir", required=True, type=Path, help="Base output directory (we create 'hitmaps' under this)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of bins to process")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing target NPYs")
    ap.add_argument("--dead-fibers-txt", type=Path, default=None, help="Optional path to Dead_Fibers.txt")
    ap.add_argument("--exclude-dead", action="store_true", help="Zero out dead fibers listed in the txt file")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Number of parallel worker processes")
    args = ap.parse_args()

    pred_dir: Path = args.pred_dir
    out_dir: Path = args.out_dir
    hitmaps_dir = out_dir / "hitmaps"
    hitmaps_dir.mkdir(parents=True, exist_ok=True)

    energy_by_bin, pos_by_bin = scan_predictions(pred_dir)
    common_bins = sorted(set(energy_by_bin).intersection(pos_by_bin))
    if args.limit is not None:
        common_bins = common_bins[: args.limit]

    if not common_bins:
        logger.error("No matching pos/energy prediction pairs found in %s", pred_dir)
        return 1

    logger.info("Found %d matched bins (pos+energy)", len(common_bins))

    processed = skipped = failed = 0
    had_errors_but_copied = 0

    # Submit work in parallel
    logger.info("Starting multiprocessing with %d workers", args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for bid in common_bins:
            futures.append(
                ex.submit(
                    _process_one_bin,
                    bid,
                    energy_by_bin[bid],
                    pos_by_bin[bid],
                    out_dir,
                    args.dead_fibers_txt,
                    args.exclude_dead,
                    args.overwrite,
                )
            )

        for fut in as_completed(futures):
            bid, success, skipped_bin, had_gen_err, message = fut.result()
            if skipped_bin:
                skipped += 1
                logger.info("bin %03d: skipped (%s)", bid, message)
            elif success:
                processed += 1
                if had_gen_err:
                    had_errors_but_copied += 1
                    logger.warning("bin %03d: copied successfully but generation had errors", bid)
                else:
                    logger.info("bin %03d: done", bid)
            else:
                failed += 1
                logger.error("bin %03d: failed (%s)", bid, message)

    logger.info(
        "Summary: processed=%d | skipped=%d | failed=%d | with_gen_errors=%d | output=%s",
        processed, skipped, failed, had_errors_but_copied, hitmaps_dir,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
