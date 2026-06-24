#!/usr/bin/env python3
"""Create reduced-statistics hitmap replicas by random cluster subsampling.

This script samples clusters directly from the NN prediction arrays before hitmap
construction. For each requested fraction and replica, it builds independent
beamtime and/or spot hitmaps while preserving the existing output directory
shape expected by downstream tools.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from NativeHitmapCreator import generate_hitmaps


DEFAULT_RESULTS_ROOT = Path("/home/philippe/Desktop/results_without_archive")
DEFAULT_OUTPUT_ROOT = Path("/home/philippe/Desktop/results_reduced_statistics_hitmaps")
DEFAULT_FRACTIONS = (0.01, 0.02, 0.04, 0.10, 0.20, 0.50, 1.00)
DEFAULT_REPS = 100

SPOT_DATASETS = (
    "OptimisedGeometry_CodedMaskHIT_Spot1_1e10_protons_MK",
    "OptimisedGeometry_CodedMaskHIT_Spot2_1e10_protons_MK",
    "OptimisedGeometry_CodedMaskHIT_Spot3_1e10_protons_MK",
    "OptimisedGeometry_CodedMaskHIT_Spot4_1e10_protons_MK",
    "OptimisedGeometry_CodedMaskHIT_Spot5_1e10_protons_MK",
    "OptimisedGeometry_CodedMaskHIT_Spot6_1e10_protons_MK",
    "OptimisedGeometry_CodedMaskHIT_Spot7_1e10_protons_MK",
)

BEAMTIME_DATASETS = (
    "run00566_sifi",
    "run00567_sifi",
    "run00568_sifi",
    "run00569_sifi",
    "run00570_sifi",
    "run00571_sifi",
    "run00575_sifi",
    "run00576_sifi",
    "run00577_sifi",
    "run00578_sifi",
    "run00579_sifi",
    "run00580_sifi",
    "run00581_sifi",
)


def init_logger() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("build_reduced_statistics_hitmaps")


def fraction_label(fraction: float) -> str:
    pct = fraction * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct)):03d}pct"
    return f"{pct:06.2f}pct".replace(".", "p")


def replica_label(rep_idx: int) -> str:
    return f"rep_{rep_idx:03d}"


def stable_seed(seed_base: int, network: str, dataset_name: str, fraction: float, rep_idx: int) -> int:
    key = f"{seed_base}|{network}|{dataset_name}|{fraction:.8f}|{rep_idx}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 32)


def dataset_output_prefix(
    output_root: Path,
    network: str,
    fraction: float,
    rep_idx: int,
    dataset_group: str,
    dataset_name: str,
) -> Path:
    return (
        output_root
        / network
        / fraction_label(fraction)
        / replica_label(rep_idx)
        / dataset_group
        / dataset_name
        / dataset_name
    )


def iter_networks(results_root: Path, requested: Iterable[str]) -> list[str]:
    if requested:
        return list(requested)
    return sorted(path.name for path in results_root.glob("Norm_*") if path.is_dir())


def iter_dataset_specs(mode: str, requested: Optional[set[str]]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if mode in {"beamtime", "both"}:
        specs.extend(("beamtimeHitmaps", name) for name in BEAMTIME_DATASETS)
    if mode in {"simulation", "both"}:
        specs.extend(("spotHitmaps", name) for name in SPOT_DATASETS)
    if requested:
        specs = [spec for spec in specs if spec[1] in requested]
    return specs


def build_prediction_paths(network_dir: Path, dataset_name: str) -> tuple[Path, Path, Optional[Path]]:
    dataset_dir = network_dir / dataset_name
    energy_path = dataset_dir / f"{dataset_name}_regE_pred.npy"
    pos_path = dataset_dir / f"{dataset_name}_ClassXZ_pred.npy"
    metadata_path = dataset_dir / f"{dataset_name}_ClassXZ_pred_topk.npz"
    if not metadata_path.exists():
        metadata_path = None
    return energy_path, pos_path, metadata_path


def build_task_list(
    results_root: Path,
    output_root: Path,
    networks: list[str],
    fractions: list[float],
    n_reps: int,
    mode: str,
    seed_base: int,
    requested_datasets: Optional[set[str]],
    requested_replica_indices: Optional[list[int]],
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for network in networks:
        network_dir = results_root / network
        specs = iter_dataset_specs(mode, requested_datasets)
        for dataset_group, dataset_name in specs:
            energy_path, pos_path, metadata_path = build_prediction_paths(network_dir, dataset_name)
            if not energy_path.exists() or not pos_path.exists():
                raise FileNotFoundError(
                    f"Missing prediction files for {network}/{dataset_name}: {energy_path} or {pos_path}"
                )
            for fraction in fractions:
                if requested_replica_indices is not None:
                    if float(fraction) >= 1.0:
                        replica_indices = [rep_idx for rep_idx in requested_replica_indices if rep_idx == 0]
                    else:
                        replica_indices = [rep_idx for rep_idx in requested_replica_indices if rep_idx >= 0]
                else:
                    fraction_reps = 1 if float(fraction) >= 1.0 else n_reps
                    replica_indices = list(range(fraction_reps))

                for rep_idx in replica_indices:
                    tasks.append(
                        {
                            "network": network,
                            "dataset_group": dataset_group,
                            "dataset_name": dataset_name,
                            "energy_path": energy_path,
                            "pos_path": pos_path,
                            "metadata_path": metadata_path,
                            "fraction": fraction,
                            "rep_idx": rep_idx,
                            "output_prefix": dataset_output_prefix(
                                output_root,
                                network,
                                fraction,
                                rep_idx,
                                dataset_group,
                                dataset_name,
                            ),
                            "random_seed": stable_seed(seed_base, network, dataset_name, fraction, rep_idx),
                        }
                    )
    return tasks


def process_one_task(
    task: dict[str, object],
    overwrite: bool,
    e_threshold: float,
    min_confidence: float,
    min_margin: float,
    max_entropy: float,
) -> tuple[str, str, float, int, str]:
    network = str(task["network"])
    dataset_name = str(task["dataset_name"])
    fraction = float(task["fraction"])
    rep_idx = int(task["rep_idx"])
    output_prefix = Path(str(task["output_prefix"]))
    if dataset_name.startswith("run"):
        run_id = dataset_name.split("_", 1)[0]
    else:
        parts = dataset_name.split("_")
        run_id = parts[2] if len(parts) > 2 else dataset_name
    final_root = output_prefix.parent / f"{run_id}_hitmaps.root"

    if final_root.exists() and not overwrite:
        return network, dataset_name, fraction, rep_idx, "skipped"

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    generate_hitmaps(
        pred_energy_path=str(task["energy_path"]),
        pred_pos_path=str(task["pos_path"]),
        output_prefix=str(output_prefix),
        e_threshold=e_threshold,
        label_path=None,
        position_metadata_path=str(task["metadata_path"]) if task["metadata_path"] is not None else None,
        save_extras=False,
        reduced_statistics=fraction,
        random_seed=int(task["random_seed"]),
        min_confidence=min_confidence,
        min_margin=min_margin,
        max_entropy=max_entropy,
    )
    return network, dataset_name, fraction, rep_idx, "done"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build reduced-statistics hitmap replicas from NN clusters")
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--networks", nargs="*", default=[])
    ap.add_argument("--fractions", nargs="*", type=float, default=list(DEFAULT_FRACTIONS))
    ap.add_argument("--n-reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--mode", choices=("beamtime", "simulation", "both"), default="both")
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--replica-indices", nargs="*", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--e-threshold", type=float, default=7000.0)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--max-entropy", type=float, default=1.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logger = init_logger()

    requested_datasets = set(args.datasets) if args.datasets else None
    requested_replica_indices = list(args.replica_indices) if args.replica_indices is not None else None
    networks = iter_networks(args.results_root, args.networks)
    if not networks:
        logger.error("No networks found under %s", args.results_root)
        return 1

    tasks = build_task_list(
        args.results_root,
        args.output_root,
        networks,
        [float(value) for value in args.fractions],
        int(args.n_reps),
        str(args.mode),
        int(args.seed_base),
        requested_datasets,
        requested_replica_indices,
    )
    if not tasks:
        logger.error("No tasks generated for mode=%s", args.mode)
        return 1

    logger.info(
        "Prepared %d hitmap tasks for %d network(s), fractions=%s, reps=%d, mode=%s",
        len(tasks),
        len(networks),
        ",".join(fraction_label(float(value)) for value in args.fractions),
        args.n_reps,
        args.mode,
    )

    completed = 0
    skipped = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_one_task,
                task,
                bool(args.overwrite),
                float(args.e_threshold),
                float(args.min_confidence),
                float(args.min_margin),
                float(args.max_entropy),
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            try:
                network, dataset_name, fraction, rep_idx, status = future.result()
            except Exception:
                failed += 1
                logger.exception("Reduced-statistics hitmap task failed")
                continue

            if status == "skipped":
                skipped += 1
            else:
                completed += 1
            logger.info(
                "%s | %s | %s | %s | %s",
                status,
                network,
                dataset_name,
                fraction_label(fraction),
                replica_label(rep_idx),
            )

    logger.info("Summary: completed=%d skipped=%d failed=%d output_root=%s", completed, skipped, failed, args.output_root)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())