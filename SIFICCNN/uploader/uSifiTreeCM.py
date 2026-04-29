"""
Create a SiFi tree (ROOT) from NN predictions.

Inputs
- Energy prediction file (NumPy .npy)
- Position prediction file (NumPy .npy of fiber IDs)

Output
- ROOT file with SFibersHit category filled per predicted cluster

Notes
- Energy is written in keV (prediction assumed to be MeV, converted by x1000).
- Fiber ID is converted to (layer, fiber) by divmod(fibers_per_layer).
"""


from __future__ import annotations

import argparse
import ctypes
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from tqdm import tqdm

import ROOT  # type: ignore

# Optimize ROOT behavior for non-interactive scripts
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.StartGUIThread = False
ROOT.PyConfig.IgnoreCommandLineOptions = True

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _compile_root_helpers() -> None:
    """Make sure required C++ headers and helper are available to ROOT."""
    ROOT.gInterpreter.ProcessLine("#include <SiFi.h>")
    ROOT.gInterpreter.ProcessLine("#include <SCategoryManager.h>")
    ROOT.gInterpreter.ProcessLine("#include <SLocator.h>")
    ROOT.gInterpreter.ProcessLine("#include <SFibersHit.h>")

    ROOT.gInterpreter.ProcessLine(
        r'''
        SFibersHit* AddSFibersHit(SCategory* cat, const SLocator& loc) {
            TObject*& slot = cat->getSlot(loc);
            new (slot) SFibersHit();
            return reinterpret_cast<SFibersHit*>(slot);
        }
        '''
    )


def _register_sfibers_category(modules: int, layers: int, fibers_per_layer: int) -> Tuple[object, object]:
    """Register and build the SFibersHit category; return (sifi, pCatFibHit)."""
    sifi = ROOT.sifi()

    sizes_np = np.array([modules, layers, fibers_per_layer], dtype=np.uint64)
    sizes_ptr = sizes_np.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong))

    dm = ROOT.SiFi.instance()
    ok = dm.registerCategory(
        ROOT.SCategory.CatFibersHit,
        "SFibersHit",
        3,
        sizes_ptr,
        False,
    )
    if not ok:
        raise RuntimeError("Could not register SFibersHit category")

    pCatFibHit = sifi.buildCategory(ROOT.SCategory.CatFibersHit)
    if not pCatFibHit:
        raise RuntimeError("No CatFibersHit category found after build")

    return sifi, pCatFibHit




def write_sifi_tree(
    layers: np.ndarray,
    fibers: np.ndarray,
    energies_mev: np.ndarray,
    output_file: Path,
    modules: int,
    layers_count: int,
    fibers_per_layer: int,
) -> None:
    """Write predictions to a SiFi ROOT file as SFibersHit entries.

    - layers/fibers: per-cluster indices
    - energies_mev: energy per cluster (MeV)
    """
    if not (len(layers) == len(fibers) == len(energies_mev)):
        raise ValueError("layers, fibers and energies must have same length")

    _compile_root_helpers()
    sifi = ROOT.sifi()
    # Match original order: set output name before booking
    sifi.setOutputFileName(str(output_file))
    sifi.setTree(ROOT.TTree())
    sifi.book()
    sifi, pCatFibHit = _register_sfibers_category(modules, layers_count, fibers_per_layer)

    n_events = len(energies_mev)
    sifi.loop(n_events)

    for i in tqdm(range(n_events), desc="Writing events"):
        pCatFibHit.clear()

        # One hit per event for now
        mod = 0
        lay = int(layers[i])
        fib = int(fibers[i])

        loc = ROOT.SLocator(3)
        loc[0] = mod
        loc[1] = lay
        loc[2] = fib

        pHit = ROOT.AddSFibersHit(pCatFibHit, loc)
        pHit.setAddress(mod, lay, fib)

        # Convert MeV to keV for ROOT
        energy_kev = float(energies_mev[i]) * 1000.0

        # Dummy placeholders for unused fields
        pHit.setTime(int(-100), float(-100.0))
        pHit.setE(energy_kev, float(-100.0))
        pHit.setU(int(-100), float(-100.0))

        sifi.fill()

    sifi.save()
    logging.info(f"Saved SiFi tree: {output_file}")



def _load_predictions(
    prediction_dir: Path,
    dataset_name: str,
    fibers_per_layer: int,
    energy_file: Optional[Path] = None,
    position_file: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load energy (MeV) and position (fiber id) predictions and split to (layer, fiber)."""
    if energy_file is None:
        energy_file = prediction_dir / f"regE_bin_{dataset_name}.npy"
    if position_file is None:
        position_file = prediction_dir / f"pos_clas_bin_{dataset_name}.npy"

    if not energy_file.exists():
        raise FileNotFoundError(f"Energy prediction file not found: {energy_file}")
    if not position_file.exists():
        raise FileNotFoundError(f"Position prediction file not found: {position_file}")

    energies_mev = np.load(energy_file)
    pos_ids = np.load(position_file)

    # Ensure 1D arrays
    energies_mev = np.asarray(energies_mev).reshape(-1)
    pos_ids = np.asarray(pos_ids).reshape(-1)

    if len(energies_mev) != len(pos_ids):
        raise ValueError(
            f"Mismatched lengths: energies={len(energies_mev)} vs positions={len(pos_ids)}"
        )

    layers, fibers = np.divmod(pos_ids, fibers_per_layer)
    return layers.astype(int), fibers.astype(int), energies_mev.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a SiFi ROOT tree from NN predictions.")
    parser.add_argument("--prediction_dir", type=Path, required=True, help="Directory with prediction .npy files")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name used in prediction filenames")
    parser.add_argument("--output", type=Path, help="Output ROOT file path (default: <dataset_name>.root)")

    # Geometry
    parser.add_argument("--modules", type=int, default=1, help="Number of modules (default: 1)")
    parser.add_argument("--layers", dest="layers_count", type=int, default=7, help="Number of layers (default: 7)")
    parser.add_argument("--fibers_per_layer", type=int, default=55, help="Fibers per layer (default: 55)")

    # Optional custom file names
    parser.add_argument("--energy_file", type=Path, help="Path to energy .npy (MeV)")
    parser.add_argument("--position_file", type=Path, help="Path to position .npy (fiber IDs)")

    args = parser.parse_args()

    output_file = args.output or Path(f"{args.dataset_name}.root")

    logging.info("Loading predictions…")
    layers, fibers, energies_mev = _load_predictions(
        args.prediction_dir,
        args.dataset_name,
        args.fibers_per_layer,
        args.energy_file,
        args.position_file,
    )

    logging.info("Writing ROOT (SiFi tree)…")
    write_sifi_tree(
        layers=layers,
        fibers=fibers,
        energies_mev=energies_mev,
        output_file=output_file,
        modules=args.modules,
        layers_count=args.layers_count,
        fibers_per_layer=args.fibers_per_layer,
    )


if __name__ == "__main__":
    main()





