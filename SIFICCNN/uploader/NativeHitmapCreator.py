"""
This script generates hitmaps and energy maps from predicted energy and position data,
saving the results as .npy, .npz, .png, and a ROOT file with TH2D histograms matching
the structure and names of the provided PyROOT macro.

It can be used as an alternative to the ROOT-based hitmap creator and should be faster.

New: writes a ROOT file named "<output_prefix>_hitmaps_sim.root" with histograms:
  - hFiberHitMap_AE_t0000, hFiberHitMap_AE_t0500, ...
  - hEnergyDepoMap_AE_t0000, ...
  - hFiberHitMap_NS_t.... and hEnergyDepoMap_NS_t....
  - hFiberHitMap_AE_t0000_ERR (error map from AE t0000)

uproot is used to write TH2D histograms.

Functions:
----------
generate_hitmaps(pred_energy_path, pred_pos_path, output_prefix=None, e_threshold=120000,
                 label_path=None, dead_fibers_txt=None, exclude_dead_fibers=False,
                 save_extras=False)
    Generates hitmaps and energy maps for different energy thresholds from prediction files
    and writes a ROOT file mirroring the PyROOT macro output.

Parameters
----------
pred_energy_path : str
    Path to the .npy file containing predicted energy values (in MeV, will be converted to keV).
pred_pos_path : str
    Path to the .npy file containing predicted position values (flat fiber IDs).
output_prefix : str, optional
    Prefix for output files (also used to name the ROOT file: <prefix>_hitmaps_sim.root).
e_threshold : float or int, optional
    Upper energy threshold (in keV) for the last bin. Default is 120000 to match the macro.
label_path : str, optional
    Path to a .npy array of integer labels; NS maps are filled only for labels in {1,2} if provided.
dead_fibers_txt : str, optional
    Path to "Dead_Fibers.txt" (two integers per line: yChannel xChannel). If provided with
    exclude_dead_fibers=True, corresponding bins are zeroed in all maps and the error map.
exclude_dead_fibers : bool, optional
    Whether to zero out dead fibers specified by dead_fibers_txt.
save_extras : bool, optional
    If True, also write NPY/NPZ/PNG outputs; by default only the ROOT file is written.

Outputs
-------
- .npy files for hitmaps and energy maps for each threshold.
- .png images visualizing the hitmaps and energy maps.
- A combined .npz archive containing all hitmaps and energy maps.
- A ROOT file <output_prefix>_hitmaps_sim.root with TH2D histograms named like the macro.

Usage:
------
Run as a script with the following arguments:
    --energy_npy         Path to regE_bin_<dataset>.npy (required)
    --position_npy       Path to pos_clas_bin_<dataset>.npy (required)
    --output_prefix      Prefix for output files (required)
    --e_threshold        Upper energy threshold in keV (optional, default: 120000)
    --label_npy          Optional labels .npy for NS maps (labels in {1,2} are counted)
    --dead_fibers_txt    Optional path to Dead_Fibers.txt
    --exclude_dead       If set, zero out dead fibers from the txt file
    --save_extras        Also write npy/npz/png outputs (default off)
"""
import os
import numpy as np
import matplotlib
# Use a non-interactive backend to avoid Qt/Wayland plugin issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import inspect
import argparse
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)

# Constants
N_LAYERS_PER_MODULE = 7
N_FIBERS_PER_LAYER = 55
THRESHOLDS = [0, 500, 1000, 1500]
HITMAP_TYPES = ["AE", "NS"]

try:
    import uproot  # type: ignore
    from uproot.writing.identify import to_TH2x as _to_TH2x, to_TAxis as _to_TAxis  # histogram writers
    _HAS_UPROOT = True
except Exception:  # pragma: no cover
    _HAS_UPROOT = False


def _apply_dead_fibers_mask(arrays: Dict[str, Dict[int, np.ndarray]],
                            dead_fibers_txt: str) -> None:
    """Zero out bins listed in the dead fibers file for all provided arrays.

    The file should contain lines: "yChannel xChannel" (layer fiber).
    """
    if not os.path.isfile(dead_fibers_txt):
        raise FileNotFoundError(f"Dead fibers file not found: {dead_fibers_txt}")

    with open(dead_fibers_txt, "r") as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            y = int(parts[0])
            x = int(parts[1])
            # Zero out for all types and thresholds
            for t in arrays.values():
                for m in t.values():
                    if 0 <= y < N_LAYERS_PER_MODULE and 0 <= x < N_FIBERS_PER_LAYER:
                        m[y, x] = 0


def _write_root_with_uproot(out_path: str,
                            hitmaps: Dict[str, Dict[int, np.ndarray]],
                            energymaps: Dict[str, Dict[int, np.ndarray]],
                            err_map: np.ndarray) -> None:
    """Write TH2D histograms using uproot with explicit axis metadata and stats."""
    if not _HAS_UPROOT:
        raise RuntimeError("uproot not available")

    # Define axes like in the macro
    xedges = np.linspace(0.0, float(N_FIBERS_PER_LAYER), N_FIBERS_PER_LAYER + 1)
    yedges = np.linspace(0.0, float(N_LAYERS_PER_MODULE), N_LAYERS_PER_MODULE + 1)
    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    ycenters = (yedges[:-1] + yedges[1:]) / 2.0
    fXaxis = _to_TAxis(
        fNbins=len(xedges) - 1,
        fXmin=float(xedges[0]),
        fXmax=float(xedges[-1]),
        fXbins=xedges,
        fName="x",
        fTitle="",
    )
    fYaxis = _to_TAxis(
        fNbins=len(yedges) - 1,
        fXmin=float(yedges[0]),
        fXmax=float(yedges[-1]),
        fXbins=yedges,
        fName="y",
        fTitle="",
    )

    def _make_th2(values: np.ndarray, title: str, entries: float):
        vals = values.astype("float64", copy=False)
        # Compute histogram statistics
        fTsumw = float(vals.sum())
        fTsumw2 = float((vals * vals).sum())
        fTsumwx = float((vals * xcenters[None, :]).sum())
        fTsumwy = float((vals * ycenters[:, None]).sum())
        fTsumwx2 = float((vals * (xcenters[None, :] ** 2)).sum())
        fTsumwy2 = float((vals * (ycenters[:, None] ** 2)).sum())
        fTsumwxy = float((vals * ycenters[:, None] * xcenters[None, :]).sum())
        # Build 1D storage including under/overflow bins using ROOT's global bin formula:
        # globalBin = (ny+2) * binx + biny, where binx in [0..nx+1], biny in [0..ny+1].
        ny, nx = vals.shape
        data_flat = np.zeros((nx + 2) * (ny + 2), dtype="float64")
        # Fill only in-range bins (shift by +1 to account for underflow)
        for y in range(ny):
            for x in range(nx):
                v = vals[y, x]
                if v != 0.0:
                    binx = x + 1  # 1..nx
                    biny = y + 1  # 1..ny
                    # ROOT: TH2::GetBin(binx, biny) = (nx+2) * biny + binx
                    idx = (nx + 2) * biny + binx
                    data_flat[idx] = v
        # Keyword-based call aligned with uproot 5.6.0 signature (requires fName)
        try:
            return _to_TH2x(
                fName=title,
                fTitle=title,
                data=data_flat,
                fEntries=float(entries),
                fTsumw=fTsumw,
                fTsumw2=fTsumw2,
                fTsumwx=fTsumwx,
                fTsumwx2=fTsumwx2,
                fTsumwy=fTsumwy,
                fTsumwy2=fTsumwy2,
                fTsumwxy=fTsumwxy,
                fSumw2=None,
                fXaxis=fXaxis,
                fYaxis=fYaxis,
            )
        except TypeError as e:
            sig = None
            try:
                sig = str(inspect.signature(_to_TH2x))
            except Exception:
                sig = "<unavailable>"
            raise TypeError(f"Failed to construct TH2 with uproot identify.to_TH2x: {e}. Signature: {sig}")

    with uproot.recreate(out_path) as fout:
        for htype in HITMAP_TYPES:
            for thr in THRESHOLDS:
                name_hit = f"hFiberHitMap_{htype}_t{thr:04d}"
                name_ede = f"hEnergyDepoMap_{htype}_t{thr:04d}"
                entries = float(np.nansum(hitmaps[htype][thr]))
                fout[name_hit] = _make_th2(hitmaps[htype][thr], name_hit, entries)
                fout[name_ede] = _make_th2(energymaps[htype][thr], name_ede, entries)

        # For error map, use entries from AE threshold 0 (t0000)
        ae_entries = float(np.nansum(hitmaps["AE"][THRESHOLDS[0]]))
        fout["hFiberHitMap_AE_t0000_ERR"] = _make_th2(err_map, "hFiberHitMap_AE_t0000_ERR", ae_entries)


## PyROOT fallback removed per user preference (uproot-only writer)

def generate_hitmaps(
    pred_energy_path: str,
    pred_pos_path: str,
    output_prefix: Optional[str] = None,
    e_threshold: float = 120000,
    label_path: Optional[str] = None,
    dead_fibers_txt: Optional[str] = None,
    exclude_dead_fibers: bool = False,
    save_extras: bool = False,
    reduced_statistics: float = 1.0,
):
    print(f"output_prefix: {output_prefix}, e_path: {pred_energy_path}, p_path: {pred_pos_path}")
    """Build hitmaps/energy maps for thresholds and write outputs including a ROOT file."""
    if output_prefix is None:
        raise ValueError("output_prefix is required")

    logging.info(
        "Loading predictions:\n  Energy: %s\n  Position: %s%s",
        pred_energy_path,
        pred_pos_path,
        f"\n  Labels: {label_path}" if label_path else "",
    )

    # Ensure 1D arrays to avoid deprecation warnings on scalar conversion
    energy = np.asarray(np.load(pred_energy_path)).ravel() * 1000.0  # MeV -> keV
    pos = np.load(pred_pos_path) 
    if pos.ndim > 1:
        pos = pos[:,1] # take fiber id if multiple outputs

    # Index array for random selection (kept for consistent slicing of labels)
    idx = None

    # Reduce statistics if requested
    if 0.0 < reduced_statistics < 1.0:
        n_events = energy.shape[0]
        n_select = int(n_events * reduced_statistics)
        logging.info(f"Reducing statistics: selecting {n_select} / {n_events} events")
        # Select a uniformly spaced subset of indices across the event range
        # (deterministic, reproducible). Use linspace and cast to int.
        if n_select <= 0:
            idx = np.array([], dtype=int)
            energy = energy[:0]
            pos = pos[:0]
        else:
            # np.linspace with dtype=int produces evenly spaced integer indices
            # within [0, n_events-1]. This ensures a uniform coverage.
            idx = np.linspace(0, n_events - 1, num=n_select, dtype=int)
            energy = energy[idx]
            pos = pos[idx]


    if energy.shape[0] != pos.shape[0]:
        raise ValueError("Mismatch between energy and position array lengths")

    labels = None
    if label_path and os.path.isfile(label_path):
        labels = np.asarray(np.load(label_path)).ravel()
        if 0.0 < reduced_statistics < 1.0:
            # Apply the same random indices if we sampled randomly earlier.
            if idx is None:
                # Fallback: use last-chunk behavior (shouldn't happen)
                if n_select <= 0:
                    labels = labels[:0]
                else:
                    labels = labels[-n_select:]
            else:
                labels = labels[idx]
        if labels.shape[0] != energy.shape[0]:
            logging.warning(
                "Labels array length (%d) != energy/pos length (%d); ignoring labels",
                labels.shape[0], energy.shape[0]
            )
            labels = None

    # Decode flat fiber index -> (layer, fiber)
    layer, fiber = np.divmod(pos.astype(int), N_FIBERS_PER_LAYER)

    # Initialize maps for both types
    hitmaps: Dict[str, Dict[int, np.ndarray]] = {
        htype: {thr: np.zeros((N_LAYERS_PER_MODULE, N_FIBERS_PER_LAYER), dtype=float) for thr in THRESHOLDS}
        for htype in HITMAP_TYPES
    }
    energy_maps: Dict[str, Dict[int, np.ndarray]] = {
        htype: {thr: np.zeros((N_LAYERS_PER_MODULE, N_FIBERS_PER_LAYER), dtype=float) for thr in THRESHOLDS}
        for htype in HITMAP_TYPES
    }

    # Fill maps
    for i in range(energy.shape[0]):
        e = float(energy[i])
        l = int(layer[i])
        f = int(fiber[i])
        if l < 0 or l >= N_LAYERS_PER_MODULE or f < 0 or f >= N_FIBERS_PER_LAYER:
            continue

        for j, thr in enumerate(THRESHOLDS):
            upper = THRESHOLDS[j + 1] if j + 1 < len(THRESHOLDS) else e_threshold
            if thr < e <= upper:
                # AE maps: always fill
                hitmaps["AE"][thr][l, f] += 1.0
                energy_maps["AE"][thr][l, f] += e

                # NS maps: only if label in {1,2}; if labels missing, mirror AE (inform user once)
                do_ns = False
                if labels is not None:
                    lab = int(labels[i])
                    do_ns = (lab > 0 and lab < 3)
                else:
                    do_ns = False
                if do_ns:
                    hitmaps["NS"][thr][l, f] += 1.0
                    energy_maps["NS"][thr][l, f] += e
                break

    logging.info("Finished building hitmaps and energy maps.")

    # Error map from AE t0000: 1/sqrt(count) where count>0, else 0
    ae_t0 = hitmaps["AE"][THRESHOLDS[0]]
    with np.errstate(divide="ignore"):
        err_map = np.where(ae_t0 > 0, 1.0 / np.sqrt(ae_t0), 0.0)

    # Optionally zero dead fibers
    if exclude_dead_fibers and dead_fibers_txt:
        all_maps = {
            **{f"hit_{ht}_{thr}": hitmaps[ht][thr] for ht in HITMAP_TYPES for thr in THRESHOLDS},
            **{f"ene_{ht}_{thr}": energy_maps[ht][thr] for ht in HITMAP_TYPES for thr in THRESHOLDS},
            "err": err_map,
        }
        # Reshape into type->thr mapping for mask helper
        structured = {
            "hit_AE": {thr: hitmaps["AE"][thr] for thr in THRESHOLDS},
            "hit_NS": {thr: hitmaps["NS"][thr] for thr in THRESHOLDS},
            "ene_AE": {thr: energy_maps["AE"][thr] for thr in THRESHOLDS},
            "ene_NS": {thr: energy_maps["NS"][thr] for thr in THRESHOLDS},
            "err": {0: err_map},
        }
        _apply_dead_fibers_mask(structured, dead_fibers_txt)
        err_map = structured["err"][0]

    # Ensure output directory exists (exclude filename part)
    output_dir = os.path.dirname(os.path.abspath(output_prefix))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if pred_energy_path.split("/")[-1].startswith("run"):
        run_id = pred_energy_path.split("/")[-1].split("_")[0]  # e.g. run00581
    else:
        run_id = pred_energy_path.split("/")[-1].split("_")[2]  
    output_file_prefix = os.path.join(output_prefix, run_id)

    # Optionally save NPY/PNG for AE maps
    if save_extras:
        for thr in THRESHOLDS:
            np.save(f"{output_file_prefix}_hitmap_AE_thr{thr:04d}.npy", hitmaps["AE"][thr])
            np.save(f"{output_file_prefix}_energy_AE_thr{thr:04d}.npy", energy_maps["AE"][thr])

            # Plot hitmap
            plt.figure(figsize=(10, 2))
            plt.imshow(hitmaps["AE"][thr], origin="lower", aspect="equal", cmap="viridis")
            plt.colorbar(label="Hit Count")
            plt.xlabel("Fiber Index")
            plt.ylabel("Layer Index")
            plt.tight_layout()
            plt.savefig(f"{output_file_prefix}_hitmap_AE_thr{thr:04d}.png")
            plt.close()

            # Plot energy map
            plt.figure(figsize=(10, 2))
            plt.imshow(energy_maps["AE"][thr], origin="lower", aspect="equal", cmap="plasma")
            plt.colorbar(label="Energy [keV]")
            plt.xlabel("Fiber Index")
            plt.ylabel("Layer Index")
            plt.tight_layout()
            plt.savefig(f"{output_file_prefix}_energy_AE_thr{thr:04d}.png")
            plt.close()

    # Save all maps in a single .npz archive (AE + NS)
    if save_extras:
        combined_dict = {}
        for htype in HITMAP_TYPES:
            for thr in THRESHOLDS:
                combined_dict[f"hitmap_{htype}_thr{thr:04d}"] = hitmaps[htype][thr]
                combined_dict[f"energy_{htype}_thr{thr:04d}"] = energy_maps[htype][thr]
        combined_dict["error_map_AE_t0000"] = err_map
        np.savez(f"{output_file_prefix}_allmaps.npz", **combined_dict)

    # Write ROOT file with the same histogram names as the macro (uproot-only)
    out_root = f"{output_file_prefix}_hitmaps.root"
    if not _HAS_UPROOT:
        raise RuntimeError("uproot is required to write ROOT histograms; please install it (pip install uproot)")
    _write_root_with_uproot(out_root, hitmaps, energy_maps, err_map)
    logging.info("Wrote ROOT histograms with uproot: %s", out_root)

    if save_extras:
        logging.info(f"Saved ROOT + .npy/.npz/.png to prefix: {output_file_prefix}")
    else:
        logging.info(f"Saved ROOT file to prefix: {output_file_prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create native hitmaps from prediction .npy files.")
    parser.add_argument("--energy_npy", required=True, help="Path to regEnergyFile.npy")
    parser.add_argument("--position_npy", required=True, help="Path to regPositionFile.npy")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output files (e.g. output/run00566)")
    parser.add_argument("--e_threshold", type=int, default=120000, help="Upper energy threshold (keV) for the last bin; default 120000")
    parser.add_argument("--label_npy", required=False, default=None, help="Optional labels .npy for NS maps (labels in {1,2} are counted)")
    parser.add_argument("--dead_fibers_txt", required=False, default=None, help="Optional path to Dead_Fibers.txt (yChannel xChannel per line)")
    parser.add_argument("--exclude_dead", action="store_true", help="Zero out dead fibers from the txt file if provided")
    parser.add_argument("--save_extras", action="store_true", help="Also write npy/npz/png outputs (default off)")
    parser.add_argument("--reduced_statistics", type=float, default=1.0, help="Fraction of events to process (for testing); default 1.0 (all events)")
    args = parser.parse_args()

    generate_hitmaps(
        args.energy_npy,
        args.position_npy,
        args.output_prefix,
        args.e_threshold,
        label_path=args.label_npy,
        dead_fibers_txt=args.dead_fibers_txt,
        exclude_dead_fibers=args.exclude_dead,
        save_extras=args.save_extras,
        reduced_statistics=args.reduced_statistics,
    )