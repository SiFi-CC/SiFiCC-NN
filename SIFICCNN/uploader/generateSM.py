import numpy as np
import ROOT
import os
import glob

# Configuration
NUM_FILES = 200
N_GAMMAS_TOTAL = 37798000

# Base directory for this dataset (outputs will be saved here)
BASE_DIR = "/net/scratch_g4rt1/clement/SM/paper2025/CCBestClassdo40/systemMatrix"

# Directory where input ROOT files are stored (one per bin, containing histograms)
# The script will search using INPUT_ROOTS_GLOB formatted with bin_id,
# e.g., bin-specific subfolders produced by NativeHitmapCreator:
#   <ROOTS_DIR>/bin_000/*hitmaps*.root
ROOTS_DIR = os.path.join(BASE_DIR, "hitmaps")
INPUT_ROOTS_GLOB = os.path.join(ROOTS_DIR, "bin_{bin_id}/*hitmaps*.root")

# Optional fallback: directory where input hitmap .npy files are stored
HITMAPS_DIR = os.path.join(BASE_DIR, "hitmaps")

# Output paths (use BASE_DIR for saving)
OUTPUT_ROOT = os.path.join(BASE_DIR, "system_matrix_from_numpy.root")
OUTPUT_PDF_DIR = os.path.join(BASE_DIR, "heatmaps")
SOURCE_HIST_PATH = os.path.join(BASE_DIR, "sourceHist_wideSource_200pixels.root")

# Ensure base and output directories exist
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_PDF_DIR, exist_ok=True)

# Thresholds used in native hitmap generation
THRESHOLDS = [0, 500, 1000, 1500]

# Enable batch mode for ROOT to avoid GUI requirements
ROOT.gROOT.SetBatch(True)

# Output ROOT file
output_file = ROOT.TFile(OUTPUT_ROOT, "RECREATE")

# Load the source histogram once (optional but recommended for reconstruction input)
source_hist = None
if os.path.isfile(SOURCE_HIST_PATH):
    src_file = ROOT.TFile.Open(SOURCE_HIST_PATH)
    if src_file and src_file.IsOpen():
        tmp = src_file.Get("sourceHist")
        if tmp:
            # Clone to decouple from the input file (and allow closing it safely)
            source_hist = tmp.Clone("sourceHist")
            source_hist.SetDirectory(0)
        src_file.Close()
    else:
        print(f"Warning: Could not open source histogram file: {SOURCE_HIST_PATH}")
else:
    print(f"Warning: Source histogram file not found: {SOURCE_HIST_PATH}")

def _find_root_for_bin(bin_id_str: str) -> str | None:
    # Direct candidate matching our builder output
    direct = os.path.join(ROOTS_DIR, f"bin_{bin_id_str}_hitmaps.root")
    if os.path.isfile(direct):
        return direct
    pattern = INPUT_ROOTS_GLOB.format(bin_id=bin_id_str)
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[0]
    # Fallbacks: try without subfolder and alternative naming
    alt_patterns = [
        os.path.join(ROOTS_DIR, f"bin_{bin_id_str}*hitmaps*.root"),
        os.path.join(ROOTS_DIR, f"*{bin_id_str}*hitmaps*.root"),
    ]
    for pat in alt_patterns:
        m = sorted(glob.glob(pat))
        if m:
            return m[0]
    return None


def _load_hist_as_array(root_path: str, hist_name: str) -> np.ndarray | None:
    f = ROOT.TFile.Open(root_path)
    if not f or not f.IsOpen():
        return None
    h = f.Get(hist_name)
    if not h:
        f.Close()
        return None
    nx = h.GetNbinsX()
    ny = h.GetNbinsY()
    arr = np.zeros((ny, nx), dtype=float)
    for iy in range(1, ny + 1):
        for ix in range(1, nx + 1):
            arr[iy - 1, ix - 1] = float(h.GetBinContent(ix, iy))
    f.Close()
    return arr


for thr in THRESHOLDS:
    for map_type in ["hitmap", "energy"]:
        postfix = "FiberHitMap" if map_type == "hitmap" else "EnergyDepoMap"
        hist_name = f"h{postfix}_AE_t{thr:04d}"

        reshaped_histograms = []

        for i in range(NUM_FILES):
            bin_id = f"{i:03d}"
            # Prefer ROOT input, but fall back to NPY if not found
            array = None
            root_file = _find_root_for_bin(bin_id)
            if root_file and os.path.isfile(root_file):
                array = _load_hist_as_array(root_file, hist_name)
                if array is None:
                    print(f"Missing histogram {hist_name} in {root_file}")
            if array is None:
                # Fallback: NPY from HITMAPS_DIR
                npy_file = os.path.join(HITMAPS_DIR, f"bin_{bin_id}utinfkeV_{map_type}_thr{thr:04d}.npy")
                if os.path.isfile(npy_file):
                    array = np.load(npy_file)
                else:
                    print(f"Missing ROOT and NPY for bin {bin_id} (ROOT pattern: {INPUT_ROOTS_GLOB})")
                    reshaped_histograms.append(np.zeros((7 * 55,)))
                    continue

            flat = array.flatten()
            normed = flat / N_GAMMAS_TOTAL
            reshaped_histograms.append(normed)

        system_matrix = np.column_stack(reshaped_histograms)
        nbins_y, nbins_x = system_matrix.shape

        # Primary output: TH2D under the canonical name so it's directly openable
        output_file.cd()
        h2big = ROOT.TH2D(hist_name, hist_name, nbins_x, -0.5, nbins_x-0.5, nbins_y, -0.5, nbins_y-0.5)
        for y in range(nbins_y):
            for x in range(nbins_x):
                h2big.SetBinContent(x+1, y+1, float(system_matrix[y, x]))
        bytes_main = h2big.Write("", ROOT.TObject.kOverwrite)
        if bytes_main <= 0:
            print(f"ERROR: Failed to write {hist_name} to {OUTPUT_ROOT}")

        # Also save this system matrix into its own ROOT file with matrixH and sourceHist entries
        single_root_path = os.path.join(BASE_DIR, f"{hist_name}.root")
        single_file = ROOT.TFile(single_root_path, "RECREATE")
        single_file.cd()
        # Write TMatrixD as 'matrixH' for compatibility with raux.get_hmat (expects fElements)
        matrixH = ROOT.TMatrixD(nbins_y, nbins_x)
        for y in range(nbins_y):
            for x in range(nbins_x):
                matrixH[y][x] = float(system_matrix[y, x])
        matrixH.Write("matrixH", ROOT.TObject.kOverwrite)
        # Also write a TH2D for easy open/visualization under a different name
        h2file = ROOT.TH2D("matrixH_h2", "matrixH_h2", nbins_x, -0.5, nbins_x-0.5, nbins_y, -0.5, nbins_y-0.5)
        for y in range(nbins_y):
            for x in range(nbins_x):
                h2file.SetBinContent(x+1, y+1, float(system_matrix[y, x]))
        bytes_single = h2file.Write("", ROOT.TObject.kOverwrite)
        if bytes_single <= 0:
            print(f"ERROR: Failed to write matrixH to {single_root_path}")
        # Optionally include the source histogram if available
        if source_hist is not None:
            source_hist.Write("sourceHist")
        single_file.Write()
        single_file.Close()
        print(f"Saved standalone ROOT with matrixH and sourceHist: {single_root_path}")

        # Save heatmap as PDF using ROOT
        pdf_path = os.path.join(OUTPUT_PDF_DIR, f"{hist_name}.pdf")
        h2 = ROOT.TH2D(hist_name+"_h2", hist_name, nbins_x, -0.5, nbins_x-0.5, nbins_y, -0.5, nbins_y-0.5)
        for y in range(nbins_y):
            for x in range(nbins_x):
                # TH2 bins are 1-indexed
                h2.SetBinContent(x+1, y+1, float(system_matrix[y, x]))
        c = ROOT.TCanvas(hist_name+"_c", hist_name, 1200, 700)
        c.SetRightMargin(0.15)
        h2.GetXaxis().SetTitle("Bin index (0-based)")
        h2.GetYaxis().SetTitle("AE index (7x55 flattened)")
        h2.Draw("COLZ")
        c.SaveAs(pdf_path)
        c.Close()
        print(f"Saved heatmap PDF: {pdf_path}")

wbytes = output_file.Write()
if wbytes <= 0:
    print(f"ERROR: Failed to write file directory to {OUTPUT_ROOT}")
output_file.Close()
print(f"All matrices written to {OUTPUT_ROOT}")

