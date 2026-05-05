#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

SIFICCNN_ROOT=${SIFICCNN_ROOT:-/home/philippe/RWTHscratch/SiFiCCNN}
RESULTS_ROOT=${RESULTS_ROOT:-$SIFICCNN_ROOT/results}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/philippe/RWTHscratch/HitMapsSMForPaper26}
SM_BINS=${SM_BINS:-200}
N_GAMMAS_TOTAL=${GENERATE_SM_N_GAMMAS_TOTAL:-37798000}
HITMAP_WORKERS=${HITMAP_WORKERS:-8}

RUNS=(
  Norm_SiFiECRN3V1_do00
  Norm_SiFiECRN3V1_do10
  Norm_SiFiECRN3V1_do20
  Norm_SiFiECRN3V1_do30
  Norm_SiFiECRN3V1_do40
  Norm_SiFiECRN3V2_do00
  Norm_SiFiECRN3V2_do10
  Norm_SiFiECRN3V2_do20
  Norm_SiFiECRN3V2_do30
  Norm_SiFiECRN3V2_do40
)

export SIFICCNN_ROOT

mkdir -p "$OUTPUT_ROOT"

for run_name in "${RUNS[@]}"; do
  pred_dir="$RESULTS_ROOT/$run_name/SystemMatrix_CodedMaskHIT_simv5_linesource_0to29999"
  out_dir="$OUTPUT_ROOT/$run_name/systemMatrix"

  echo "=== [$run_name] ==="
  mkdir -p "$out_dir"

  if ! find "$pred_dir" -maxdepth 1 -name '*regE_pred_bin*.npy' | grep -q .; then
    python "$REPO_ROOT/analysis/EdgeConvResNetSiPM/RegressionEnergy.py" \
      --mode CM \
      --name "$run_name" \
      --evaluate_training_set \
      --sm_bins "$SM_BINS"
  else
    echo "Energy bin predictions already present, skipping"
  fi

  if ! find "$pred_dir" -maxdepth 1 -name '*pos_clas_pred_bin*.npy' | grep -q .; then
    python "$REPO_ROOT/analysis/EdgeConvResNetSiPM/ClassificationPosition.py" \
      --mode CM \
      --name "$run_name" \
      --evaluate_training_set \
      --sm_bins "$SM_BINS"
  else
    echo "Position bin predictions already present, skipping"
  fi

  if [[ ! -f "$out_dir/hitmaps/bin_199_hitmaps.root" ]]; then
    python "$REPO_ROOT/SIFICCNN/uploader/build_hitmaps_for_SM.py" \
      --pred-dir "$pred_dir" \
      --out-dir "$out_dir" \
      --workers "$HITMAP_WORKERS"
  else
    echo "Hitmaps already present, skipping"
  fi

  if [[ ! -f "$out_dir/system_matrix_from_numpy.root" ]]; then
    GENERATE_SM_BASE_DIR="$out_dir" \
    GENERATE_SM_NUM_FILES="$SM_BINS" \
    GENERATE_SM_N_GAMMAS_TOTAL="$N_GAMMAS_TOTAL" \
    python "$REPO_ROOT/SIFICCNN/uploader/generateSM.py"
  else
    echo "System matrix already present, skipping"
  fi
done