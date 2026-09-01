#!/usr/bin/env bash
# Batch recon + denoising for the Manchester 3T 129Xe Cartesian data.
#
#   recon  (siemens_twix_recon.py)
#     -> NAFNet denoise@100  (run_cartesian_nafnet.py, torch venv)
#     -> ARDL   denoise@128  (run_cartesian_ardl.py,   ardl  venv)
#
# The ARDL step uses the c3d TensorFlow SavedModel for BOTH 3D and 2D data and
# runs in its own venv ($ARDL_VENV, TF-only) so it never touches the torch venv.
#
# Usage:
#   ./run_manchester_batch.sh                 # all subjects under RAW/
#   ./run_manchester_batch.sh HV03 HV04       # only these subject folders
#
# Continues past any single-file failure; prints a summary at the end.

set -u
# Layout (2026-07-14): recon/denoise front-end scripts + data live under
# PythonRecon/; siemens_twix_recon.py stays at the repo root ($ROOT).
ROOT=/home/raja/xe129_dldegibbs
RECON_SRC=$ROOT/PythonRecon
RAW=$RECON_SRC/3T_vent_data_Manchester/RAW
OUT=$RECON_SRC/3T_vent_data_Manchester/RECON
VENV=$ROOT/xe129_dldegibbs_venv
ARDL_VENV=$ROOT/xe129_ardl_venv       # TF-only venv for the ARDL step
ARDL_PY=$ARDL_VENV/bin/python
INFER=100
ARDL_INFER=128                        # ARDL needs a multiple of 8 (~2x native)

source "$VENV/bin/activate"
cd "$RECON_SRC" || exit 1

# Build the file list (optionally restricted to subject folders given as args).
if [ "$#" -gt 0 ]; then
    DATS=()
    for subj in "$@"; do
        while IFS= read -r f; do DATS+=("$f"); done < <(find "$RAW/$subj" -iname '*.dat' | sort)
    done
else
    mapfile -t DATS < <(find "$RAW" -iname '*.dat' | sort)
fi

ok=(); fail=()
for dat in "${DATS[@]}"; do
    subj=$(basename "$(dirname "$dat")")
    tag=$(basename "$dat" .dat)
    odir="$OUT/$subj"
    mkdir -p "$odir"
    echo "==================================================================="
    echo "=== $subj / $tag"
    echo "==================================================================="
    if ! python "$ROOT/siemens_twix_recon.py" --dat "$dat" --outdir "$odir"; then
        echo "!!! RECON FAILED: $subj/$tag"; fail+=("$subj/$tag (recon)"); continue
    fi
    if ! python "$RECON_SRC/run_cartesian_nafnet.py" \
            --mat "$odir/${tag}_recon.mat" \
            --out "$odir/${tag}_denoised.mat" \
            --infer_size "$INFER"; then
        echo "!!! DENOISE FAILED: $subj/$tag"; fail+=("$subj/$tag (denoise)"); continue
    fi
    # ARDL (TF c3d model) in its own venv; failure here is non-fatal to the run.
    if ! "$ARDL_PY" "$RECON_SRC/run_cartesian_ardl.py" \
            --mat "$odir/${tag}_recon.mat" \
            --out "$odir/${tag}_ardl.mat" \
            --infer_size "$ARDL_INFER"; then
        echo "!!! ARDL FAILED: $subj/$tag"; fail+=("$subj/$tag (ardl)"); continue
    fi
    ok+=("$subj/$tag")
done

echo
echo "==================== SUMMARY ===================="
echo "OK (${#ok[@]}):";   printf '   %s\n' "${ok[@]}"
echo "FAILED (${#fail[@]}):"; printf '   %s\n' "${fail[@]}"
