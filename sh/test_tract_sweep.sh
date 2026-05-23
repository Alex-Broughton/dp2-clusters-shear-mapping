#!/bin/bash
#SBATCH --job-name=test_tract_sweep   # Job name
#SBATCH --output=sweep_output.txt     # Standard output file (not output.txt)
#SBATCH --error=sweep_error.txt       # Standard error file
#SBATCH --partition=roma              # Partition or queue name
#SBATCH --account=rubin:commissioning
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --time=2:00:00                # Maximum runtime (15-case large-Rs grid; coarser bins)

REPO="/sdf/home/a/abrought/dp2/dp2-clusters-shear-mapping"
FITS="${REPO}/_data/shear_table_xmatch_gold.fits"
SWEEP_PY="${REPO}/python/schirmer_snr_sweep.py"

echo "=== test_tract_sweep.sh ==="
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "cwd: $(pwd)"

# Do not use set -u before setup: conda activate.d scripts reference unset vars.
set -eo pipefail
source /sdf/group/rubin/sw/tag/v30_0_5_rc1/loadLSST.sh
setup lsst_sitcom -t v30_0_5_rc1
set -u

if [[ ! -f "${SWEEP_PY}" ]]; then
  echo "ERROR: sweep script not found: ${SWEEP_PY}" >&2
  echo "Sync repo to SDF (git pull). Do not run schirmer_snr_weight.py with only FITS + 32." >&2
  exit 1
fi
if [[ ! -f "${FITS}" ]]; then
  echo "ERROR: FITS not found: ${FITS}" >&2
  exit 1
fi

# Parameter sweep (no dedupe). Outputs:
#   _data/sweep_output/shear_sweep_summary.{csv,png,pdf}
#   _data/sweep_output/maps/shear_M_ap_*.{png,pdf}  (E/B per case)
echo "Running: python ${SWEEP_PY} ${FITS} 32"
python "${SWEEP_PY}" --dedupe "${FITS}" 32

echo "Done. Check: ${REPO}/_data/sweep_output/"
