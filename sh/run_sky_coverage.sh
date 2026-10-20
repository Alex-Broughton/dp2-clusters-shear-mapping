#!/bin/bash
#SBATCH --job-name=coverage                       # Job name
# Use absolute log paths: slurm opens these before the script body runs, so a
# relative path would resolve against the submit dir (which may lack output/).
#SBATCH --output=/sdf/home/a/abrought/dp2/v2/dp2-clusters-shear-mapping/output/coverage_slurm.out
#SBATCH --error=/sdf/home/a/abrought/dp2/v2/dp2-clusters-shear-mapping/output/coverage_slurm.err
#SBATCH --partition=roma                          # Partition or queue name
#SBATCH --account=rubin:commissioning
#SBATCH --nodes=2                                 # Single node (multiprocessing.Pool)
#SBATCH --cpus-per-task=16                        # Must match CONFIG["N_PROC"] in the script
# SBATCH --mem=128G                                # Each worker holds one tract object table
#SBATCH --time=4:00:00                            # Whole DP2 footprint, one tract per task

# Slurm copies this script to a spool dir before running it, so cd into the repo
# so the python path below (and any relative paths) resolve correctly.
# NOTE: the output/ directory must already exist for the #SBATCH logs above;
# create it once with `mkdir -p output` before the first submission.
cd /sdf/home/a/abrought/dp2/v2/dp2-clusters-shear-mapping

# Set up the Rubin (LSST) stack so the Butler is importable.
source /sdf/group/rubin/sw/tag/v30_0_8_rc4/loadLSST.sh
setup lsst_sitcom -t v30_0_8_rc4

# Keep the math libraries single-threaded: the parallelism here is one process
# per tract, and letting each of them spawn its own threads only adds contention.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# All controls live in the CONFIG block at the top of the script; no arguments.
python /sdf/home/a/abrought/dp2/v2/dp2-clusters-shear-mapping/python/sky_coverage.py
