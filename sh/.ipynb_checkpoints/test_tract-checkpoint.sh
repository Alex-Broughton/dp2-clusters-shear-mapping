#!/bin/bash
#SBATCH --job-name=test_tract         # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=roma              # Partition or queue name
#SBATCH --account=rubin:commissioning
#SBATCH --nodes=1                     # Number of nodes
# #SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --time=1:00:00                # Maximum runtime (D-HH:MM:SS)

source /sdf/group/rubin/sw/tag/v30_0_5_rc1/loadLSST.sh
setup lsst_sitcom -t v30_0_5_rc1

python /sdf/home/a/abrought/dp2/dp2-clusters-shear-mapping/python/schirmer_snr_weight.py /sdf/home/a/abrought/dp2/dp2-clusters-shear-mapping/_data/dp2_tract9813_shear_catalog.fits 10000 32 3

