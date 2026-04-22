#!/bin/bash
#SBATCH --job-name=test_tract         # Job name
#SBATCH --output=/sdf/home/a/abrought/dp2/dp2-clusters-shear-mapping/sh/test_tract_output/output.txt          # Standard output file
#SBATCH --error=/sdf/home/a/abrought/dp2/dp2-clusters-shear-mapping/sh/test_tract_output/error.txt             # Standard error file
#SBATCH --partition=roma              # Partition or queue name
#SBATCH --account=rubin:commissioning
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=4             # Number of CPU cores per task (1 python launch = 1 task)
#SBATCH --time=1:00:00                # Maximum runtime (D-HH:MM:SS)

source /sdf/group/rubin/sw/tag/v30_0_5_rc1/loadLSST.sh
setup lsst_sitcom -t v30_0_5_rc1

python /sdf/home/a/abrought/dp2/dp2-clusters-shear-mapping/python/map.py

