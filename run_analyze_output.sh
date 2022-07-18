#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=galraz@mit.edu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=analyze_output.sh
#SBATCH --time=4:00:00

module load openmind/singularity/3.2.0

python analyze_output.py 