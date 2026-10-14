#!/bin/bash
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>            # CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-15:00:00
#SBATCH --job-name=cc3
#SBATCH --output=logs/cc3_output_%j.log
#SBATCH --error=logs/cc3_output_%j.log

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

module load python
source activate tensorflow

python main.py --config_path=configs/extendedCrissCross_3dim.json --eta_upper=$1 --mu=$2 --discount=$3

