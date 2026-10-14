#!/bin/bash
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition> 
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=1-11:00:00 
#SBATCH --job-name=cc10
#SBATCH --output=logs/cc10_output_%j.log
#SBATCH --error=logs/cc10_output_%j.log

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

module load python
source activate tensorflow

python main.py --config_path=configs/extendedCrissCross_10dim_6.json --eta_upper=$1 --mu=$2 --discount=$3

