#!/bin/bash
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>           # CPU partition
#SBATCH --cpus-per-task=4             
#SBATCH --mem=20G                     # TensorFlow + data
#SBATCH --time=0-05:00:00
#SBATCH --job-name=rs
#SBATCH --output=logs/rs_%j.log
#SBATCH --error=logs/rs_%j.log

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

module load python
source activate tensorflow

python main.py --config_path=configs/RybkoStolyar.json --eta_upper=$1 --mu=$2 --discount=$3 --v1=$4 --v2=$5 --pricing_multiply=$6

