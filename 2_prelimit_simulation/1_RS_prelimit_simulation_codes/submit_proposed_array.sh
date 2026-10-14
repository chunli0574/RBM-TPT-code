#!/bin/bash
# =============================================================================
# submit_proposed_array.sh
#
# SLURM array job: run proposed NN policy parameter search.
# Each array task handles one (safety, epsilon, rho) combination;
# the inner (lambda_tilde × beta_rate_1 × beta_rate_2) loop runs inside.
#
# Usage:
#   bash submit_proposed_array.sh
#
# Before running on server:
#   1. Build sim_proposed in codes/ (WORK_DIR is derived from this script's location)
#   2. Set TF_LIB to the directory containing libtensorflow.so
#   3. Adjust #SBATCH directives (account, partition, time, mem) as needed
#   4. Check model_new_base_dir in codes/proposed_policy_config.txt points to the
#      directory of *_grad.weights.bin models (rho must be in 0.95, 0.96, ..., 1.05)
# =============================================================================

# --- User settings -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/codes"
TF_LIB="$HOME/libs/libtensorflow/lib"        # path to libtensorflow.so

# --- Search space (edit lists to run full or test subset) --------------------
# Full lists (example):
#   safety_values=(0.0 0.5 1.0)
#   epsilon_values=(0.1 0.15 0.2 0.25 0.3)
#   rho_values=(0.95 0.97 0.98 0.99 1.0 1.01 1.02)
#
# Test subset:
safety_values=(0.0 1.0 2.0)
epsilon_values=(0 0.11 0.22 0.34 0.45)
rho_values=(0.99 1.0 1.01 1.02 1.03 1.04 1.05)

CONFIG_FILE="proposed_policy_config.txt"

# --- Pre-flight checks -------------------------------------------------------
if [ ! -x "${WORK_DIR}/sim_proposed" ]; then
    echo "ERROR: missing executable ${WORK_DIR}/sim_proposed"
    echo "Build it first with: cd \"${WORK_DIR}\" && make sim_proposed"
    exit 1
fi
if [ ! -f "${WORK_DIR}/${CONFIG_FILE}" ]; then
    echo "ERROR: missing config ${WORK_DIR}/${CONFIG_FILE}"
    exit 1
fi
if [ ! -d "${TF_LIB}" ]; then
    echo "ERROR: TF_LIB directory not found: ${TF_LIB}"
    exit 1
fi

# NN models: one *_eta={rho}_grad.weights.bin per rho in model_new_base_dir.
# (theta is chosen inside sim_proposed from eta_theta_mapping in nn_model_paths.h)
MODEL_DIR=$(sed -n 's/^model_new_base_dir *= *//p' "${WORK_DIR}/${CONFIG_FILE}" | sed 's/ *#.*//; s/ *$//')
if [ -z "${MODEL_DIR}" ]; then
    echo "WARNING: model_new_base_dir not set in ${CONFIG_FILE}; skipping NN model check"
else
    for rho in "${rho_values[@]}"; do
        rho_py=$(awk -v x="$rho" 'BEGIN{s=sprintf("%.2f",x); sub(/0+$/,"",s); sub(/\.$/,".0",s); print s}')
        if ! ls "${MODEL_DIR}"/*_eta=${rho_py}_grad.weights.bin >/dev/null 2>&1; then
            echo "ERROR: no NN model for rho=${rho} in ${MODEL_DIR}"
            exit 1
        fi
    done
fi

# --- Generate parameter file (one line per array task) -----------------------
log_dir="${WORK_DIR}/../logs"
mkdir -p "$log_dir"
param_file="${WORK_DIR}/params_proposed.txt"

> "$param_file"
for safety in "${safety_values[@]}"; do
    for epsilon in "${epsilon_values[@]}"; do
        for rho in "${rho_values[@]}"; do
            echo "$safety $epsilon $rho" >> "$param_file"
        done
    done
done

total=$(wc -l < "$param_file")
echo "Submitting $total proposed-abs jobs"
echo "safety:   ${safety_values[*]}"
echo "epsilon:  ${epsilon_values[*]}"
echo "rho:      ${rho_values[*]}"

# --- Submit array job --------------------------------------------------------
sbatch --array=0-$(( total - 1 )) <<EOF
#!/bin/bash
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-20:00:00
#SBATCH --job-name=rs_proposed
#SBATCH --output=${log_dir}/proposed_%A_%a.log

module load gcc/13.2.0
cd "${WORK_DIR}"

# Read (safety, epsilon, rho) for this task
line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: safety=\$safety  epsilon=\$epsilon  rho=\$rho ==="
echo "Host: \$(hostname)  Date: \$(date)"

export LD_LIBRARY_PATH=${TF_LIB}:\$LD_LIBRARY_PATH

./sim_proposed ${CONFIG_FILE} \$safety \$epsilon \$rho
EOF
