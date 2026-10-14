#!/bin/bash
# =============================================================================
# submit_one_combo_multi.sh
#
# Single SLURM job that runs sim_one_combo SEQUENTIALLY for each policy in
# the `policies` array below. All policies share the same one_combo_input.txt
# but their own [section] block within that file.
#
# Each invocation produces its own pair of CSVs in output_dir (named with the
# policy as prefix), so they don't overwrite each other.
#
# Usage:
#   bash submit_one_combo_multi.sh
#
# Before running on server:
#   1. Compile sim_one_combo in codes/:  module load gcc/13.2.0 && make sim_one_combo
#      (WORK_DIR is derived from this script's location)
#   2. Edit codes/one_combo_input.txt:
#        - For each policy in `policies` below, fill its [section] with values
#        - Top-level fields (n_paths, hist_bucket, output_dir, ...) apply to all
#   3. Edit `policies` below to list the policies you want (a subset of
#      proposed, A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, D1, D2)
#   4. Adjust #SBATCH directives (time / mem) — total time should cover ALL
#      policies in the list, since they run one after another in the same job
# =============================================================================

# --- User settings -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/codes"
TF_LIB="$HOME/libs/libtensorflow/lib"   # required because sim_one_combo links libtensorflow
INPUT_FILE="one_combo_input.txt"

# --- Policies to run, in order (edit this list) ------------------------------
# Valid values: proposed | A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 D1 D2
policies=(proposed A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 D1 D2)

# --- Pre-flight checks -------------------------------------------------------
if [ ! -x "${WORK_DIR}/sim_one_combo" ]; then
    echo "ERROR: missing executable ${WORK_DIR}/sim_one_combo"
    echo "Build it first with: cd \"${WORK_DIR}\" && make sim_one_combo"
    exit 1
fi
if [ ! -f "${WORK_DIR}/${INPUT_FILE}" ]; then
    echo "ERROR: missing input ${WORK_DIR}/${INPUT_FILE}"
    exit 1
fi
if [ ! -d "${TF_LIB}" ]; then
    echo "ERROR: TF_LIB directory not found: ${TF_LIB}"
    exit 1
fi

# NN model check (only when 'proposed' is in the list): rho from the [proposed]
# section must have a *_eta={rho}_grad.weights.bin in model_new_base_dir.
if [[ " ${policies[*]} " == *" proposed "* ]]; then
    MODEL_DIR=$(sed -n 's/^model_new_base_dir *= *//p' "${WORK_DIR}/${INPUT_FILE}" | sed 's/ *#.*//; s/ *$//')
    rho=$(awk '{sub(/#.*/,""); gsub(/[ \t\r]/,"")} /^\[/{sec=$0} sec=="[proposed]" && /^rho=/{sub(/^rho=/,""); print; exit}' \
          "${WORK_DIR}/${INPUT_FILE}")
    if [ -z "${MODEL_DIR}" ] || [ -z "${rho}" ]; then
        echo "WARNING: model_new_base_dir or [proposed] rho not found in ${INPUT_FILE}; skipping NN model check"
    else
        rho_py=$(awk -v x="$rho" 'BEGIN{s=sprintf("%.2f",x); sub(/0+$/,"",s); sub(/\.$/,".0",s); print s}')
        if ! ls "${MODEL_DIR}"/*_eta=${rho_py}_grad.weights.bin >/dev/null 2>&1; then
            echo "ERROR: no NN model for [proposed] rho=${rho} in ${MODEL_DIR}"
            exit 1
        fi
    fi
fi

# --- Generate logs path ------------------------------------------------------
log_dir="${WORK_DIR}/../logs"
mkdir -p "$log_dir"

# Materialize the array as a space-separated string for the heredoc.
policies_str="${policies[*]}"

# --- Submit a single SLURM job that loops over all listed policies -----------
sbatch <<EOF
#!/bin/bash
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-08:00:00
#SBATCH --job-name=rs_one_combo_multi
#SBATCH --output=${log_dir}/one_combo_multi_%j.log

module load gcc/13.2.0
cd "${WORK_DIR}"

policies=(${policies_str})

echo "=== sim_one_combo_multi: input=${INPUT_FILE} policies=\${policies[*]} ==="
echo "Host: \$(hostname)  Date: \$(date)"

export LD_LIBRARY_PATH=${TF_LIB}:\$LD_LIBRARY_PATH

for p in "\${policies[@]}"; do
    echo
    echo "============================================================"
    echo ">>> Running policy: \$p"
    echo "============================================================"
    ./sim_one_combo ${INPUT_FILE} \$p
    if [ \$? -ne 0 ]; then
        echo "ERROR: sim_one_combo exited non-zero on policy=\$p; continuing with next."
    fi
done

echo
echo "=== All policies complete. Date: \$(date) ==="
EOF
