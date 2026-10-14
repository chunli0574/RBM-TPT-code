#!/bin/bash
# =============================================================================
# submit_proposed_array_cc3.sh   [CC3 Extended Criss-Cross]
#
# Submit proposed NN-policy parameter search × {abs, rel} on the HPC cluster.
# Each (sub-bmk, variant) sub-block has its OWN safety / epsilon / rho lists.
# Array dim per sub-block = N_safety × N_epsilon × N_rho (1 (s,e,rho) per task).
#
# The variant ("abs" or "rel") is passed as the 5th CLI arg to sim_proposed,
# which overrides config wmp_extra_relative / beta_rate_symmetric:
#   abs → wmp_extra_relative=0, beta_rate_symmetric=2
#   rel → wmp_extra_relative=1, beta_rate_symmetric=1
# The binary auto-picks the matching *_abs / *_rel WMP scheduling lists in
# proposed_policy_config.txt accordingly.
#
# Single bash invocation submits up to 2 separate sbatch array jobs:
# proposed-abs, proposed-rel.
#
# Usage:
#   bash submit_proposed_array_cc3.sh                # both abs + rel
#   bash submit_proposed_array_cc3.sh proposed-abs   # only abs
#
# Before running on server:
#   1. WORK_DIR is auto-set to <script dir>/codes (sim_proposed + config files)
#   2. Set TF_LIB to the directory containing libtensorflow.so
#   3. Update model_base_dir / new_model_base_dir in proposed_policy_config.txt
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/codes"
TF_LIB="$HOME/libs/libtensorflow/lib"

SELECTED="${1:-}"
if [ -n "$SELECTED" ]; then
    case "$SELECTED" in
        proposed-abs|proposed-rel) ;;
        *) echo "ERROR: filter '$SELECTED' must be 'proposed-abs' or 'proposed-rel'"; exit 1 ;;
    esac
fi

log_dir="${WORK_DIR}/../logs"
mkdir -p "$log_dir"

# HPC-cluster SBATCH settings
SBATCH_ACCOUNT="<your_account>"
SBATCH_PARTITION="<your_partition>"
GCC_MODULE="gcc/13.2.0"

if [ ! -x "${WORK_DIR}/sim_proposed" ]; then
    echo "ERROR: missing executable ${WORK_DIR}/sim_proposed"
    echo "Build the binary first with: cd \"${WORK_DIR}\" && make DIM_K_SELECT=3 sim_proposed"
    exit 1
fi
if [ ! -f "${WORK_DIR}/proposed_policy_config.txt" ]; then
    echo "ERROR: missing config ${WORK_DIR}/proposed_policy_config.txt"
    exit 1
fi

submit_pv() {
    local variant="$1"           # abs or rel
    local safety_arr_name="$2"
    local eps_arr_name="$3"
    local rho_arr_name="$4"
    local sbatch_time="$5"
    local tag="proposed-${variant}"

    [ -n "$SELECTED" ] && [ "$SELECTED" != "$tag" ] && return

    eval "local safeties=(\"\${${safety_arr_name}[@]}\")"
    eval "local epsilons=(\"\${${eps_arr_name}[@]}\")"
    eval "local rhos=(\"\${${rho_arr_name}[@]}\")"

    local param_file="${WORK_DIR}/params_${tag}_cc3_ser.txt"
    > "$param_file"
    for safety in "${safeties[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            for rho in "${rhos[@]}"; do
                echo "$safety $epsilon $rho" >> "$param_file"
            done
        done
    done
    local total=$(wc -l < "$param_file")

    echo "Submitting $total ${tag} array tasks  (variant=${variant}, 1 (s,e,rho) per task)"
    echo "  safety:  ${safeties[*]}"
    echo "  epsilon: ${epsilons[*]}"
    echo "  rho:     ${rhos[*]}"

    sbatch --array=0-$(( total - 1 )) <<EOF
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=${sbatch_time}
#SBATCH --job-name=cc3_${tag}
#SBATCH --output=${log_dir}/${tag}_cc3_%A_%a.log

module load mkl
export LD_LIBRARY_PATH=\$MKLROOT/lib/intel64:\$LD_LIBRARY_PATH

module load ${GCC_MODULE}
cd "${WORK_DIR}" || exit 1

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: ${tag} safety=\$safety epsilon=\$epsilon rho=\$rho ==="
echo "Host: \$(hostname)  Date: \$(date)"

export LD_LIBRARY_PATH=${TF_LIB}:\$LD_LIBRARY_PATH

./sim_proposed proposed_policy_config.txt \$safety \$epsilon \$rho ${variant}
rc=\$?
if [ \$rc -ne 0 ]; then
    echo "WARNING: sim_proposed exited \$rc"
fi

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
}

# =============================================================================
# proposed-abs
# =============================================================================
proposed_abs_safety=(0.0 1.0)
proposed_abs_epsilon=(0 0.11 0.22 0.34)
proposed_abs_rho=(0.87 0.88 0.89 0.90 0.91 0.92 0.93)
submit_pv abs proposed_abs_safety proposed_abs_epsilon proposed_abs_rho "0-20:00:00"

# =============================================================================
# proposed-rel
# =============================================================================
proposed_rel_safety=(0.0 1.0)
proposed_rel_epsilon=(0 0.11 0.22 0.34)
proposed_rel_rho=(0.87 0.88 0.89 0.90 0.91 0.92 0.93)
submit_pv rel proposed_rel_safety proposed_rel_epsilon proposed_rel_rho "0-20:00:00"
