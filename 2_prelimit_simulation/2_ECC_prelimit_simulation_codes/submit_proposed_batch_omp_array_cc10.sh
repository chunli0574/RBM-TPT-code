#!/bin/bash
# =============================================================================
# submit_proposed_batch_omp_array_cc10.sh   [CC10 Extended Criss-Cross]
#
# OMP + batch-NN version. Submit proposed NN-policy parameter search × {abs, rel}
# on the HPC cluster.
# Uses the BATCH+OMP binary sim_proposed_batch_omp (built with
# -DBATCH_NN=1 -DUSE_BLAS=1 -DUSE_OMP=1 -DDIM_K_SELECT=10).
#
# Per SLURM task: OMP parallelizes the inner (lambda_tilde × beta_rate_1 ×
# beta_rate_2) combo sweep for each fixed (rho, safety, epsilon) across
# --cpus-per-task threads. MKL is forced single-threaded so it does not
# spawn nested threads on top of OMP. The MKL batched cblas_sgemm per layer
# still runs inside each combo.
#
# Strict CPU control (matches sim_benchmark_omp submit scripts):
#   OMP_NUM_THREADS = SLURM_CPUS_PER_TASK    (cap OMP at cores SLURM gave us)
#   MKL_NUM_THREADS = 1                       (MKL runs serial in each thread)
#   OMP_DYNAMIC     = false                   (no thread-count adjustments)
#   srun --cpu-bind=cores                     (pin threads to allocated cores)
#
# Usage:
#   bash submit_proposed_batch_omp_array_cc10.sh                # both abs + rel
#   bash submit_proposed_batch_omp_array_cc10.sh proposed-abs   # only abs
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
CPUS_PER_TASK=8

if [ ! -x "${WORK_DIR}/sim_proposed_batch_omp" ]; then
    echo "ERROR: missing executable ${WORK_DIR}/sim_proposed_batch_omp"
    echo "Build the binary first with: cd \"${WORK_DIR}\" && make sim_proposed_batch_omp"
    exit 1
fi
if [ ! -f "${WORK_DIR}/proposed_policy_config.txt" ]; then
    echo "ERROR: missing config ${WORK_DIR}/proposed_policy_config.txt"
    exit 1
fi

submit_pv() {
    local variant="$1"
    local safety_arr_name="$2"
    local eps_arr_name="$3"
    local rho_arr_name="$4"
    local sbatch_time="$5"
    local tag="proposed-${variant}"

    [ -n "$SELECTED" ] && [ "$SELECTED" != "$tag" ] && return

    eval "local safeties=(\"\${${safety_arr_name}[@]}\")"
    eval "local epsilons=(\"\${${eps_arr_name}[@]}\")"
    eval "local rhos=(\"\${${rho_arr_name}[@]}\")"

    local param_file="${WORK_DIR}/params_${tag}_cc10_omp.txt"
    > "$param_file"
    for safety in "${safeties[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            for rho in "${rhos[@]}"; do
                echo "$safety $epsilon $rho" >> "$param_file"
            done
        done
    done
    local total=$(wc -l < "$param_file")

    echo "Submitting $total ${tag} array tasks  (variant=${variant}, batch+OMP, 1 (s,e,rho) per task, ${CPUS_PER_TASK} threads/task)"
    echo "  safety:  ${safeties[*]}"
    echo "  epsilon: ${epsilons[*]}"
    echo "  rho:     ${rhos[*]}"

    sbatch --array=0-$(( total - 1 )) <<EOF
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=2G
#SBATCH --time=${sbatch_time}
#SBATCH --job-name=cc10_${tag}_omp
#SBATCH --output=${log_dir}/${tag}_omp_cc10_%A_%a.log

module load mkl
export LD_LIBRARY_PATH=\$MKLROOT/lib/intel64:\$LD_LIBRARY_PATH

# --- Strict CPU control ---
# OMP gets exactly the cores SLURM allocated; MKL is forced single-threaded
# so it does not spawn nested threads on top of OMP. OMP_DYNAMIC=false keeps
# the thread count fixed regardless of runtime load.
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=false

export W_DUMP_PATH="./W_counter_${tag}_omp.bin"
export W_DUMP_GRID=0.005

module load ${GCC_MODULE}
cd "${WORK_DIR}" || exit 1

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: ${tag} safety=\$safety epsilon=\$epsilon rho=\$rho  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}  MKL_NUM_THREADS=\${MKL_NUM_THREADS}"

export LD_LIBRARY_PATH=${TF_LIB}:\$LD_LIBRARY_PATH

srun --cpu-bind=cores ./sim_proposed_batch_omp proposed_policy_config.txt \$safety \$epsilon \$rho ${variant}
rc=\$?
if [ \$rc -ne 0 ]; then
    echo "WARNING: sim_proposed_batch_omp exited \$rc"
fi

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
}

# =============================================================================
# proposed-abs  (CC10 rho center = 0.93)
# =============================================================================
proposed_abs_safety=(0.0 1.0)
proposed_abs_epsilon=(0 0.11 0.22 0.34)
proposed_abs_rho=(0.90 0.91 0.92 0.93 0.94 0.95 0.96)
submit_pv abs proposed_abs_safety proposed_abs_epsilon proposed_abs_rho "1-11:00:00"

# =============================================================================
# proposed-rel  (CC10 rho center = 0.92)
# =============================================================================
proposed_rel_safety=(0.0 1.0)
proposed_rel_epsilon=(0 0.11 0.22 0.34)
proposed_rel_rho=(0.89 0.90 0.91 0.92 0.93 0.94 0.95)
submit_pv rel proposed_rel_safety proposed_rel_epsilon proposed_rel_rho "1-11:00:00"
