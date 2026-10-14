#!/bin/bash
# =============================================================================
# submit_groupB_omp_array_cc10.sh   [CC10 Extended Criss-Cross]
#
# OMP version of submit_groupB_array_cc10.sh.
# Uses sim_benchmark_omp (built with -DUSE_OMP=1). Each SLURM task gets
# --cpus-per-task=5; OMP parallelizes the inner combo loop (within each
# rho call) using all 5 cores. Requires parallel_threads >= 5 in
# benchmark_search_config.txt (resolve_omp_threads caps at SLURM_CPUS_PER_TASK).
#
# Strict CPU control (matches submit_benchmarkB_omp_array.sh):
#   OMP_NUM_THREADS = SLURM_CPUS_PER_TASK    (cap OMP at cores SLURM gave us)
#   OMP_DYNAMIC     = false                   (no thread-count adjustments)
#   srun --cpu-bind=cores                     (pin threads to allocated cores)
# MKL_NUM_THREADS not needed — sim_benchmark_omp links nn_model_stub (no MKL).
#
# Submit Group B benchmarks (B1, B2, B3, B4) on the HPC cluster.
# B-group uses ThresholdBased outsourcing — no epsilon search (the C++ side
# ignores the epsilon arg for B benchmarks, but we still pass a placeholder
# 0 for CLI uniformity). Array dim per benchmark = N_safety (one task per
# safety value); each task loops over rho serially.
#
# Single bash invocation submits up to 4 separate sbatch array jobs.
#
# Usage:
#   bash submit_groupB_omp_array_cc10.sh         # submit B1+B2+B3+B4
#   bash submit_groupB_omp_array_cc10.sh B2      # submit only B2
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/codes"

SELECTED="${1:-}"
if [ -n "$SELECTED" ]; then
    case "$SELECTED" in
        B1|B2|B3|B4) ;;
        *) echo "ERROR: filter '$SELECTED' must be one of: B1 B2 B3 B4"; exit 1 ;;
    esac
fi

log_dir="${WORK_DIR}/../logs"
mkdir -p "$log_dir"

SBATCH_ACCOUNT="<your_account>"
SBATCH_PARTITION="<your_partition>"
GCC_MODULE="gcc/14.1.0"
CPUS_PER_TASK=5

if [ ! -x "${WORK_DIR}/sim_benchmark_omp" ]; then
    echo "ERROR: missing executable ${WORK_DIR}/sim_benchmark_omp"
    echo "Build the binary first with: cd \"${WORK_DIR}\" && make DIM_K_SELECT=10 sim_benchmark_omp"
    exit 1
fi
if [ ! -f "${WORK_DIR}/benchmark_search_config.txt" ]; then
    echo "ERROR: missing config ${WORK_DIR}/benchmark_search_config.txt"
    exit 1
fi

# =============================================================================
# Helper: emit one sbatch array for a B-benchmark.
# Usage: submit_b <benchmark> <safety_arr_name> <rho_arr_name> <sbatch_time>
# =============================================================================
submit_b() {
    local benchmark="$1"
    local safety_arr_name="$2"
    local rho_arr_name="$3"
    local sbatch_time="$4"

    [ -n "$SELECTED" ] && [ "$SELECTED" != "$benchmark" ] && return

    eval "local safeties=(\"\${${safety_arr_name}[@]}\")"
    eval "local rhos=(\"\${${rho_arr_name}[@]}\")"

    local param_file="${WORK_DIR}/params_${benchmark}_cc10_omp.txt"
    > "$param_file"
    local rho_csv=$(IFS=,; echo "${rhos[*]}")
    for safety in "${safeties[@]}"; do
        echo "$safety $rho_csv" >> "$param_file"
    done
    local total=$(wc -l < "$param_file")

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task, rho looped IN-task; no epsilon search)"
    echo "  safety: ${safeties[*]}"
    echo "  rho:    ${rhos[*]}"

    sbatch --array=0-$(( total - 1 )) <<EOF
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=2G
#SBATCH --time=${sbatch_time}
#SBATCH --job-name=cc10_${benchmark}_omp
#SBATCH --output=${log_dir}/benchmark${benchmark}_omp_cc10_%A_%a.log

module load ${GCC_MODULE}

# --- Strict CPU control ---
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=false

cd "${WORK_DIR}" || exit 1

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety rho_csv <<< \$line
IFS=',' read -ra rhos <<< "\$rho_csv"

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety rhos=\$rho_csv  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

for rho in "\${rhos[@]}"; do
    echo "--- ${benchmark} safety=\$safety rho=\$rho   start: \$(date) ---"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety 0 \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho"
    fi
done

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
}

# =============================================================================
# B1 — BCPlinear, ThresholdBased, LRS_DelayTime
# =============================================================================
B1_safety=(0.0 1.0)
B1_rho=(1.02 1.03 1.04 1.05 1.06 1.07 1.08)
submit_b B1 B1_safety B1_rho "1-10:00:00"

# =============================================================================
# B2 — BCPlinear, ThresholdBased, LRS_QueueLen
# =============================================================================
B2_safety=(0.0 1.0)
B2_rho=(0.93 0.94 0.95 0.96 0.97 0.98 0.99)
submit_b B2 B2_safety B2_rho "1-10:00:00"

# =============================================================================
# B3 — HighLow, ThresholdBased, LRS_DelayTime
# =============================================================================
B3_safety=(0.0 1.0)
B3_rho=(1.02 1.03 1.04 1.05 1.06 1.07 1.08)
submit_b B3 B3_safety B3_rho "1-10:00:00"

# =============================================================================
# B4 — HighLow, ThresholdBased, LRS_QueueLen
# =============================================================================
B4_safety=(0.0 1.0)
B4_rho=(0.88 0.89 0.90 0.91 0.92 0.93 0.94)
submit_b B4 B4_safety B4_rho "1-10:00:00"
