#!/bin/bash
# =============================================================================
# submit_groupD_omp_array_cc10.sh   [CC10 Extended Criss-Cross]
#
# OMP version of submit_groupD_array_cc10.sh.
# Uses sim_benchmark_omp (built with -DUSE_OMP=1). Each SLURM task gets
# --cpus-per-task=5; OMP parallelizes the inner (lambda_tilde × beta_rate)
# combo sweep using all 5 cores. Requires parallel_threads >= 5 in
# benchmark_search_config.txt (resolve_omp_threads caps at SLURM_CPUS_PER_TASK).
#
# Strict CPU control (matches submit_benchmarkD_omp_array.sh):
#   OMP_NUM_THREADS = SLURM_CPUS_PER_TASK    (cap OMP at cores SLURM gave us)
#   OMP_DYNAMIC     = false                   (no thread-count adjustments)
#   srun --cpu-bind=cores                     (pin threads to allocated cores)
# MKL_NUM_THREADS not needed — sim_benchmark_omp links nn_model_stub (no MKL).
#
# Submit Group D benchmarks (D1, D2) × {abs, rel} on cluster B
# (gcc/14.1.0).
# D-group uses ThresholdBased outsourcing — no epsilon search (the C++ side
# ignores the epsilon arg for D benchmarks, but we still pass a placeholder
# 0 for CLI uniformity). Array dim per sub-benchmark = N_safety × N_rho
# (rho IS in the array dim — one (s, rho) per task).
#
# The variant ("abs" or "rel") is passed as the 6th CLI arg to sim_benchmark_omp,
# which overrides config wmp_extra_relative / beta_rate_symmetric.
#
# Single bash invocation submits up to 4 separate sbatch array jobs:
# D1-abs, D1-rel, D2-abs, D2-rel.
#
# Usage:
#   bash submit_groupD_omp_array_cc10.sh             # all 4 sub-benchmarks
#   bash submit_groupD_omp_array_cc10.sh D1-abs      # only D1-abs
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/codes"

SELECTED="${1:-}"
if [ -n "$SELECTED" ]; then
    case "$SELECTED" in
        D1-abs|D1-rel|D2-abs|D2-rel) ;;
        *) echo "ERROR: filter '$SELECTED' must be one of: D1-abs D1-rel D2-abs D2-rel"; exit 1 ;;
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
    echo "Build the CC10 binary first with: cd \"${WORK_DIR}\" && make DIM_K_SELECT=10 sim_benchmark_omp"
    exit 1
fi
if [ ! -f "${WORK_DIR}/benchmark_search_config.txt" ]; then
    echo "ERROR: missing config ${WORK_DIR}/benchmark_search_config.txt"
    exit 1
fi

submit_dv() {
    local benchmark="$1"   # D1 or D2
    local variant="$2"     # abs or rel
    local safety_arr_name="$3"
    local rho_arr_name="$4"
    local sbatch_time="$5"
    local tag="${benchmark}-${variant}"

    [ -n "$SELECTED" ] && [ "$SELECTED" != "$tag" ] && return

    eval "local safeties=(\"\${${safety_arr_name}[@]}\")"
    eval "local rhos=(\"\${${rho_arr_name}[@]}\")"

    local param_file="${WORK_DIR}/params_${tag}_cc10_omp.txt"
    > "$param_file"
    for safety in "${safeties[@]}"; do
        for rho in "${rhos[@]}"; do
            echo "$safety $rho" >> "$param_file"
        done
    done
    local total=$(wc -l < "$param_file")

    echo "Submitting $total ${tag} array tasks  (variant=${variant}, OMP, ${CPUS_PER_TASK} threads/task, 1 rho per task; no epsilon search)"
    echo "  safety: ${safeties[*]}"
    echo "  rho:    ${rhos[*]}"

    sbatch --array=0-$(( total - 1 )) <<EOF
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=2G
#SBATCH --time=${sbatch_time}
#SBATCH --job-name=cc10_${tag}_omp
#SBATCH --output=${log_dir}/benchmark${tag}_omp_cc10_%A_%a.log

module load ${GCC_MODULE}

# --- Strict CPU control ---
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=false

cd "${WORK_DIR}"

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety rho <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} variant=${variant} safety=\$safety rho=\$rho  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety 0 \$rho ${variant}
rc=\$?
if [ \$rc -ne 0 ]; then
    echo "WARNING: sim_benchmark_omp exited \$rc"
fi

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
}

# =============================================================================
# D1 — BCPlinear, ThresholdBased, WMP scheduling
# =============================================================================
D1_abs_safety=(0.0 1.0)
D1_abs_rho=(0.97 0.98 0.99 1.00 1.01 1.02 1.03)
submit_dv D1 abs D1_abs_safety D1_abs_rho "2-12:00:00"

D1_rel_safety=(0.0 1.0)
D1_rel_rho=(0.97 0.98 0.99 1.00 1.01 1.02 1.03)
submit_dv D1 rel D1_rel_safety D1_rel_rho "2-12:00:00"

# =============================================================================
# D2 — HighLow, ThresholdBased, WMP scheduling
# =============================================================================
D2_abs_safety=(0.0 1.0)
D2_abs_rho=(0.97 0.98 0.99 1.00 1.01 1.02 1.03)
submit_dv D2 abs D2_abs_safety D2_abs_rho "2-12:00:00"

D2_rel_safety=(0.0 1.0)
D2_rel_rho=(0.96 0.97 0.98 0.99 1.00 1.01 1.02)
submit_dv D2 rel D2_rel_safety D2_rel_rho "2-12:00:00"
