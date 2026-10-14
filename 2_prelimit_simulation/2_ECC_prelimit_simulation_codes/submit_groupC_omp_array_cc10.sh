#!/bin/bash
# =============================================================================
# submit_groupC_omp_array_cc10.sh   [CC10 Extended Criss-Cross]
#
# OMP version of submit_groupC_array_cc10.sh.
# Uses sim_benchmark_omp (built with -DUSE_OMP=1 -DDIM_K_SELECT=10). Each
# SLURM task gets --cpus-per-task=5; OMP parallelizes the inner (lambda_tilde
# × beta_rate) combo sweep using all 5 cores. Requires parallel_threads >= 5
# in benchmark_search_config.txt (resolve_omp_threads caps at SLURM_CPUS_PER_TASK).
#
# Strict CPU control (matches submit_benchmarkC_omp_array.sh):
#   OMP_NUM_THREADS = SLURM_CPUS_PER_TASK    (cap OMP at cores SLURM gave us)
#   OMP_DYNAMIC     = false                   (no thread-count adjustments)
#   srun --cpu-bind=cores                     (pin threads to allocated cores)
# MKL_NUM_THREADS not needed — sim_benchmark_omp links nn_model_stub (no MKL).
#
# Submit Group C benchmarks (C1, C2) × {abs, rel} on cluster B
# (gcc/14.1.0).
# Each (sub-benchmark, variant) combo has its OWN safety / epsilon / rho
# search lists. Array dim per sub-benchmark = N_safety × N_epsilon × N_rho
# (rho IS in the array dim — one (s,e,rho) per task).
#
# The variant ("abs" or "rel") is passed as the 6th CLI arg to sim_benchmark_omp,
# which overrides config wmp_extra_relative / beta_rate_symmetric:
#   abs → wmp_extra_relative=0, beta_rate_symmetric=2
#   rel → wmp_extra_relative=1, beta_rate_symmetric=1
#
# Single bash invocation submits up to 4 separate sbatch array jobs:
# C1-abs, C1-rel, C2-abs, C2-rel.
#
# Usage:
#   bash submit_groupC_omp_array_cc10.sh             # all 4 sub-benchmarks
#   bash submit_groupC_omp_array_cc10.sh C1-abs      # only C1-abs
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/codes"

SELECTED="${1:-}"
if [ -n "$SELECTED" ]; then
    case "$SELECTED" in
        C1-abs|C1-rel|C2-abs|C2-rel) ;;
        *) echo "ERROR: filter '$SELECTED' must be one of: C1-abs C1-rel C2-abs C2-rel"; exit 1 ;;
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

# =============================================================================
# Helper: emit one sbatch array for (benchmark, variant) tuple.
# Usage: submit_cv <benchmark> <variant> <safety_arr> <eps_arr> <rho_arr> <time>
# arrays are passed by name (eval'd).
# =============================================================================
submit_cv() {
    local benchmark="$1"   # C1 or C2
    local variant="$2"     # abs or rel
    local safety_arr_name="$3"
    local eps_arr_name="$4"
    local rho_arr_name="$5"
    local sbatch_time="$6"
    local tag="${benchmark}-${variant}"

    [ -n "$SELECTED" ] && [ "$SELECTED" != "$tag" ] && return

    # Pull arrays by name via eval (portable; nameref needs bash 4.3+)
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

    echo "Submitting $total ${tag} array tasks  (variant=${variant}, OMP, ${CPUS_PER_TASK} threads/task, 1 (s,e,rho) per task)"
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
#SBATCH --output=${log_dir}/benchmark${tag}_omp_cc10_%A_%a.log

module load ${GCC_MODULE}

# --- Strict CPU control ---
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=false

cd "${WORK_DIR}"

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} variant=${variant} safety=\$safety epsilon=\$epsilon rho=\$rho  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho ${variant}
rc=\$?
if [ \$rc -ne 0 ]; then
    echo "WARNING: sim_benchmark_omp exited \$rc"
fi

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
}

# =============================================================================
# C1 — BCPlinear, SinglePushing, WMP scheduling
# =============================================================================
C1_abs_safety=(0.0 1.0)
C1_abs_epsilon=(0 0.11 0.22 0.34)
C1_abs_rho=(0.89 0.90 0.91 0.92 0.93 0.94 0.95)
submit_cv C1 abs C1_abs_safety C1_abs_epsilon C1_abs_rho "2-12:00:00"

C1_rel_safety=(0.0 1.0)
C1_rel_epsilon=(0 0.11 0.22 0.34)
C1_rel_rho=(0.92 0.93 0.94 0.95 0.96 0.97 0.98)
submit_cv C1 rel C1_rel_safety C1_rel_epsilon C1_rel_rho "2-12:00:00"

# =============================================================================
# C2 — HighLow, SinglePushing, WMP scheduling
# =============================================================================
C2_abs_safety=(0.0 1.0)
C2_abs_epsilon=(0 0.11 0.22 0.34)
C2_abs_rho=(0.90 0.91 0.92 0.93 0.94 0.95 0.96)
submit_cv C2 abs C2_abs_safety C2_abs_epsilon C2_abs_rho "2-12:00:00"

C2_rel_safety=(0.0 1.0)
C2_rel_epsilon=(0 0.11 0.22 0.34)
C2_rel_rho=(0.90 0.91 0.92 0.93 0.94 0.95 0.96)
submit_cv C2 rel C2_rel_safety C2_rel_epsilon C2_rel_rho "2-12:00:00"
