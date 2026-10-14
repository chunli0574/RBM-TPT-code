#!/bin/bash
# =============================================================================
# submit_groupC_omp_array.sh
#
# OMP version of submit_groupC_array.sh.
# Uses sim_benchmark_omp (built with `make sim_benchmark_omp` → -fopenmp -DUSE_OMP=1).
# Each SLURM task gets --cpus-per-task=5; OMP parallelizes pricing-search combos
# and scheduling-search combos using those 5 cores.
#
# Requires:
#   - benchmark_search_config.txt has `parallel_threads = 5` (or any value >= 1;
#     resolve_omp_threads caps at SLURM_CPUS_PER_TASK).
#   - sim_benchmark_omp built (make sim_benchmark_omp).
#
# Strict CPU control (matches CC3 reference):
#   OMP_NUM_THREADS = SLURM_CPUS_PER_TASK    (cap OMP at cores SLURM gave us)
#   OMP_DYNAMIC     = false                   (no thread-count adjustments)
#   srun --cpu-bind=cores                     (pin threads to allocated cores)
#
# Submit Group C benchmarks (C1, C2) on cluster A.
# Each benchmark has its OWN safety / epsilon / rho search lists.
# Array dimension per benchmark = N_safety × N_epsilon × N_rho (1 rho per task;
# rho is part of the array dim, not looped inside the task).
#
# Single bash invocation submits up to 2 separate sbatch jobs (one per benchmark).
#
# Usage:
#   bash submit_groupC_omp_array.sh         # submit both C1 and C2
#   bash submit_groupC_omp_array.sh C2      # submit only C2
#
# NOTE: sim_benchmark_omp links against nn_model_stub.cpp (no-op NN) and does
# NOT need libtensorflow at runtime. No LD_LIBRARY_PATH setup required.
# =============================================================================

# --- User settings -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/codes"

# --- Optional CLI filter -----------------------------------------------------
SELECTED="${1:-}"
if [ -n "$SELECTED" ]; then
    case "$SELECTED" in
        C1|C2) ;;
        *) echo "ERROR: filter '$SELECTED' must be one of: C1 C2"; exit 1 ;;
    esac
fi

log_dir="${WORK_DIR}/../logs"
mkdir -p "$log_dir"

# cluster A SLURM settings
SBATCH_ACCOUNT="<your_account>"
SBATCH_PARTITION="<your_partition>"
GCC_MODULE="gcc/13.2.0"
CPUS_PER_TASK=5

if [ ! -x "${WORK_DIR}/sim_benchmark_omp" ]; then
    echo "ERROR: missing executable ${WORK_DIR}/sim_benchmark_omp"
    echo "Build it first with: cd \"${WORK_DIR}\" && make sim_benchmark_omp"
    exit 1
fi
if [ ! -f "${WORK_DIR}/benchmark_search_config.txt" ]; then
    echo "ERROR: missing config ${WORK_DIR}/benchmark_search_config.txt"
    exit 1
fi

# =============================================================================
# C1 — BCPlinear, SinglePushing, WMP scheduling
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "C1" ]; then
    benchmark="C1"
    safety_values=(0.0 1.0 2.0)
    epsilon_values=(0 0.11 0.22 0.34 0.45)
    rho_values=(0.98 0.99 1.00 1.01 1.02 1.03 1.04)
    sbatch_time="1-10:00:00"   # WMP scheduling search; single rho per task with OMP

    param_file="${WORK_DIR}/params_${benchmark}_ser_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            for rho in "${rho_values[@]}"; do
                echo "$safety $epsilon $rho" >> "$param_file"
            done
        done
    done
    total=$(wc -l < "$param_file")

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task, 1 rho per task)"
    echo "  safety:  ${safety_values[*]}"
    echo "  epsilon: ${epsilon_values[*]}"
    echo "  rho:     ${rho_values[*]}"

    sbatch --array=0-$(( total - 1 )) <<EOF
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=2G
#SBATCH --time=${sbatch_time}
#SBATCH --job-name=rs_${benchmark}_omp
#SBATCH --output=${log_dir}/benchmark${benchmark}_omp_%A_%a.log

module load ${GCC_MODULE}

# --- Strict CPU control ---
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=false

cd "${WORK_DIR}"

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon rho=\$rho  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
rc=\$?
if [ \$rc -ne 0 ]; then
    echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho"
fi

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
fi

# =============================================================================
# C2 — HighLow, SinglePushing, WMP scheduling
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "C2" ]; then
    benchmark="C2"
    safety_values=(0.0 1.0 2.0)
    epsilon_values=(0 0.11 0.22 0.34 0.45)
    rho_values=(0.98 0.99 1.00 1.01 1.02 1.03 1.04)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_ser_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            for rho in "${rho_values[@]}"; do
                echo "$safety $epsilon $rho" >> "$param_file"
            done
        done
    done
    total=$(wc -l < "$param_file")

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task, 1 rho per task)"
    echo "  safety:  ${safety_values[*]}"
    echo "  epsilon: ${epsilon_values[*]}"
    echo "  rho:     ${rho_values[*]}"

    sbatch --array=0-$(( total - 1 )) <<EOF
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=2G
#SBATCH --time=${sbatch_time}
#SBATCH --job-name=rs_${benchmark}_omp
#SBATCH --output=${log_dir}/benchmark${benchmark}_omp_%A_%a.log

module load ${GCC_MODULE}

# --- Strict CPU control ---
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=false

cd "${WORK_DIR}"

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon rho=\$rho  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
rc=\$?
if [ \$rc -ne 0 ]; then
    echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho"
fi

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
fi
