#!/bin/bash
# =============================================================================
# submit_groupD_omp_array.sh
#
# OMP version of submit_groupD_array.sh.
# Uses sim_benchmark_omp (--cpus-per-task=5); OMP parallelizes pricing-search
# and scheduling-search combos. rho IS in the array dim (1 rho per task).
#
# Requires `make sim_benchmark_omp` and `parallel_threads >= 1` in config
# (resolve_omp_threads caps at SLURM_CPUS_PER_TASK).
#
# Submit Group D benchmarks (D1, D2) on cluster A.
# Each benchmark has its OWN safety / rho search lists.
# Array dimension per benchmark = N_safety × N_rho (1 rho per task;
# rho is part of the array dim, not looped inside the task).
# Note: D-group is ThresholdBased outsourcing — epsilon is NOT searched; the
# C++ binary receives 0.0 as a placeholder for epsilon.
# Also uses WMP scheduling — allocate enough --time per rho accordingly.
#
# Single bash invocation submits up to 2 separate sbatch jobs (one per benchmark).
#
# Usage:
#   bash submit_groupD_omp_array.sh         # submit both D1 and D2
#   bash submit_groupD_omp_array.sh D2      # submit only D2
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
        D1|D2) ;;
        *) echo "ERROR: filter '$SELECTED' must be one of: D1 D2"; exit 1 ;;
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
# D1 — BCPlinear, ThresholdBased, WMP scheduling
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "D1" ]; then
    benchmark="D1"
    safety_values=(0.0 1.0 2.0)
    rho_values=(1.10 1.11 1.12 1.13 1.14 1.15 1.16)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_sr_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        for rho in "${rho_values[@]}"; do
            echo "$safety $rho" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task, 1 rho per task; rho is in the array dim)"
    echo "  safety: ${safety_values[*]}"
    echo "  rho:    ${rho_values[*]}"

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
read safety rho <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety rho=\$rho ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Start: \$(date)"

# Pass 0.0 as placeholder epsilon (D-group is ThresholdBased; epsilon unused).
srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety 0.0 \$rho
rc=\$?
if [ \$rc -ne 0 ]; then
    echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho"
fi

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
fi

# =============================================================================
# D2 — HighLow, ThresholdBased, WMP scheduling
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "D2" ]; then
    benchmark="D2"
    safety_values=(0.0 1.0 2.0)
    rho_values=(1.07 1.08 1.09 1.10 1.11 1.12 1.13)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_sr_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        for rho in "${rho_values[@]}"; do
            echo "$safety $rho" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task, 1 rho per task; rho is in the array dim)"
    echo "  safety: ${safety_values[*]}"
    echo "  rho:    ${rho_values[*]}"

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
read safety rho <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety rho=\$rho ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Start: \$(date)"

# Pass 0.0 as placeholder epsilon (D-group is ThresholdBased; epsilon unused).
srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety 0.0 \$rho
rc=\$?
if [ \$rc -ne 0 ]; then
    echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho"
fi

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
fi
