#!/bin/bash
# =============================================================================
# submit_groupA_omp_array.sh
#
# OMP version of submit_groupA_array.sh.
# Uses sim_benchmark_omp (built with `make sim_benchmark_omp` → -fopenmp -DUSE_OMP=1).
# Each SLURM task gets --cpus-per-task=5; OMP parallelizes pricing-search combos
# using those 5 cores. rho is still looped inside each task (one srun per rho).
#
# Requires:
#   - benchmark_search_config.txt has `parallel_threads = 5` (or any value >= 1;
#     resolve_omp_threads caps at SLURM_CPUS_PER_TASK).
#   - sim_benchmark_omp built (make sim_benchmark_omp).
#
# Strict CPU control:
#   OMP_NUM_THREADS = SLURM_CPUS_PER_TASK
#   OMP_DYNAMIC     = false
#   srun --cpu-bind=cores
#
# Submit Group A benchmarks (A1, A2, A3, A4) on the HPC cluster (<your_account>).
# Each benchmark has its OWN safety / epsilon / rho search lists.
# Array dimension per benchmark = N_safety × N_epsilon (rho is looped inside
# each array task, NOT part of the array dimension).
#
# Single bash invocation submits up to 4 separate sbatch jobs (one per benchmark).
#
# Usage:
#   bash submit_groupA_omp_array.sh         # submit all 4 (A1, A2, A3, A4)
#   bash submit_groupA_omp_array.sh A2      # submit only A2
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
        A1|A2|A3|A4) ;;
        *) echo "ERROR: filter '$SELECTED' must be one of: A1 A2 A3 A4"; exit 1 ;;
    esac
fi

log_dir="${WORK_DIR}/../logs"
mkdir -p "$log_dir"

# HPC-cluster (<your_account>) SBATCH settings (kept as local vars so each sbatch block can interpolate)
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
# A1 — BCPlinear, SinglePushing, LRS_DelayTime
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "A1" ]; then
    benchmark="A1"
    safety_values=(0.0 1.0 2.0)
    epsilon_values=(0 0.11 0.22 0.34 0.45)
    rho_values=(0.93 0.94 0.95 0.96 0.97 0.98 0.99)
    sbatch_time="1-10:00:00"   # ~N_rho * 1h + buffer

    param_file="${WORK_DIR}/params_${benchmark}_se_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            echo "$safety $epsilon" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")
    rho_str="${rho_values[*]}"

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task; rho looped inside each task)"
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
read safety epsilon <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Looping rho values: ${rho_str}"

for rho in ${rho_str}; do
    echo
    echo "--- ${benchmark}  safety=\$safety  epsilon=\$epsilon  rho=\$rho ---"
    echo "Start: \$(date)"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho; continuing"
    fi
done

echo
echo "=== All rhos done for task \$SLURM_ARRAY_TASK_ID. Date: \$(date) ==="
EOF
fi

# =============================================================================
# A2 — BCPlinear, SinglePushing, LRS_QueueLen
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "A2" ]; then
    benchmark="A2"
    safety_values=(0.0 1.0 2.0)
    epsilon_values=(0 0.11 0.22 0.34 0.45)
    rho_values=(0.93 0.94 0.95 0.96 0.97 0.98 0.99)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_se_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            echo "$safety $epsilon" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")
    rho_str="${rho_values[*]}"

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task; rho looped inside each task)"
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
read safety epsilon <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Looping rho values: ${rho_str}"

for rho in ${rho_str}; do
    echo
    echo "--- ${benchmark}  safety=\$safety  epsilon=\$epsilon  rho=\$rho ---"
    echo "Start: \$(date)"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho; continuing"
    fi
done

echo
echo "=== All rhos done for task \$SLURM_ARRAY_TASK_ID. Date: \$(date) ==="
EOF
fi

# =============================================================================
# A3 — HighLow, SinglePushing, LRS_DelayTime
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "A3" ]; then
    benchmark="A3"
    safety_values=(0.0 1.0 2.0)
    epsilon_values=(0 0.11 0.22 0.34 0.45)
    rho_values=(0.92 0.93 0.94 0.95 0.96 0.97 0.98)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_se_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            echo "$safety $epsilon" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")
    rho_str="${rho_values[*]}"

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task; rho looped inside each task)"
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
read safety epsilon <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Looping rho values: ${rho_str}"

for rho in ${rho_str}; do
    echo
    echo "--- ${benchmark}  safety=\$safety  epsilon=\$epsilon  rho=\$rho ---"
    echo "Start: \$(date)"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho; continuing"
    fi
done

echo
echo "=== All rhos done for task \$SLURM_ARRAY_TASK_ID. Date: \$(date) ==="
EOF
fi

# =============================================================================
# A4 — HighLow, SinglePushing, LRS_QueueLen
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "A4" ]; then
    benchmark="A4"
    safety_values=(0.0 1.0 2.0)
    epsilon_values=(0 0.11 0.22 0.34 0.45)
    rho_values=(0.92 0.93 0.94 0.95 0.96 0.97 0.98)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_se_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            echo "$safety $epsilon" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")
    rho_str="${rho_values[*]}"

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task; rho looped inside each task)"
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
read safety epsilon <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Looping rho values: ${rho_str}"

for rho in ${rho_str}; do
    echo
    echo "--- ${benchmark}  safety=\$safety  epsilon=\$epsilon  rho=\$rho ---"
    echo "Start: \$(date)"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho; continuing"
    fi
done

echo
echo "=== All rhos done for task \$SLURM_ARRAY_TASK_ID. Date: \$(date) ==="
EOF
fi
