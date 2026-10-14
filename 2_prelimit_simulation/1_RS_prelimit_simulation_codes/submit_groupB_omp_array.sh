#!/bin/bash
# =============================================================================
# submit_groupB_omp_array.sh
#
# OMP version of submit_groupB_array.sh.
# Uses sim_benchmark_omp (--cpus-per-task=5); OMP parallelizes pricing-search combos.
# rho is still looped inside each task (one srun per rho).
#
# Requires `make sim_benchmark_omp` and `parallel_threads >= 1` in config
# (resolve_omp_threads caps at SLURM_CPUS_PER_TASK).
#
# Submit Group B benchmarks (B1, B2, B3, B4) on the HPC cluster (<your_account>).
# Each benchmark has its OWN safety / rho search lists.
# Array dimension per benchmark = N_safety (rho is looped inside each task).
# Note: B-group is ThresholdBased outsourcing — epsilon is NOT searched; the
# C++ binary receives 0.0 as a placeholder for epsilon.
#
# Single bash invocation submits up to 4 separate sbatch jobs (one per benchmark).
#
# Usage:
#   bash submit_groupB_omp_array.sh         # submit all 4 (B1, B2, B3, B4)
#   bash submit_groupB_omp_array.sh B2      # submit only B2
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
        B1|B2|B3|B4) ;;
        *) echo "ERROR: filter '$SELECTED' must be one of: B1 B2 B3 B4"; exit 1 ;;
    esac
fi

log_dir="${WORK_DIR}/../logs"
mkdir -p "$log_dir"

# HPC-cluster (<your_account>) SBATCH settings
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
# B1 — BCPlinear, ThresholdBased, LRS_DelayTime
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "B1" ]; then
    benchmark="B1"
    safety_values=(0.0 1.0 2.0)
    rho_values=(1.02 1.03 1.04 1.05 1.06 1.07 1.08)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_s_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        echo "$safety" >> "$param_file"
    done
    total=$(wc -l < "$param_file")
    rho_str="${rho_values[*]}"

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task; rho looped inside each task)"
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
read safety <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Looping rho values: ${rho_str}"

# Pass 0.0 as placeholder epsilon (B-group is ThresholdBased; epsilon unused).
for rho in ${rho_str}; do
    echo
    echo "--- ${benchmark}  safety=\$safety  rho=\$rho ---"
    echo "Start: \$(date)"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety 0.0 \$rho
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
# B2 — BCPlinear, ThresholdBased, LRS_QueueLen
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "B2" ]; then
    benchmark="B2"
    safety_values=(0.0 1.0 2.0)
    rho_values=(1.02 1.03 1.04 1.05 1.06 1.07 1.08)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_s_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        echo "$safety" >> "$param_file"
    done
    total=$(wc -l < "$param_file")
    rho_str="${rho_values[*]}"

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task; rho looped inside each task)"
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
read safety <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Looping rho values: ${rho_str}"

for rho in ${rho_str}; do
    echo
    echo "--- ${benchmark}  safety=\$safety  rho=\$rho ---"
    echo "Start: \$(date)"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety 0.0 \$rho
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
# B3 — HighLow, ThresholdBased, LRS_DelayTime
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "B3" ]; then
    benchmark="B3"
    safety_values=(0.0 1.0 2.0)
    rho_values=(1.02 1.03 1.04 1.05 1.06 1.07 1.08)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_s_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        echo "$safety" >> "$param_file"
    done
    total=$(wc -l < "$param_file")
    rho_str="${rho_values[*]}"

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task; rho looped inside each task)"
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
read safety <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Looping rho values: ${rho_str}"

for rho in ${rho_str}; do
    echo
    echo "--- ${benchmark}  safety=\$safety  rho=\$rho ---"
    echo "Start: \$(date)"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety 0.0 \$rho
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
# B4 — HighLow, ThresholdBased, LRS_QueueLen
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "B4" ]; then
    benchmark="B4"
    safety_values=(0.0 1.0 2.0)
    rho_values=(1.02 1.03 1.04 1.05 1.06 1.07 1.08)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_s_omp.txt"
    > "$param_file"
    for safety in "${safety_values[@]}"; do
        echo "$safety" >> "$param_file"
    done
    total=$(wc -l < "$param_file")
    rho_str="${rho_values[*]}"

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task; rho looped inside each task)"
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
read safety <<< \$line

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "Looping rho values: ${rho_str}"

for rho in ${rho_str}; do
    echo
    echo "--- ${benchmark}  safety=\$safety  rho=\$rho ---"
    echo "Start: \$(date)"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety 0.0 \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho; continuing"
    fi
done

echo
echo "=== All rhos done for task \$SLURM_ARRAY_TASK_ID. Date: \$(date) ==="
EOF
fi
