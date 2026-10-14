#!/bin/bash
# =============================================================================
# submit_groupA_omp_array_cc3.sh   [CC3 Extended Criss-Cross]
#
# OMP version of submit_groupA_array_cc3.sh.
# Uses sim_benchmark_omp (built with -DUSE_OMP=1). Each SLURM task gets
# --cpus-per-task=5; OMP parallelizes the inner combo loop (within each
# rho call) using all 5 cores. Requires parallel_threads >= 5 in
# benchmark_search_config.txt (resolve_omp_threads caps at SLURM_CPUS_PER_TASK).
#
# Strict CPU control (matches submit_benchmarkA_omp_array.sh):
#   OMP_NUM_THREADS = SLURM_CPUS_PER_TASK    (cap OMP at cores SLURM gave us)
#   OMP_DYNAMIC     = false                   (no thread-count adjustments)
#   srun --cpu-bind=cores                     (pin threads to allocated cores)
# MKL_NUM_THREADS not needed — sim_benchmark_omp links nn_model_stub (no MKL).
#
# Submit Group A benchmarks (A1, A2, A3, A4) on the HPC cluster.
# Each benchmark has its OWN safety / epsilon / rho search lists, defined
# below. Array dimension per benchmark = N_safety × N_epsilon (rho is NOT
# in the array dim — each array task loops over rho serially in-task).
#
# Single bash invocation submits up to 4 separate sbatch array jobs
# (one sbatch per benchmark).
#
# Usage:
#   bash submit_groupA_omp_array_cc3.sh         # submit A1+A2+A3+A4
#   bash submit_groupA_omp_array_cc3.sh A2      # submit only A2
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

# HPC-cluster SBATCH settings
SBATCH_ACCOUNT="<your_account>"
SBATCH_PARTITION="<your_partition>"
GCC_MODULE="gcc/13.2.0"
CPUS_PER_TASK=5

if [ ! -x "${WORK_DIR}/sim_benchmark_omp" ]; then
    echo "ERROR: missing executable ${WORK_DIR}/sim_benchmark_omp"
    echo "Build the binary first with: cd \"${WORK_DIR}\" && make DIM_K_SELECT=3 sim_benchmark_omp"
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
    safety_values=(0.0 1.0)
    epsilon_values=(0 0.11 0.22 0.34)
    rho_values=(0.92 0.93 0.94 0.95 0.96 0.97 0.98)         # inner-loop, NOT in array dim
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_cc3_omp.txt"
    > "$param_file"
    rho_csv=$(IFS=,; echo "${rho_values[*]}")
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            echo "$safety $epsilon $rho_csv" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task, rho looped IN-task)"
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
#SBATCH --job-name=cc3_${benchmark}_omp
#SBATCH --output=${log_dir}/benchmark${benchmark}_omp_cc3_%A_%a.log

module load ${GCC_MODULE}

# --- Strict CPU control ---
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=false

cd "${WORK_DIR}" || exit 1

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho_csv <<< \$line
IFS=',' read -ra rhos <<< "\$rho_csv"

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon rhos=\$rho_csv  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

for rho in "\${rhos[@]}"; do
    echo "--- ${benchmark} safety=\$safety eps=\$epsilon rho=\$rho   start: \$(date) ---"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho"
    fi
done

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
fi

# =============================================================================
# A2 — BCPlinear, SinglePushing, LRS_QueueLen
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "A2" ]; then
    benchmark="A2"
    safety_values=(0.0 1.0)
    epsilon_values=(0 0.11 0.22 0.34)
    rho_values=(0.78 0.79 0.80 0.81 0.82 0.83 0.84)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_cc3_omp.txt"
    > "$param_file"
    rho_csv=$(IFS=,; echo "${rho_values[*]}")
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            echo "$safety $epsilon $rho_csv" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task, rho looped IN-task)"
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
#SBATCH --job-name=cc3_${benchmark}_omp
#SBATCH --output=${log_dir}/benchmark${benchmark}_omp_cc3_%A_%a.log

module load ${GCC_MODULE}

# --- Strict CPU control ---
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=false

cd "${WORK_DIR}" || exit 1

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho_csv <<< \$line
IFS=',' read -ra rhos <<< "\$rho_csv"

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon rhos=\$rho_csv  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

for rho in "\${rhos[@]}"; do
    echo "--- ${benchmark} safety=\$safety eps=\$epsilon rho=\$rho   start: \$(date) ---"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho"
    fi
done

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
fi

# =============================================================================
# A3 — HighLow, SinglePushing, LRS_DelayTime
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "A3" ]; then
    benchmark="A3"
    safety_values=(0.0 1.0)
    epsilon_values=(0 0.11 0.22 0.34)
    rho_values=(0.90 0.91 0.92 0.93 0.94 0.95 0.96)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_cc3_omp.txt"
    > "$param_file"
    rho_csv=$(IFS=,; echo "${rho_values[*]}")
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            echo "$safety $epsilon $rho_csv" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task, rho looped IN-task)"
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
#SBATCH --job-name=cc3_${benchmark}_omp
#SBATCH --output=${log_dir}/benchmark${benchmark}_omp_cc3_%A_%a.log

module load ${GCC_MODULE}

# --- Strict CPU control ---
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=false

cd "${WORK_DIR}" || exit 1

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho_csv <<< \$line
IFS=',' read -ra rhos <<< "\$rho_csv"

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon rhos=\$rho_csv  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

for rho in "\${rhos[@]}"; do
    echo "--- ${benchmark} safety=\$safety eps=\$epsilon rho=\$rho   start: \$(date) ---"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho"
    fi
done

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
fi

# =============================================================================
# A4 — HighLow, SinglePushing, LRS_QueueLen
# =============================================================================
if [ -z "$SELECTED" ] || [ "$SELECTED" = "A4" ]; then
    benchmark="A4"
    safety_values=(0.0 1.0)
    epsilon_values=(0 0.11 0.22 0.34)
    rho_values=(0.80 0.81 0.82 0.83 0.84 0.85 0.86)
    sbatch_time="1-10:00:00"

    param_file="${WORK_DIR}/params_${benchmark}_cc3_omp.txt"
    > "$param_file"
    rho_csv=$(IFS=,; echo "${rho_values[*]}")
    for safety in "${safety_values[@]}"; do
        for epsilon in "${epsilon_values[@]}"; do
            echo "$safety $epsilon $rho_csv" >> "$param_file"
        done
    done
    total=$(wc -l < "$param_file")

    echo "Submitting $total ${benchmark} array tasks (OMP, ${CPUS_PER_TASK} threads/task, rho looped IN-task)"
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
#SBATCH --job-name=cc3_${benchmark}_omp
#SBATCH --output=${log_dir}/benchmark${benchmark}_omp_cc3_%A_%a.log

module load ${GCC_MODULE}

# --- Strict CPU control ---
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=false

cd "${WORK_DIR}" || exit 1

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${param_file}")
read safety epsilon rho_csv <<< \$line
IFS=',' read -ra rhos <<< "\$rho_csv"

echo "=== Task \$SLURM_ARRAY_TASK_ID: benchmark=${benchmark} safety=\$safety epsilon=\$epsilon rhos=\$rho_csv  cpus=\${SLURM_CPUS_PER_TASK} ==="
echo "Host: \$(hostname)  Date: \$(date)"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

for rho in "\${rhos[@]}"; do
    echo "--- ${benchmark} safety=\$safety eps=\$epsilon rho=\$rho   start: \$(date) ---"
    srun --cpu-bind=cores ./sim_benchmark_omp benchmark_search_config.txt ${benchmark} \$safety \$epsilon \$rho
    rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "WARNING: sim_benchmark_omp exited \$rc on rho=\$rho"
    fi
done

echo "=== Task \$SLURM_ARRAY_TASK_ID done. Date: \$(date) ==="
EOF
fi
