#!/bin/bash
# =============================================================================
# submit_one_combo_multi_cc10.sh  [CC10 Extended Criss-Cross]
#
# Single SLURM job that loops over many CC10 policies in series, using the
# batch (BATCH_NN=1 USE_BLAS=1) build for [proposed]. Benchmark policies
# don't need NN — but we use the same binary throughout for simplicity.
#
# Variant selector:
#   bash submit_one_combo_multi_cc10.sh abs   → one_combo_input_cc10_abs.txt
#                                               policies = all 13
#   bash submit_one_combo_multi_cc10.sh rel   → one_combo_input_cc10_rel.txt
#                                               policies = 5 WMP only
#   bash submit_one_combo_multi_cc10.sh       → defaults to "abs"
# =============================================================================

# --- Variant selector --------------------------------------------------------
VARIANT="${1:-abs}"
case "$VARIANT" in
    abs)
        INPUT_FILE="one_combo_input_cc10_abs.txt"
        policies=(proposed A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 D1 D2)
        ;;
    rel)
        INPUT_FILE="one_combo_input_cc10_rel.txt"
        policies=(proposed C1 C2 D1 D2)
        ;;
    *)
        echo "Usage: $0 [abs|rel]    (default: abs)"
        exit 1
        ;;
esac

# --- User settings -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/codes"
TF_LIB="$HOME/libs/libtensorflow/lib"
BIN="./sim_one_combo_batch"

log_dir="${WORK_DIR}/../logs"
mkdir -p "$log_dir"

if [ ! -x "${WORK_DIR}/sim_one_combo_batch" ]; then
    echo "ERROR: missing executable ${WORK_DIR}/sim_one_combo_batch"
    echo "Build the binary first with: cd \"${WORK_DIR}\" && make sim_one_combo_batch"
    exit 1
fi
if [ ! -f "${WORK_DIR}/${INPUT_FILE}" ]; then
    echo "ERROR: missing input ${WORK_DIR}/${INPUT_FILE}"
    exit 1
fi

echo "Variant: ${VARIANT}    Input: ${INPUT_FILE}"
echo "Policies: ${policies[*]}"

# --- Submit a single job that loops --------------------------------------------
sbatch <<EOF
#!/bin/bash
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-20:00:00
#SBATCH --job-name=cc10_one_combo_multi_${VARIANT}
#SBATCH --output=${log_dir}/one_combo_multi_cc10_${VARIANT}_%j.log

module load gcc/13.2.0 
module load mkl
cd "${WORK_DIR}" || exit 1

echo "=== sim_one_combo_batch (CC10, ${VARIANT}, multi): input=${INPUT_FILE} ==="
echo "Host: \$(hostname)  Date: \$(date)"

export LD_LIBRARY_PATH=${TF_LIB}:\$LD_LIBRARY_PATH

for p in ${policies[@]}; do
    echo "------ policy = \$p ------"
    ${BIN} ${INPUT_FILE} \$p
done
EOF
