# Dynamic Control of a Make-to-Order Manufacturing System under Throughput Time Constraints: An Effective Computational Method in the High-Dimensional Case

This repository contains the code and data accompanying the paper above. It provides what is needed to
reproduce the numerical experiments reported in the paper: neural-network
training of the computational method, C++ pre-limit simulation of
the proposed policy and the benchmark policies, and the Python analysis that
produces the tables and figures.


---

## 1. Repository layout

```
.
├── README.md                  ← this file
├── LICENSE                    ← MIT license
│
├── 1_NN_training/             ← Neural-network training (Python / TensorFlow)
│   ├── 1_RS_NN_training_codes/     Rybko–Stolyar (RS) training code
│   ├── 2_ECC_NN_training_codes/    Extended criss-cross (ECC) training code
│   ├── 3_RS_saved_NNs/             Trained NN models used for RS results
│   └── 4_ECC_saved_NNs/            Trained NN models used for ECC results
│
├── 2_prelimit_simulation/     ← Pre-limit simulation (C++) + saved results
│   ├── 1_RS_prelimit_simulation_codes/   RS simulation code + SLURM scripts
│   ├── 2_ECC_prelimit_simulation_codes/  ECC simulation code + SLURM scripts
│   ├── 3_RS_saved_results/               RS simulation output
│   └── 4_ECC_saved_results/              ECC simulation output
│
└── 3_support/                 ← Analysis & plotting (Python / Jupyter)
    ├── 1_plot_training_results/            illustrative & solution figures
    └── 2_find_optimal_parameter_and_plot_delays/
                                            summarize search results, pick
                                            optimal parameters, plot delays
```


`RS / ECC3 / ECC10` throughout: the three test problems reported in the
paper (Rybko–Stolyar, 3-server extended criss-cross, 10-server extended
criss-cross).

---

## 2. Requirements

### 2.1 Hardware

All experiments ran as SLURM jobs on **CPU** nodes of two HPC clusters
(referred to below as cluster A and cluster B); **no GPU is required**. The
per-job resource requests used in the submit scripts were as follows.

**Neural-network training (`1_NN_training`, CPU):**

| Problem | CPUs | Memory | Walltime |
|---|---|---|---|
| RS    |  10 |  20 GB | ~3 h  |
| ECC3  |  5 |  48 GB | ~6 h  |
| ECC10 | 20 | 100 GB | ~24 h |

**Pre-limit simulation (`2_prelimit_simulation`, CPU SLURM arrays; resources per array task):**

| Stage | CPUs/task | Memory | Walltime |
|---|---|---|---|
| Proposed policy — RS / ECC3           | 1 | 1 GB   | <20 h |
| Proposed policy — ECC10 (batched, OpenMP) | 8 | 2 GB | <36 h |
| Benchmarks — RS / ECC3 (OpenMP)       | 5 | 2 GB   | <36 h |
| Benchmarks — ECC10 (OpenMP)           | 5 | 2 GB   | <72 h |
| Delay distribution (one-combo)        | 1 | 1 GB | ~3 h |

Each search stage is submitted as a large SLURM job array (one array task per
parameter subset), so total wall-clock time depends on cluster throughput.
Reproducing the *full* parameter search is time-consuming; see §5 for how the
experiments are divided into subsets and what is needed to reproduce only the
reported (optimal-parameter) results.

- **Analysis (`3_support`)** runs on a standard laptop/desktop; a few GB of
  RAM is sufficient.

### 2.2 Software

**Python (training and analysis)**
- Python **3.12** (loaded on the cluster via `module load python`).
- TensorFlow **2.17.0** (current Keras models). Some NN models were originally
  trained on TensorFlow **2.12.0** (legacy SavedModel format); see §3.
- Additional packages: `numpy`, `pandas`, `matplotlib`, `scipy`,
  `statsmodels`, `munch`, `absl-py`, `Pillow` (PIL).

**C++ (pre-limit simulation)**
- A C++17 compiler. Experiments used **gcc 13.2.0** (cluster A) and
  **gcc 14.1.0** (cluster B): `module load gcc/13.2.0` or `gcc/14.1.0`.
- **cppflow** (header-only) + **libtensorflow** — required only to evaluate
  the legacy TensorFlow *SavedModel* NNs from C++. Expected under
  `$HOME/libs/cppflow/include` and `$HOME/libs/libtensorflow/{include,lib}`
  (see the Makefile header). At run time:
  `export LD_LIBRARY_PATH=$HOME/libs/libtensorflow/lib` (Linux) or
  `DYLD_LIBRARY_PATH=...` (macOS).
- **BLAS** (optional, accelerates the native NN forward pass): Intel **MKL**
  (`module load mkl`) or **OpenBLAS**. Enable with `make USE_BLAS=1`.
- **OpenMP** for the parallel (`_omp`) simulation variants.

**Operating system**: Linux (HPC clusters) for the simulation; the C++ code
also builds on macOS. Analysis notebooks are OS-independent.

---

## 3. Trained neural-network models (`1_NN_training/3_*`, `4_*`)

The folders `1_NN_training/3_RS_saved_NNs/` and `1_NN_training/4_ECC_saved_NNs/`
hold the trained models referenced by the C++ simulation. The layout is flat:
all model files / SavedModel directories sit directly in these folders (there
are no `model_base_dir/` / `model_new_base_dir/` subfolders). Three formats may appear:

1. **SavedModel** — trained on TensorFlow 2.12.0 (no longer available online);
   evaluated from C++ via **cppflow**.
2. **Keras** — trained on TensorFlow 2.17.0.
3. **Native** (`*.weights.bin`) — transformed from (1)/(2) by the Python
   scripts in the C++ code folder (`export_keras_weights*.py`,
   `export_savedmodel_weights.py`); used for fast NN evaluation inside the C++
   code (optionally BLAS-accelerated).

**Which models are included.** The proposed-policy parameter search sweeps a
range of tuning parameter `rho`, and **each `rho` uses a different
trained NN**. To keep the archive small, this repository ships only the NN
models needed to reproduce the **reported (optimal-parameter) results** and
the **illustrative figures** — i.e. the models at the optimal `rho` for each
problem (RS `rho = 1.02`, ECC3 `rho = 0.91`, ECC10 `rho = 0.93`) plus the
baseline models `rho = 1.0` used for illustrations. The additional per-`rho` models used
only during the parameter *search* are **not** included; they can be
regenerated with the training code in `1_NN_training` (see §4, step 1). The
search results themselves are preserved as CSV files (see §5).

---

## 4. How to reproduce the experiments

The workflow has four stages. Stages 1 and 2 run on an HPC cluster; stages 3
and 4 run locally.

### Step 1 (optional) — Train the neural networks

Trained models are already provided (§3), so this step is only needed to
regenerate them. Training code is in `1_NN_training/{1_RS,2_ECC}_NN_training_codes/`;
configurations are in `configs/*.json`. Submit the training sweep with:

```bash
cd 1_NN_training/1_RS_NN_training_codes      # or 2_ECC_NN_training_codes
bash run_all.sh                              # RS
# ECC: bash run_all_3dim.sh   /   bash run_all_10dim.sh
```

### Step 2 — Pre-limit simulation on the server (C++)

Code is in `2_prelimit_simulation/{1_RS,2_ECC}_prelimit_simulation_codes/`.
Load modules and build in `codes/`, then submit the SLURM array scripts in the
parent folder. Select the ECC variant at compile time via `DIM_K_SELECT`
(3 = ECC3, 10 = ECC10).

**Load modules / clean**
```bash
# cluster A
module load gcc/13.2.0
module load mkl
make clean
# cluster B
srun --account=<your_account> --pty bash --login
module load gcc/14.1.0
make clean
```

**Proposed policy**
```bash
cd codes
# RS and ECC3:
make sim_proposed
# ECC10 (batched NN + BLAS):
make BATCH_NN=1 USE_BLAS=1 DIM_K_SELECT=10 sim_proposed_batch
make sim_proposed_batch_omp          # OpenMP version for ECC10
# (add PROFILING=1 for time profiling)
```
Submit with `submit_proposed_*array_*.sh`.

**Benchmarks (12 policies, grouped A/B/C/D)**
```bash
cd codes
# RS / ECC3:
make sim_benchmark_omp      # parallel
# ECC10:
make DIM_K_SELECT=10 sim_benchmark_omp
```
Submit with `submit_group{A,B,C,D}_*array_*.sh`

### Step 3 — Summarize search results & find optimal parameters (Python)

Run the notebooks in
`3_support/2_find_optimal_parameter_and_plot_delays/`
(`Formal_run_5000paths_*_20260921.ipynb`). They read the consolidated search
grids (`*_full.csv`, see §5), summarize proposed / proposed-extra (RS only,
`Formal_run_5000paths_RS_proposed_extra_20260921.ipynb`) / benchmarks /
benchmarks-extra, select the optimal parameters, and perform the boundary
check (re-run step 2 if an optimum lies on the search boundary).

### Step 4 — Delay distribution under optimal parameters (C++, then Python)

Fill the searched optimal parameters into
`codes/one_combo_input*.txt`, then build and submit:

```bash
cd codes
# RS:    make sim_one_combo                 → submit_one_combo_multi.sh
# ECC3:  make sim_one_combo                 → submit_one_combo_multi_cc3.sh abs   (13 policies)
#                                             submit_one_combo_multi_cc3.sh rel   (5 remaining)
# ECC10: make DIM_K_SELECT=10 sim_one_combo_batch
#                                           → submit_one_combo_multi_cc10.sh abs / rel
```
Finally, summarize the delay times into the paper figures with the delay
notebooks in `3_support/2_find_optimal_parameter_and_plot_delays/`
(output → `delay_distribution_plots/`).

Illustrative figures (state space, drift/singular controls, limiting-workload
solution, gradient slices) are produced by the notebooks in
`3_support/1_plot_training_results/`.

---

## 5. Saved results (`2_prelimit_simulation/3_*`, `4_*`)

For each problem `*`:

- `1_*_proposed_policy_5000p/` — proposed policy search results
- `1_RS_proposed_policy_extra_5000p/` — extra proposed policy runs (RS only;
  extends the `lambda` grid to 22, 24, 26 beyond the main search's 2–20)
- `2_*_benchmarks_5000p/` — benchmark search results
- `2_*_benchmarks_5000p_extra/` — extra benchmark runs (when an optimum lay on
  the search boundary)
- `3_*_opt_delay_distribution_5000p/` — delay times under the optimal
  parameters, including `delay_distribution_results/` (`*_summary.csv`,
  `*_delay_hist.csv`)

**What is included and why.** The parameter search produced thousands of raw
per-parameter output files on the server. To keep the archive manageable, the
search results are provided as the **consolidated grids** that the analysis
notebooks actually read: one `*_full.csv` per policy (produced by the first
part of each `Formal_run_5000paths_*` notebook). These are sufficient to
reproduce steps 3–4 (finding the optima and all reported tables/figures).
The raw per-run server files and full run logs are large and are **not**
shipped; they can be regenerated with the C++ code and SLURM scripts in
step 2. 

---

## 6. Configuration to adjust before running

**NN model paths.** The NN model directories in `codes/one_combo_input*.txt`,
`codes/proposed_policy_config.txt`, and the
default `NN_MODEL_PATH` in `codes/params.h` are given as paths **relative to
the `codes/` directory**, pointing into the included model folders, e.g.

```
model_new_base_dir  = ../../../1_NN_training/3_RS_saved_NNs        # RS
model_base_dir      = ../../../1_NN_training/4_ECC_saved_NNs       # ECC
new_model_base_dir  = ../../../1_NN_training/4_ECC_saved_NNs       # ECC
```

These resolve correctly when the simulation is built and run from `codes/`
(as the submit scripts do). If you run from a different directory, change them
to absolute paths.

**SLURM submit scripts.** The scripts in
`2_prelimit_simulation/*/` and `1_NN_training/*/` contain placeholders
`<your_account>` and `<your_partition>` for the SLURM account and partition.
Set these to values valid on your cluster before submitting.

---


