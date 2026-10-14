# Makefile for the CC3/CC10 Extended Criss-Cross queueing simulation
#
# Usage:
#   make                        → compile single-run binary ./sim  (DIM_K=3 default)
#   make sim_proposed           → compile proposed NN policy search binary ./sim_proposed
#   make sim_benchmark          → compile benchmark search binary ./sim_benchmark (no TF needed)
#   make clean                  → remove compiled files
#   make run                    → compile and run ./sim
#   make run_proposed           → compile and run ./sim_proposed
#   make run_benchmark          → compile and run ./sim_benchmark
#
# To select dim_K=10 (CC10), pass DIM_K_SELECT=10 on the command line:
#   make DIM_K_SELECT=10 sim_proposed
#   make DIM_K_SELECT=10 sim_benchmark
#   make DIM_K_SELECT=10
#
# Default is dim_K=3 (CC3).
#
# To run with the TensorFlow shared library:
#   macOS:  DYLD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib ./sim
#   Linux:  LD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib ./sim
# Or set it permanently in your shell (Linux/server):
#   export LD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib
#
# Compiler flags:
#   -std=c++17    → C++17 standard (required by cppflow)
#   -O2           → standard optimization
#   -Wall         → show all compiler warnings
#   -g            → include debug info

CXX      = g++

# DIM_K_SELECT: 3 (CC3, default) or 10 (CC10).  Override with: make DIM_K_SELECT=10
DIM_K_SELECT ?= 3

# Include paths:
#   cppflow headers:       $(HOME)/libs/cppflow/include
#   libtensorflow headers: $(HOME)/libs/libtensorflow/include
CPPFLAGS = -I$(HOME)/libs/cppflow/include \
            -I$(HOME)/libs/libtensorflow/include

# -DNDEBUG: strip all assert() in release builds. The only asserts in this
# project are sanity checks in apply_transition / generate_next_event that
# never fire in valid runs; keeping them active costs measurable cycles
# inside the hot path.
CXXFLAGS = -std=c++17 -O2 -Wall -g -DNDEBUG -DDIM_K_SELECT=$(DIM_K_SELECT)

# ---------------------------------------------------------------------------
# BLAS acceleration for native NN forward pass (optional).
# To enable:  make USE_BLAS=1 DIM_K_SELECT=10 sim_proposed
# Requires:   module load mkl   (or openblas)
#
# MKL:      BLAS_CFLAGS = -DUSE_BLAS -I${MKLROOT}/include
#           BLAS_LDFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_rt
# OpenBLAS: BLAS_CFLAGS = -DUSE_BLAS   BLAS_LDFLAGS = -lopenblas
# ---------------------------------------------------------------------------
USE_BLAS ?= 0
ifeq ($(USE_BLAS),1)
  # Default to MKL if MKLROOT is set, otherwise OpenBLAS
  ifneq ($(MKLROOT),)
    BLAS_CFLAGS  = -DUSE_BLAS -I$(MKLROOT)/include
    BLAS_LDFLAGS = -L$(MKLROOT)/lib/intel64 -lmkl_rt -lgomp -lpthread -lm -ldl
  else
    BLAS_CFLAGS  = -DUSE_BLAS
    BLAS_LDFLAGS = -lopenblas
  endif
  CXXFLAGS += $(BLAS_CFLAGS)
endif

# ---------------------------------------------------------------------------
# OpenMP parallelism for benchmark search (combo-level parallelization).
# To enable: make USE_OMP=1 sim_benchmark_omp   (or DIM_K_SELECT=10 ...)
# Only affects the sim_benchmark_omp target; the plain sim_benchmark target
# and all other binaries are built without OpenMP.
# ---------------------------------------------------------------------------
USE_OMP ?= 0

# ---------------------------------------------------------------------------
# Section-level profiling for the batch run_simulation (CC10 + BATCH_NN only).
# To enable:  make PROFILING=1 BATCH_NN=1 USE_BLAS=1 DIM_K_SELECT=10 sim_proposed_batch
# Adds per-section wall-clock timers (Phase 1 / Phase 2 NN / Phase 3 policy /
# event / cost / update / aggregate) and prints a summary at the end of the run.
# When PROFILING=0 (default), all timers compile to no-ops (zero overhead).
# ---------------------------------------------------------------------------
PROFILING ?= 0
ifeq ($(PROFILING),1)
  CXXFLAGS += -DENABLE_PROFILING=1
endif

# ---------------------------------------------------------------------------
# Diagnostic: dump the distribution of NN-input W vectors to a binary file.
# To enable: make DUMP_W=1 ... sim_proposed (or sim_proposed_batch)
# Runtime overrides via env vars:
#   W_DUMP_PATH=./my_counter.bin   (default: ./W_counter.bin)
#   W_DUMP_GRID=0.01               (default: 0.005)
# Automatically writes the file on program exit. No-op when DUMP_W=0.
# ---------------------------------------------------------------------------
DUMP_W ?= 0
ifeq ($(DUMP_W),1)
  CXXFLAGS += -DENABLE_W_DUMP=1
endif

# ---------------------------------------------------------------------------
# Batch NN path (CC10 proposed policy only).
# To enable: make BATCH_NN=1 USE_BLAS=1 DIM_K_SELECT=10 sim_proposed_batch
# This switches run_simulation to a step-major loop that batches N_PATHS
# NN queries into a single cblas_sgemm per layer. Only affects the binary
# built with this flag; CC3, benchmark and denser binaries are unchanged.
# Requires USE_BLAS=1 and DIM_K_SELECT=10.
# ---------------------------------------------------------------------------
BATCH_NN ?= 0

# Linker flags: link against libtensorflow (+ BLAS if enabled)
LDFLAGS  = -L$(HOME)/libs/libtensorflow/lib -ltensorflow $(BLAS_LDFLAGS)

# ---------------------------------------------------------------------------
# Single-run simulation (main.cpp)
# ---------------------------------------------------------------------------

SRCS   = main.cpp simulation.cpp policy.cpp nn_model.cpp run_config.cpp model_path.cpp
OBJS   = $(SRCS:.cpp=.o)
TARGET = sim

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# ---------------------------------------------------------------------------
# Proposed NN policy search binary (proposed_policy_main.cpp)
# ---------------------------------------------------------------------------

SEARCH_SRCS   = proposed_policy_main.cpp simulation.cpp policy.cpp nn_model.cpp run_config.cpp model_path.cpp
SEARCH_OBJS   = $(SEARCH_SRCS:.cpp=.o)
SEARCH_TARGET = sim_proposed

$(SEARCH_TARGET): $(SEARCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $(SEARCH_TARGET) $(SEARCH_OBJS) $(LDFLAGS)

# ---------------------------------------------------------------------------
# Batch-NN proposed policy search (CC10 only).
# Re-invokes make with ENABLE_BATCH_NN=1, DIM_K_SELECT=10, USE_BLAS=1, and
# outputs a distinct binary sim_proposed_batch, leaving sim_proposed untouched.
# Object files are built into a separate batch_obj/ directory so normal and
# batch builds never share objects.
# ---------------------------------------------------------------------------
BATCH_TARGET = sim_proposed_batch
BATCH_OBJDIR = batch_obj
BATCH_OBJS   = $(addprefix $(BATCH_OBJDIR)/,$(SEARCH_SRCS:.cpp=.o))

ifeq ($(BATCH_NN),1)
  BATCH_CXXFLAGS = $(CXXFLAGS) -DENABLE_BATCH_NN=1
else
  BATCH_CXXFLAGS = $(CXXFLAGS)
endif

$(BATCH_TARGET):
	@mkdir -p $(BATCH_OBJDIR)
	$(MAKE) BATCH_NN=1 USE_BLAS=1 DIM_K_SELECT=10 PROFILING=$(PROFILING) DUMP_W=$(DUMP_W) $(BATCH_OBJS)
	$(MAKE) USE_BLAS=1 DIM_K_SELECT=10 _link_$(BATCH_TARGET)

# Internal helper: runs the final link with USE_BLAS=1 visible so that
# $(LDFLAGS) expands to include $(BLAS_LDFLAGS). Called by the parent recipe
# above via $(MAKE) so the link step doesn't depend on the user passing
# USE_BLAS=1 on the outer command line.
.PHONY: _link_$(BATCH_TARGET)
_link_$(BATCH_TARGET):
	$(CXX) $(CXXFLAGS) -DENABLE_BATCH_NN=1 -o $(BATCH_TARGET) $(BATCH_OBJS) $(LDFLAGS)

# Pattern rule for batch object files
$(BATCH_OBJDIR)/%.o: %.cpp
	@mkdir -p $(BATCH_OBJDIR)
	$(CXX) $(BATCH_CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# ---------------------------------------------------------------------------
# OpenMP-enabled batch-NN proposed policy search (sim_proposed_batch_omp).
# CC10 only.  Combines BATCH_NN=1 + USE_BLAS=1 + USE_OMP=1: each OMP thread
# runs one (lambda_tilde, beta_rate_1, beta_rate_2) combo end-to-end via the
# batched cblas_sgemm/sgemv path.
#
# Set MKL_NUM_THREADS=1 (and OMP_NUM_THREADS=$$SLURM_CPUS_PER_TASK) in the
# submit script so MKL does NOT spawn nested threads on top of OMP.
#
# To build: make sim_proposed_batch_omp
# Object files go to batch_omp_obj/ so other builds are unaffected.
# ---------------------------------------------------------------------------
BATCH_OMP_TARGET = sim_proposed_batch_omp
BATCH_OMP_OBJDIR = batch_omp_obj
BATCH_OMP_OBJS   = $(addprefix $(BATCH_OMP_OBJDIR)/,$(SEARCH_SRCS:.cpp=.o))

$(BATCH_OMP_TARGET):
	@mkdir -p $(BATCH_OMP_OBJDIR)
	$(MAKE) BATCH_NN=1 USE_BLAS=1 USE_OMP=1 DIM_K_SELECT=10 \
	        PROFILING=$(PROFILING) DUMP_W=$(DUMP_W) $(BATCH_OMP_OBJS)
	$(MAKE) USE_BLAS=1 USE_OMP=1 DIM_K_SELECT=10 _link_$(BATCH_OMP_TARGET)

# Internal link helper (see _link_$(BATCH_TARGET) for rationale).
.PHONY: _link_$(BATCH_OMP_TARGET)
_link_$(BATCH_OMP_TARGET):
	$(CXX) $(CXXFLAGS) -fopenmp -DENABLE_BATCH_NN=1 -DUSE_OMP=1 \
	    -o $(BATCH_OMP_TARGET) $(BATCH_OMP_OBJS) $(LDFLAGS) -fopenmp

# Pattern rule for batch+OMP object files (BATCH_NN=1, USE_BLAS=1, USE_OMP=1)
$(BATCH_OMP_OBJDIR)/%.o: %.cpp
	@mkdir -p $(BATCH_OMP_OBJDIR)
	$(CXX) $(BATCH_CXXFLAGS) -fopenmp -DUSE_OMP=1 $(CPPFLAGS) -c $< -o $@

# ---------------------------------------------------------------------------
# Benchmark policy search binary (benchmark_search_main.cpp)
# Uses nn_model_stub.o (no TensorFlow) instead of nn_model.o.
# Benchmarks never call nn_predict(), so no TF link is needed.
# ---------------------------------------------------------------------------

BMK_SRCS   = benchmark_search_main.cpp simulation.cpp policy.cpp nn_model_stub.cpp run_config.cpp benchmark_kinds.cpp
BMK_OBJS   = $(BMK_SRCS:.cpp=.o)
BMK_TARGET = sim_benchmark

$(BMK_TARGET): $(BMK_OBJS)
	$(CXX) $(CXXFLAGS) -o $(BMK_TARGET) $(BMK_OBJS)

# ---------------------------------------------------------------------------
# OpenMP-enabled benchmark search (sim_benchmark_omp).
# Parallelizes combo-level loops in scheduling search and pricing search.
# Object files go to omp_obj/ so the non-OMP sim_benchmark stays intact.
# ---------------------------------------------------------------------------
OMP_BMK_TARGET = sim_benchmark_omp
OMP_OBJDIR     = omp_obj
OMP_BMK_OBJS   = $(addprefix $(OMP_OBJDIR)/,$(BMK_SRCS:.cpp=.o))

$(OMP_BMK_TARGET):
	@mkdir -p $(OMP_OBJDIR)
	$(MAKE) USE_OMP=1 DIM_K_SELECT=$(DIM_K_SELECT) $(OMP_BMK_OBJS)
	$(CXX) $(CXXFLAGS) -fopenmp -DUSE_OMP=1 -o $(OMP_BMK_TARGET) $(OMP_BMK_OBJS) -fopenmp

# Pattern rule for OMP object files (compiled with -fopenmp and USE_OMP=1)
$(OMP_OBJDIR)/%.o: %.cpp
	@mkdir -p $(OMP_OBJDIR)
	$(CXX) $(CXXFLAGS) -fopenmp -DUSE_OMP=1 $(CPPFLAGS) -c $< -o $@

# ---------------------------------------------------------------------------
# Denser simulation binary (denser_sim_main.cpp)
# One combo per invocation; designed for SLURM array jobs.
# Usage: ./sim_denser <safety> <rho> <epsilon> <lambda> <beta_rate_1> <beta_rate_2>
# ---------------------------------------------------------------------------

DENSER_SRCS   = denser_sim_main.cpp simulation.cpp policy.cpp nn_model.cpp run_config.cpp model_path.cpp
DENSER_OBJS   = $(DENSER_SRCS:.cpp=.o)
DENSER_TARGET = sim_denser

$(DENSER_TARGET): $(DENSER_OBJS)
	$(CXX) $(CXXFLAGS) -o $(DENSER_TARGET) $(DENSER_OBJS) $(LDFLAGS)

# ---------------------------------------------------------------------------
# One-combo simulation binary (one_combo_main.cpp)
# Reads a single parameter set from a txt file (one section per policy),
# runs n_paths paths, and writes a delay-time histogram + summary CSV.
# Links nn_model.cpp + libtensorflow because [proposed] needs the NN.
# Usage: ./sim_one_combo INPUT.txt [POLICY]
# ---------------------------------------------------------------------------
ONECOMBO_SRCS   = one_combo_main.cpp simulation.cpp policy.cpp nn_model.cpp run_config.cpp model_path.cpp benchmark_kinds.cpp
ONECOMBO_OBJS   = $(ONECOMBO_SRCS:.cpp=.o)
ONECOMBO_TARGET = sim_one_combo

$(ONECOMBO_TARGET): $(ONECOMBO_OBJS)
	$(CXX) $(CXXFLAGS) -o $(ONECOMBO_TARGET) $(ONECOMBO_OBJS) $(LDFLAGS)

# ---------------------------------------------------------------------------
# Batch-NN one-combo binary (sim_one_combo_batch).
# Mirrors sim_proposed_batch: re-invokes make with ENABLE_BATCH_NN=1,
# DIM_K_SELECT=10, USE_BLAS=1; objects go to onecombo_batch_obj/ to keep
# the normal sim_one_combo build untouched. Recommended for CC10 [proposed].
# ---------------------------------------------------------------------------
ONECOMBO_BATCH_TARGET = sim_one_combo_batch
ONECOMBO_BATCH_OBJDIR = onecombo_batch_obj
ONECOMBO_BATCH_OBJS   = $(addprefix $(ONECOMBO_BATCH_OBJDIR)/,$(ONECOMBO_SRCS:.cpp=.o))

$(ONECOMBO_BATCH_TARGET):
	@mkdir -p $(ONECOMBO_BATCH_OBJDIR)
	$(MAKE) BATCH_NN=1 USE_BLAS=1 DIM_K_SELECT=10 PROFILING=$(PROFILING) DUMP_W=$(DUMP_W) $(ONECOMBO_BATCH_OBJS)
	$(MAKE) USE_BLAS=1 DIM_K_SELECT=10 _link_$(ONECOMBO_BATCH_TARGET)

# Internal link helper (see _link_$(BATCH_TARGET) for rationale).
.PHONY: _link_$(ONECOMBO_BATCH_TARGET)
_link_$(ONECOMBO_BATCH_TARGET):
	$(CXX) $(CXXFLAGS) -DENABLE_BATCH_NN=1 -o $(ONECOMBO_BATCH_TARGET) $(ONECOMBO_BATCH_OBJS) $(LDFLAGS)

# Pattern rule for batch one-combo object files (compiled with ENABLE_BATCH_NN=1)
$(ONECOMBO_BATCH_OBJDIR)/%.o: %.cpp
	@mkdir -p $(ONECOMBO_BATCH_OBJDIR)
	$(CXX) $(BATCH_CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# ---------------------------------------------------------------------------
# Compilation rule (.cpp → .o)
# ---------------------------------------------------------------------------
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Header dependencies: if a header changes, recompile affected .cpp files
main.o:                    main.cpp params.h simulation.h policy.h nn_model.h
proposed_policy_main.o:    proposed_policy_main.cpp params.h run_config.h simulation.h nn_model.h model_path.h
benchmark_search_main.o:   benchmark_search_main.cpp params.h run_config.h simulation.h nn_model.h benchmark_kinds.h
nn_model_stub.o:           nn_model_stub.cpp nn_model.h params.h
benchmark_kinds.o:         benchmark_kinds.cpp benchmark_kinds.h params.h run_config.h
model_path.o:              model_path.cpp model_path.h params.h
denser_sim_main.o:         denser_sim_main.cpp params.h run_config.h simulation.h nn_model.h model_path.h
one_combo_main.o:          one_combo_main.cpp params.h run_config.h simulation.h nn_model.h model_path.h benchmark_kinds.h
simulation.o:              simulation.cpp simulation.h params.h policy.h run_config.h
policy.o:                  policy.cpp policy.h params.h nn_model.h run_config.h
nn_model.o:                nn_model.cpp nn_model.h params.h
run_config.o:              run_config.cpp run_config.h params.h

# ---------------------------------------------------------------------------
# Convenience targets
# ---------------------------------------------------------------------------

run: $(TARGET)
	LD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib DYLD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib ./$(TARGET)

run_proposed: $(SEARCH_TARGET)
	LD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib DYLD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib ./$(SEARCH_TARGET)

run_benchmark: $(BMK_TARGET)
	./$(BMK_TARGET)

clean:
	rm -f $(OBJS) $(SEARCH_OBJS) $(BMK_OBJS) $(DENSER_OBJS) $(ONECOMBO_OBJS) \
	      $(TARGET) $(SEARCH_TARGET) $(BMK_TARGET) $(DENSER_TARGET) \
	      $(BATCH_TARGET) $(OMP_BMK_TARGET) $(BATCH_OMP_TARGET) \
	      $(ONECOMBO_TARGET) $(ONECOMBO_BATCH_TARGET)
	rm -rf $(BATCH_OBJDIR) $(OMP_OBJDIR) $(BATCH_OMP_OBJDIR) $(ONECOMBO_BATCH_OBJDIR)

.PHONY: run run_proposed run_benchmark clean
