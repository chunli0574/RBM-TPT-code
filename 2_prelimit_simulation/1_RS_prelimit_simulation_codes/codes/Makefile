# Makefile for the W-network queueing simulation
#
# Usage:
#   make                → compile single-run binary ./sim
#   make sim_proposed   → compile proposed NN policy search binary ./sim_proposed
#   make sim_benchmark  → compile benchmark search binary ./sim_benchmark
#   make sim_denser     → compile denser single-combo simulation binary ./sim_denser
#   make clean          → remove compiled files
#   make run            → compile and run ./sim
#   make run_proposed   → compile and run ./sim_proposed
#   make run_benchmark  → compile and run ./sim_benchmark
#
# To run with the TensorFlow shared library (macOS):
#   DYLD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib ./sim
# Or set it permanently in your shell:
#   export DYLD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib
#
# Compiler flags:
#   -std=c++17    → C++17 standard (required by cppflow)
#   -O2           → standard optimization (fast but not too aggressive)
#   -Wall         → show all compiler warnings (helps catch bugs)
#   -g            → include debug info (lets you use gdb/lldb if needed)

CXX      = g++

# Include paths:
#   cppflow headers:      $(HOME)/libs/cppflow/include
#   libtensorflow headers:$(HOME)/libs/libtensorflow/include
CPPFLAGS = -I$(HOME)/libs/cppflow/include \
            -I$(HOME)/libs/libtensorflow/include

CXXFLAGS = -std=c++17 -O2 -Wall -g

# Profiling flags: enable 3-level TIC/TOC via `make PROFILING=1 <target>`.
# When PROFILING is unset, PROF_FLAGS is empty and all PROF_* macros expand to no-ops.
PROF_FLAGS = $(if $(PROFILING),-DENABLE_PROFILING=1,)

# Linker flags: link against libtensorflow (used by sim / sim_proposed / sim_denser)
LDFLAGS  = -L$(HOME)/libs/libtensorflow/lib -ltensorflow

# Benchmark binaries (sim_benchmark / sim_benchmark_omp) link against a
# no-op nn_model_stub.cpp instead of nn_model.cpp, so they do NOT need
# libtensorflow at build or run time.
BENCH_LDFLAGS =

# ---------------------------------------------------------------------------
# Single-run simulation (main.cpp)
# ---------------------------------------------------------------------------

SRCS   = main.cpp simulation.cpp policy.cpp nn_model.cpp run_config.cpp profiling.cpp
OBJS   = $(SRCS:.cpp=.o)
TARGET = sim

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# ---------------------------------------------------------------------------
# Proposed NN policy search binary (proposed_policy_main.cpp)
# ---------------------------------------------------------------------------

SEARCH_SRCS   = proposed_policy_main.cpp simulation.cpp policy.cpp nn_model.cpp run_config.cpp profiling.cpp
SEARCH_OBJS   = $(SEARCH_SRCS:.cpp=.o)
SEARCH_TARGET = sim_proposed

$(SEARCH_TARGET): $(SEARCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $(SEARCH_TARGET) $(SEARCH_OBJS) $(LDFLAGS)

# ---------------------------------------------------------------------------
# Compilation rule (.cpp → .o)
# ---------------------------------------------------------------------------
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(PROF_FLAGS) $(CPPFLAGS) -c $< -o $@

# ---------------------------------------------------------------------------
# Benchmark search binary (benchmark_search_main.cpp)
# ---------------------------------------------------------------------------

BENCH_SRCS   = benchmark_search_main.cpp simulation.cpp policy.cpp nn_model_stub.cpp run_config.cpp profiling.cpp benchmark_kinds.cpp
BENCH_OBJS   = $(BENCH_SRCS:.cpp=.o)
BENCH_TARGET = sim_benchmark

$(BENCH_TARGET): $(BENCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $(BENCH_TARGET) $(BENCH_OBJS) $(BENCH_LDFLAGS)

# ---------------------------------------------------------------------------
# Benchmark search binary with OpenMP combo-level parallelization.
# Built with -fopenmp -DUSE_OMP=1 from the same sources; object files go
# into omp_obj/ to keep the serial sim_benchmark build untouched.
# ---------------------------------------------------------------------------

OMP_FLAGS    = -fopenmp -DUSE_OMP=1
OMP_OBJ_DIR  = omp_obj
OMP_BMK_OBJS = $(addprefix $(OMP_OBJ_DIR)/, $(BENCH_SRCS:.cpp=.o))
OMP_BMK_TARGET = sim_benchmark_omp

$(OMP_OBJ_DIR):
	mkdir -p $(OMP_OBJ_DIR)

$(OMP_OBJ_DIR)/%.o: %.cpp | $(OMP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(PROF_FLAGS) $(CPPFLAGS) -c $< -o $@

$(OMP_BMK_TARGET): $(OMP_BMK_OBJS)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -o $(OMP_BMK_TARGET) $(OMP_BMK_OBJS) $(BENCH_LDFLAGS)

# ---------------------------------------------------------------------------
# Denser single-combo simulation binary (denser_sim_main.cpp)
# ---------------------------------------------------------------------------

DENSER_SRCS   = denser_sim_main.cpp simulation.cpp policy.cpp nn_model.cpp run_config.cpp profiling.cpp
DENSER_OBJS   = $(DENSER_SRCS:.cpp=.o)
DENSER_TARGET = sim_denser

$(DENSER_TARGET): $(DENSER_OBJS)
	$(CXX) $(CXXFLAGS) -o $(DENSER_TARGET) $(DENSER_OBJS) $(LDFLAGS)

# ---------------------------------------------------------------------------
# One-combo simulation binary (one_combo_main.cpp)
# Records a delay-time histogram for ONE parameter set across all 13 supported
# policies (proposed + A1..D2). Links real nn_model.cpp because [proposed]
# loads a NN at runtime.
# ---------------------------------------------------------------------------

ONECOMBO_SRCS   = one_combo_main.cpp simulation.cpp policy.cpp nn_model.cpp run_config.cpp profiling.cpp benchmark_kinds.cpp
ONECOMBO_OBJS   = $(ONECOMBO_SRCS:.cpp=.o)
ONECOMBO_TARGET = sim_one_combo

$(ONECOMBO_TARGET): $(ONECOMBO_OBJS)
	$(CXX) $(CXXFLAGS) -o $(ONECOMBO_TARGET) $(ONECOMBO_OBJS) $(LDFLAGS)

# Header dependencies: if a header changes, recompile affected .cpp files
main.o:                   main.cpp params.h simulation.h policy.h nn_model.h
proposed_policy_main.o:   proposed_policy_main.cpp params.h run_config.h simulation.h nn_model.h nn_model_paths.h
benchmark_search_main.o:  benchmark_search_main.cpp params.h run_config.h simulation.h nn_model.h profiling.h benchmark_kinds.h
benchmark_kinds.o:        benchmark_kinds.cpp benchmark_kinds.h params.h run_config.h
one_combo_main.o:         one_combo_main.cpp params.h run_config.h simulation.h nn_model.h benchmark_kinds.h nn_model_paths.h
denser_sim_main.o:        denser_sim_main.cpp params.h run_config.h simulation.h nn_model.h nn_model_paths.h
simulation.o:             simulation.cpp simulation.h params.h policy.h nn_model.h run_config.h profiling.h
policy.o:                 policy.cpp policy.h params.h nn_model.h run_config.h profiling.h
nn_model.o:               nn_model.cpp nn_model.h
nn_model_stub.o:          nn_model_stub.cpp nn_model.h
run_config.o:             run_config.cpp run_config.h params.h
profiling.o:              profiling.cpp profiling.h

# ---------------------------------------------------------------------------
# Convenience targets
# ---------------------------------------------------------------------------

# Compile and run single-run simulation
run: $(TARGET)
	DYLD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib ./$(TARGET)

# Compile and run proposed policy search
run_proposed: $(SEARCH_TARGET)
	DYLD_LIBRARY_PATH=$(HOME)/libs/libtensorflow/lib ./$(SEARCH_TARGET)

# Compile and run benchmark search (no libtensorflow needed — stub is linked in)
run_benchmark: $(BENCH_TARGET)
	./$(BENCH_TARGET)

# Remove all compiled files
clean:
	rm -f $(OBJS) $(SEARCH_OBJS) $(BENCH_OBJS) $(DENSER_OBJS) $(ONECOMBO_OBJS) \
	      $(TARGET) $(SEARCH_TARGET) $(BENCH_TARGET) $(DENSER_TARGET) \
	      $(OMP_BMK_TARGET) $(ONECOMBO_TARGET)
	rm -rf $(OMP_OBJ_DIR)

.PHONY: run run_proposed run_benchmark clean
