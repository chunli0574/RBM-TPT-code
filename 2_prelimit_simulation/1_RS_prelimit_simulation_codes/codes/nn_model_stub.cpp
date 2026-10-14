/*
 * nn_model_stub.cpp
 *
 * No-op stand-in for nn_model.cpp, used by the benchmark binaries
 * (sim_benchmark, sim_benchmark_omp) which never evaluate the NN.
 *
 * Why this exists:
 *   simulation.cpp and policy.cpp include "nn_model.h" and can call
 *   nn_predict / nn_model_load / etc. — but the benchmark policies
 *   (BCP, High-Low, WMP/LQ scheduling, threshold outsourcing, hardstop)
 *   never exercise those code paths at runtime. Linking against the real
 *   nn_model.cpp would needlessly pull in libtensorflow / cppflow.
 *
 *   By linking against this stub instead, the benchmark binaries build
 *   and run with NO libtensorflow dependency. If a benchmark execution
 *   path ever reaches nn_predict by mistake, it aborts loudly so the
 *   bug is obvious.
 *
 * Signatures must stay bit-identical to nn_model.h.
 */

#include "nn_model.h"
#include <cstdio>
#include <cstdlib>

void nn_model_load(const std::string& /*model_path*/)
{
    // no-op: benchmark builds never need to load a NN.
}

void nn_model_reload(const std::string& /*model_path*/)
{
    // no-op: benchmark builds never swap NN models.
}

bool nn_model_path_exists(const std::string& /*model_path*/)
{
    // benchmark builds have no NN on disk to check.
    return false;
}

void nn_predict(double /*W0*/, double /*W1*/, double /*out*/[2])
{
    std::fprintf(stderr,
        "[nn_model_stub] nn_predict() called in a NN-less benchmark build.\n"
        "  This indicates a bug: a benchmark policy reached an NN code path.\n");
    std::abort();
}

void nn_cache_stats()
{
    // no-op: nothing cached in a stub build.
}

void nn_cache_clear()
{
    // no-op: nothing cached in a stub build.
}
