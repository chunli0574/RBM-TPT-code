/*
 * nn_model_stub.cpp
 *
 * No-op stub implementations of nn_model.h interface.
 * Used in benchmark binaries (sim_benchmark) that never call nn_predict().
 * Avoids linking against libtensorflow for the benchmark build.
 *
 * Functions are no-ops; nn_predict returns zeros (should never be called
 * by any benchmark policy — benchmarks use BCPlinear / HighLow pricing
 * and SinglePushing / ThresholdBased outsourcing, which don't use NN).
 */

#include "nn_model.h"
#include <cstring>
#include <cstdio>

void nn_model_load(const std::string& /*model_path*/) {
    // No-op for benchmark builds: NN is not used by any benchmark policy.
}

void nn_predict(const double /*W*/[DIM_K], double out[DIM_K]) {
    // Should never be called by benchmark policies.
    // Return zeros as a safe fallback.
    for (int i = 0; i < DIM_K; ++i) out[i] = 0.0;
}

void nn_cache_stats() {
    printf("[nn_model_stub] cache stats: N/A (stub build)\n");
}

void nn_cache_clear() {
    // No-op.
}

void nn_model_reload(const std::string& /*model_path*/) {
    // No-op.
}

bool nn_model_path_exists(const std::string& /*model_path*/) {
    // Stub: always report exists (benchmark build never checks model files).
    return true;
}
