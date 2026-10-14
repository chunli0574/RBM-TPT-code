/*
 * nn_model.cpp
 *
 * Dual-backend NN inference wrapper:
 *
 *   Backend A — cppflow / TF SavedModel  (old format, path = directory "_grad")
 *   Backend B — native C++ forward pass  (new format, path = file "_grad.weights.bin")
 *
 * Only one backend is active at a time per model slot.
 * The inference cache is backend-agnostic: same quantised key→value logic.
 *
 * New-format weights binary layout (written by export_keras_weights.py):
 *   float64[NN_DIM × NN_HIDDEN]      W1  (input→hidden)
 *   float64[NN_HIDDEN × NN_HIDDEN]   W2  (hidden→hidden)
 *   float64[NN_HIDDEN × NN_HIDDEN]   W3  (hidden→hidden)
 *   float64[NN_HIDDEN × NN_DIM]      W4  (hidden→output)
 * Row-major, Keras kernel convention: output = input @ W  (no transpose).
 * Activations: ELU between hidden layers, linear on output.
 */

#include "nn_model.h"
#include <cppflow/model.h>
#include <cppflow/tensor.h>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <array>
#include <cmath>     // std::expm1
#include <cstring>   // std::memcpy
#include <cstddef>   // size_t
#include <cstdint>   // uint64_t
#include <cstdio>    // printf, fopen
#include <sys/stat.h>  // stat(), S_ISDIR, S_ISREG

// ===========================================================================
// Native backend: NativeNN struct + ELU forward pass
// ===========================================================================

static constexpr int NN_DIM    = 2;    // RS workload dimension (W0, W1)
static constexpr int NN_HIDDEN = 100;  // hidden layer width

// Weights stored in Keras kernel convention, row-major:
//   output = input @ W  →  W[i][j] = W_flat[i*cols + j]
struct NativeNN {
    std::vector<double> W1;  // (NN_DIM,    NN_HIDDEN)
    std::vector<double> W2;  // (NN_HIDDEN, NN_HIDDEN)
    std::vector<double> W3;  // (NN_HIDDEN, NN_HIDDEN)
    std::vector<double> W4;  // (NN_HIDDEN, NN_DIM)
};

static inline double elu(double x) noexcept {
    return x > 0.0 ? x : std::expm1(x);  // expm1(x) = exp(x)-1, accurate near 0
}

// Forward pass: out = W4 @ ELU(W3 @ ELU(W2 @ ELU(W1 @ in)))
static void native_forward(const NativeNN& net,
                            const double in[NN_DIM],
                            double out[NN_DIM]) noexcept {
    double h1[NN_HIDDEN], h2[NN_HIDDEN], h3[NN_HIDDEN];

    // h1 = ELU(in @ W1),  W1 shape (NN_DIM, NN_HIDDEN)
    for (int j = 0; j < NN_HIDDEN; ++j) {
        double s = 0.0;
        for (int i = 0; i < NN_DIM; ++i) s += in[i] * net.W1[i * NN_HIDDEN + j];
        h1[j] = elu(s);
    }
    // h2 = ELU(h1 @ W2),  W2 shape (NN_HIDDEN, NN_HIDDEN)
    for (int j = 0; j < NN_HIDDEN; ++j) {
        double s = 0.0;
        for (int i = 0; i < NN_HIDDEN; ++i) s += h1[i] * net.W2[i * NN_HIDDEN + j];
        h2[j] = elu(s);
    }
    // h3 = ELU(h2 @ W3),  W3 shape (NN_HIDDEN, NN_HIDDEN)
    for (int j = 0; j < NN_HIDDEN; ++j) {
        double s = 0.0;
        for (int i = 0; i < NN_HIDDEN; ++i) s += h2[i] * net.W3[i * NN_HIDDEN + j];
        h3[j] = elu(s);
    }
    // out = h3 @ W4,  W4 shape (NN_HIDDEN, NN_DIM)
    for (int j = 0; j < NN_DIM; ++j) {
        double s = 0.0;
        for (int i = 0; i < NN_HIDDEN; ++i) s += h3[i] * net.W4[i * NN_DIM + j];
        out[j] = s;
    }
}

// Detect new-format path: ends with ".weights.bin"
static bool is_native_path(const std::string& path) {
    static const std::string suffix = ".weights.bin";
    return path.size() >= suffix.size() &&
           path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Load native weights from binary file
static NativeNN* load_native_weights(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open weights file: %s\n", path.c_str());
        exit(1);
    }
    NativeNN* nn = new NativeNN();
    nn->W1.resize(NN_DIM    * NN_HIDDEN);
    nn->W2.resize(NN_HIDDEN * NN_HIDDEN);
    nn->W3.resize(NN_HIDDEN * NN_HIDDEN);
    nn->W4.resize(NN_HIDDEN * NN_DIM);

    size_t r = 0;
    r += fread(nn->W1.data(), sizeof(double), (size_t)NN_DIM    * NN_HIDDEN, f);
    r += fread(nn->W2.data(), sizeof(double), (size_t)NN_HIDDEN * NN_HIDDEN, f);
    r += fread(nn->W3.data(), sizeof(double), (size_t)NN_HIDDEN * NN_HIDDEN, f);
    r += fread(nn->W4.data(), sizeof(double), (size_t)NN_HIDDEN * NN_DIM,    f);
    fclose(f);

    size_t expected = (size_t)(NN_DIM * NN_HIDDEN + NN_HIDDEN * NN_HIDDEN +
                               NN_HIDDEN * NN_HIDDEN + NN_HIDDEN * NN_DIM);
    if (r != expected) {
        fprintf(stderr,
            "ERROR: weights file truncated: %s\n"
            "       read %zu doubles, expected %zu\n",
            path.c_str(), r, expected);
        exit(1);
    }
    return nn;
}

// ===========================================================================
// File-scope model state
// ===========================================================================

static cppflow::model* g_nn_model  = nullptr;   // cppflow backend
static NativeNN*       g_nn_native = nullptr;   // native backend

// ===========================================================================
// Inference cache (backend-agnostic)
// ===========================================================================

static constexpr std::size_t MAX_CACHE_SIZE = 500000;

struct PairDoubleHash {
    std::size_t operator()(const std::pair<double,double>& p) const noexcept {
        std::uint64_t a, b;
        std::memcpy(&a, &p.first,  sizeof(a));
        std::memcpy(&b, &p.second, sizeof(b));
        std::size_t h = a;
        h ^= b + std::size_t{0x9e3779b97f4a7c15ULL} + (h << 6) + (h >> 2);
        return h;
    }
};

using NNCache = std::unordered_map<
    std::pair<double,double>,
    std::array<double,2>,
    PairDoubleHash
>;

static NNCache g_nn_cache;

// ===========================================================================
// Internal backend dispatch helpers
// ===========================================================================

static void run_outsource_backend(double W0, double W1, double out[2]) {
    if (g_nn_native) {
        const double in[2] = {W0, W1};
        native_forward(*g_nn_native, in, out);
    } else {
        std::vector<double> input_data = {W0, W1};
        cppflow::tensor input_tensor(input_data, {1, 2});
        cppflow::tensor output_tensor = (*g_nn_model)(
            {{"serving_default_input_1", input_tensor}},
            {"StatefulPartitionedCall"})[0];
        auto output_data = output_tensor.get_data<double>();
        out[0] = output_data[0];
        out[1] = output_data[1];
    }
}

// ===========================================================================
// Public API: outsourcing model
// ===========================================================================

void nn_model_load(const std::string& model_path) {
    if (is_native_path(model_path)) {
        g_nn_native = load_native_weights(model_path);
    } else {
        g_nn_model = new cppflow::model(model_path);
    }
    g_nn_cache.reserve(1 << 17);
}

void nn_model_reload(const std::string& model_path) {
    delete g_nn_model;   g_nn_model  = nullptr;
    delete g_nn_native;  g_nn_native = nullptr;
    g_nn_cache.clear();
    g_nn_cache.reserve(1 << 17);
    if (is_native_path(model_path)) {
        g_nn_native = load_native_weights(model_path);
    } else {
        g_nn_model = new cppflow::model(model_path);
    }
}

void nn_predict(double W0, double W1, double out[2]) {
    if (!g_nn_model && !g_nn_native)
        throw std::runtime_error("nn_predict: call nn_model_load() first");

    const auto key = std::make_pair(W0, W1);
    const auto it  = g_nn_cache.find(key);
    if (it != g_nn_cache.end()) {
        out[0] = it->second[0];
        out[1] = it->second[1];
        return;
    }

    run_outsource_backend(W0, W1, out);

    if (g_nn_cache.size() < MAX_CACHE_SIZE)
        g_nn_cache.emplace(key, std::array<double,2>{out[0], out[1]});
}

void nn_cache_clear() {
    g_nn_cache.clear();
    g_nn_cache.reserve(1 << 17);
}

// ===========================================================================
// Utility
// ===========================================================================

bool nn_model_path_exists(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    // Accept both directories (old SavedModel) and regular files (.weights.bin)
    return S_ISDIR(st.st_mode) || S_ISREG(st.st_mode);
}

void nn_cache_stats() {
    printf("  NN cache [%s]: %zu unique (W0,W1) points\n",
           g_nn_native ? "native" : "cppflow",
           g_nn_cache.size());
}
