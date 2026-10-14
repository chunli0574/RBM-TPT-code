/*
 * nn_model.cpp
 *
 * Dual-backend NN inference wrapper:
 *
 *   Backend A — cppflow / TF SavedModel  (old format, path = directory "_grad")
 *   Backend B — native C++ forward pass  (new format, path = file "_grad.weights.bin")
 *
 * Only one backend is active at a time.
 * The inference cache is backend-agnostic: same quantised key→value logic.
 *
 * Single model slot:
 *   outsourcing slot  — always loaded;  used by nn_predict()
 *                       (NN pricing uses the same slot)
 *
 * New-format weights binary layout (written by export_keras_weights.py):
 *   float64[DIM_K × NN_HIDDEN]    W1  (input→hidden)
 *   float64[NN_HIDDEN × NN_HIDDEN] W2  (hidden→hidden)
 *   float64[NN_HIDDEN × NN_HIDDEN] W3  (hidden→hidden)
 *   float64[NN_HIDDEN × DIM_K]    W4  (hidden→output)
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
#include <cmath>     // std::expm1, std::round
#include <cstring>   // std::memcpy
#include <cstddef>   // size_t
#include <cstdint>   // uint64_t
#include <cstdio>    // printf, fopen
#include <string>
#include <sys/stat.h>  // stat()

#ifdef USE_BLAS
#include <mkl_cblas.h>
#endif

// ===========================================================================
// Optional W-distribution counter (enable with -DENABLE_W_DUMP=1).
// Accumulates a quantized counter over every NN-input W across the entire
// run, then dumps to a binary file on program exit.
//
// File format (little-endian, packed):
//   int32 dim_k
//   int32 n_unique
//   then n_unique records, each:
//       int16 key[dim_k]    (quantized W, key[i] = round(W[i] / grid_step))
//       uint32 count
//
// Default dump path: ./W_counter.bin    (override with env var W_DUMP_PATH)
// Default grid step: 0.005              (override with env var W_DUMP_GRID, e.g. "0.01")
// ===========================================================================
#ifdef ENABLE_W_DUMP
#include <cstdlib>  // getenv, atof, atexit

struct ArrayInt16Hash {
    std::size_t operator()(const std::array<int16_t, DIM_K>& a) const noexcept {
        std::size_t h = 0;
        for (int i = 0; i < DIM_K; ++i) {
            std::uint64_t bits = (std::uint64_t)(std::uint16_t)a[i];
            h ^= bits + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

using WCounter = std::unordered_map<
    std::array<int16_t, DIM_K>,
    std::uint32_t,
    ArrayInt16Hash
>;

static WCounter g_W_counter;
static double   g_W_dump_grid_step = 0.005;
static std::string g_W_dump_path   = "W_counter.bin";
static std::uint64_t g_W_dump_total_bumps = 0;

static void W_counter_dump_now();

static void W_counter_init_once() {
    static bool done = false;
    if (done) return;
    done = true;

    if (const char* s = std::getenv("W_DUMP_GRID")) {
        double v = std::atof(s);
        if (v > 0.0) g_W_dump_grid_step = v;
    }
    if (const char* s = std::getenv("W_DUMP_PATH")) {
        g_W_dump_path = s;
    }
    // Reserve to reduce rehashing overhead during simulation
    g_W_counter.reserve(1 << 20);  // 1M buckets initial
    std::atexit(W_counter_dump_now);
}

static inline void W_counter_bump(const double W[DIM_K]) {
    if (g_W_dump_total_bumps == 0) W_counter_init_once();
    std::array<int16_t, DIM_K> key;
    const double inv = 1.0 / g_W_dump_grid_step;
    for (int i = 0; i < DIM_K; ++i) {
        long q = std::lround(W[i] * inv);
        if (q >  32767) q =  32767;
        if (q < -32768) q = -32768;
        key[i] = (int16_t)q;
    }
    ++g_W_counter[key];
    ++g_W_dump_total_bumps;
}

static void W_counter_dump_now() {
    if (g_W_counter.empty()) return;
    FILE* f = std::fopen(g_W_dump_path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "W_counter_dump: cannot open %s\n", g_W_dump_path.c_str());
        return;
    }
    std::int32_t dim_k = (std::int32_t)DIM_K;
    std::int32_t n     = (std::int32_t)g_W_counter.size();
    std::fwrite(&dim_k, sizeof(dim_k), 1, f);
    std::fwrite(&n,     sizeof(n),     1, f);
    for (const auto& kv : g_W_counter) {
        std::fwrite(kv.first.data(), sizeof(int16_t), DIM_K, f);
        std::fwrite(&kv.second,      sizeof(uint32_t), 1,    f);
    }
    std::fclose(f);
    std::printf("W_counter_dump: %llu total bumps, %d unique keys, grid=%.6f, "
                "wrote to %s (%.2f MB)\n",
                (unsigned long long)g_W_dump_total_bumps, n,
                g_W_dump_grid_step, g_W_dump_path.c_str(),
                (double)(8 + (size_t)n * (sizeof(int16_t)*DIM_K + sizeof(uint32_t))) / 1e6);
}

#define W_COUNTER_BUMP(W)  W_counter_bump(W)
#else
#define W_COUNTER_BUMP(W)  do {} while(0)
#endif

// ===========================================================================
// Native backend: NativeNN struct + ELU forward pass
// ===========================================================================

// Hidden layer width and depth differ between CC3 and CC10.
//   CC3:  100_100_100_D3 → 3 hidden layers (width 100) + 1 output = NN_DEPTH=4
//   CC10: 200_200_200_D4 → 4 hidden layers (width 200) + 1 output = NN_DEPTH=5
#if DIM_K_SELECT == 3
static constexpr int NN_HIDDEN = 100;
static constexpr int NN_DEPTH  = 4;   // total Dense layers
#elif DIM_K_SELECT == 10
static constexpr int NN_HIDDEN = 200;
static constexpr int NN_DEPTH  = 5;
#endif

// Weights stored in Keras kernel convention, row-major:
//   output = input @ W  →  W[i][j] = W_flat[i*cols + j]
// Layer 0:           (DIM_K,     NN_HIDDEN)
// Layers 1..DEPTH-2: (NN_HIDDEN, NN_HIDDEN)
// Layer DEPTH-1:     (NN_HIDDEN, DIM_K)
//
// Weights are stored as float (single precision) for faster inference.
// The .weights.bin file is still float64; conversion happens at load time.
struct NativeNN {
    std::vector<float> weights[NN_DEPTH];
};

static inline float elu_f(float x) noexcept {
    return x > 0.0f ? x : std::expm1f(x);
}

// Forward pass: out = W_last @ ELU(... @ ELU(W2 @ ELU(W1 @ in)))
// Uses float internally for speed; input/output are double (API contract).
//
// Weights are in Keras convention (row-major): output = input @ W
//   out_j = sum_i input_i * W[i*out_dim + j]
//   ⟺  cblas_sgemv(CblasTrans, in_dim, out_dim, 1.0, W, out_dim, input, 1, 0.0, output, 1)
static void native_forward(const NativeNN& net,
                            const double in[DIM_K],
                            double out[DIM_K]) noexcept {
    // Convert input to float
    float in_f[DIM_K];
    for (int i = 0; i < DIM_K; ++i) in_f[i] = static_cast<float>(in[i]);

    float buf_a[NN_HIDDEN], buf_b[NN_HIDDEN];
    float* h_cur  = buf_a;
    float* h_next = buf_b;

    // Layer 0: in(DIM_K) → h_cur(NN_HIDDEN), with ELU
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasTrans,
                DIM_K, NN_HIDDEN, 1.0f,
                net.weights[0].data(), NN_HIDDEN,
                in_f, 1, 0.0f, h_cur, 1);
#else
    for (int j = 0; j < NN_HIDDEN; ++j) {
        float s = 0.0f;
        for (int i = 0; i < DIM_K; ++i)
            s += in_f[i] * net.weights[0][i * NN_HIDDEN + j];
        h_cur[j] = s;
    }
#endif
    for (int j = 0; j < NN_HIDDEN; ++j) h_cur[j] = elu_f(h_cur[j]);

    // Hidden layers 1..NN_DEPTH-2: h(NN_HIDDEN) → h(NN_HIDDEN), with ELU
    for (int layer = 1; layer < NN_DEPTH - 1; ++layer) {
#ifdef USE_BLAS
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    NN_HIDDEN, NN_HIDDEN, 1.0f,
                    net.weights[layer].data(), NN_HIDDEN,
                    h_cur, 1, 0.0f, h_next, 1);
#else
        for (int j = 0; j < NN_HIDDEN; ++j) {
            float s = 0.0f;
            for (int i = 0; i < NN_HIDDEN; ++i)
                s += h_cur[i] * net.weights[layer][i * NN_HIDDEN + j];
            h_next[j] = s;
        }
#endif
        for (int j = 0; j < NN_HIDDEN; ++j) h_next[j] = elu_f(h_next[j]);
        float* tmp = h_cur; h_cur = h_next; h_next = tmp;
    }

    // Output layer NN_DEPTH-1: h(NN_HIDDEN) → out(DIM_K), linear
    float out_f[DIM_K];
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasTrans,
                NN_HIDDEN, DIM_K, 1.0f,
                net.weights[NN_DEPTH - 1].data(), DIM_K,
                h_cur, 1, 0.0f, out_f, 1);
#else
    for (int j = 0; j < DIM_K; ++j) {
        float s = 0.0f;
        for (int i = 0; i < NN_HIDDEN; ++i)
            s += h_cur[i] * net.weights[NN_DEPTH - 1][i * DIM_K + j];
        out_f[j] = s;
    }
#endif
    // Convert output back to double
    for (int i = 0; i < DIM_K; ++i) out[i] = static_cast<double>(out_f[i]);
}

// Detect new-format path: ends with ".weights.bin"
static bool is_native_path(const std::string& path) {
    static const std::string suffix = ".weights.bin";
    return path.size() >= suffix.size() &&
           path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Load native weights from binary file (float64 on disk → float32 in memory)
static NativeNN* load_native_weights(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open weights file: %s\n", path.c_str());
        exit(1);
    }
    NativeNN* nn = new NativeNN();

    // Layer sizes (in number of elements)
    size_t layer_sizes[NN_DEPTH];
    layer_sizes[0] = (size_t)DIM_K * NN_HIDDEN;
    for (int l = 1; l < NN_DEPTH - 1; ++l)
        layer_sizes[l] = (size_t)NN_HIDDEN * NN_HIDDEN;
    layer_sizes[NN_DEPTH - 1] = (size_t)NN_HIDDEN * DIM_K;

    size_t total_read = 0;
    for (int l = 0; l < NN_DEPTH; ++l) {
        // Read as double from file, then convert to float
        std::vector<double> tmp(layer_sizes[l]);
        size_t r = fread(tmp.data(), sizeof(double), layer_sizes[l], f);
        total_read += r;
        nn->weights[l].resize(layer_sizes[l]);
        for (size_t i = 0; i < layer_sizes[l]; ++i)
            nn->weights[l][i] = static_cast<float>(tmp[i]);
    }
    fclose(f);

    size_t expected = 0;
    for (int l = 0; l < NN_DEPTH; ++l) expected += layer_sizes[l];
    if (total_read != expected) {
        fprintf(stderr,
            "ERROR: weights file truncated: %s\n"
            "       read %zu doubles, expected %zu\n",
            path.c_str(), total_read, expected);
        exit(1);
    }
    printf("  Native weights loaded: %d layers, NN_HIDDEN=%d, %zu params (float32)\n",
           NN_DEPTH, NN_HIDDEN, expected);
    return nn;
}

// ===========================================================================
// File-scope model state
// Single slot; at most one backend active.
// ===========================================================================

// -- Outsourcing slot (shared by pricing) --
static cppflow::model* g_nn_model  = nullptr;   // cppflow backend
static NativeNN*       g_nn_native = nullptr;   // native backend

// ===========================================================================
// Inference cache (backend-agnostic: same key/value for both backends)
// ===========================================================================

// CC3: cache is highly effective (3D state space, high hit rate, fine grid 0.005).
// CC10: cache disabled — tested grid_step=0.005 (0.38% hit), 0.1 (4.38% hit, adds error). Not worth it.
#if DIM_K_SELECT == 10
static constexpr std::size_t MAX_CACHE_SIZE = 0;
#else
static constexpr std::size_t MAX_CACHE_SIZE = 500000;
#endif

// Cache hit/miss counters
static uint64_t g_cache_hits   = 0;
static uint64_t g_cache_misses = 0;

struct ArrayDoubleHash {
    std::size_t operator()(const std::array<double, DIM_K>& a) const noexcept {
        std::size_t h = 0;
        for (int i = 0; i < DIM_K; i++) {
            std::uint64_t bits;
            std::memcpy(&bits, &a[i], sizeof(bits));
            h ^= bits + std::size_t{0x9e3779b97f4a7c15ULL} + (h << 6) + (h >> 2);
        }
        return h;
    }
};

using NNCache = std::unordered_map<
    std::array<double, DIM_K>,
    std::array<double, DIM_K>,
    ArrayDoubleHash
>;

static NNCache g_nn_cache;          // outsourcing cache (shared with pricing)

// Shared cache grid step for quantisation
static constexpr double CACHE_GRID_STEP = 0.005;

// ===========================================================================
// Internal helpers shared by predict functions
// ===========================================================================

// Compute quantised cache key
static std::array<double, DIM_K> make_cache_key(const double W[DIM_K]) noexcept {
    std::array<double, DIM_K> key;
    for (int i = 0; i < DIM_K; i++)
        key[i] = std::round(W[i] / CACHE_GRID_STEP) * CACHE_GRID_STEP;
    return key;
}

// Run backend inference (cppflow or native) for the single slot
static void run_outsource_backend(const double W[DIM_K], double out[DIM_K]) {
    if (g_nn_native) {
        native_forward(*g_nn_native, W, out);
    } else {
        std::vector<double> input_data(W, W + DIM_K);
        cppflow::tensor input_tensor(input_data, {1, DIM_K});
        cppflow::tensor output_tensor = (*g_nn_model)(
            {{"serving_default_input_1", input_tensor}},
            {"StatefulPartitionedCall"})[0];
        auto output_data = output_tensor.get_data<double>();
        for (int i = 0; i < DIM_K; i++) out[i] = output_data[i];
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
    // Clean up both backends for this slot
    delete g_nn_model;   g_nn_model  = nullptr;
    delete g_nn_native;  g_nn_native = nullptr;
    g_nn_cache.clear();
    g_nn_cache.reserve(1 << 17);
    // Load new backend
    if (is_native_path(model_path)) {
        g_nn_native = load_native_weights(model_path);
    } else {
        g_nn_model = new cppflow::model(model_path);
    }
}

void nn_predict(const double W[DIM_K], double out[DIM_K]) {
    if (!g_nn_model && !g_nn_native)
        throw std::runtime_error("nn_predict: call nn_model_load() first");

    W_COUNTER_BUMP(W);

    if (MAX_CACHE_SIZE > 0) {
        auto key = make_cache_key(W);
        auto it  = g_nn_cache.find(key);
        if (it != g_nn_cache.end()) {
            for (int i = 0; i < DIM_K; i++) out[i] = it->second[i];
            ++g_cache_hits;
            return;
        }

        ++g_cache_misses;
        run_outsource_backend(W, out);

        std::array<double, DIM_K> result;
        for (int i = 0; i < DIM_K; i++) result[i] = out[i];
        if (g_nn_cache.size() < MAX_CACHE_SIZE)
            g_nn_cache.emplace(key, result);
    } else {
        // Cache disabled
        run_outsource_backend(W, out);
    }
}

void nn_cache_clear() {
    g_nn_cache.clear();
    g_nn_cache.reserve(1 << 17);
    g_cache_hits = 0;
    g_cache_misses = 0;
}

// ===========================================================================
// Utility
// ===========================================================================

bool nn_model_path_exists(const std::string& model_path) {
    struct stat st;
    if (stat(model_path.c_str(), &st) != 0) return false;
    // Accept both directories (old SavedModel) and regular files (.weights.bin)
    return S_ISDIR(st.st_mode) || S_ISREG(st.st_mode);
}

// ===========================================================================
// Batch inference (CC10 + ENABLE_BATCH_NN only)
// ===========================================================================

#if DIM_K_SELECT == 10 && defined(ENABLE_BATCH_NN)

// Scratch buffers, resized on first call and reused thereafter.
// Made thread_local so OMP combo-level parallelism (sim_proposed_batch_omp)
// can call nn_predict_batch concurrently from multiple threads — each thread
// owns its own scratch. Single-threaded callers see no behavioral change.
// NOTE: W_COUNTER (if DUMP_W enabled) is NOT thread-safe; do not use DUMP_W
// together with OMP. NN weights (g_nn_native) are read-only and safe to share.
static thread_local std::vector<float> g_batch_in_f;     // [n * DIM_K]
static thread_local std::vector<float> g_batch_hA_f;     // [n * NN_HIDDEN]
static thread_local std::vector<float> g_batch_hB_f;     // [n * NN_HIDDEN]
static thread_local std::vector<float> g_batch_out_f;    // [n * DIM_K]

void nn_predict_batch(const double W_batch[], double G_batch[], int n) {
    if (!g_nn_native) {
        throw std::runtime_error(
            "nn_predict_batch: native backend required (cppflow backend not supported)");
    }
    if (n <= 0) return;

    // W distribution counter (bump once per path in this batch).
    for (int m = 0; m < n; ++m) {
        W_COUNTER_BUMP(&W_batch[m * DIM_K]);
    }

    const std::size_t n_in     = (std::size_t)n * DIM_K;
    const std::size_t n_hidden = (std::size_t)n * NN_HIDDEN;
    if (g_batch_in_f.size()  < n_in)     g_batch_in_f.resize(n_in);
    if (g_batch_hA_f.size()  < n_hidden) g_batch_hA_f.resize(n_hidden);
    if (g_batch_hB_f.size()  < n_hidden) g_batch_hB_f.resize(n_hidden);
    if (g_batch_out_f.size() < n_in)     g_batch_out_f.resize(n_in);

    // double → float
    for (std::size_t k = 0; k < n_in; ++k)
        g_batch_in_f[k] = static_cast<float>(W_batch[k]);

    const NativeNN& net = *g_nn_native;
    float* h_cur  = g_batch_hA_f.data();
    float* h_next = g_batch_hB_f.data();

    // Layer 0: in (n × DIM_K) @ W0 (DIM_K × NN_HIDDEN) -> h_cur (n × NN_HIDDEN)
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, NN_HIDDEN, DIM_K,
                1.0f,
                g_batch_in_f.data(), DIM_K,
                net.weights[0].data(), NN_HIDDEN,
                0.0f,
                h_cur, NN_HIDDEN);
#else
    // Fallback (shouldn't happen: batch build requires USE_BLAS, but keep safe)
    for (int m = 0; m < n; ++m) {
        for (int j = 0; j < NN_HIDDEN; ++j) {
            float s = 0.0f;
            for (int i = 0; i < DIM_K; ++i)
                s += g_batch_in_f[m * DIM_K + i] * net.weights[0][i * NN_HIDDEN + j];
            h_cur[m * NN_HIDDEN + j] = s;
        }
    }
#endif
    // ELU
    for (std::size_t k = 0; k < n_hidden; ++k)
        h_cur[k] = h_cur[k] > 0.0f ? h_cur[k] : std::expm1f(h_cur[k]);

    // Hidden layers 1..NN_DEPTH-2
    for (int layer = 1; layer < NN_DEPTH - 1; ++layer) {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, NN_HIDDEN, NN_HIDDEN,
                    1.0f,
                    h_cur, NN_HIDDEN,
                    net.weights[layer].data(), NN_HIDDEN,
                    0.0f,
                    h_next, NN_HIDDEN);
#else
        for (int m = 0; m < n; ++m) {
            for (int j = 0; j < NN_HIDDEN; ++j) {
                float s = 0.0f;
                for (int i = 0; i < NN_HIDDEN; ++i)
                    s += h_cur[m * NN_HIDDEN + i] * net.weights[layer][i * NN_HIDDEN + j];
                h_next[m * NN_HIDDEN + j] = s;
            }
        }
#endif
        for (std::size_t k = 0; k < n_hidden; ++k)
            h_next[k] = h_next[k] > 0.0f ? h_next[k] : std::expm1f(h_next[k]);
        std::swap(h_cur, h_next);
    }

    // Output layer: h_cur (n × NN_HIDDEN) @ W_out (NN_HIDDEN × DIM_K)
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, DIM_K, NN_HIDDEN,
                1.0f,
                h_cur, NN_HIDDEN,
                net.weights[NN_DEPTH - 1].data(), DIM_K,
                0.0f,
                g_batch_out_f.data(), DIM_K);
#else
    for (int m = 0; m < n; ++m) {
        for (int j = 0; j < DIM_K; ++j) {
            float s = 0.0f;
            for (int i = 0; i < NN_HIDDEN; ++i)
                s += h_cur[m * NN_HIDDEN + i] * net.weights[NN_DEPTH - 1][i * DIM_K + j];
            g_batch_out_f[m * DIM_K + j] = s;
        }
    }
#endif

    // float → double
    for (std::size_t k = 0; k < n_in; ++k)
        G_batch[k] = static_cast<double>(g_batch_out_f[k]);
}

#endif  // DIM_K_SELECT == 10 && ENABLE_BATCH_NN

void nn_cache_stats() {
    auto print_cache = [](const char* label, const NNCache& c,
                          uint64_t hits, uint64_t misses) {
        std::size_t sz = c.size();
        uint64_t total = hits + misses;
        double hit_rate = total > 0 ? 100.0 * hits / total : 0.0;
        printf("  %-22s: %zu / %zu entries (%.1f MB%s)\n",
               label, sz, MAX_CACHE_SIZE,
               sz * (2.0 * DIM_K * 8 + 100) / 1e6,
               sz >= MAX_CACHE_SIZE ? ", limit REACHED" : "");
        printf("  %-22s  hits=%llu  misses=%llu  total=%llu  hit_rate=%.2f%%\n",
               "", (unsigned long long)hits, (unsigned long long)misses,
               (unsigned long long)total, hit_rate);
    };
    print_cache("NN outsource cache", g_nn_cache,
                g_cache_hits, g_cache_misses);
}
