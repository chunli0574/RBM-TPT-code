/*
 * nn_model.h
 *
 * Thin wrapper around a TensorFlow SavedModel loaded via cppflow.
 * The model is a DIM_K-input → DIM_K-output network trained on the W-space
 * (workload vector), where DIM_K = 3 for the CC3 network.
 *
 * CC3 difference from RS:
 *   RS:  nn_predict(W0, W1, out[2])   — 2D input/output
 *   CC3: nn_predict(W[DIM_K], out[DIM_K]) — 3D input/output (W is a pointer)
 *
 * Usage:
 *   1. Call nn_model_load(path) once at program start.
 *   2. Call nn_predict(W, G) to get the gradient estimate for workload W.
 *
 * Python equivalent:
 *   model = load_savedmodel_as_keras(model_path)    (functions_general.py)
 *   G = model(W).numpy()                             (NNPricingPolicy / NNOutsourcingPolicy)
 *
 * Input tensor:  shape [1, DIM_K], float64  — workload values W[0..DIM_K-1]
 * Output tensor: shape [1, DIM_K], float64  — gradient G[0..DIM_K-1]
 */

#pragma once

#include "params.h"   // for DIM_K
#include <string>

// ---------------------------------------------------------------------------
// nn_model_load
// Load the TensorFlow SavedModel from model_path.
// Must be called ONCE before any call to nn_predict.
// ---------------------------------------------------------------------------
void nn_model_load(const std::string& model_path);

// ---------------------------------------------------------------------------
// nn_predict
// Run a single forward pass: input W[DIM_K], output gradient G written to out[DIM_K].
// Results are cached by the W vector — TF is only called on the first encounter of
// each distinct W; subsequent calls return the cached value immediately.
// Python: G = model(W).numpy()  where W has shape (1, DIM_K)
// ---------------------------------------------------------------------------
void nn_predict(const double W[DIM_K], double out[DIM_K]);

// ---------------------------------------------------------------------------
// nn_cache_stats
// Print the number of unique W vectors stored in the inference cache.
// ---------------------------------------------------------------------------
void nn_cache_stats();

// ---------------------------------------------------------------------------
// nn_cache_clear
// Discard all cached entries (model stays loaded).
// ---------------------------------------------------------------------------
void nn_cache_clear();

// ---------------------------------------------------------------------------
// nn_model_reload
// Unload current model, clear cache, then load a new model from model_path.
// ---------------------------------------------------------------------------
void nn_model_reload(const std::string& model_path);

// ---------------------------------------------------------------------------
// nn_model_path_exists
// Return true if model_path is an existing directory (TF SavedModel format).
// Used for pre-flight validation before the simulation loop starts.
// ---------------------------------------------------------------------------
bool nn_model_path_exists(const std::string& model_path);

// ---------------------------------------------------------------------------
// nn_predict_batch
// Batch version of nn_predict: runs a single forward pass over n W vectors,
// returning n G vectors. Input/output are row-major double arrays of size
// n * DIM_K.
// Only available under the batch build (CC10 + ENABLE_BATCH_NN) where the
// native backend is active; uses cblas_sgemm internally. Bypasses the
// quantised cache (CC10 cache is disabled anyway).
// ---------------------------------------------------------------------------
#if DIM_K_SELECT == 10 && defined(ENABLE_BATCH_NN)
void nn_predict_batch(const double W_batch[], double G_batch[], int n);
#endif
