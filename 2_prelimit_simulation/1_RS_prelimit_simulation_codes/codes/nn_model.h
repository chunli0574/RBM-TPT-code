/*
 * nn_model.h
 *
 * Thin wrapper around a TensorFlow SavedModel loaded via cppflow.
 * The model is a 2-input → 2-output network trained on the W-space (workload).
 *
 * Usage pattern (matches the reference project continuous_review_dynamic_highdim_24.cpp):
 *   1. Call nn_model_load(path) once at program start (in main()).
 *   2. Call nn_predict(W0, W1, out) whenever you need a gradient estimate.
 *
 * Python equivalent:
 *   model = load_savedmodel_as_keras(model_path)   (functions_general.py)
 *   G = model(W).numpy()                            (NNPricingPolicy / NNOutsourcingPolicy)
 *
 * Input tensor:  shape [1, 2], float32  — the two workload values W[0], W[1]
 * Output tensor: shape [1, 2], float32  — the gradient G[0], G[1]
 *
 * Tensor names used (same as SavedModel serving_default signature):
 *   Input  key:  "serving_default_inputs"
 *   Output key:  "StatefulPartitionedCall"
 */

#pragma once

#include <string>

// ---------------------------------------------------------------------------
// nn_model_load
// Load the TensorFlow SavedModel from model_path.
// Must be called ONCE before any call to nn_predict.
// Python: tf.saved_model.load(savedmodel_dir)
// ---------------------------------------------------------------------------
void nn_model_load(const std::string& model_path);

// ---------------------------------------------------------------------------
// nn_predict
// Run a single forward pass: input W = (W0, W1), output gradient G written to out[2].
// Results are cached by (W0, W1) — TF is only called on the first encounter of
// each distinct point; subsequent calls return the cached value immediately.
// Python: G = model(W).numpy()  where W has shape (1, 2)
// ---------------------------------------------------------------------------
void nn_predict(double W0, double W1, double out[2]);

// ---------------------------------------------------------------------------
// nn_cache_stats
// Print the number of unique (W0, W1) points stored in the inference cache.
// Call after run_simulation() to see how many distinct NN evaluations occurred.
// ---------------------------------------------------------------------------
void nn_cache_stats();

// ---------------------------------------------------------------------------
// nn_cache_clear
// Discard all cached (W0, W1) → G entries (model stays loaded).
// Call between grid-search iterations when the model changes.
// ---------------------------------------------------------------------------
void nn_cache_clear();

// ---------------------------------------------------------------------------
// nn_model_reload
// Unload current model, clear cache, then load a new model from model_path.
// Use when rho (and therefore the model path) changes across grid-search runs.
// ---------------------------------------------------------------------------
void nn_model_reload(const std::string& model_path);

// ---------------------------------------------------------------------------
// nn_model_path_exists
// Returns true if model_path is an existing directory (old SavedModel format)
// OR an existing regular file (new .weights.bin format).
// Use before nn_model_load to fail fast with a clear error message.
// ---------------------------------------------------------------------------
bool nn_model_path_exists(const std::string& model_path);

