/*
 * nn_model_paths.h
 *
 * rho → theta mapping and NN model file-name construction, shared by
 * proposed_policy_main.cpp, one_combo_main.cpp and denser_sim_main.cpp.
 *
 * Models come from RS_training_2026Sept_21st/models; each _grad.keras was
 * converted to _grad.weights.bin by export_keras_weights.py and is loaded
 * by the native backend in nn_model.cpp.
 *
 * Python (training side):
 *   suffix = "RybkoStolyar_none_100_{theta}_..._eta={rho}"
 *   NN_value_grad.save("models/" + suffix + "_grad.keras")
 */

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

// Default model directory (relative to codes/); overridden by model_new_base_dir in config files.
inline const std::string NN_MODEL_DIR_DEFAULT =
    "../../../1_NN_training/3_RS_saved_NNs";

// ---------------------------------------------------------------------------
// eta_theta_mapping: rho → theta for selecting the NN model file.
// Only these 11 (rho, theta) models are available.
// ---------------------------------------------------------------------------
inline const std::unordered_map<double, double>& eta_theta_map() {
    static const std::unordered_map<double, double> m = {
        {0.95, -0.8}, {0.96, -0.8}, {0.97, -0.7}, {0.98, -1.0}, {0.99, -0.9},
        {1.00, -0.9}, {1.01, -0.8}, {1.02, -1.0}, {1.03, -1.0}, {1.04, -0.9},
        {1.05, -1.0}
    };
    return m;
}

// Format a number as Python str(float): e.g. 1.0 → "1.0", 0.95 → "0.95".
// precision controls %.Nf before trailing zeros are stripped (keeps one decimal digit).
inline std::string format_py_float(double x, int precision) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", precision, x);
    std::string s(buf);
    size_t dot = s.find('.');
    if (dot != std::string::npos) {
        size_t last_nz = s.find_last_not_of('0');
        s = (last_nz == dot) ? s.substr(0, dot + 2) : s.substr(0, last_nz + 1);
    }
    return s;
}

// Build the .weights.bin path for a given rho. Exits if rho is not in eta_theta_map().
inline std::string make_model_path(double rho, const std::string& model_dir) {
    const auto& eta_theta = eta_theta_map();
    double rho_key = std::round(rho * 100.0) / 100.0;
    auto it = eta_theta.find(rho_key);
    if (std::fabs(rho - rho_key) > 1e-9 || it == eta_theta.end()) {
        fprintf(stderr,
            "ERROR: rho=%.6f not found in eta_theta_mapping "
            "(available: 0.95, 0.96, ..., 1.05)\n", rho);
        exit(1);
    }
    std::string file_name =
        std::string("RybkoStolyar_none_100_") +
        format_py_float(it->second, 1) +
        "_0.0_0.01_100_0.1_64_6000"
        "_dim=2_TD1_elu_x0=1.0_v1=1.0_v2=2.5"
        "_addcost=0.05_0.2_lr0.0005_eta=" +
        format_py_float(rho_key, 2) +
        "_grad.weights.bin";
    const std::string& dir = model_dir.empty() ? NN_MODEL_DIR_DEFAULT : model_dir;
    return dir + "/" + file_name;
}
