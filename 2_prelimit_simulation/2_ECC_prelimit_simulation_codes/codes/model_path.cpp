/*
 * model_path.cpp
 *
 * Implementation of the shared NN model-file naming logic.
 * Pure refactor — copied verbatim from proposed_policy_main.cpp /
 * denser_sim_main.cpp so behaviour is bit-identical.
 */

#include "model_path.h"
#include "params.h"   // for DIM_K_SELECT

#include <cmath>
#include <cstdio>
#include <cstdlib>

// ===========================================================================
// rho → theta maps + rho sets
// ===========================================================================

#if DIM_K_SELECT == 3
// CC3: all rho values map to theta = -0.8 (D3 models)
std::unordered_map<double, double> build_eta_theta_map() {
    std::unordered_map<double, double> m;
    // Old-format rhos (2-decimal, TF SavedModel directory)
    double rhos[] = {
        0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
        0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.10, 1.15, 1.20
    };
    for (double r : rhos) m[r] = -0.8;
    // New-format rhos (3-decimal, .weights.bin native backend)
    m[0.885] = -0.8;
    m[0.895] = -0.8;
    m[0.905] = -0.8;
    m[0.915] = -0.8;
    m[0.925] = -0.8;
    return m;
}

// CC3: separate b50 map (not used for CC3 but defined for API consistency)
std::unordered_map<double, double> build_eta_theta_b50_map() {
    return {};
}

// New-format rhos for CC3: use .weights.bin native backend + new_model_base_dir.
static const std::unordered_set<double> CC3_NEW_FORMAT_RHOS = {
    0.885, 0.895, 0.905, 0.915, 0.925
};

const std::unordered_set<double>& new_format_rho_set() { return CC3_NEW_FORMAT_RHOS; }

// CC10-only sets — empty on CC3.
static const std::unordered_set<double>          CC10_PACE20_RHOS_EMPTY = {};
static const std::unordered_map<double, double>  CC10_PACE20_3DEC_EMPTY = {};
const std::unordered_set<double>& cc10_pace20_rhos()                  { return CC10_PACE20_RHOS_EMPTY; }
const std::unordered_map<double, double>& cc10_pace20_3dec_rho_theta(){ return CC10_PACE20_3DEC_EMPTY; }

#elif DIM_K_SELECT == 10
// CC10: unique theta per rho from eta_theta_mapping in parameters_model.py
// Two maps: main (rho>=0.8) and b50 (rho<0.8)
std::unordered_map<double, double> build_eta_theta_map() {
    std::unordered_map<double, double> m;
    // eta_theta_mapping (rho >= 0.8 use this; also includes rho<0.8 keys but b50 preferred)
    m[0.60]=-0.4; m[0.65]=-0.4; m[0.66]=-0.6; m[0.67]=-0.4; m[0.68]=-0.4;
    m[0.69]=-0.6; m[0.70]=-0.4; m[0.71]=-0.3; m[0.72]=-0.6; m[0.73]=-0.5;
    m[0.74]=-0.3; m[0.75]=-0.6; m[0.76]=-0.6; m[0.77]=-0.4; m[0.78]=-0.3;
    m[0.79]=-0.5;
    m[0.80]=-0.3;
    // 2026-Apr batch (pace=20): rho 0.81-0.84 supplied by
    // CC_training_2026Feb_24th/CC_training_2026_clusterA/models (.weights.bin).
    // File name uses elu_100pace=20_ suffix; see CC10_PACE20_RHOS in make_model_path.
    m[0.81]=-0.5; m[0.82]=-0.5; m[0.83]=-0.5; m[0.84]=-0.4;
    m[0.85]=-0.4; m[0.86]=-0.4; m[0.87]=-0.3; m[0.88]=-0.4;
    m[0.89]=-0.5; m[0.90]=-0.6; m[0.91]=-0.4; m[0.92]=-0.5; m[0.93]=-0.6;
    m[0.94]=-0.4; m[0.95]=-0.4; m[0.96]=-0.6; m[0.97]=-0.5; m[0.98]=-0.5;
    m[0.99]=-0.6; m[1.00]=-0.4;
    m[1.01]=-0.5; m[1.02]=-0.6; m[1.03]=-0.5; m[1.04]=-0.6; m[1.05]=-0.5;
    m[1.06]=-0.5; m[1.07]=-0.5; m[1.08]=-0.5; m[1.09]=-0.5; m[1.10]=-0.6;
    m[1.11]=-0.5; m[1.12]=-0.5; m[1.13]=-0.5; m[1.14]=-0.5; m[1.15]=-0.5;
    m[1.20]=-0.5;
    return m;
}

// CC10 pace=20 rhos: rho >= 0.8 branch uses elu_100pace=20_ suffix instead of
// the default elu_100_. 2026-Apr batch (.weights.bin, exact matches only).
static const std::unordered_set<double> CC10_PACE20_RHOS = {
    0.81, 0.82, 0.83, 0.84
};

// CC10 3-decimal pace=20 rho → theta map: same physical batch as
// CC10_PACE20_RHOS but with 3-decimal eta string and per-rho theta.
// 2026-Apr supplemental batch.
static const std::unordered_map<double, double> CC10_PACE20_3DEC_RHO_THETA = {
    {0.905, -0.4},
    {0.915, -0.4},
    {0.925, -0.4},
    {0.935, -0.4},
    {0.945, -0.4},
};

// eta_theta_mapping_b50: used for rho < 0.8
std::unordered_map<double, double> build_eta_theta_b50_map() {
    std::unordered_map<double, double> m;
    m[0.60]=-0.3; m[0.61]=-0.6; m[0.62]=-0.5; m[0.63]=-0.6; m[0.64]=-0.5;
    m[0.65]=-0.6; m[0.66]=-0.6; m[0.67]=-0.3; m[0.68]=-0.4; m[0.69]=-0.4;
    m[0.70]=-0.5; m[0.71]=-0.3; m[0.72]=-0.4; m[0.73]=-0.6; m[0.74]=-0.5;
    m[0.75]=-0.3; m[0.76]=-0.6; m[0.77]=-0.5; m[0.78]=-0.4; m[0.79]=-0.5;
    return m;
}

// No CC10 new-format rhos defined yet (placeholder for future models).
static const std::unordered_set<double> CC10_NEW_FORMAT_RHOS = {};

const std::unordered_set<double>& new_format_rho_set()                      { return CC10_NEW_FORMAT_RHOS; }
const std::unordered_set<double>& cc10_pace20_rhos()                        { return CC10_PACE20_RHOS; }
const std::unordered_map<double, double>& cc10_pace20_3dec_rho_theta()      { return CC10_PACE20_3DEC_RHO_THETA; }
#endif

// ===========================================================================
// rho formatting helpers
// ===========================================================================

std::string fmt_rho_2dec(double rho) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", rho);
    std::string s(buf);
    size_t dot = s.find('.');
    if (dot != std::string::npos) {
        size_t last_nz = s.find_last_not_of('0');
        if (last_nz == dot)
            s = s.substr(0, dot + 2);
        else
            s = s.substr(0, last_nz + 1);
    }
    return s;
}

std::string fmt_rho_3dec(double rho) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", rho);
    std::string s(buf);
    size_t dot = s.find('.');
    if (dot != std::string::npos) {
        size_t last_nz = s.find_last_not_of('0');
        if (last_nz == dot)
            s = s.substr(0, dot + 2);
        else
            s = s.substr(0, last_nz + 1);
    }
    return s;
}

double rho_3dec_key(double rho) {
    return std::round(rho * 1000.0) / 1000.0;
}

// ===========================================================================
// make_model_path
// ===========================================================================

#if DIM_K_SELECT == 3
std::string make_model_path(double rho,
                            const std::unordered_map<double,double>& eta_theta,
                            const std::unordered_map<double,double>& /*eta_theta_b50*/,
                            const std::string& model_base_dir,
                            const std::string& new_model_base_dir) {
    // ---- New-format rho? ----
    if (new_format_rho_set().count(rho_3dec_key(rho)) > 0) {
        std::string rho_str = fmt_rho_3dec(rho);
        std::string file_name =
            std::string("RBMSteadyState_extendedCrissCross_none_100_100_100_D3_") +
            "-0.8_0.0_0.01_200_0.1_64_6000_dim=3_TD1_elu_50pace=10_" +
            "addcost=0.05_0.2_0.3_lr140000_eta=" + rho_str +
            "_pricing=Y_delay=2.25_1.5_grad.weights.bin";
        if (new_model_base_dir.empty()) {
            fprintf(stderr,
                "ERROR: rho=%.4f requires new_model_base_dir (not set in config)\n", rho);
            exit(1);
        }
        return new_model_base_dir + "/" + file_name;
    }

    // ---- Old-format rho ----
    double rho_key = std::round(rho * 100.0) / 100.0;
    auto it = eta_theta.find(rho_key);
    if (it == eta_theta.end()) {
        fprintf(stderr, "ERROR: rho=%.4f not found in eta_theta_mapping\n", rho);
        exit(1);
    }
    char theta_buf[32];
    snprintf(theta_buf, sizeof(theta_buf), "%.1f", it->second);
    std::string rho_str = fmt_rho_2dec(rho);

    bool use_pace10 = ((rho > 0.95 + 0.001) && (rho < 1.0 - 0.001))
                   || ((rho > 1.0 + 0.001) && (rho < 1.05 - 0.001));
    std::string elu_suffix = use_pace10 ? "elu_50pace=10_" : "elu_50_";

    std::string file_name =
        std::string("RBMSteadyState_extendedCrissCross_none_100_100_100_D3_") +
        theta_buf + "_0.0_0.01_200_0.1_64_6000_dim=3_TD1_" +
        elu_suffix +
        "addcost=0.05_0.2_0.3_lr140000_eta=" + rho_str +
        "_pricing=Y_delay=2.25_1.5_grad";

    return model_base_dir + "/" + file_name;
}

#elif DIM_K_SELECT == 10
std::string make_model_path(double rho,
                            const std::unordered_map<double,double>& eta_theta,
                            const std::unordered_map<double,double>& eta_theta_b50,
                            const std::string& model_base_dir,
                            const std::string& new_model_base_dir) {
    // ---- 3-decimal pace=20 supplemental rho? (loaded from model_base_dir) ----
    {
        const auto& m = cc10_pace20_3dec_rho_theta();
        auto it = m.find(rho_3dec_key(rho));
        if (it != m.end()) {
            char theta_buf[32];
            snprintf(theta_buf, sizeof(theta_buf), "%.1f", it->second);
            std::string rho_str = fmt_rho_3dec(rho);
            std::string file_name =
                std::string("RBMSteadyState_extendedCrissCross_none_200_200_200_D4_") +
                theta_buf + "_0.0_0.01_100_0.1_64_9000_dim=10_TD1_elu_100pace=20_" +
                "addcost=0.1_0.2_0.3_lr270000_eta=" + rho_str +
                "_pricing=Y_delay=6.0_1.5_grad.weights.bin";
            return model_base_dir + "/" + file_name;
        }
    }

    // ---- New-format rho? (placeholder: CC10_NEW_FORMAT_RHOS currently empty) ----
    if (new_format_rho_set().count(rho_3dec_key(rho)) > 0) {
        if (new_model_base_dir.empty()) {
            fprintf(stderr,
                "ERROR: rho=%.4f requires new_model_base_dir (not set in config)\n", rho);
            exit(1);
        }
        // TODO: construct CC10 new-format filename when CC10 new models are available.
        fprintf(stderr, "ERROR: CC10 new-format path not yet implemented for rho=%.4f\n", rho);
        exit(1);
    }

    double rho_key = std::round(rho * 100.0) / 100.0;
    double theta;
    std::string n_param;
    std::string elu_suffix;

    if (rho >= 0.8 - 0.001) {
        auto it = eta_theta.find(rho_key);
        if (it == eta_theta.end()) {
            fprintf(stderr, "ERROR: rho=%.4f not found in eta_theta_mapping (CC10)\n", rho);
            exit(1);
        }
        theta      = it->second;
        n_param    = "100";
        elu_suffix = (cc10_pace20_rhos().count(rho_key) > 0) ? "elu_100pace=20_"
                                                             : "elu_100_";
    } else {
        auto it = eta_theta_b50.find(rho_key);
        if (it == eta_theta_b50.end()) {
            fprintf(stderr, "ERROR: rho=%.4f not found in eta_theta_mapping_b50 (CC10)\n", rho);
            exit(1);
        }
        theta      = it->second;
        n_param    = "50.0";
        elu_suffix = "elu_100pace=40_";
    }

    char theta_buf[32];
    snprintf(theta_buf, sizeof(theta_buf), "%.1f", theta);
    std::string rho_str = fmt_rho_2dec(rho);

    std::string file_name =
        std::string("RBMSteadyState_extendedCrissCross_none_200_200_200_D4_") +
        theta_buf + "_0.0_0.01_" + n_param + "_0.1_64_9000_dim=10_TD1_" +
        elu_suffix +
        "addcost=0.1_0.2_0.3_lr270000_eta=" + rho_str +
        "_pricing=Y_delay=6.0_1.5_grad.weights.bin";

    return model_base_dir + "/" + file_name;
}
#endif

// ===========================================================================
// rho_in_map (pre-flight check)
// ===========================================================================

bool rho_in_map(double rho,
                const std::unordered_map<double,double>& eta_theta,
                const std::unordered_map<double,double>& eta_theta_b50) {
    // New-format rhos are valid regardless of the eta_theta map.
    if (new_format_rho_set().count(rho_3dec_key(rho)) > 0) return true;
#if DIM_K_SELECT == 10
    // CC10 3-decimal pace=20 supplemental rhos are also valid.
    if (cc10_pace20_3dec_rho_theta().count(rho_3dec_key(rho)) > 0) return true;
#endif
    double rho_key = std::round(rho * 100.0) / 100.0;
#if DIM_K_SELECT == 3
    (void)eta_theta_b50;
    return eta_theta.count(rho_key) > 0;
#elif DIM_K_SELECT == 10
    if (rho >= 0.8 - 0.001) return eta_theta.count(rho_key) > 0;
    else                     return eta_theta_b50.count(rho_key) > 0;
#endif
}
