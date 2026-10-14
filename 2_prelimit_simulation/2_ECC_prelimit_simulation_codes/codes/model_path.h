/*
 * model_path.h
 *
 * Shared NN model-file naming logic used by:
 *   - proposed_policy_main.cpp  (sim_proposed / sim_proposed_batch)
 *   - denser_sim_main.cpp       (sim_denser)
 *   - one_combo_main.cpp        (sim_one_combo / sim_one_combo_batch)
 *
 * All three binaries previously held their own copy of the eta_theta maps and
 * make_model_path; this header centralises the logic so adding/removing rhos
 * or batches needs to be done in only one place.
 *
 * Behaviour-equivalent to the original definitions (pure refactor):
 *   - CC3 (DIM_K_SELECT == 3):
 *       theta = -0.8 for ALL rhos.
 *       New-format rhos (CC3_NEW_FORMAT_RHOS): 3-decimal rhos served from
 *         new_model_base_dir as .weights.bin (elu_50pace=10_).
 *       Old-format rhos: SavedModel directory in model_base_dir; pace=10 vs
 *         default elu_50_ chosen by intermediate-rho heuristic.
 *
 *   - CC10 (DIM_K_SELECT == 10):
 *       eta_theta_mapping (rho >= 0.8 keys) and eta_theta_mapping_b50 (rho < 0.8).
 *       Branches: rho<0.8 (b50) | CC10_PACE20_RHOS (2-decimal pace=20)
 *               | CC10_PACE20_3DEC_RHO_THETA (3-decimal pace=20, theta from map)
 *               | default elu_100_
 *               | CC10_NEW_FORMAT_RHOS (currently empty placeholder).
 */

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

// ---------------------------------------------------------------------------
// rho → theta maps (selected by DIM_K_SELECT inside the .cpp)
// ---------------------------------------------------------------------------

// Main map: CC3 uses this for all rhos; CC10 uses this for rho >= 0.8.
std::unordered_map<double, double> build_eta_theta_map();

// b50 map: CC10 only — used for rho < 0.8.  CC3 returns an empty map.
std::unordered_map<double, double> build_eta_theta_b50_map();

// ---------------------------------------------------------------------------
// rho-set helpers
// ---------------------------------------------------------------------------

// 3-decimal new-format rho registry (CC3_NEW_FORMAT_RHOS / CC10_NEW_FORMAT_RHOS).
// CC3: {0.885, 0.895, 0.905, 0.915, 0.925}.   CC10: {} (placeholder).
const std::unordered_set<double>& new_format_rho_set();

// CC10 only: 2-decimal rhos that use the elu_100pace=20_ suffix (2026-Apr batch
// 0.81-0.84). Empty on CC3.
const std::unordered_set<double>& cc10_pace20_rhos();

// CC10 only: 3-decimal pace=20 supplemental rho → theta map (2026-Apr batch
// 0.905..0.945, theta = -0.4). Empty on CC3.
const std::unordered_map<double, double>& cc10_pace20_3dec_rho_theta();

// ---------------------------------------------------------------------------
// rho formatting
// ---------------------------------------------------------------------------

// Format rho to up to 2 decimals, trailing zeros stripped (keeps at least one).
//   0.9 → "0.9", 0.95 → "0.95", 1.0 → "1.0"
std::string fmt_rho_2dec(double rho);

// Format rho to up to 3 decimals, trailing zeros stripped (keeps at least one).
//   0.905 → "0.905", 0.9 → "0.9", 0.85 → "0.85"
std::string fmt_rho_3dec(double rho);

// Round rho to 3 decimals as the canonical key for new-format / 3-decimal
// rho-set lookups. Stable under float-literal == double-literal comparisons.
double rho_3dec_key(double rho);

// ---------------------------------------------------------------------------
// Path construction
// ---------------------------------------------------------------------------

// Construct the NN model file path for a given rho.
//
// Both maps must be supplied; the b50 map is only consulted when DIM_K_SELECT==10
// AND rho < 0.8. new_model_base_dir may be empty for branches that don't use it,
// but must be set when rho is in the new-format set.
//
// Aborts with an explanatory stderr message + exit(1) if rho is not registered
// or required base directories are missing.
std::string make_model_path(double rho,
                            const std::unordered_map<double,double>& eta_theta,
                            const std::unordered_map<double,double>& eta_theta_b50,
                            const std::string& model_base_dir,
                            const std::string& new_model_base_dir);

// Pre-flight check: returns true iff `rho` is registered in any branch of
// make_model_path's lookup logic. Lets callers (proposed_policy_main pre-flight,
// one_combo_main validation) emit a nicer error before make_model_path's exit().
bool rho_in_map(double rho,
                const std::unordered_map<double,double>& eta_theta,
                const std::unordered_map<double,double>& eta_theta_b50);
