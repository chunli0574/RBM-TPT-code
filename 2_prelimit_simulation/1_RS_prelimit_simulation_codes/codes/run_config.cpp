/*
 * run_config.cpp
 *
 * Definitions of the runtime-mutable policy parameters.
 * Default values match the constexpr values in params.h.
 */

#include "run_config.h"
#include "params.h"   // for OUTSOURCE_THRESHOLD, EPSILON_BAND, etc.

double RC_SAFETY_STOCK        = SAFETY_STOCK;
double RC_EPSILON_BAND        = EPSILON_BAND;
double RC_OUTSOURCE_THRESHOLD = OUTSOURCE_THRESHOLD;
// thread_local — written per-combo inside benchmark OpenMP parallel region.
thread_local double RC_LAMBDA_TILDE = LAMBDA_TILDE;
thread_local double RC_BETA_TILDE_1 = BETA_TILDE_1;
thread_local double RC_BETA_TILDE_2 = BETA_TILDE_2;

// Policy mode selectors — default to the compile-time enums from params.h.
int RC_PRICING_MODE      = (int)PRICING_MODE;
int RC_OUTSOURCING_MODE  = (int)OUTSOURCING_MODE;
int RC_SCHEDULING_MODE   = (int)SCHEDULING_MODE;
int RC_INWARD_REFLECTION = 0;  // 0=hold_last_no_takeover (outward), 1=inward
int RC_WMP_LIMITING      = 0;  // 0='N': full WMP formula; 1='Y': simplified limiting formula
int RC_WMP_EXTRA_RELATIVE = 0;  // 0=original extra term; 1=extra term scaled by 1/(BETA_CONST*rho)
int RC_EPSILON_RELATIVE      = 0;  // 0=absolute eps (× EPSILON_SCALING), 1=relative eps (× FACE_BOUNDARY)
int RC_N_PATHS           = N_PATHS;  // number of paths per run (default 1000)
int RC_PRINT_DETAILED_INFO = 0;  // 0=basic CSV (param_*, cost, std, delay, delay_std);
                                 // 1=full CSV with cost breakdown + hardstop columns
int RC_RECORD_DELAY_HIST   = 0;  // 0=no per-job delay histogram (default, zero overhead);
                                 // 1=collect per-route delay histograms (sim_one_combo only)

// Pricing parameter arrays — default to the constexpr values from params.h.
// thread_local — rewritten per-combo inside benchmark OpenMP parallel region.
// Compact 2-scalar-per-route layout: [route][0]=phi_own, [route][1]=phi_other.
thread_local double RC_PRICING_PARA[2][2] = {
    {PRICING_PARA[0][0], PRICING_PARA[0][1]},
    {PRICING_PARA[1][0], PRICING_PARA[1][1]}
};
thread_local double RC_HL_PARA[2][3] = {
    {HL_PARA[0][0], HL_PARA[0][1], HL_PARA[0][2]},
    {HL_PARA[1][0], HL_PARA[1][1], HL_PARA[1][2]}
};
