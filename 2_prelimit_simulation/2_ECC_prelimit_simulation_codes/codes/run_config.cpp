/*
 * run_config.cpp
 *
 * Definitions of the runtime-mutable policy parameters.
 * Default values match the constexpr values in params.h.
 */

#include "run_config.h"
#include "params.h"

double RC_SAFETY_STOCK        = SAFETY_STOCK;          // = -1.0
double RC_EPSILON_BAND        = EPSILON_BAND;          // = 0.2
double RC_OUTSOURCE_THRESHOLD = OUTSOURCE_THRESHOLD;   // = 1.0
// thread_local: each thread sees its own value; main thread's default is
// used for non-benchmark binaries (sim_proposed/sim_denser).
thread_local double RC_LAMBDA_TILDE        = LAMBDA_TILDE;          // = 4.0
thread_local double RC_BETA_TILDE_V        = BETA_TILDE_V;          // = 0.75 * B_V
thread_local double RC_BETA_TILDE_H        = BETA_TILDE_H;          // = 0.75 * B_H

int RC_PRICING_MODE     = (int)PRICING_MODE;      // = NN_PRICING = 2
int RC_OUTSOURCING_MODE = (int)OUTSOURCING_MODE;  // = NN_OUTSOURCING = 2
int RC_SCHEDULING_MODE  = (int)SCHEDULING_MODE;   // = WMP_OBJ2 = 0
int RC_INWARD_REFLECTION = 0;                     // 0 = hold_last (outward band), 1 = inward band
int RC_EPSILON_RELATIVE  = 0;                     // 0 = absolute eps (×scaling), 1 = relative eps (×FACE_CONST)
int RC_WMP_RELATIVE           = 0;   // 0 = absolute (w=W_ARR), 1 = relative (w=W_ARR/b_arr)
int RC_WMP_NO_QUEUE           = 0;   // 0 = v1 full pressure, 1 = v2 no-queue pressure
int RC_WMP_EXTRA_RELATIVE     = 0;   // 0 = plain v1; 1 = normalize v1 pressure term by (B_V*rho) / (B_H*rho)
int RC_N_PATHS = N_PATHS;

// Detailed CSV/computation switch. Defaults to 0 (lean output): skip pricing/
// outsourcing cost splits and per-route outsource time — cost_mean, delay_prob,
// cost_std are unaffected. Each main overrides from its config file.
int RC_PRINT_DETAILED_INFO = 0;

// Delay-histogram intent flag (see run_config.h). Hot path uses the
// FinishedInfo::delay_hist pointer instead of this; kept for callers that
// want to inspect the recording intent declaratively.
int RC_RECORD_DELAY_HIST = 0;

// BCPlinear pricing parameters (compact): RC_PRICING_PHI[r][0] = phi_r,
// RC_PRICING_PHI[r][1] = phi_{-r}. All zero by default (set by benchmark search).
thread_local double RC_PRICING_PHI[N_ROUTES][2] = {};

// High-Low pricing parameters — all zero by default (set by benchmark search)
thread_local double RC_HL_PARA[N_ROUTES][3] = {};
