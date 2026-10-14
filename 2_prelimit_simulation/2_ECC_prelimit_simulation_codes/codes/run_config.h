/*
 * run_config.h
 *
 * Runtime-mutable parameters for the proposed policy and benchmark searches.
 * Supports DIM_K_SELECT = 3 (CC3) and DIM_K_SELECT = 10 (CC10).
 *
 * Usage: set RC_* globals before calling run_simulation().
 */

#pragma once
#include "params.h"   // for N_ROUTES, N_BUFFERS, B_V, B_H, DELAY_BOUND_V, DELAY_BOUND_H

// beta_const for vertical routes: b_v = delay_bound_v * lambda_ast * inv_sqrt_n = 1.425
// (same for CC3 and CC10, since delay_bound_v=60 and lambda_ast=0.475 are both unchanged)
// Python: beta_const_v = b_v
constexpr double BETA_CONST_V = B_V;   // = 1.425

// beta_const for horizontal route: b_h = delay_bound_h * lambda_ast * inv_sqrt_n
// CC3:  delay_bound_h = 90  → b_h = 2.1375
// CC10: delay_bound_h = 240 → b_h = 5.7
// Python: beta_const_h = b_h
constexpr double BETA_CONST_H = B_H;   // CC3: 2.1375 | CC10: 5.7

// queue_len_bound[r] = delay_bound[r] * lambda_ast
// Vertical routes: 60 * 0.475 = 28.5  (same for CC3 and CC10)
// Horizontal route:
//   CC3:  90  * 0.475 = 42.75
//   CC10: 240 * 0.475 = 114.0
// Python: queue_len_bound = delay_bound * lambda_ast
#if DIM_K_SELECT == 3
constexpr double QUEUE_LEN_BOUND[N_ROUTES] = {28.5, 28.5, 28.5, 42.75};
#elif DIM_K_SELECT == 10
constexpr double QUEUE_LEN_BOUND[N_ROUTES] = {
    28.5, 28.5, 28.5, 28.5, 28.5,
    28.5, 28.5, 28.5, 28.5, 28.5,
    114.0
};
#endif

// ---------------------------------------------------------------------------
// Mutable runtime parameters (set before each run_simulation() call)
// ---------------------------------------------------------------------------

extern double RC_SAFETY_STOCK;         // safety_stock (default -1.0 → never fires)
extern double RC_EPSILON_BAND;         // epsilon band width (default EPSILON_BAND)
extern double RC_OUTSOURCE_THRESHOLD;  // rho: face boundary scale (default 1.0)
// Scheduling parameters: thread_local so benchmark_search_main's parallel
// combo loops can write each thread's copy independently.
extern thread_local double RC_LAMBDA_TILDE;   // WMP scheduling lambda_tilde (default 4.0)
extern thread_local double RC_BETA_TILDE_V;   // WMP scheduling beta for vertical routes
extern thread_local double RC_BETA_TILDE_H;   // WMP scheduling beta for horizontal route

// ---------------------------------------------------------------------------
// Runtime policy mode selectors
// Integer values match the underlying enum ordinals in params.h:
//   RC_PRICING_MODE:    0=BCP_LINEAR, 1=HIGH_LOW, 2=NN_PRICING
//   RC_OUTSOURCING_MODE:0=THRESHOLD_BASED, 1=SINGLE_PUSHING, 2=NN_OUTSOURCING
//   RC_SCHEDULING_MODE: 0=WMP_OBJ2, 1=LRS_QUEUE_LEN, 2=LRS_DELAY_TIME, 3=LRS_REMAIN_DELAY
// ---------------------------------------------------------------------------
extern int RC_PRICING_MODE;
extern int RC_OUTSOURCING_MODE;
extern int RC_SCHEDULING_MODE;
extern int RC_INWARD_REFLECTION;  // 0 = hold_last (outward band), 1 = inward band
extern int RC_EPSILON_RELATIVE;   // epsilon band: 0=absolute (±eps×scaling), 1=relative (FACE_CONST×(rho±eps))
extern int RC_WMP_RELATIVE;            // WMP w weights: 0=absolute (w=mu_arr), 1=relative (w=mu_arr/b_arr)
extern int RC_WMP_NO_QUEUE;            // WMP pressure: 0=v1 full (Z+λ*(Z-β)), 1=v2 no-queue (λ*(Z-β) only)
extern int RC_WMP_EXTRA_RELATIVE;      // WMP v1 last back: 0=original, 1=scale pressure by BETA_CONST_V/BETA_CONST_H
extern int RC_N_PATHS;                 // number of paths to run (default N_PATHS)

// ---------------------------------------------------------------------------
// RC_PRINT_DETAILED_INFO: controls whether auxiliary cost/outsourcing statistics
// are computed by the simulation and written to output CSVs.
//   1           = full output — compute and log pricing_cost, outsourcing_cost,
//                 hardstop columns, and outsource_frac[] per route.
//   0 (default) = lean output — skip accumulation of pricing/outsourcing cost
//                 splits and t_outsourced[]; CSVs omit the corresponding columns.
//                 cost_mean, delay_prob_mean[], and cost_std are unaffected.
// Not thread_local: the value is set once at startup from each main's config.
// ---------------------------------------------------------------------------
extern int RC_PRINT_DETAILED_INFO;

// ---------------------------------------------------------------------------
// RC_RECORD_DELAY_HIST: declarative on/off flag for delay-time histogram
// recording. Currently advisory only — the actual hot-path gate is the
// FinishedInfo::delay_hist pointer (zero-overhead null check). Existed for
// one_combo_main to expose the intent in stdout / config readers; kept here
// in case a future caller wants to read the flag without inspecting pointer
// state. Default 0; one_combo_main sets to 1.
// ---------------------------------------------------------------------------
extern int RC_RECORD_DELAY_HIST;

// ---------------------------------------------------------------------------
// Runtime BCPlinear pricing parameters (RC_PRICING_MODE == BCP_LINEAR)
//
// Compact (phi_r, phi_{-r}) representation per route r:
//   RC_PRICING_PHI[r][0] = phi_r     (coefficient of route-r's "own" buffer(s))
//   RC_PRICING_PHI[r][1] = phi_{-r}  (coefficient of all other buffers)
//
// This is equivalent to the structured K×m matrix Φ in the paper, where
// Φ[k, (k', l')] = phi_k if k' == k else phi_{-k}. All 2K free parameters
// are stored here; the full N_ROUTES × N_BUFFERS matrix is never materialised.
//
// "Own" buffer semantics:
//   - Vertical route r = 0..DIM_K-1  : own = {Z[r]}
//   - Horizontal route r = DIM_K     : own = {Z[DIM_K], Z[DIM_K+1], ..., Z[2*DIM_K-1]}
//
// Pricing formula evaluated in bcp_linear_pricing:
//   pc[r] = (phi_r * Σ Z_own[r] + phi_{-r} * Σ Z_other[r]) * INV_SQRT_N
//
// thread_local so benchmark's parallel pricing search can write each thread's
// copy independently.
// ---------------------------------------------------------------------------
extern thread_local double RC_PRICING_PHI[N_ROUTES][2];

// ---------------------------------------------------------------------------
// Runtime High-Low pricing parameters (RC_PRICING_MODE == HIGH_LOW)
// RC_HL_PARA[route][col]:
//   col 0 = threshold (Z-scale: already multiplied by INV_SQRT_N at search time,
//           so runtime comparison is inp.Z[...] > threshold directly)
//   col 1 = low drift
//   col 2 = high drift
// pricing_control[r] = (Z_sum[r] > threshold) ? high : low  * INV_SQRT_N
// Python: HL_para = pricing_para.reshape((num_routes, 3))
// thread_local so benchmark's parallel pricing search can write each thread's
// copy independently.
// ---------------------------------------------------------------------------
extern thread_local double RC_HL_PARA[N_ROUTES][3];
