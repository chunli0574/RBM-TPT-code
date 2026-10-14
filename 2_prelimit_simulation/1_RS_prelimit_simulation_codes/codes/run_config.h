/*
 * run_config.h
 *
 * Runtime-mutable parameters for proposed policy and benchmark parameter searches.
 *
 * These shadow the constexpr defaults in params.h for the six search parameters:
 *   safety_stock, epsilon_band, outsource_threshold (rho),
 *   lambda_tilde, beta_tilde_1, beta_tilde_2
 *
 * policy.cpp uses these globals instead of the params.h constexprs,
 * so proposed_policy_main.cpp / benchmark_search_main.cpp can change them
 * before each run_simulation() call without recompiling.
 *
 * Usage in proposed_policy_main.cpp / benchmark_search_main.cpp:
 *   RC_SAFETY_STOCK       = safety;
 *   RC_EPSILON_BAND       = epsilon;
 *   RC_OUTSOURCE_THRESHOLD= rho;
 *   RC_LAMBDA_TILDE       = lambda_tilde;
 *   RC_BETA_TILDE_1       = beta_rate_1 * BETA_CONST * rho;
 *   RC_BETA_TILDE_2       = beta_rate_2 * BETA_CONST * rho;
 *   run_simulation();
 *
 * Default values match params.h so the original `sim` binary is unaffected.
 */

#pragma once

// beta_const = 19/9  (from parameters_model.py)
// beta_tilde = beta_rate * beta_const * rho
constexpr double BETA_CONST = 19.0 / 9.0;

// ---------------------------------------------------------------------------
// Mutable runtime parameters (set by proposed_policy_main.cpp / benchmark_search_main.cpp before each run)
// ---------------------------------------------------------------------------

extern double RC_SAFETY_STOCK;         // safety_stock (default 0.0)
extern double RC_EPSILON_BAND;         // epsilon band width (default 0.25)
extern double RC_OUTSOURCE_THRESHOLD;  // rho: W-space face bound (default 1.01)
// The three scheduling scalars below are written per-combo inside the benchmark
// OpenMP parallel region — mark thread_local so each worker thread has its own
// copy. All other RC globals are only written on the outer (serial) loops over
// safety/epsilon/rho and read-only inside the parallel region.
extern thread_local double RC_LAMBDA_TILDE;  // WMP scheduling lambda_tilde (default 4.0)
extern thread_local double RC_BETA_TILDE_1;  // WMP scheduling beta_tilde_1 (default 1.279)
extern thread_local double RC_BETA_TILDE_2;  // WMP scheduling beta_tilde_2 (default 1.279)

// ---------------------------------------------------------------------------
// Runtime policy mode selectors (used by benchmark_search_main to switch
// among pricing/outsourcing/scheduling policies without recompiling).
// Integer values match the underlying enum ordinals in params.h:
//   RC_PRICING_MODE:    0=BCP_LINEAR, 1=HIGH_LOW, 2=NN_PRICING
//   RC_OUTSOURCING_MODE:0=THRESHOLD_BASED, 1=SINGLE_PUSHING, 2=NN_OUTSOURCING
//   RC_SCHEDULING_MODE: 0=WMP_OBJ2, 1=LRS_QUEUE_LEN, 2=LRS_DELAY_TIME, 3=LRS_REMAIN_DELAY
// Defaults match the constexpr enums in params.h so the single-run 'sim' binary
// is unaffected.
// ---------------------------------------------------------------------------
extern int RC_PRICING_MODE;
extern int RC_OUTSOURCING_MODE;
extern int RC_SCHEDULING_MODE;

// WMP scheduling variant (proposed_scheduling_limiting in Python).
//   0 = 'N': full WMP formula using lambda_tilde and beta_tilde
//   1 = 'Y': simplified limiting formula — pos[i] = (total queue of OTHER route < beta_tilde_1)
// Default: 0 ('N'), matching the existing wmp_scheduling implementation.
extern int RC_WMP_LIMITING;

// WMP full-formula extra-term scaling (only meaningful when RC_WMP_LIMITING == 0).
//   0 = original:  extra term = lambda_tilde * (Z_other_total - beta_tilde_other)
//   1 = scaled:    extra term = lambda_tilde * (Z_other_total - beta_tilde_other) / (BETA_CONST * rho)
// Ignored when RC_WMP_LIMITING == 1 (limiting variant has no such extra term).
// Default: 0.
extern int RC_WMP_EXTRA_RELATIVE;

// idle_new_reflection mode for SinglePushingPerFacePolicy and NNOutsourcingPolicy.
//   0 = epsilon_band_hold_last_outsource_simultaneously_no_takeover  (outward band)
//   1 = epsilon_band_inward_outsource_simultaneously                  (inward band)
// Default: 0 (hold_last_no_takeover), matching the constexpr NN_OUTSOURCING_VARIANT in params.h.
extern int RC_INWARD_REFLECTION;

// Epsilon band scaling mode (applies to both inward and outward variants).
//   0 = absolute: threshold = FACE_BOUNDARY[f]*rho ± epsilon * EPSILON_SCALING[f]
//                 EPSILON_SCALING = {1, 1, sqrt(5), sqrt(5)}  (norm of each face normal)
//   1 = relative: threshold = FACE_BOUNDARY[f] * (rho ± epsilon)
//                 epsilon scales proportionally with the face boundary
// Default: 0 (absolute), matching the original Python implementation.
extern int RC_EPSILON_RELATIVE;

// Number of simulation paths to run per combo (overrides compile-time N_PATHS).
// Default: N_PATHS (1000). Set to a larger value for denser/finer runs.
extern int RC_N_PATHS;

// Print-detailed-info switch:
//   0 = CSV output only contains param_*, cost, std, delay1/2, delay*_std;
//       internally skip the per-step sub-accumulator updates (pricing_cost,
//       outsourcing_cost, band, outer, route_cost[2]) and their aggregation.
//       cost/std/delay results are bit-identical.
//   1 = full detail (current behavior): include all sub-cost columns.
// Set once at startup (non-thread_local — read-only inside OMP parallel region).
extern int RC_PRINT_DETAILED_INFO;

// Record-delay-histogram switch:
//   0 = no per-job delay-time histogram is collected (zero overhead, default).
//       sim, sim_proposed, sim_benchmark, sim_denser all leave this at 0.
//   1 = inside qi_process_departure, every completed-after-warm-up job pushes
//       its delay = (finish_time - enter_time) into a per-route histogram.
//       Used by sim_one_combo to draw delay-time histograms for one parameter set.
extern int RC_RECORD_DELAY_HIST;

// Runtime BCP-linear pricing parameters (RC_PRICING_MODE == 0 = BCP_LINEAR).
// RC_PRICING_PARA[route][0] = phi_own,  RC_PRICING_PARA[route][1] = phi_other
// (compact 2-scalar-per-route layout; see PRICING_PARA in params.h for details).
// thread_local: rewritten per-combo inside the benchmark OpenMP parallel region.
extern thread_local double RC_PRICING_PARA[2][2];

// Runtime High-Low pricing parameters (RC_PRICING_MODE == 1 = HIGH_LOW).
// RC_HL_PARA[route][col] where col 0=threshold, 1=low, 2=high — same as HL_PARA.
// thread_local: rewritten per-combo inside the benchmark OpenMP parallel region.
extern thread_local double RC_HL_PARA[2][3];
