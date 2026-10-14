/*
 * policy.h
 *
 * Declarations for the policy functions and data structures
 * for the CC3 (Extended Criss-Cross, dim_K=3) queueing network.
 *
 * Python equivalent: policy classes in functions_general.py (CC3_python_code)
 *
 * CC3 differences from RS policy.h:
 *   - PolicyInputs: Z[6], W[3], delay_time[6], W_last[3]
 *   - PolicyOutputs: pricing_control[4], outsourcing_control[7], scheduling_control[3]
 *   - Scheduling codes: station kk → value (kk+1) for vertical, (DIM_K+kk+1) for horizontal
 *   - Outsourcing: 6 faces (3 diagonal + 3 off-diagonal)
 *
 * SCHEDULING_CONTROL ENCODING (for all 3 stations):
 *   station kk: (kk+1)       = serve buf[kk]       (vertical "front" buffer)
 *   station kk: (DIM_K+kk+1) = serve buf[DIM_K+kk] (horizontal "back" buffer)
 *   station kk: 0             = idle
 * Concretely:
 *   Station 0: 1=buf[0], 4=buf[3], 0=idle
 *   Station 1: 2=buf[1], 5=buf[4], 0=idle
 *   Station 2: 3=buf[2], 6=buf[5], 0=idle
 *
 * OUTSOURCING_CONTROL ENCODING:
 *   [0..N_ROUTES-1 = 0..3]: outsource route flags (0=no, 1=yes, 2=hardstop)
 *   [N_ROUTES..N_ROUTES+DIM_K-1 = 4..6]: idle station flags
 *     [4]: idle station 0  (set by NN outsourcing, cleared by idling_policy)
 *     [5]: idle station 1
 *     [6]: idle station 2
 *   NOTE: columns 4..6 are only meaningful BEFORE idling_policy runs.
 *         Simulation code only reads columns 0..3 for cost/arrival decisions.
 */

#pragma once

#include "params.h"

// ---------------------------------------------------------------------------
// PolicyInputs: what the policy functions receive
// ---------------------------------------------------------------------------
struct PolicyInputs {
    double Z[N_BUFFERS];            // scaled queue lengths: Z[k] = states[k] * INV_SQRT_N
    double W[DIM_K];                // workload: W[i] = sum_k M_MATRIX[i][k] * Z[k]
    double delay_time[N_BUFFERS];   // head-of-line sojourn time per buffer (0 if empty)
                                    // used by LRS delay-time scheduling policies
    double W_last[DIM_K];           // workload at last step when state was INSIDE state space
                                    // used by NN pricing (hold_last_price) and NN outsourcing
    double G_cached[DIM_K];         // cached NN output evaluated at W_last:
                                    //   G_cached == NN(W_last)
                                    // Shared by nn_pricing and nn_outsourcing (hold_last).
                                    // Refreshed by compute_policy only when state is inside.
    bool   G_at_W_valid;            // true if G_cached is also NN(W) (i.e. W_last == W).
                                    // Used by nn_outsourcing_inward to avoid a fresh NN
                                    // evaluation when inside the state space.
    double sum_Z_h;                 // Σ Z[DIM_K..2*DIM_K-1]  (horizontal buffer total).
                                    // Filled by compute_Z_and_W; reused by policies that
                                    // need the horizontal route's aggregated Z (BCP /
                                    // HighLow pricing, WMP scheduling, threshold /
                                    // hardstop outsourcing) to avoid recomputing.
};

// ---------------------------------------------------------------------------
// PolicyOutputs: what the policy functions produce
// ---------------------------------------------------------------------------
struct PolicyOutputs {
    // pricing_control[r]: additive adjustment to arrival rate for route r.
    // Effective arrival rate = LAMBDA[r] + pricing_control[r].
    double pricing_control[N_ROUTES];  // shape (4,)

    // outsourcing_control:
    //   [0..3]: route outsourcing flags (0=no outsourcing, 1=outsource, 2=hardstop)
    //   [4..6]: idle station flags (internal, used by idling_policy only)
    int outsourcing_control[N_OUTSOURCE_CTRL];  // shape (7,)

    // scheduling_control[kk]: assignment for station kk (0,1,2).
    //   (kk+1)       = serve vertical buffer buf[kk]
    //   (DIM_K+kk+1) = serve horizontal buffer buf[DIM_K+kk]
    //   0             = idle
    int scheduling_control[DIM_K];  // shape (3,)
};

// ---------------------------------------------------------------------------
// Function declarations
// ---------------------------------------------------------------------------

// compute_Z_and_W: compute scaled state Z and workload W from integer states.
// Python: calculate_scaled_queues_and_workload() in functions_general.py
void compute_Z_and_W(const int states[N_BUFFERS], PolicyInputs& inputs);

// ---- Pricing policies ----

// nn_pricing: NNPricingPolicy with pricing_outside='hold_last_price'.
// Formula: pricing_control[r] = -0.5 * sum_k( G[k] * NN_PRICING_MATRIX[r][k] ) * INV_SQRT_N
//   where G[DIM_K] = nn_predict(inp.W_last)
// Python: NNPricingPolicy.calculate_pricing_control() (functions_general.py lines 58-64)
void nn_pricing(const PolicyInputs& inputs, double pricing_control[N_ROUTES]);

// bcp_linear_pricing: Z-space linear feedback (BCPLinearPricingPolicy), compact form.
// pricing_control[r] = (phi_r * Σ Z_own[r] + phi_{-r} * Σ Z_other[r]) * INV_SQRT_N
//   where (phi_r, phi_{-r}) = (RC_PRICING_PHI[r][0], RC_PRICING_PHI[r][1]).
// "Own" buffers: Z[r] for vertical routes (r<DIM_K), Z[DIM_K..2*DIM_K-1] for horizontal route.
// Python: BCPLinearPricingPolicy.calculate_pricing_control()
void bcp_linear_pricing(const PolicyInputs& inputs, double pricing_control[N_ROUTES]);

// high_low_pricing: two-level drift based on the (scaled) total queue length.
// pricing_control[r] = (Z_sum[r] > RC_HL_PARA[r][0]) ? RC_HL_PARA[r][2] : RC_HL_PARA[r][1]
//                      * INV_SQRT_N
//   Z_sum[r] = Z[r]                       for vertical routes
//   Z_sum[r] = Σ Z[DIM_K..2*DIM_K-1]      for the horizontal route
// RC_HL_PARA[r][0] is stored in Z-scale (i.e. integer-count threshold already
// multiplied by INV_SQRT_N at benchmark-search time).
// Python: HighLowPricingPolicy.calculate_pricing_control()
void high_low_pricing(const PolicyInputs& inputs, double pricing_control[N_ROUTES]);

// ---- Outsourcing policies ----

// nn_outsourcing_hold_last_no_takeover: NNOutsourcingPolicy with
//   idle_new_reflection = 'epsilon_band_hold_last_outsource_simultaneously_no_takeover'
//
// Uses W_last to query NN for gradient G.  Outward band: [threshold, threshold+eps].
// Regions:
//   Inside state space (W < threshold): no outsourcing
//   Within outward band & exactly 1 face violated: NN-guided workload solution
//     (off-diagonal faces first apply a safety-stock guard mirroring
//      idling_policy: if idling station kk_I would starve its downstream
//      buffer, force outsourcing instead of idling; kk_I=DIM_K-1 has no
//      downstream and is treated as safe)
//   Outside band (W > threshold+eps) or multiple faces violated: forced
//     outsourcing — NN cost comparison per face among outsourcing options only
//     (diagonal: route kk vs horizontal DIM_K; off-diagonal: always route kk_U,
//      idling excluded)
//
// Python: NNOutsourcingPolicy.calculate_outsourcing_control() (lines 489-553)
void nn_outsourcing_hold_last_no_takeover(const PolicyInputs& inp,
                                          int outsourcing_control[N_OUTSOURCE_CTRL]);

// nn_outsourcing_inward: NNOutsourcingPolicy with
//   idle_new_reflection = 'epsilon_band_inward_outsource_simultaneously'
//
// Uses current W (not W_last) to query NN for gradient G.
// Inward band: [threshold-eps, threshold].
// Regions:
//   Far inside state space (W < threshold-eps): no outsourcing
//   Within inward band & exactly 1 face approaching: NN-guided workload solution
//   Outside state space (W >= threshold) or multiple faces in band: forced outsourcing
//
// Python: NNOutsourcingPolicy.calculate_outsourcing_control() (lines 368-429)
void nn_outsourcing_inward(const PolicyInputs& inp,
                           int outsourcing_control[N_OUTSOURCE_CTRL]);

// threshold_outsourcing: ThresholdBasedPolicy.
// Outsource route r when Z_groups[r] >= RC_OUTSOURCE_THRESHOLD * B_VEC[r].
// Python: ThresholdBasedPolicy.calculate_outsourcing_control()
void threshold_outsourcing(const PolicyInputs& inputs,
                           int outsourcing_control[N_OUTSOURCE_CTRL]);

// single_pushing_outsourcing: SinglePushingPerFacePolicy with
//   idle_new_reflection = 'epsilon_band_hold_last_outsource_simultaneously_no_takeover'
// Outward band [threshold, threshold+eps].  No NN guidance:
//   mask_workload_solu: diagonal face kk → outsource route kk; off-diagonal → idle kk_I
//   mask_forced: diagonal → outsource kk; off-diagonal → outsource kk_U
// Python: SinglePushingPerFacePolicy.calculate_outsourcing_control() (lines 631-707)
void single_pushing_outsourcing(const PolicyInputs& inputs,
                                int outsourcing_control[N_OUTSOURCE_CTRL]);

// single_pushing_outsourcing_inward: SinglePushingPerFacePolicy with
//   idle_new_reflection = 'epsilon_band_inward_outsource_simultaneously'
// Inward band [threshold-eps, threshold].  No NN guidance:
//   mask_workload_solu: diagonal face kk → outsource route kk; off-diagonal → idle kk_I
//   mask_forced: diagonal → outsource kk; off-diagonal → outsource kk_U
// Python: SinglePushingPerFacePolicy.calculate_outsourcing_control() (lines 577-629)
void single_pushing_outsourcing_inward(const PolicyInputs& inputs,
                                       int outsourcing_control[N_OUTSOURCE_CTRL]);

// ---- Scheduling policies ----

// wmp_scheduling: WMPObj2 workload-based scheduling.
//
// For each station kk:
//   front_value[kk] = (Z[kk] + lambda_tilde*(Z[kk] - beta_tilde_v)) * w_arr[kk]
//   back_value[kk]  = depends on kk:
//     kk < DIM_K-1: (Z[DIM_K+kk] - Z[DIM_K+kk+1]) * w_arr[DIM_K+kk]
//     kk = DIM_K-1: (Z[2*DIM_K-1] + lambda_tilde*(sum(Z[DIM_K:2*DIM_K]) - beta_tilde_h)) * w_arr[2*DIM_K-1]
//   pos[kk] = (front_value[kk] > back_value[kk])
//   → if pos[kk]: serve buf[kk] (vertical); else: serve buf[DIM_K+kk] (horizontal)
//
// Then work-conserving overrides (if one buffer empty, serve the other).
//
// Python: WMPObj2Policy.calculate_scheduling_control() (lines 156-178)
//         + SchedulingPolicy._assign_scheduling_control() (lines 135-151)
void wmp_scheduling(const PolicyInputs& inputs, int scheduling_control[DIM_K]);

// lrs_queue_len_scheduling: LeastRelativeSlackQueueLenPolicy.
// For station kk: pos[kk] = Z[kk]/B_VEC[kk] > cumsum(Z[DIM_K..DIM_K+kk])/B_VEC[DIM_K]
// Python: LeastRelativeSlackQueueLenPolicy.calculate_scheduling_control()
void lrs_queue_len_scheduling(const PolicyInputs& inputs, int scheduling_control[DIM_K]);

// lrs_delay_time_scheduling: LeastRelativeSlackDelayTimePolicy.
// For station kk: pos[kk] = delay_time[kk]/DELAY_BOUND[kk] > delay_time[DIM_K+kk]/DELAY_BOUND[DIM_K]
// Python: LeastRelativeSlackDelayTimePolicy.calculate_scheduling_control()
void lrs_delay_time_scheduling(const PolicyInputs& inputs, int scheduling_control[DIM_K]);

// hardstop_finite_buffer: emergency guard — forces outsourcing if total scaled queue
// length exceeds HARDSTOP_MULTIPLE * B_VEC[r] for any route r.
// Python: hardstop_finite_buffer() in functions_general.py (lines 621-638)
void hardstop_finite_buffer(const PolicyInputs& inp,
                            int outsourcing_control[N_OUTSOURCE_CTRL]);

// idling_policy: post-processing to enforce work-conserving and outsourcing-idle rules.
//
// CC3 specifics (apply_exit_buffer='N', apply_safety_stock='Always', safety_stock=-1):
//   1. exit_buffer='N': if outsourcing_control[N_ROUTES+kk] == 1, set scheduling_control[kk]=0
//   2. safety_stock=-1: ss = -1*INV_SQRT_N + 1e-4 < 0 → pos_Z_safety always false → no override
//   3. Trim: simulation only reads outsourcing_control[0..N_ROUTES-1] afterward
//      (columns N_ROUTES..N_ROUTES+DIM_K-1 are effectively cleared/unused after this)
//
// Python: idling_policy() in functions_general.py (lines 640-691)
void idling_policy(const PolicyInputs& inp, PolicyOutputs& out);

// compute_policy: convenience wrapper that:
//   1. Computes Z and W from integer states
//   2. Updates W_last if current state is inside the state space
//   3. Refreshes G_last = NN(W_last) on the same condition (so outside steps
//      reuse the cached G_last, avoiding a NN call)
//   4. Calls nn_pricing / wmp_scheduling / nn_outsourcing based on RC_* modes
//   5. Calls hardstop_finite_buffer and idling_policy
//
// W_last[DIM_K], G_last[DIM_K]: maintained across iterations in run_one_path.
//   W_last updated here: if current W is inside state space → W_last = W.
//   G_last updated here: on the same condition → G_last = NN(W).
//   When outside, both stay unchanged and sub-policies use the cached values.
//   G_last must be pre-seeded by the caller (nn_predict(W_last, G_last))
//   when NN pricing or NN outsourcing is in use.
//
// Python: policy() function in functions_general.py (lines 738-781)
void compute_policy(const int states[N_BUFFERS],
                    const double delay_time[N_BUFFERS],
                    double W_last[DIM_K],
                    double G_last[DIM_K],
                    PolicyOutputs& outputs);
