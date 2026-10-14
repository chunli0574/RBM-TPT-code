/*
 * policy.h
 *
 * Declarations for the three policy functions and the data structures
 * they use to communicate with the simulation loop.
 *
 * Python equivalent:
 *   BCPLinearPricingPolicy  (class in to_cpp_single_parameter.py, lines 46-49)
 *   WMPObj2Policy           (class in to_cpp_single_parameter.py, lines 114-123)
 *   ThresholdBasedPolicy    (class in to_cpp_single_parameter.py, lines 181-195)
 *   _assign_scheduling_control (method in SchedulingPolicy, lines 90-111)
 *
 * In Python these are classes.  Here we use plain structs for data and
 * plain functions for logic, which is simpler for a Python reader to follow.
 *
 * POLICY OVERVIEW:
 *
 *   1. Pricing policy (BCPLinearPricingPolicy):
 *      Adjusts how fast customers arrive based on queue lengths.
 *      If queues are long, reduce demand (pricing_control < 0).
 *
 *   2. Outsourcing policy (ThresholdBasedPolicy):
 *      If total queue length for a route exceeds a threshold, outsource that route.
 *      Outsourcing means: new arrivals for that route go to an external server
 *      (so we stop accepting arrivals internally = set arrival rate to 0).
 *
 *   3. Scheduling policy (WMPObj2Policy):
 *      Each station decides which buffer to serve next.
 *      Station 1 chooses between buf[0] and buf[3].
 *      Station 2 chooses between buf[2] and buf[1].
 */

#pragma once

#include "params.h"

// ---------------------------------------------------------------------------
// PolicyInputs: what the policy functions receive
// ---------------------------------------------------------------------------
// Python equivalent: the Z and W arguments passed to calculate_*_control()

struct PolicyInputs {
    double Z[N_BUFFERS];          // scaled queue lengths: Z[k] = states[k] * INV_SQRT_N
                                  // Python: Z = states * inv_sqrt_n
    double W[N_ROUTES];           // workload: W[i] = sum_k M_MATRIX[i][k] * Z[k]
                                  // Python: W = Z @ M_matrix.T
    double Z_route_total[N_ROUTES]; // per-route total scaled queue:
                                  //   Z_route_total[r] = sum of Z[k] over route-r's buffers
                                  //   (= SUM_OP[r] @ Z).  Precomputed in compute_Z_and_W
                                  //   and reused by bcp_linear_pricing, high_low_pricing,
                                  //   threshold_outsourcing, hardstop_finite_buffer.
    double delay_time[N_BUFFERS]; // head-of-line sojourn time per buffer (time in system so far)
                                  // delay_time[k] = current_t - entry_time of oldest job in buf k
                                  // = 0.0 if buf k is empty
                                  // Used by LRS delay-time scheduling policies.
                                  // Python: delay_time[:, k] passed to scheduling policies
    double G_cached[N_ROUTES];    // cached NN output G = NN(W_last); populated by compute_policy
                                  // (NN policies only — skipped entirely when no NN policy is active).
                                  // Shared by nn_pricing and nn_outsourcing_hold_last_no_takeover;
                                  // refreshed only when the state is inside the state space.
    bool   G_at_W_valid;          // true iff W_last == W (inside), in which case G_cached is also
                                  // NN(W). Used by nn_outsourcing_inward to skip a redundant NN call.
    // Note: W_last itself is maintained in run_one_path (simulation.cpp) and passed to
    // compute_policy directly; it's not stored in PolicyInputs because sub-policies only
    // ever need G_cached (= NN(W_last)), not W_last itself.
};

// ---------------------------------------------------------------------------
// PolicyOutputs: what the policy functions produce
// ---------------------------------------------------------------------------
// Python equivalent: the three arrays returned by policy()

struct PolicyOutputs {
    // pricing_control[r]: additive adjustment to arrival rate for route r.
    // Effective arrival rate = LAMBDA[r] + pricing_control[r].
    // Can be negative (reduce demand) or positive (increase demand).
    // Python: pricing_control  shape (M, 2)
    double pricing_control[N_ROUTES];

    // outsourcing_control[r]: whether to outsource route r, and which zone.
    //   0   = no outsourcing (normal operation)
    //   1.0 = outsourcing from epsilon band, single face (band cost)
    //   1.5 = outsourcing from forced zone (outside band or multiple faces) (outer cost)
    //   2.0 = hardstop (emergency outsourcing)
    // Indices 2 and 3 are used for idling (L1/L2) by SinglePushing/NN policies; always 0 or 1.
    // Python: outsourcing_control  shape (M, 4)
    double outsourcing_control[4];

    // scheduling_control[s]: which buffer station s should serve this step.
    // For station 0 (Station1): 1 = serve buf[0],  4 = serve buf[3],  0 = idle
    // For station 1 (Station2): 2 = serve buf[1],  3 = serve buf[2],  0 = idle
    // These codes match exactly those in Python's _assign_scheduling_control().
    // Python: scheduling_control  shape (M, 2)
    int scheduling_control[N_ROUTES];
};

// ---------------------------------------------------------------------------
// Function declarations
// ---------------------------------------------------------------------------

// compute_Z_and_W: compute scaled state Z and workload W from integer states.
// Python: calculate_scaled_queues_and_workload() in to_cpp_single_parameter.py
void compute_Z_and_W(const int states[N_BUFFERS], PolicyInputs& inputs);

// ---- Pricing policies ----

// bcp_linear_pricing: Z-space linear feedback with own/other simplification (eq. (98)).
// Formula: pricing_control[j] = (phi_own[j] * sum_{k in route j} Z[k]
//                              + phi_other[j] * sum_{k not in route j} Z[k]) * INV_SQRT_N
// PRICING_PARA[j][0] = phi_own[j], PRICING_PARA[j][1] = phi_other[j].
// Python: BCPLinearPricingPolicy.calculate_pricing_control()
void bcp_linear_pricing(const PolicyInputs& inputs, double pricing_control[N_ROUTES]);

// high_low_pricing: two-level pricing based on Z-space total queue per route.
// pricing_control[r] = HL_PARA[r][low_or_high] * INV_SQRT_N
//   where low_or_high depends on (z_total_route > HL_PARA[r][0]),
//   with z_total_route = sum of Z[k] over route-r buffers and HL_PARA[r][0]
//   being the Z-space threshold (real-valued).
// Python: HighLowPricingPolicy.calculate_pricing_control() (functions_general.py lines 79-88)
void high_low_pricing(const PolicyInputs& inputs, double pricing_control[N_ROUTES]);

// ---- Outsourcing policies ----

// threshold_outsourcing: decide whether to outsource each route.
// Formula: outsource route r if (Z[0]+Z[1] >= OUTSOURCE_THRESHOLD * B_VEC[0])
// Python: ThresholdBasedPolicy.calculate_outsourcing_control()
void threshold_outsourcing(const PolicyInputs& inputs, double outsourcing_control[4]);

// ---- Scheduling policies ----

// wmp_scheduling: WMPObj2 workload-based scheduling (proposed_scheduling_limiting == 'N').
// Python: WMPObj2Policy.calculate_scheduling_control()
//         + SchedulingPolicy._assign_scheduling_control()
void wmp_scheduling(const PolicyInputs& inputs, int scheduling_control[N_ROUTES]);

// wmp_scheduling_limiting: simplified limiting formula (proposed_scheduling_limiting == 'Y').
// pos[0] = (Z[2]+Z[3] < beta_tilde_1),  pos[1] = (Z[0]+Z[1] < beta_tilde_1)
void wmp_scheduling_limiting(const PolicyInputs& inputs, int scheduling_control[N_ROUTES]);

// lrs_queue_len_scheduling: LeastRelativeSlackQueueLenPolicy — no parameters needed.
// Station 1: serve buf[0] if Z[0]/B_VEC[0] > (Z[2]+Z[3])/B_VEC[1]
// Station 2: serve buf[2] if Z[2]/B_VEC[1] > (Z[0]+Z[1])/B_VEC[0]
// Python: LeastRelativeSlackQueueLenPolicy.calculate_scheduling_control() (lines 175-183)
void lrs_queue_len_scheduling(const PolicyInputs& inputs, int scheduling_control[N_ROUTES]);

// lrs_delay_time_scheduling: LeastRelativeSlackDelayTimePolicy — uses delay_time.
// Station 1: serve buf[0] if delay_time[0]/DELAY_BOUND[0] > delay_time[3]/DELAY_BOUND[1]
// Station 2: serve buf[2] if delay_time[2]/DELAY_BOUND[1] > delay_time[1]/DELAY_BOUND[0]
// Python: LeastRelativeSlackDelayTimePolicy.calculate_scheduling_control() (lines 165-173)
void lrs_delay_time_scheduling(const PolicyInputs& inputs, int scheduling_control[N_ROUTES]);

// lrs_remain_delay_scheduling: LeastRelativeSlackRemainDelayTimePolicy — uses delay_time.
// Adds expected remaining service time to delay before normalizing.
// Python: LeastRelativeSlackRemainDelayTimePolicy.calculate_scheduling_control() (lines 185-193)
void lrs_remain_delay_scheduling(const PolicyInputs& inputs, int scheduling_control[N_ROUTES]);

// hardstop_finite_buffer: emergency safety guard — forces outsourcing if total
// scaled queue length exceeds HARDSTOP_MULTIPLE * B_VEC[r].
// Python: hardstop_finite_buffer() in to_cpp_single_parameter.py (lines 229–256)
// do_hardstop = 'Y': raises ValueError in Python if triggered; assert(false) in C++.
// For a stable policy this should NEVER fire; its presence is purely defensive.
// Behavior: EXACTLY the same as Python (minus the ValueError message detail).
void hardstop_finite_buffer(const PolicyInputs& inp, double outsourcing_control[4]);

// single_pushing_outsourcing: W-space face-based outsourcing policy.
// Implements the 'epsilon_band_hold_last_outsource_simultaneously_no_takeover' branch
// of SinglePushingPerFacePolicy.calculate_outsourcing_control() (lines 940–991
// of functions_general.py).
//
// The W-space is divided into:
//   Inside state space (no face violated):  no outsourcing
//   In epsilon band, 1 face violated:       workload solution (may idle a station)
//   Outside band or multiple faces violated: forced outsourcing (U1 or U2 only)
//
// outsourcing_control columns (same meaning as Python):
//   [0] = outsource route 1 (U1)
//   [1] = outsource route 2 (U2)
//   [2] = idle station 1    (L1) — used by F2 in band; cleared to 0 by idling_policy after
//   [3] = idle station 2    (L2) — used by F5 in band; cleared to 0 by idling_policy after
//
// Behavior: EXACTLY the same as Python for idle_new_reflection =
//   'epsilon_band_hold_last_outsource_simultaneously_no_takeover'.
void single_pushing_outsourcing(const PolicyInputs& inp, double outsourcing_control[4]);
// idle_new_reflection == 'epsilon_band_inward_outsource_simultaneously'
void single_pushing_outsourcing_inward(const PolicyInputs& inp, double outsourcing_control[4]);

// ---- NN policies ----

// nn_pricing: NNPricingPolicy with pricing_outside='hold_last_price'.
// Formula: pricing_control[j] = -0.5 * (G @ NN_pricing_matrix.T)[j] * INV_SQRT_N
//   where G = inp.G_cached (= NN(W_last), refreshed once per inside step in compute_policy).
// Python: NNPricingPolicy.calculate_pricing_control() (functions_general.py lines 53-60)
void nn_pricing(const PolicyInputs& inputs, double pricing_control[N_ROUTES]);

// nn_outsourcing_hold_last_no_takeover: NNOutsourcingPolicy with
//   idle_new_reflection = 'epsilon_band_hold_last_outsource_simultaneously_no_takeover'
//
// Uses inp.G_cached (= NN(W_last), the NN gradient at the last inside-state workload).
// Band is defined outward from the boundary (same epsilon band as single_pushing).
// Regions:
//   Inside & one face in epsilon band → NN-guided workload solution (same formulas)
//   Inside & multiple faces or outside band → forced outsourcing (no NN)
// Python: NNOutsourcingPolicy.calculate_outsourcing_control(), lines 774-838
void nn_outsourcing_hold_last_no_takeover(const PolicyInputs& inp, double outsourcing_control[4]);

// nn_outsourcing_inward: NNOutsourcingPolicy with
//   idle_new_reflection = 'epsilon_band_inward_outsource_simultaneously'
//
// Uses CURRENT W (inp.W) to query the NN.
// Band is defined inward from the boundary: fires when W is close to boundary from inside.
// Regions:
//   Inside & one face within epsilon inward → NN-guided workload solution
//   Inside & multiple faces or outside state space → forced outsourcing
// Python: NNOutsourcingPolicy.calculate_outsourcing_control(), lines 659-716
void nn_outsourcing_inward(const PolicyInputs& inp, double outsourcing_control[4]);

// idling_policy: post-processing step that overrides scheduling decisions to
// prevent pipeline starvation and to enforce outsourcing-based idle rules.
//
// Python: idling_policy() in to_cpp_single_parameter.py (lines 258–331)
//
// IMPLEMENTED (matches our Python configuration exactly):
//   apply_exit_buffer  = 'N'      → if outsourcing_control[2 or 3] == 1, set station idle
//                                    (for ThresholdBasedPolicy this never fires)
//   apply_safety_stock = 'Always' → anti-starvation override:
//     if buf[1] is empty AND buf[0] is not empty → station 1 must serve buf[0]
//     if buf[3] is empty AND buf[2] is not empty → station 2 must serve buf[2]
//
// NOT IMPLEMENTED (other branches of the if-else):
//   apply_exit_buffer = 'Y'          (TODO if needed)
//   apply_safety_stock = 'N' / 'Always_Inside' / 'Always_Gated' / 'Cautious'  (TODO)
//   idle_new_reflection = 'Y'        (skipped — our config uses 'epsilon_band_...')
void idling_policy(const PolicyInputs& inp, PolicyOutputs& out);

// compute_policy: convenience wrapper that calls all three policies + idling_policy.
// Dispatches based on PRICING_MODE, SCHEDULING_MODE, OUTSOURCING_MODE in params.h.
// Python equivalent: the policy() function in to_cpp_single_parameter.py (lines 384–430)
//
// delay_time[k]: head-of-line sojourn time per buffer (0.0 if empty).
//   Used by LRS_DELAY_TIME and LRS_REMAIN_DELAY scheduling modes.
//
// W_last[N_ROUTES]: last workload when state was inside the state space.
//   Updated in-place: if current W is inside → W_last is set to current W.
//   Used by NN_PRICING and NN_HOLD_LAST_NO_TAKEOVER modes.
//   Ignored by all non-NN modes (but still maintained for forward compatibility).
//   Python: last_states / W_last updated in policy() before calling NN policies.
// G_last[N_ROUTES]: cached NN output G = NN(W_last).
//   Updated in-place: refreshed via a single nn_predict call only when the state
//   is inside the state space; otherwise preserved from the previous step.
//   Used by NN pricing / outsourcing sub-policies (via inp.G_cached) to avoid
//   redundant NN evaluations both within a step and across outside steps.
void compute_policy(const int states[N_BUFFERS],
                    const double delay_time[N_BUFFERS],
                    double W_last[N_ROUTES],
                    double G_last[N_ROUTES],
                    PolicyOutputs& outputs);
