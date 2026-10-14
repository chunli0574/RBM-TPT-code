/*
 * policy.cpp
 *
 * Implementations of the three policy functions.
 *
 * Python equivalent sections:
 *   - compute_Z_and_W     → calculate_scaled_queues_and_workload() (lines 220-224)
 *   - bcp_linear_pricing  → BCPLinearPricingPolicy.calculate_pricing_control() (lines 46-49)
 *   - threshold_outsourcing → ThresholdBasedPolicy.calculate_outsourcing_control() (lines 181-195)
 *   - wmp_scheduling      → WMPObj2Policy.calculate_scheduling_control() (lines 114-123)
 *                           + _assign_scheduling_control() (lines 90-111)
 *   - compute_policy      → policy() function (lines 384-431)
 *
 * SIMPLIFICATIONS vs. Python:
 *   - No identify_mode() (inside/outside state space check).  TODO: add in step 4.
 *   - No idling_policy() safety stock / exit buffer logic.     TODO: add in step 4.
 *   - No hardstop_finite_buffer().                             TODO: add after core loop works.
 *   - Single path (M=1): all arrays are 1D, no matrix batch ops.
 */

#include "policy.h"
#include "nn_model.h"
#include "run_config.h"
#include "profiling.h"   // optional TIC/TOC (zero overhead when disabled)
#include <cmath>     // for sqrt(), fabs()
#include <algorithm> // for std::max()
#include <cassert>   // for assert()
#include <cstdio>    // for fprintf(stderr, ...) in hardstop_finite_buffer

// ---------------------------------------------------------------------------
// compute_Z_and_W
// ---------------------------------------------------------------------------
// Python:
//   Z = states * inv_sqrt_n             (calculate_scaled_queues)
//   W = Z @ M_matrix.T                  (calculate_workload)
//
// In Python these use NumPy vectorized ops over M paths simultaneously.
// Here we use simple loops because M=1.

void compute_Z_and_W(const int states[N_BUFFERS], PolicyInputs& inp) {
    // Step 1: scale queue lengths to get Z
    // Z[k] = states[k] / sqrt(n)   (divides by 20 when n=400)
    for (int k = 0; k < N_BUFFERS; k++) {
        inp.Z[k] = states[k] * INV_SQRT_N;
    }

    // Step 2: compute workload W = M_MATRIX @ Z
    // W[i] = sum over k of: M_MATRIX[i][k] * Z[k]
    for (int i = 0; i < N_ROUTES; i++) {
        inp.W[i] = 0.0;
        for (int k = 0; k < N_BUFFERS; k++) {
            inp.W[i] += M_MATRIX[i][k] * inp.Z[k];
        }
    }

    // Step 3: per-route total scaled queue = SUM_OP @ Z
    // W-network fixed layout: route 0 owns buf[0..1], route 1 owns buf[2..3].
    // Precomputed here once so sub-policies (bcp_linear, high_low, threshold,
    // hardstop_finite_buffer) can read inp.Z_route_total[r] instead of redoing
    // the SUM_OP[r] @ Z dot product.
    inp.Z_route_total[0] = inp.Z[0] + inp.Z[1];
    inp.Z_route_total[1] = inp.Z[2] + inp.Z[3];
}

// ---------------------------------------------------------------------------
// bcp_linear_pricing
// ---------------------------------------------------------------------------
// Compact formula using the simplification φ_{k,(k',l')} = φ_k if k'=k else φ_{-k}
// (eq. (98) of the paper). For each route j:
//   z_own[j]   = sum of Z[k] for buffers belonging to route j   (via SUM_OP[j])
//   z_other[j] = z_total - z_own[j]                             (sum of Z for non-route-j buffers)
//   pricing_control[j] = (phi_own[j] * z_own[j] + phi_other[j] * z_other[j]) * INV_SQRT_N
//
// RC_PRICING_PARA[j][0] = phi_own[j],  RC_PRICING_PARA[j][1] = phi_other[j]
//
// Intuition: pricing_control < 0 means "reduce arrival rate" (price up),
//            pricing_control > 0 means "increase arrival rate" (price down).
// The effective arrival rate is clamped to ≥ 0 later in simulation.cpp.

void bcp_linear_pricing(const PolicyInputs& inp, double pricing_control[N_ROUTES]) {
    // z_total = sum of all scaled queue lengths (= sum of per-route totals)
    const double z_total = inp.Z_route_total[0] + inp.Z_route_total[1];

    for (int j = 0; j < N_ROUTES; j++) {
        // z_own[j] / z_other[j] precomputed via inp.Z_route_total
        double z_own   = inp.Z_route_total[j];
        double z_other = z_total - z_own;
        pricing_control[j] = (RC_PRICING_PARA[j][0] * z_own
                            + RC_PRICING_PARA[j][1] * z_other) * INV_SQRT_N;
    }
}

// ---------------------------------------------------------------------------
// high_low_pricing
// ---------------------------------------------------------------------------
// Python: HighLowPricingPolicy.calculate_pricing_control() (functions_general.py lines 79-88)
//
//   point = Z * sqrt_n  (= integer queue lengths)
//   HL_para = pricing_para.reshape((2, 3))  — [threshold, low_drift, high_drift] per route
//   pricing_control = where(point @ SUM_OP.T > HL_para[:,0], HL_para[:,2], HL_para[:,1])
//                     * inv_sqrt_n
//
// C++ convention: we keep everything in Z-space by pre-scaling HL_PARA[r][0]:
//   HL_PARA[r][0] = Z-space threshold (real-valued; = Q_loop × Q_BOUND_MAX × rho × INV_SQRT_N)
//   HL_PARA[r][1] = low drift,   HL_PARA[r][2] = high drift
//   Comparison: z_total_route > HL_PARA[r][0]  (instead of int_queue > int_threshold)
// Both sides of the original comparison are scaled by the same positive factor INV_SQRT_N,
// so the truth value is unchanged; this saves the per-step division by INV_SQRT_N.
// (benchmark_search_main.cpp::highlow_pricing_search computes
//  z_thresh = Q_loop * Q_BOUND_MAX * rho * INV_SQRT_N directly and writes it
//  into RC_HL_PARA[r][0] — no integer rounding.)

void high_low_pricing(const PolicyInputs& inp, double pricing_control[N_ROUTES]) {
    for (int r = 0; r < N_ROUTES; r++) {
        // z_total_route = total Z-space queue for route r (precomputed in compute_Z_and_W)
        double z_total_route = inp.Z_route_total[r];

        // Choose drift level based on threshold (compared in Z-space)
        double drift = (z_total_route > RC_HL_PARA[r][0]) ? RC_HL_PARA[r][2] : RC_HL_PARA[r][1];
        pricing_control[r] = drift * INV_SQRT_N;  // Python: ... * inv_sqrt_n
    }
}

// ---------------------------------------------------------------------------
// threshold_outsourcing
// ---------------------------------------------------------------------------
// Python ThresholdBasedPolicy.calculate_outsourcing_control():
//   Z_groups = Z @ sum_queue_len_operator.T
//   pos_matrix = Z_groups >= outsourcing_ratio_threshold * b_vec
//   outsourcing_control[:, 0] = where(pos_matrix[:, 0], 1, 0)
//   outsourcing_control[:, 1] = where(pos_matrix[:, 1], 1, 0)
//
// Z_groups[r] = total scaled queue for route r:
//   Z_groups[0] = Z[0] + Z[1]   (route 1: buf0 + buf1)
//   Z_groups[1] = Z[2] + Z[3]   (route 2: buf2 + buf3)
//
// Outsource route r when total queue exceeds RC_OUTSOURCE_THRESHOLD * B_VEC[r].

void threshold_outsourcing(const PolicyInputs& inp, double outsourcing_control[4]) {
    // Initialize all to 0 (no outsourcing)
    for (int c = 0; c < 4; c++) outsourcing_control[c] = 0;

    // Z_groups[r] = Z @ SUM_OP[r]  (precomputed in compute_Z_and_W as inp.Z_route_total[r])
    for (int r = 0; r < N_ROUTES; r++) {
        double Z_group = inp.Z_route_total[r];
        // Outsource if total queue exceeds threshold boundary
        if (Z_group >= RC_OUTSOURCE_THRESHOLD * B_VEC[r]) {
            outsourcing_control[r] = 1;  // columns 0 and 1 only
        }
    }
    // outsourcing_control[2] and [3] stay 0 (used by other policy types in Python)
}

// ---------------------------------------------------------------------------
// assign_scheduling_control  (static helper — used by all scheduling policies)
// ---------------------------------------------------------------------------
// Python: SchedulingPolicy._assign_scheduling_control() (lines 90-111)
//
// All scheduling policies compute two booleans (pos0, pos1) and then call
// this shared helper to:
//   1. Map booleans → scheduling codes: pos0 → 1/4, pos1 → 3/2
//   2. Apply idle overrides when a buffer is empty
//
// Scheduling codes:
//   scheduling_control[0]: 1 = station1 serves buf[0], 4 = buf[3], 0 = idle
//   scheduling_control[1]: 2 = station2 serves buf[1], 3 = buf[2], 0 = idle

static void assign_scheduling_control(bool pos0, bool pos1,
                                       const PolicyInputs& inp,
                                       int scheduling_control[N_ROUTES]) {
    // Map booleans to codes
    // Python: np.where(pos[:, 0], 1, 4) and np.where(pos[:, 1], 3, 2)
    scheduling_control[0] = pos0 ? 1 : 4;
    scheduling_control[1] = pos1 ? 3 : 2;

    // Idle overrides: if a buffer is empty, serve the non-empty one
    // Python: _assign_scheduling_control(), lines 101-109
    bool buf_empty[N_BUFFERS];
    for (int k = 0; k < N_BUFFERS; k++) {
        buf_empty[k] = (inp.Z[k] < 1e-4);
    }

    // Station 1 (buf[0] vs buf[3]):
    if (buf_empty[3]) scheduling_control[0] = 1;
    if (buf_empty[0]) scheduling_control[0] = 4;
    if (buf_empty[0] && buf_empty[3]) scheduling_control[0] = 0;

    // Station 2 (buf[1] vs buf[2]):
    if (buf_empty[1]) scheduling_control[1] = 3;
    if (buf_empty[2]) scheduling_control[1] = 2;
    if (buf_empty[2] && buf_empty[1]) scheduling_control[1] = 0;
}

// ---------------------------------------------------------------------------
// wmp_scheduling
// ---------------------------------------------------------------------------
// Python WMPObj2Policy.calculate_scheduling_control() (lines 114-123):
//   pos[:, 0] = (Z[:,0]-Z[:,1])*w[0] > (Z[:,3]+lam*(Z[:,2]+Z[:,3]-b2))*w[3]
//   pos[:, 1] = (Z[:,2]-Z[:,3])*w[2] > (Z[:,1]+lam*(Z[:,0]+Z[:,1]-b1))*w[1]
//
// RC_WMP_EXTRA_RELATIVE controls how the lambda_tilde-scaled penalty term is formed:
//   0 (original): extra_term_s = lambda_tilde * (Z_other_total - beta_tilde_other)
//   1 (scaled):   extra_term_s = lambda_tilde * (Z_other_total - beta_tilde_other)
//                                / (BETA_CONST * rho)
//   where BETA_CONST = 19/9 and rho = RC_OUTSOURCE_THRESHOLD.

void wmp_scheduling(const PolicyInputs& inp, int scheduling_control[N_ROUTES]) {
    // Extra-term scaling: 1.0 in original mode; 1/(BETA_CONST*rho) in scaled mode.
    const double scale = RC_WMP_EXTRA_RELATIVE
                       ? 1.0 / (BETA_CONST * RC_OUTSOURCE_THRESHOLD)
                       : 1.0;

    // Station 1: prefer buf[0] when route1 early-stage pressure > route2 late-stage pressure
    // (Z[2]+Z[3] = Z_route_total[1], precomputed in compute_Z_and_W)
    double extra0 = RC_LAMBDA_TILDE * (inp.Z_route_total[1] - RC_BETA_TILDE_2) * scale;
    bool pos0 = (inp.Z[0] - inp.Z[1]) * W_ARR[0]
              > (inp.Z[3] + extra0) * W_ARR[3];
    // Station 2: prefer buf[2] when route2 early-stage pressure > route1 late-stage pressure
    // (Z[0]+Z[1] = Z_route_total[0], precomputed in compute_Z_and_W)
    double extra1 = RC_LAMBDA_TILDE * (inp.Z_route_total[0] - RC_BETA_TILDE_1) * scale;
    bool pos1 = (inp.Z[2] - inp.Z[3]) * W_ARR[2]
              > (inp.Z[1] + extra1) * W_ARR[1];

    assign_scheduling_control(pos0, pos1, inp, scheduling_control);
}

// ---------------------------------------------------------------------------
// wmp_scheduling_limiting
// ---------------------------------------------------------------------------
// Python WMPObj2Policy.calculate_scheduling_control() — proposed_scheduling_limiting == 'Y':
//   pos[:, 0] = (Z[:, 2] + Z[:, 3] < beta_tilde_1)
//   pos[:, 1] = (Z[:, 0] + Z[:, 1] < beta_tilde_1)
//
// Station 1 serves buf[0] (route1) when OTHER route's total queue < beta_tilde_1.
// Station 2 serves buf[2] (route2) when OTHER route's total queue < beta_tilde_1.

void wmp_scheduling_limiting(const PolicyInputs& inp, int scheduling_control[N_ROUTES]) {
    // Z[2]+Z[3] = Z_route_total[1], Z[0]+Z[1] = Z_route_total[0]
    // (precomputed in compute_Z_and_W).
    bool pos0 = (inp.Z_route_total[1] < RC_BETA_TILDE_1);
    bool pos1 = (inp.Z_route_total[0] < RC_BETA_TILDE_1);
    assign_scheduling_control(pos0, pos1, inp, scheduling_control);
}

// ---------------------------------------------------------------------------
// lrs_queue_len_scheduling
// ---------------------------------------------------------------------------
// Python: LeastRelativeSlackQueueLenPolicy.calculate_scheduling_control() (lines 175-183)
//
//   pos[:, 0] = Z[:, 0] / b1 > (Z[:, 2] + Z[:, 3]) / b3
//   pos[:, 1] = Z[:, 2] / b3 > (Z[:, 0] + Z[:, 1]) / b1
//
// Intuition: a station serves the route whose CURRENT queue is larger relative
// to that route's allowed total queue (B_VEC[r]).
// No policy parameters needed.

void lrs_queue_len_scheduling(const PolicyInputs& inp, int scheduling_control[N_ROUTES]) {
    // Station 1: buf[0] (route1 stage1) vs buf[3] (route2 stage2)
    // Python: Z[:,0] / b1 > (Z[:,2]+Z[:,3]) / b3
    // (Z[2]+Z[3] = Z_route_total[1], precomputed in compute_Z_and_W)
    bool pos0 = (inp.Z[0] / B_VEC[0]) > (inp.Z_route_total[1] / B_VEC[1]);

    // Station 2: buf[2] (route2 stage1) vs buf[1] (route1 stage2)
    // Python: Z[:,2] / b3 > (Z[:,0]+Z[:,1]) / b1
    // (Z[0]+Z[1] = Z_route_total[0], precomputed in compute_Z_and_W)
    bool pos1 = (inp.Z[2] / B_VEC[1]) > (inp.Z_route_total[0] / B_VEC[0]);

    assign_scheduling_control(pos0, pos1, inp, scheduling_control);
}

// ---------------------------------------------------------------------------
// lrs_delay_time_scheduling
// ---------------------------------------------------------------------------
// Python: LeastRelativeSlackDelayTimePolicy.calculate_scheduling_control() (lines 165-173)
//
//   pos[:, 0] = delay_time[:, 0] / delay_bound1 > delay_time[:, 3] / delay_bound3
//   pos[:, 1] = delay_time[:, 2] / delay_bound3 > delay_time[:, 1] / delay_bound1
//
// delay_time[k] = time head-of-line job in buf[k] has been in the system (0 if empty).
// A station serves the route whose oldest job is relatively MORE delayed vs its deadline.

void lrs_delay_time_scheduling(const PolicyInputs& inp, int scheduling_control[N_ROUTES]) {
    // Station 1: buf[0] (route1) vs buf[3] (route2)
    // Python: delay_time[:,0]/delay_bound1 > delay_time[:,3]/delay_bound3
    bool pos0 = (inp.delay_time[0] / DELAY_BOUND[0]) > (inp.delay_time[3] / DELAY_BOUND[1]);

    // Station 2: buf[2] (route2) vs buf[1] (route1)
    // Python: delay_time[:,2]/delay_bound3 > delay_time[:,1]/delay_bound1
    bool pos1 = (inp.delay_time[2] / DELAY_BOUND[1]) > (inp.delay_time[1] / DELAY_BOUND[0]);

    assign_scheduling_control(pos0, pos1, inp, scheduling_control);
}

// ---------------------------------------------------------------------------
// lrs_remain_delay_scheduling
// ---------------------------------------------------------------------------
// Python: LeastRelativeSlackRemainDelayTimePolicy.calculate_scheduling_control() (lines 185-193)
//
//   pos[:, 0] = (delay_time[:,0] + m1 + m2) / delay_bound1
//             > (delay_time[:,3] + m4) / delay_bound3
//   pos[:, 1] = (delay_time[:,2] + m3 + m4) / delay_bound3
//             > (delay_time[:,1] + m2) / delay_bound1
//
// Adds EXPECTED REMAINING service time before normalizing:
//   buf[0] job still needs: m1 = 1/MUS[0] (stage1) + m2 = 1/MUS[1] (stage2)
//   buf[3] job still needs: m4 = 1/MUS[3] (stage2 only — already passed stage1)
//   buf[2] job still needs: m3 = 1/MUS[2] (stage1) + m4 = 1/MUS[3] (stage2)
//   buf[1] job still needs: m2 = 1/MUS[1] (stage2 only — already passed stage1)

void lrs_remain_delay_scheduling(const PolicyInputs& inp, int scheduling_control[N_ROUTES]) {
    const double m1 = 1.0 / MUS[0];  // expected service time at buf[0]: 1/mu1 = 1.0
    const double m2 = 1.0 / MUS[1];  // expected service time at buf[1]: 1/mu2 = 2.0
    const double m3 = 1.0 / MUS[2];  // expected service time at buf[2]: 1/mu3 = 1.0
    const double m4 = 1.0 / MUS[3];  // expected service time at buf[3]: 1/mu4 = 2.0

    // Station 1: buf[0] (route1, needs m1+m2 more) vs buf[3] (route2, needs m4 more)
    bool pos0 = ((inp.delay_time[0] + m1 + m2) / DELAY_BOUND[0])
              > ((inp.delay_time[3] + m4)       / DELAY_BOUND[1]);

    // Station 2: buf[2] (route2, needs m3+m4 more) vs buf[1] (route1, needs m2 more)
    bool pos1 = ((inp.delay_time[2] + m3 + m4) / DELAY_BOUND[1])
              > ((inp.delay_time[1] + m2)       / DELAY_BOUND[0]);

    assign_scheduling_control(pos0, pos1, inp, scheduling_control);
}

// ---------------------------------------------------------------------------
// hardstop_finite_buffer
// ---------------------------------------------------------------------------
// Python: hardstop_finite_buffer(outsourcing_control, Z) — lines 229–256
//
// Purpose: safety guard to prevent the circular delay-tracking buffer from
// overflowing.  If total scaled queue for route r exceeds
//   HARDSTOP_MULTIPLE * B_VEC[r]  (= 10 * 19/9 ≈ 21.1)
// the system is unstable under the current policy.  Python raises ValueError;
// C++ fires an assert.
//
// Python:
//   Z_groups = Z @ SUM_OP.T   (= [Z[0]+Z[1], Z[2]+Z[3]])
//   pos_matrix = Z_groups >= hardstop_multiple * b_vec
//   outsourcing_control[pos, 0] = 2   (route r → hardstop mode)
//   if any pos: raise ValueError
//
// Behavior: EXACTLY the same as Python (using assert instead of ValueError).
// For a stable policy this never fires in practice.
//
// NOTE: called AFTER single_pushing_outsourcing / threshold_outsourcing, and
//       BEFORE idling_policy — matches Python's call order in policy().

void hardstop_finite_buffer(const PolicyInputs& inp, double outsourcing_control[4]) {
    for (int r = 0; r < N_ROUTES; r++) {
        // Z_group[r] = Z[0]+Z[1]  or  Z[2]+Z[3]  (total scaled queue for route r)
        // Python: Z_groups = Z @ SUM_OP.T  (precomputed in compute_Z_and_W)
        double Z_group = inp.Z_route_total[r];

        // If total queue exceeds hardstop threshold, path is unstable.
        // Python: pos_matrix = Z_groups >= hardstop_multiple * b_vec  → raise ValueError
        // C++: we no longer abort on hardstop — just set hardstop mode (outsourcing=2,
        // which the downstream code treats identically to regular outsourcing=1), rate-
        // limit a warning to stderr so logs aren't flooded, and let the simulation
        // continue. The resulting path will typically violate the delay constraint
        // and be discarded by find_min_cost, but the search run keeps going instead
        // of killing the whole process.
        if (Z_group >= HARDSTOP_MULTIPLE * B_VEC[r]) {
            outsourcing_control[r] = 2;   // Python: outsourcing_control[pos, r] = 2
            thread_local static int hardstop_warn_count = 0;
            constexpr int HARDSTOP_WARN_LIMIT = 5;
            if (hardstop_warn_count < HARDSTOP_WARN_LIMIT) {
                ++hardstop_warn_count;
                fprintf(stderr,
                    "WARNING: hardstop triggered for route %d "
                    "(Z_group=%.4f >= %.4f = HARDSTOP_MULTIPLE * B_VEC). "
                    "Policy may be unstable; simulation continues.%s\n",
                    r, Z_group, HARDSTOP_MULTIPLE * B_VEC[r],
                    (hardstop_warn_count == HARDSTOP_WARN_LIMIT)
                        ? " [further warnings from this thread suppressed]" : "");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// eps_threshold: compute the epsilon-band boundary for face f
// ---------------------------------------------------------------------------
// sign = +1.0 for outward band:  threshold = boundary*(rho + eps)  [relative]
//                                           = boundary*rho + eps*scaling  [absolute]
// sign = -1.0 for inward band:   threshold = boundary*(rho - eps)  [relative]
//                                           = boundary*rho - eps*scaling  [absolute]
//
// RC_EPSILON_RELATIVE == 0 (absolute, default):
//   threshold = FACE_BOUNDARY[f] * RC_OUTSOURCE_THRESHOLD + sign * RC_EPSILON_BAND * EPSILON_SCALING[f]
//   EPSILON_SCALING = {1, 1, sqrt(5), sqrt(5)}  (L2 norm of each face normal)
//
// RC_EPSILON_RELATIVE == 1 (relative):
//   threshold = FACE_BOUNDARY[f] * (RC_OUTSOURCE_THRESHOLD + sign * RC_EPSILON_BAND)
//   epsilon scales proportionally with the face boundary (no EPSILON_SCALING factor)
// ---------------------------------------------------------------------------
static inline double eps_threshold(int f, double sign) {
    if (RC_EPSILON_RELATIVE)
        return FACE_BOUNDARY[f] * (RC_OUTSOURCE_THRESHOLD + sign * RC_EPSILON_BAND);
    else
        return FACE_BOUNDARY[f] * RC_OUTSOURCE_THRESHOLD
             + sign * RC_EPSILON_BAND * EPSILON_SCALING[f];
}

// ---------------------------------------------------------------------------
// single_pushing_outsourcing
// ---------------------------------------------------------------------------
// Python: SinglePushingPerFacePolicy.calculate_outsourcing_control()
//         branch: idle_new_reflection == 'epsilon_band_hold_last_outsource_simultaneously_no_takeover'
//         lines 940–991 of functions_general.py
//
// STEP-BY-STEP LOGIC:
//
//   1. Compute pos_matrix[f] = whether face f is violated
//      Python: pos_matrix = (W @ condition_matrix.T > state_space_boundary * rho)
//
//   2. Count violations → mask_one = (exactly 1 face violated)
//
//   3. Compute epsilon-band membership per face
//      pos_matrix_eps[f] = W @ condition_matrix[f] < FACE_BOUNDARY[f]*rho + RC_EPSILON_BAND*EPSILON_SCALING[f]
//      mask_state_space_eps = all pos_matrix_eps true  (within the extended state space)
//      mask_band   = mask_state_space_eps AND any violation  (in the epsilon band)
//      mask_outside= NOT mask_state_space_eps               (strictly outside band)
//
//   4. Compute zone masks:
//      mask_workload_solu = mask_one AND mask_band          (band + single violation)
//      mask_forced        = (NOT mask_one AND mask_band) OR mask_outside
//
//   5. Apply controls based on which zone and which face:
//      In workload-solution zone:
//        F2 violated → outsourcing_control[2] = 1  (L1: idle station 1)
//        F3 violated → outsourcing_control[0] = 1  (U1: outsource route 1)
//        F4 violated → outsourcing_control[0] = 1  (U1: outsource route 1)
//        F5 violated → outsourcing_control[3] = 1  (L2: idle station 2)
//      In forced zone:
//        F2 violated → outsourcing_control[0] = 1  (U1: outsource route 1)
//        F3 violated → outsourcing_control[0] = 1  (U1: outsource route 1)
//        F4 violated → outsourcing_control[0] = 1  (U1: outsource route 1)
//        F5 violated → outsourcing_control[1] = 1  (U2: outsource route 2)
//
// Columns 2 and 3 (L1, L2) are later processed by idling_policy:
//   apply_exit_buffer='N': if col2==1 → scheduling_control[0]=0 (idle station 1)
//                          if col3==1 → scheduling_control[1]=0 (idle station 2)
//   The outsourcing_control is then trimmed to columns 0,1 before cost calculation.
//
// Behavior: EXACTLY the same as Python.

void single_pushing_outsourcing(const PolicyInputs& inp, double outsourcing_control[4]) {
    // Initialize all to 0 (no outsourcing, no idling)
    for (int c = 0; c < 4; c++) outsourcing_control[c] = 0;

    // ------------------------------------------------------------------
    // Step 1: face violation flags
    // Python: pos_matrix = (W @ self.condition_matrix.T > self.state_space_boundary * self.outsourcing_para[0])
    // pos_matrix[f] = True when face f is violated (workload outside boundary on face f)
    // ------------------------------------------------------------------
    bool pos_matrix[4];
    int n_violated = 0;
    for (int f = 0; f < 4; f++) {
        double lhs = CONDITION_MATRIX[f][0] * inp.W[0]
                   + CONDITION_MATRIX[f][1] * inp.W[1];
        pos_matrix[f] = (lhs > FACE_BOUNDARY[f] * RC_OUTSOURCE_THRESHOLD);
        if (pos_matrix[f]) n_violated++;
    }

    // If no face is violated we are strictly inside the state space → no action.
    // Early exit avoids computing epsilon band unnecessarily.
    if (n_violated == 0) return;

    // ------------------------------------------------------------------
    // Step 2: count violations
    // Python: mask_one = (np.sum(pos_matrix, axis=1) == 1)
    // (n_violated accumulated in Step 1 above.)
    // ------------------------------------------------------------------
    bool mask_one = (n_violated == 1);

    // ------------------------------------------------------------------
    // Step 3: epsilon band check per face
    // Python: pos_matrix_eps = (W @ condition_matrix.T <
    //           state_space_boundary * rho + epsilon_band * [1,1,sqrt5,sqrt5])
    // mask_state_space_eps = all pos_matrix_eps (within extended state space)
    // mask_band   = mask_state_space_eps AND any violation
    // mask_outside= NOT mask_state_space_eps
    // ------------------------------------------------------------------
    bool pos_matrix_eps[4];
    for (int f = 0; f < 4; f++) {
        double lhs = CONDITION_MATRIX[f][0] * inp.W[0]
                   + CONDITION_MATRIX[f][1] * inp.W[1];
        pos_matrix_eps[f] = (lhs < eps_threshold(f, +1.0));
    }
    bool mask_state_space_eps = pos_matrix_eps[0] && pos_matrix_eps[1]
                              && pos_matrix_eps[2] && pos_matrix_eps[3];

    bool mask_band    = mask_state_space_eps;   // (any violation implied since n_violated > 0)
    bool mask_outside = !mask_state_space_eps;

    // ------------------------------------------------------------------
    // Step 4: zone classification
    // Python:
    //   mask_workload_solu = mask_one & mask_band
    //   mask_forced = ((~mask_one) & mask_band) | mask_outside
    // ------------------------------------------------------------------
    bool mask_workload_solu = mask_one && mask_band;
    bool mask_forced        = ((!mask_one) && mask_band) || mask_outside;

    // ------------------------------------------------------------------
    // Step 5a: workload-solution zone actions
    // Python (lines 960–970):
    //   outsourcing_control[mask_workload_solu & pos_matrix[:,2], 2] = 1  F2 → L1
    //   outsourcing_control[mask_workload_solu & pos_matrix[:,1], 0] = 1  F3 → U1
    //   outsourcing_control[mask_workload_solu & pos_matrix[:,0], 0] = 1  F4 → U1
    //   outsourcing_control[mask_workload_solu & pos_matrix[:,3], 3] = 1  F5 → L2
    //
    // F2/F5 safety-stock override: before defaulting to idle, check whether the
    // server's downstream buffer needs replenishment, using the SAME test as
    // idling_policy (safety_empty && !buf_empty). If so, idling is not a valid
    // choice (idling_policy would force the server to serve upstream anyway),
    // so we outsource that face's route instead (F2 → U1, F5 → U2). Otherwise
    // we keep the original idle action (F2 → L1, F5 → L2). F3/F4 are unaffected.
    // Threshold: ss = RC_SAFETY_STOCK * INV_SQRT_N - 1e-4 (idling_policy convention).
    // ------------------------------------------------------------------
    if (mask_workload_solu) {
        const double ss = RC_SAFETY_STOCK * INV_SQRT_N - 1e-4;
        if (pos_matrix[2]) {  // F2: idle station 1 (L1) vs outsource route 1 (U1)
            bool ss_need_1 = (inp.Z[1] < ss) && !(inp.Z[0] < 1e-4);  // safety_empty[1] && !buf_empty[0]
            if (ss_need_1) outsourcing_control[0] = 1;  // U1 (downstream needs safety stock)
            else           outsourcing_control[2] = 1;  // L1 (idle station 1)
        }
        if (pos_matrix[1]) outsourcing_control[0] = 1;  // F3: outsource route 1 (U1)
        if (pos_matrix[0]) outsourcing_control[0] = 1;  // F4: outsource route 1 (U1)
        if (pos_matrix[3]) {  // F5: idle station 2 (L2) vs outsource route 2 (U2)
            bool ss_need_2 = (inp.Z[3] < ss) && !(inp.Z[2] < 1e-4);  // safety_empty[3] && !buf_empty[2]
            if (ss_need_2) outsourcing_control[1] = 1;  // U2 (downstream needs safety stock)
            else           outsourcing_control[3] = 1;  // L2 (idle station 2)
        }
    }

    // ------------------------------------------------------------------
    // Step 5b: forced outsourcing zone actions
    // Python (lines 979–989):
    //   outsourcing_control[mask_forced & pos_matrix[:,2], 0] = 1  F2 → U1
    //   outsourcing_control[mask_forced & pos_matrix[:,1], 0] = 1  F3 → U1
    //   outsourcing_control[mask_forced & pos_matrix[:,0], 0] = 1  F4 → U1
    //   outsourcing_control[mask_forced & pos_matrix[:,3], 1] = 1  F5 → U2
    // Note: F2 in forced zone uses U1 (not L1 as in band zone).
    //       F5 in forced zone uses U2 (not L2 as in band zone).
    // ------------------------------------------------------------------
    if (mask_forced) {
        if (pos_matrix[2]) outsourcing_control[0] = 1.5;  // F2: outsource route 1 (U1), outer
        if (pos_matrix[1]) outsourcing_control[0] = 1.5;  // F3: outsource route 1 (U1), outer
        if (pos_matrix[0]) outsourcing_control[0] = 1.5;  // F4: outsource route 1 (U1), outer
        if (pos_matrix[3]) outsourcing_control[1] = 1.5;  // F5: outsource route 2 (U2), outer
    }
    // cols 2 and 3 (L1, L2) set in workload_solu zone are consumed by idling_policy:
    //   idling_policy sets scheduling_control[0]=0 when col2==1 (L1: idle station 1)
    //   idling_policy sets scheduling_control[1]=0 when col3==1 (L2: idle station 2)
}

// ---------------------------------------------------------------------------
// single_pushing_outsourcing_inward
// ---------------------------------------------------------------------------
// Python: SinglePushingPerFacePolicy.calculate_outsourcing_control()
//         branch: idle_new_reflection == 'epsilon_band_inward_outsource_simultaneously'
//         lines 991–1037 of functions_general.py
//
// Differs from single_pushing_outsourcing (hold_last_no_takeover) only in how
// the face violation and band checks are defined:
//
//   hold_last_no_takeover (outward band):
//     pos_matrix[f]     = W @ cond[f] > boundary * rho            (face crossed outward)
//     pos_matrix_eps[f] = W @ cond[f] < boundary * rho + eps * s  (within extended state space)
//     band = inside extended state space AND any face crossed outward
//
//   inward (inward band):
//     pos_inward[f]     = W @ cond[f] > boundary * rho - eps * s  (within eps of boundary from inside)
//     inside_face[f]    = W @ cond[f] < boundary * rho            (inside original state space)
//     band = inside original state space AND any pos_inward
//
// Control actions in workload-solution / forced zones are identical to
// hold_last_no_takeover (see that function for the action table).

void single_pushing_outsourcing_inward(const PolicyInputs& inp, double outsourcing_control[4]) {
    for (int c = 0; c < 4; c++) outsourcing_control[c] = 0;

    // ------------------------------------------------------------------
    // Step 1: inward face violation flags
    // Python: pos_matrix_inward = W @ cond.T > boundary * rho - epsilon * scaling
    // pos_inward[f] = True when W is within epsilon of face f from inside
    // eps_threshold(f, -1) handles absolute vs relative epsilon via RC_EPSILON_RELATIVE.
    // ------------------------------------------------------------------
    bool pos_inward[4];
    int n_inward = 0;
    for (int f = 0; f < 4; f++) {
        double lhs = CONDITION_MATRIX[f][0] * inp.W[0]
                   + CONDITION_MATRIX[f][1] * inp.W[1];
        pos_inward[f] = (lhs > eps_threshold(f, -1.0));
        if (pos_inward[f]) n_inward++;
    }

    // Early exit: n_inward == 0 ⟹ all pos_inward[f]=false ⟹ lhs < boundary*rho for all f
    // (since eps*scaling > 0), so inside_face[f]=true for all f ⟹ mask_state_space_eps=true,
    // mask_outside=false, mask_band=false → no action needed. Skipping Step 2.
    if (n_inward == 0) return;

    // ------------------------------------------------------------------
    // Step 2: inside original state space check
    // Python: pos_matrix_inward_eps = W @ cond.T < boundary * rho
    // mask_state_space_eps = all(inside_face)
    // mask_band   = inside state space AND any inward violation
    // mask_outside= NOT inside state space
    // ------------------------------------------------------------------
    bool inside_face[4];
    for (int f = 0; f < 4; f++) {
        double lhs = CONDITION_MATRIX[f][0] * inp.W[0]
                   + CONDITION_MATRIX[f][1] * inp.W[1];
        inside_face[f] = (lhs < FACE_BOUNDARY[f] * RC_OUTSOURCE_THRESHOLD);
    }
    bool mask_state_space_eps = inside_face[0] && inside_face[1]
                              && inside_face[2] && inside_face[3];
    bool mask_band    = mask_state_space_eps;   // (any_inward implied since n_inward > 0 above)
    bool mask_outside = !mask_state_space_eps;

    // ------------------------------------------------------------------
    // Step 3: zone classification (same logic as hold_last_no_takeover)
    // ------------------------------------------------------------------
    bool mask_one           = (n_inward == 1);
    bool mask_workload_solu = mask_one && mask_band;
    bool mask_forced        = ((!mask_one) && mask_band) || mask_outside;

    // ------------------------------------------------------------------
    // Step 4a: workload-solution zone — same actions as hold_last_no_takeover
    // Python (lines 1008–1018):
    //   F2 → col2=1 (L1: idle station 1)
    //   F3 → col0=1 (U1: outsource route 1)
    //   F4 → col0=1 (U1: outsource route 1)
    //   F5 → col3=1 (L2: idle station 2)
    // ------------------------------------------------------------------
    if (mask_workload_solu) {
        if (pos_inward[2]) outsourcing_control[2] = 1;  // F2: idle station 1 (L1)
        if (pos_inward[1]) outsourcing_control[0] = 1;  // F3: outsource route 1 (U1)
        if (pos_inward[0]) outsourcing_control[0] = 1;  // F4: outsource route 1 (U1)
        if (pos_inward[3]) outsourcing_control[3] = 1;  // F5: idle station 2 (L2)
    }

    // ------------------------------------------------------------------
    // Step 4b: forced zone — same actions as hold_last_no_takeover
    // Python (lines 1026–1036):
    //   F2 → col0=1 (U1)
    //   F3 → col0=1 (U1)
    //   F4 → col0=1 (U1)
    //   F5 → col1=1 (U2: outsource route 2)
    // ------------------------------------------------------------------
    if (mask_forced) {
        if (pos_inward[2]) outsourcing_control[0] = 1.5;  // F2: outsource route 1 (U1), outer
        if (pos_inward[1]) outsourcing_control[0] = 1.5;  // F3: outsource route 1 (U1), outer
        if (pos_inward[0]) outsourcing_control[0] = 1.5;  // F4: outsource route 1 (U1), outer
        if (pos_inward[3]) outsourcing_control[1] = 1.5;  // F5: outsource route 2 (U2), outer
    }
}

// ---------------------------------------------------------------------------
// idling_policy
// ---------------------------------------------------------------------------
// Python: idling_policy() in to_cpp_single_parameter.py (lines 258–331)
//
// This is called AFTER all three policy functions (pricing, outsourcing,
// scheduling) to apply two post-processing overrides:
//
//   1. Exit-buffer rule (apply_exit_buffer = 'N'):
//      If outsourcing_control[2] or [3] == 1, force the corresponding station
//      to idle.  For ThresholdBasedPolicy columns 2,3 are always 0, so this
//      rule never fires — but the code is correct and consistent with Python.
//
//   2. Safety-stock rule (apply_safety_stock = 'Always', safety_stock = 0):
//      ss = SAFETY_STOCK * INV_SQRT_N + 1e-4  =  0 * 0.05 + 1e-4  =  1e-4
//      (with safety_stock=0, "safety empty" threshold equals "empty" threshold)
//
//      Python (lines 292–294):
//        pos_Z_safety = (Z < ss)
//        scheduling_control[pos_Z_safety[:, 1] & (~pos_Z[:, 0]), 0] = 1
//        scheduling_control[pos_Z_safety[:, 3] & (~pos_Z[:, 2]), 1] = 3
//
//      Meaning:
//        If Z[1] < ss AND Z[0] >= 1e-4 → station 1 must serve buf[0]
//          (buf[1] is nearly empty; station 1 should replenish it via buf[0])
//        If Z[3] < ss AND Z[2] >= 1e-4 → station 2 must serve buf[2]
//          (buf[3] is nearly empty; station 2 should replenish it via buf[2])
//
//      This OVERRIDES whatever wmp_scheduling() decided for that station.
//      It is an anti-starvation rule: keep the downstream buffer non-empty.
//
// Behavior: EXACTLY the same as Python for apply_exit_buffer='N',
//           apply_safety_stock='Always', safety_stock=0.

void idling_policy(const PolicyInputs& inp, PolicyOutputs& out) {

    // Pre-compute "buffer effectively empty" flags for each buffer.
    // Python: pos_Z = (Z < 1e-4)
    bool buf_empty[N_BUFFERS];
    for (int k = 0; k < N_BUFFERS; k++) {
        buf_empty[k] = (inp.Z[k] < 1e-4);
    }

    // ---- Exit-buffer rule (apply_exit_buffer = 'N') ----
    // Python:
    //   scheduling_control[(outsourcing_control[:, 2] == 1), 0] = 0
    //   scheduling_control[(outsourcing_control[:, 3] == 1), 1] = 0
    //
    // For ThresholdBasedPolicy, out.outsourcing_control[2] and [3] are always 0,
    // so these conditions are never true. Included for correctness.
    if (out.outsourcing_control[2] == 1) out.scheduling_control[0] = 0;
    if (out.outsourcing_control[3] == 1) out.scheduling_control[1] = 0;

    // ---- Safety-stock rule (apply_safety_stock = 'Always') ----
    // Python:
    //   ss = safety_stock * inv_sqrt_n + 1e-4
    //   pos_Z_safety = (Z < ss)
    //   scheduling_control[pos_Z_safety[:,1] & (~pos_Z[:,0]), 0] = 1
    //   scheduling_control[pos_Z_safety[:,3] & (~pos_Z[:,2]), 1] = 3
    //
    // C++ uses a shifted convention: safety integer k means "idle when queue <= k-1".
    //   ss = RC_SAFETY_STOCK * INV_SQRT_N - 1e-4
    //   safety=0 → ss=-1e-4 → never idle (Z >= 0 > -1e-4 always)
    //   safety=1 → ss=0.0499 → idle when queue=0  (Z=0 < ss, Z=0.05 > ss)
    //   safety=k → idle when queue <= k-1
    // This is equivalent to Python safety_stock = RC_SAFETY_STOCK - 1.
    const double ss = RC_SAFETY_STOCK * INV_SQRT_N - 1e-4;
    bool safety_empty[N_BUFFERS];
    for (int k = 0; k < N_BUFFERS; k++) {
        safety_empty[k] = (inp.Z[k] < ss);
    }

    // Station 1 safety-stock override:
    //   safety_empty[1] = (Z[1] < ss)  i.e. buf[1] nearly empty
    //   !buf_empty[0]   = (Z[0] >= 1e-4) i.e. buf[0] has jobs to process
    // → Force station 1 to serve buf[0] so it can replenish buf[1].
    // Python: scheduling_control[pos_Z_safety[:,1] & (~pos_Z[:,0]), 0] = 1
    if (safety_empty[1] && !buf_empty[0]) {
        out.scheduling_control[0] = 1;   // station 1 → serve buf[0]
    }

    // Station 2 safety-stock override:
    //   safety_empty[3] = (Z[3] < ss)  i.e. buf[3] nearly empty
    //   !buf_empty[2]   = (Z[2] >= 1e-4) i.e. buf[2] has jobs to process
    // → Force station 2 to serve buf[2] so it can replenish buf[3].
    // Python: scheduling_control[pos_Z_safety[:,3] & (~pos_Z[:,2]), 1] = 3
    if (safety_empty[3] && !buf_empty[2]) {
        out.scheduling_control[1] = 3;   // station 2 → serve buf[2]
    }

    // Python line 329: outsourcing_control = outsourcing_control[:, :2]
    // Trims outsourcing_control to its first 2 columns.
    // In C++ we already only use out.outsourcing_control[0] and [1] downstream,
    // so no explicit action is needed.
}

// ---------------------------------------------------------------------------
// compute_policy  (wrapper calling all four sub-functions in Python order)
// ---------------------------------------------------------------------------
// Python: policy() function (lines 384–430 in to_cpp_single_parameter.py)
//
// Python call order:
//   1. Z, W = calculate_scaled_queues_and_workload(states)   → compute_Z_and_W
//   2. pricing_control = pricing_policy.calculate_pricing_control(...)  → bcp_linear_pricing
//   3. scheduling_control = scheduling_policy.calculate_scheduling_control(...)  → wmp_scheduling
//   4. outsourcing_control = outsourcing_policy.calculate_outsourcing_control(...)
//      → threshold_outsourcing  OR  single_pushing_outsourcing  (dispatch on OUTSOURCING_MODE)
//   5. outsourcing_control = hardstop_finite_buffer(...)     → hardstop_finite_buffer
//   6. scheduling_control, outsourcing_control = idling_policy(...)  → idling_policy
//
// Note: Python computes scheduling BEFORE outsourcing (steps 3 and 4 above).
//       We follow the same order here.
//
// WHAT IS STILL TODO vs. Python's policy():
//   - identify_mode()  (TODO: needed for NN policies; not needed for Threshold/SinglePushing)

// ---------------------------------------------------------------------------
// NN Pricing
// ---------------------------------------------------------------------------
// Python: NNPricingPolicy.calculate_pricing_control() — pricing_outside='hold_last_price'
//   G = model(W_last).numpy()
//   pricing_control = -0.5 * G @ NN_pricing_matrix.T * inv_sqrt_n
//
// (G @ NN_pricing_matrix.T)[j] = sum_k G[k] * NN_PRICING_MATRIX[j][k]
// NN_PRICING_MATRIX = [[1/6, 1/3], [1/3, 1/6]]  (inv(H) @ M_tilde)
// ---------------------------------------------------------------------------
void nn_pricing(const PolicyInputs& inp, double pricing_control[N_ROUTES]) {
    // Use cached G = NN(W_last), populated by compute_policy.
    // No NN call here — shared with nn_outsourcing_hold_last in the same step.
    const double* G = inp.G_cached;

    // pricing_control[j] = -0.5 * (G @ NN_PRICING_MATRIX.T)[j] * INV_SQRT_N
    // (G @ M.T)[j] = sum_k G[k] * M[j][k]
    for (int j = 0; j < N_ROUTES; j++) {
        double dot = 0.0;
        for (int k = 0; k < N_ROUTES; k++) {
            dot += G[k] * NN_PRICING_MATRIX[j][k];
        }
        pricing_control[j] = -0.5 * dot * INV_SQRT_N;
    }
}

// ---------------------------------------------------------------------------
// Shared helper: apply NN-guided workload solution for a given face f.
// Called from both nn_outsourcing variants when mask_workload_solu is true
// and exactly one face f is violated.
//
// For each face, we compare two ratios involving G (NN gradient) and the
// face-specific cost decomposition to decide between two actions.
//
// Python (functions_general.py, NNOutsourcingPolicy.calculate_outsourcing_control):
//   pos_F2 = ... > ...   → idle server 1 (oc[2]=1) vs outsource route 1 (oc[0]=1)
//   pos_F3 = ... > ...   → U1 (oc[0]=1)  vs U2 (oc[1]=1)
//   pos_F4 = ... > ...   → U1 (oc[0]=1)  vs U2 (oc[1]=1)
//   pos_F5 = ... > ...   → idle server 2 (oc[3]=1) vs outsource route 2 (oc[1]=1)
//
// F2/F5 safety-stock override: before the idle-vs-outsource comparison, check
// whether the server's downstream buffer needs replenishment, using the SAME
// test as idling_policy (safety_empty && !buf_empty). If so, idling is not a
// valid choice (idling_policy would force the server to serve upstream anyway),
// so we outsource that face's route directly (F2 → U1, F5 → U2) and skip the
// comparison. F3/F4 (outsource-only faces) are unaffected.
//
// Face index: 0=F4, 1=F3, 2=F2, 3=F5  (matches CONDITION_MATRIX ordering)
//
// Constants used (from parameters_model.py):
//   m1 = M_LIM[0] = 20/19  (effective service time, first type)
//   m2 = M_LIM[1] = 40/19  (effective service time, second type)
//   OUTSOURCE_COST_RATE_LIM[0] = 1.0   (= lambda1 * outsourcing_cost_rate[0] * sqrt_n)
//   OUTSOURCE_COST_RATE_LIM[1] = 2.5   (= lambda2 * outsourcing_cost_rate[1] * sqrt_n)
//   SIGMA_F[f]: sigma vector for face f
// ---------------------------------------------------------------------------
static void nn_apply_workload_solution(const PolicyInputs& inp,
                                       const double G[N_ROUTES],
                                       int f,
                                       double outsourcing_control[4]) {
    const double m1 = M_LIM[0];   // 20/19
    const double m2 = M_LIM[1];   // 40/19
    const double c1 = OUTSOURCE_COST_RATE_LIM[0];  // 1.0
    const double c2 = OUTSOURCE_COST_RATE_LIM[1];  // 2.5

    // Safety-stock threshold, same convention as idling_policy (policy.cpp).
    //   ss = RC_SAFETY_STOCK * INV_SQRT_N - 1e-4
    // Used by the F2/F5 branches to detect whether the server's downstream
    // buffer needs replenishment (safety_empty && !buf_empty).
    const double ss = RC_SAFETY_STOCK * INV_SQRT_N - 1e-4;

    if (f == 2) {
        // F2: sigma_F2 = [1/2, -1]
        // pos_F2: L1 vs U1
        // lhs = ((-m1)*G[0] + (-m2)*G[1] + c1) / (sigma[0]*(-m1) + sigma[1]*(-m2))
        // rhs = G[0] / sigma[0]
        // sigma[0]=-m1/2+m2 = -10/19 + 40/19 = 30/19
        // sigma[1]*(-m2) = -(-m2) = m2 — wait:
        //   sigma_F2[0]*(-m1) + sigma_F2[1]*(-m2) = (1/2)*(-m1) + (-1)*(-m2) = -m1/2 + m2
        //   = -10/19 + 40/19 = 30/19
        // rhs = G[0] / (1/2) = 2*G[0]
        // Safety-stock pre-check on server 1's downstream buffer:
        //   safety_empty[1] && !buf_empty[0]  (same test as idling_policy)
        // If buf[1] is nearly empty while buf[0] has jobs, server 1 must stay
        // busy replenishing buf[1] (idling_policy would force it anyway), so we
        // skip the idle option and outsource route 1 (U1) directly.
        bool ss_need_1 = (inp.Z[1] < ss) && !(inp.Z[0] < 1e-4);
        if (ss_need_1) {
            outsourcing_control[0] = 1;  // U1 (downstream needs safety stock)
        } else {
            double denom = SIGMA_F[2][0] * (-m1) + SIGMA_F[2][1] * (-m2);
            double lhs   = ((-m1) * G[0] + (-m2) * G[1] + c1) / denom;
            double rhs   = G[0] / SIGMA_F[2][0];
            if (lhs > rhs) outsourcing_control[2] = 1;  // idle server 1 (L1)
            else           outsourcing_control[0] = 1;  // outsource route 1 (U1)
        }

    } else if (f == 1) {
        // F3: sigma_F3 = [0, -1]
        // pos_F3: U1 vs U2
        // lhs = ((-m2)*G[0] + (-m1)*G[1] + c2) / (sigma[0]*(-m2) + sigma[1]*(-m1))
        //      = ((-m2)*G[0] + (-m1)*G[1] + c2) / (0*(-m2) + (-1)*(-m1))
        //      = ((-m2)*G[0] + (-m1)*G[1] + c2) / m1
        // rhs  = ((-m1)*G[0] + (-m2)*G[1] + c1) / (sigma[0]*(-m1) + sigma[1]*(-m2))
        //      = ((-m1)*G[0] + (-m2)*G[1] + c1) / m2
        double denom_u2 = SIGMA_F[1][0] * (-m2) + SIGMA_F[1][1] * (-m1);  // m1
        double denom_u1 = SIGMA_F[1][0] * (-m1) + SIGMA_F[1][1] * (-m2);  // m2
        double lhs = ((-m2) * G[0] + (-m1) * G[1] + c2) / denom_u2;
        double rhs = ((-m1) * G[0] + (-m2) * G[1] + c1) / denom_u1;
        if (lhs > rhs) outsourcing_control[0] = 1;  // U1
        else           outsourcing_control[1] = 1;  // U2

    } else if (f == 0) {
        // F4: sigma_F4 = [-1, 0]
        // pos_F4: U1 vs U2
        // denom for U2 side: sigma_F4[0]*(-m2) + sigma_F4[1]*(-m1) = (-1)*(-m2) = m2
        // denom for U1 side: sigma_F4[0]*(-m1) + sigma_F4[1]*(-m2) = (-1)*(-m1) = m1
        double denom_u2 = SIGMA_F[0][0] * (-m2) + SIGMA_F[0][1] * (-m1);  // m2
        double denom_u1 = SIGMA_F[0][0] * (-m1) + SIGMA_F[0][1] * (-m2);  // m1
        double lhs = ((-m2) * G[0] + (-m1) * G[1] + c2) / denom_u2;
        double rhs = ((-m1) * G[0] + (-m2) * G[1] + c1) / denom_u1;
        if (lhs > rhs) outsourcing_control[0] = 1;  // U1
        else           outsourcing_control[1] = 1;  // U2

    } else {
        // f == 3: F5: sigma_F5 = [-1, 1/2]
        // pos_F5: L2 vs U2
        // denom: sigma_F5[0]*(-m2) + sigma_F5[1]*(-m1) = (-1)*(-m2) + (1/2)*(-m1) = m2 - m1/2
        //      = 40/19 - 10/19 = 30/19
        // rhs = G[1] / sigma_F5[1] = G[1] / (1/2) = 2*G[1]
        // Safety-stock pre-check on server 2's downstream buffer:
        //   safety_empty[3] && !buf_empty[2]  (same test as idling_policy)
        // If buf[3] is nearly empty while buf[2] has jobs, server 2 must stay
        // busy replenishing buf[3] (idling_policy would force it anyway), so we
        // skip the idle option and outsource route 2 (U2) directly.
        bool ss_need_2 = (inp.Z[3] < ss) && !(inp.Z[2] < 1e-4);
        if (ss_need_2) {
            outsourcing_control[1] = 1;  // U2 (downstream needs safety stock)
        } else {
            double denom = SIGMA_F[3][0] * (-m2) + SIGMA_F[3][1] * (-m1);
            double lhs   = ((-m2) * G[0] + (-m1) * G[1] + c2) / denom;
            double rhs   = G[1] / SIGMA_F[3][1];
            if (lhs > rhs) outsourcing_control[3] = 1;  // idle server 2 (L2)
            else           outsourcing_control[1] = 1;  // outsource route 2 (U2)
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helper: apply forced outsourcing for all violated faces.
// Used when outside the epsilon band or multiple faces are violated.
//
// Like nn_apply_workload_solution, the action for each violated face is chosen
// by a cost comparison using the NN gradient G — but restricted to OUTSOURCING
// options only (idling is not allowed in the forced region):
//   - F3, F4: choose the cheaper of U1 vs U2 (both outsourcing) using the SAME
//             cost comparison as nn_apply_workload_solution (formulas inlined).
//   - F2, F5: nn_apply_workload_solution compares idle vs outsource; here idling
//             is disallowed, so take the outsourcing option directly
//             (F2 → U1, F5 → U2) with no comparison.
//
// Forced actions use the outer-zone marker value 1.5, so compute_cost_rate
// attributes them to the "outer" bucket; the actual outsourcing cost is
// identical to the band value 1.0.
// Constants: m1 = M_LIM[0] = 20/19, m2 = M_LIM[1] = 40/19,
//            c1 = OUTSOURCE_COST_RATE_LIM[0] = 1.0, c2 = OUTSOURCE_COST_RATE_LIM[1] = 2.5.
// ---------------------------------------------------------------------------
static void nn_apply_forced_outsourcing(const double G[N_ROUTES],
                                        const bool pos[4],
                                        double outsourcing_control[4]) {
    const double m1 = M_LIM[0];                    // 20/19
    const double m2 = M_LIM[1];                    // 40/19
    const double c1 = OUTSOURCE_COST_RATE_LIM[0];  // 1.0
    const double c2 = OUTSOURCE_COST_RATE_LIM[1];  // 2.5

    // F4 (f=0): U1 vs U2 — same cost comparison as nn_apply_workload_solution.
    if (pos[0]) {
        double denom_u2 = SIGMA_F[0][0] * (-m2) + SIGMA_F[0][1] * (-m1);  // m2
        double denom_u1 = SIGMA_F[0][0] * (-m1) + SIGMA_F[0][1] * (-m2);  // m1
        double lhs = ((-m2) * G[0] + (-m1) * G[1] + c2) / denom_u2;
        double rhs = ((-m1) * G[0] + (-m2) * G[1] + c1) / denom_u1;
        if (lhs > rhs) outsourcing_control[0] = 1.5;  // U1
        else           outsourcing_control[1] = 1.5;  // U2
    }
    // F3 (f=1): U1 vs U2 — same cost comparison as nn_apply_workload_solution.
    if (pos[1]) {
        double denom_u2 = SIGMA_F[1][0] * (-m2) + SIGMA_F[1][1] * (-m1);  // m1
        double denom_u1 = SIGMA_F[1][0] * (-m1) + SIGMA_F[1][1] * (-m2);  // m2
        double lhs = ((-m2) * G[0] + (-m1) * G[1] + c2) / denom_u2;
        double rhs = ((-m1) * G[0] + (-m2) * G[1] + c1) / denom_u1;
        if (lhs > rhs) outsourcing_control[0] = 1.5;  // U1
        else           outsourcing_control[1] = 1.5;  // U2
    }
    // F2 (f=2): workload solution compares L1(idle) vs U1(outsource);
    //           forced region → take U1 directly.
    if (pos[2]) outsourcing_control[0] = 1.5;
    // F5 (f=3): workload solution compares L2(idle) vs U2(outsource);
    //           forced region → take U2 directly.
    if (pos[3]) outsourcing_control[1] = 1.5;
}

// ---------------------------------------------------------------------------
// nn_outsourcing_hold_last_no_takeover
// ---------------------------------------------------------------------------
void nn_outsourcing_hold_last_no_takeover(const PolicyInputs& inp, double outsourcing_control[4]) {
    for (int c = 0; c < 4; c++) outsourcing_control[c] = 0;

    // ---- 1. Face violation check using CURRENT W ----
    // Python: pos_matrix = W @ condition_matrix.T > boundary * rho  (F4, F3, F2, F5)
    bool pos[4];
    int n_violated = 0;
    for (int f = 0; f < 4; f++) {
        double val = CONDITION_MATRIX[f][0] * inp.W[0] + CONDITION_MATRIX[f][1] * inp.W[1];
        pos[f] = (val > FACE_BOUNDARY[f] * RC_OUTSOURCE_THRESHOLD);
        if (pos[f]) n_violated++;
    }

    // Early exit: if no face is violated (typical case, state well inside space),
    // the later mask_workload_solu / mask_forced_outsource are both guaranteed false
    // (since n_violated == 0 implies all in_eps[f] = true → mask_outside = false).
    // Skipping Step 2 avoids 4 more boundary computations per step.
    if (n_violated == 0) return;

    // ---- 2. Epsilon band check ----
    // pos_matrix_eps[f] = W @ cond[f] < boundary[f] * rho + epsilon * scaling[f]
    // mask_state_space_eps = all(pos_matrix_eps)  (within outward epsilon band)
    // mask_band = inside outward eps AND any face violated
    // mask_outside = NOT mask_state_space_eps
    bool in_eps[4];
    for (int f = 0; f < 4; f++) {
        double val = CONDITION_MATRIX[f][0] * inp.W[0] + CONDITION_MATRIX[f][1] * inp.W[1];
        in_eps[f] = (val < eps_threshold(f, +1.0));
    }
    bool mask_state_space_eps = in_eps[0] && in_eps[1] && in_eps[2] && in_eps[3];
    bool mask_band    = mask_state_space_eps;   // (any_violated implied since n_violated > 0 above)
    bool mask_outside = !mask_state_space_eps;
    bool mask_one     = (n_violated == 1);
    bool mask_workload_solu    = mask_one && mask_band;
    bool mask_forced_outsource = ((!mask_one) && mask_band) || mask_outside;

    // ---- 3. NN gradient at W_last (cached) ----
    // Python: cur_grad = self.model(W_last).numpy()  [hold_last variant]
    // Reuse G = NN(W_last) from inp.G_cached, populated by compute_policy — no NN call.
    const double* G = inp.G_cached;

    // ---- 4a. Workload solution (one face violated, in band) ----
    if (mask_workload_solu) {
        for (int f = 0; f < 4; f++) {
            if (pos[f]) {
                nn_apply_workload_solution(inp, G, f, outsourcing_control);
                break;  // exactly one face violated
            }
        }
    }

    // ---- 4b. Forced outsourcing (multiple faces or outside band) ----
    if (mask_forced_outsource) {
        nn_apply_forced_outsourcing(G, pos, outsourcing_control);
    }
}

// ---------------------------------------------------------------------------
// nn_outsourcing_inward
// ---------------------------------------------------------------------------
void nn_outsourcing_inward(const PolicyInputs& inp, double outsourcing_control[4]) {
    for (int c = 0; c < 4; c++) outsourcing_control[c] = 0;

    // ---- 1. Inward face violation check ----
    // pos_matrix_inward[f]: W is within epsilon of boundary FROM INSIDE
    // Python: pos_matrix_inward = W @ cond.T > boundary * rho - epsilon * scaling
    // eps_threshold(f, -1) handles absolute vs relative epsilon via RC_EPSILON_RELATIVE.
    bool pos_inward[4];
    int n_inward = 0;
    for (int f = 0; f < 4; f++) {
        double val = CONDITION_MATRIX[f][0] * inp.W[0] + CONDITION_MATRIX[f][1] * inp.W[1];
        pos_inward[f] = (val > eps_threshold(f, -1.0));
        if (pos_inward[f]) n_inward++;
    }

    // Early exit: n_inward == 0 ⟹ all pos_inward[f]=false ⟹ val < boundary*rho for all f
    // (since eps*scaling > 0), so inside_face[f]=true for all f ⟹ mask_state_space_eps=true,
    // mask_outside=false, mask_band=false → both action masks are false. Skipping Step 2.
    if (n_inward == 0) return;

    // ---- 2. Inside state space check ----
    // pos_matrix_inward_eps[f]: W @ cond[f] < boundary[f] * rho  (inside state space)
    // mask_state_space_eps = all(pos_matrix_inward_eps)  (fully inside)
    // mask_band = fully inside AND any inward violation (near boundary from inside)
    // mask_outside = NOT fully inside (at least one face outward-violated)
    bool inside_face[4];
    for (int f = 0; f < 4; f++) {
        double val = CONDITION_MATRIX[f][0] * inp.W[0] + CONDITION_MATRIX[f][1] * inp.W[1];
        inside_face[f] = (val < FACE_BOUNDARY[f] * RC_OUTSOURCE_THRESHOLD);
    }
    bool mask_state_space_eps = inside_face[0] && inside_face[1] && inside_face[2] && inside_face[3];
    bool mask_band    = mask_state_space_eps;   // (any_inward implied since n_inward > 0 above)
    bool mask_outside = !mask_state_space_eps;
    bool mask_one     = (n_inward == 1);
    bool mask_workload_solu    = mask_one && mask_band;
    bool mask_forced_outsource = ((!mask_one) && mask_band) || mask_outside;

    // ---- 3. NN gradient at CURRENT W ----
    // Python: cur_grad = self.model(W).numpy()  [inward variant uses current W]
    // When inside state space (G_at_W_valid=true), W_last == W so the cached
    // G_cached is already NN(W); reuse it and skip the NN call. Otherwise call NN(W).
    double G_local[N_ROUTES];
    const double* G;
    if (inp.G_at_W_valid) {
        G = inp.G_cached;
    } else {
        nn_predict(inp.W[0], inp.W[1], G_local);
        G = G_local;
    }

    // ---- 4a. Workload solution (one inward violation, inside state space) ----
    if (mask_workload_solu) {
        for (int f = 0; f < 4; f++) {
            if (pos_inward[f]) {
                nn_apply_workload_solution(inp, G, f, outsourcing_control);
                break;
            }
        }
    }

    // ---- 4b. Forced outsourcing (multiple inward violations or outside state space) ----
    if (mask_forced_outsource) {
        // For outside: use outward violation mask
        // For multiple inward: use inward violation mask
        // Python applies using pos_matrix_inward in the forced region
        nn_apply_forced_outsourcing(G, pos_inward, outsourcing_control);
    }
}

// ---------------------------------------------------------------------------
// compute_policy (updated — now takes W_last as in/out parameter)
// ---------------------------------------------------------------------------
//   1. Z, W = calculate_scaled_queues_and_workload(states)   → compute_Z_and_W
//   2. W_last update: if current W inside state space → W_last = W
//   3. pricing_control = ...  (dispatch on PRICING_MODE)
//   4. scheduling_control = ... (dispatch on SCHEDULING_MODE)
//   5. outsourcing_control = ... (dispatch on OUTSOURCING_MODE)
//   6. hardstop safety guard
//   7. idling policy

void compute_policy(const int states[N_BUFFERS],
                    const double delay_time[N_BUFFERS],
                    double W_last[N_ROUTES],
                    double G_last[N_ROUTES],
                    PolicyOutputs& out) {
    PolicyInputs inp;

    // ---- P1: compute Z, W + delay_time copy + inside check + W_last / G_last refresh ----
    {
        PROF_L2(P1_Z_W_PREP);
        // Step 1: Z, W
        compute_Z_and_W(states, inp);
        // Step 1b: copy delay_time into inp — only LRS delay-based scheduling reads it.
        // (Caller run_one_path also skips the upstream S1 computation in this case, so
        // the `delay_time` array may contain uninitialized values here; the copy is
        // unconditionally skipped to guarantee no garbage ever lands in inp.delay_time.)
        if (RC_SCHEDULING_MODE == LRS_DELAY_TIME
         || RC_SCHEDULING_MODE == LRS_REMAIN_DELAY) {
            for (int k = 0; k < N_BUFFERS; k++) {
                inp.delay_time[k] = delay_time[k];
            }
        }
        // Step 1c: inside check + W_last / G_last refresh — NN policies only.
        // Non-NN sub-policies (BCPlinear/HighLow pricing, Threshold/SinglePushing
        // outsourcing, WMP/LRS scheduling) never read inp.G_cached / inp.G_at_W_valid,
        // so the entire block is skipped when no NN policy is active.
        const bool uses_nn = (RC_PRICING_MODE == (int)NN_PRICING)
                          || (RC_OUTSOURCING_MODE == (int)NN_OUTSOURCING);
        if (uses_nn) {
            bool inside = true;
            for (int f = 0; f < 4; f++) {
                double val = CONDITION_MATRIX[f][0] * inp.W[0] + CONDITION_MATRIX[f][1] * inp.W[1];
                if (val > FACE_BOUNDARY[f] * RC_OUTSOURCE_THRESHOLD) { inside = false; break; }
            }
            if (inside) {
                W_last[0] = inp.W[0];
                W_last[1] = inp.W[1];
                nn_predict(W_last[0], W_last[1], G_last);  // single NN call per inside step
            }
            inp.G_cached[0]  = G_last[0];
            inp.G_cached[1]  = G_last[1];
            inp.G_at_W_valid = inside;
        }
    }

    // ---- P2: pricing control (dispatch on PRICING_MODE) ----
    {
        PROF_L2(P2_PRICING);
        if (RC_PRICING_MODE == BCP_LINEAR) {
            bcp_linear_pricing(inp, out.pricing_control);
        } else if (RC_PRICING_MODE == HIGH_LOW) {
            high_low_pricing(inp, out.pricing_control);
        } else {
            nn_pricing(inp, out.pricing_control);
        }
    }

    // ---- P3: scheduling control (dispatch on SCHEDULING_MODE) ----
    {
        PROF_L2(P3_SCHEDULING);
        if (RC_SCHEDULING_MODE == WMP_OBJ2) {
            if (RC_WMP_LIMITING == 1) {
                wmp_scheduling_limiting(inp, out.scheduling_control);
            } else {
                wmp_scheduling(inp, out.scheduling_control);
            }
        } else if (RC_SCHEDULING_MODE == LRS_QUEUE_LEN) {
            lrs_queue_len_scheduling(inp, out.scheduling_control);
        } else if (RC_SCHEDULING_MODE == LRS_DELAY_TIME) {
            lrs_delay_time_scheduling(inp, out.scheduling_control);
        } else {
            lrs_remain_delay_scheduling(inp, out.scheduling_control);
        }
    }

    // ---- P4: outsourcing control (dispatch on OUTSOURCING_MODE) ----
    {
        PROF_L2(P4_OUTSOURCING);
        if (RC_OUTSOURCING_MODE == THRESHOLD_BASED) {
            threshold_outsourcing(inp, out.outsourcing_control);
        } else if (RC_OUTSOURCING_MODE == SINGLE_PUSHING) {
            if (RC_INWARD_REFLECTION == 1) {
                single_pushing_outsourcing_inward(inp, out.outsourcing_control);
            } else {
                single_pushing_outsourcing(inp, out.outsourcing_control);
            }
        } else {
            if (RC_INWARD_REFLECTION == 1) {
                nn_outsourcing_inward(inp, out.outsourcing_control);
            } else {
                nn_outsourcing_hold_last_no_takeover(inp, out.outsourcing_control);
            }
        }
    }

    // ---- P5: hardstop safety guard + idling policy ----
    {
        PROF_L2(P5_HARDSTOP_IDLING);
        hardstop_finite_buffer(inp, out.outsourcing_control);
        idling_policy(inp, out);
    }
}
