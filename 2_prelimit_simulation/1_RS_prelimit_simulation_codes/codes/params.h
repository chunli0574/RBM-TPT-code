/*
 * params.h
 *
 * All system constants for the W-network queueing simulation.
 *
 * Python equivalent: parameters_model.py
 *
 * This file is the single source of truth for every number in the system.
 * If you change a parameter here, it automatically propagates everywhere.
 *
 * HOW THE NETWORK LOOKS:
 *
 *   Route 1:  arrivals → buf[0] → Station1 → buf[1] → Station2 → depart
 *   Route 2:  arrivals → buf[2] → Station2 → buf[3] → Station1 → depart
 *
 * This is called a "W-network": both stations are cross-trained.
 *   Station 1 can serve buf[0] (route 1, stage 1) OR buf[3] (route 2, stage 2).
 *   Station 2 can serve buf[2] (route 2, stage 1) OR buf[1] (route 1, stage 2).
 *
 * 6 EVENT TYPES (indexed 0-5):
 *   Event 0: arrival to buf[0]           (route 1 arrives)
 *   Event 1: arrival to buf[2]           (route 2 arrives)
 *   Event 2: Station1 serves buf[0]→buf[1]
 *   Event 3: Station2 serves buf[1]→depart
 *   Event 4: Station2 serves buf[2]→buf[3]
 *   Event 5: Station1 serves buf[3]→depart
 */

#pragma once  // only include this file once per compilation unit

#include <cmath>  // for sqrt(), exp()

// ---------------------------------------------------------------------------
// Simulation scale (from parameters_model.py)
// ---------------------------------------------------------------------------

// n: scale factor between the original system and the heavy-traffic limit.
// inv_sqrt_n is used to convert integer queue lengths → scaled state Z.
// Python: n = 400, inv_sqrt_n = 1 / sqrt(n)
constexpr int    N_SCALE    = 400;
constexpr double INV_SQRT_N = 1.0 / 20.0;   // = 1/sqrt(400)

// Number of simulation events (steps) to run.
// Python: nJumps = 100000
constexpr int N_JUMPS = 100000;

// Number of parallel simulation paths.
// Python: M = 1000
constexpr int N_PATHS = 5000;

// ---------------------------------------------------------------------------
// Network topology constants
// ---------------------------------------------------------------------------

constexpr int N_BUFFERS = 4;   // buf[0], buf[1], buf[2], buf[3]
constexpr int N_ROUTES  = 2;   // route 1 and route 2
constexpr int N_EVENTS  = 6;   // 2 arrivals + 4 services

// Initial queue lengths for all buffers.
// Python: initial_state = [10, 0, 10, 0]
constexpr int INITIAL_STATE[N_BUFFERS] = {10, 0, 10, 0};

// ---------------------------------------------------------------------------
// Arrival rates (Python: lambdas)
// ---------------------------------------------------------------------------

// Nominal arrival rates λ_1 and λ_3 for route 1 and route 2.
// Python: lambda1 = lambda3 = 0.95 * (1/3)
constexpr double LAMBDA[N_ROUTES] = {0.95 / 3.0, 0.95 / 3.0};

// ---------------------------------------------------------------------------
// Service rates (Python: mus)
// ---------------------------------------------------------------------------

// μ_1..μ_4: service rate at each buffer position.
// mus[k] = rate when serving buf[k].
//   mus[0] = μ_1 = 1.0  (Station1 serving buf[0])
//   mus[1] = μ_2 = 0.5  (Station2 serving buf[1])
//   mus[2] = μ_3 = 1.0  (Station2 serving buf[2])
//   mus[3] = μ_4 = 0.5  (Station1 serving buf[3])
// Python: mus = np.array([mu1, mu2, mu3, mu4])
constexpr double MUS[N_BUFFERS] = {1.0, 0.5, 1.0, 0.5};

// w_arr: weights used in the WMP scheduling policy.
// In this model w_arr == MUS (same values).
// Python: w_arr = [mu1, mu2, mu3, mu4]
constexpr double W_ARR[N_BUFFERS] = {1.0, 0.5, 1.0, 0.5};

// ---------------------------------------------------------------------------
// Workload matrix M_matrix (Python: M_matrix)
// ---------------------------------------------------------------------------

// Workload W = Z @ M_matrix.T  (in Python notation)
// In C++: W[i] = sum_k M_MATRIX[i][k] * Z[k]
//
// m1 = m3 = 20/19,  m2 = m4 = 40/19   (average service times in the limit)
// Python: M_matrix = np.array([[m1, 0, m4, m4], [m2, m2, m3, 0]])
constexpr double M_MATRIX[2][N_BUFFERS] = {
    {20.0/19.0,      0.0, 40.0/19.0, 40.0/19.0},  // station 1 workload
    {40.0/19.0, 40.0/19.0, 20.0/19.0,      0.0}   // station 2 workload
};

// ---------------------------------------------------------------------------
// Queue aggregation operator (Python: sum_queue_len_operator)
// ---------------------------------------------------------------------------

// Z_groups[r] = sum of scaled queue lengths for route r.
// Z_groups[0] = Z[0] + Z[1]   (route 1 total)
// Z_groups[1] = Z[2] + Z[3]   (route 2 total)
// Python: sum_queue_len_operator = np.array([[1,1,0,0],[0,0,1,1]])
constexpr double SUM_OP[N_ROUTES][N_BUFFERS] = {
    {1.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 1.0}
};

// ---------------------------------------------------------------------------
// Outsourcing policy selector
// ---------------------------------------------------------------------------
// Switch between outsourcing policies by changing OUTSOURCING_MODE.
// Only one policy is compiled-in at a time (no grid search).
//
//   THRESHOLD_BASED  → ThresholdBasedPolicy  (simple Z-space threshold)
//   SINGLE_PUSHING   → SinglePushingPerFacePolicy  (W-space face reflections)
//   NN_OUTSOURCING   → NNOutsourcingPolicy  (NN gradient-guided reflections)
//                       variant controlled by NN_OUTSOURCING_VARIANT below
//
// Python: outsourcing_type = 'threshold_based' / 'single_pushing_per_face' / 'NN'

enum OutsourcingPolicyMode { THRESHOLD_BASED, SINGLE_PUSHING, NN_OUTSOURCING };
constexpr OutsourcingPolicyMode OUTSOURCING_MODE = NN_OUTSOURCING;

// For NN_OUTSOURCING: which variant of idle_new_reflection to use.
// These are the only two variants implemented in C++.
//   NN_INWARD              → 'epsilon_band_inward_outsource_simultaneously'
//     uses current W for NN, band is inward from boundary
//   NN_HOLD_LAST_NO_TAKEOVER → 'epsilon_band_hold_last_outsource_simultaneously_no_takeover'
//     uses last-inside-state-space W for NN, band is outward from boundary
//
// Python: idle_new_reflection in parameters_model.py
enum NNOutsourcingVariant { NN_INWARD, NN_HOLD_LAST_NO_TAKEOVER };
constexpr NNOutsourcingVariant NN_OUTSOURCING_VARIANT = NN_HOLD_LAST_NO_TAKEOVER;

// ---------------------------------------------------------------------------
// Pricing policy selector
// ---------------------------------------------------------------------------
// BCP_LINEAR  → BCPLinearPricingPolicy  (Z-space linear feedback, uses PRICING_PARA)
// HIGH_LOW    → HighLowPricingPolicy    (two-level drift based on total queue, uses HL_PARA)
// NN_PRICING  → NNPricingPolicy         (NN gradient-guided, uses W_last)
// Python: pricing_type = 'bcp_linear' / 'High_low' / 'NN'

enum PricingPolicyMode { BCP_LINEAR, HIGH_LOW, NN_PRICING };
constexpr PricingPolicyMode PRICING_MODE = NN_PRICING;

// ---------------------------------------------------------------------------
// Scheduling policy selector
// ---------------------------------------------------------------------------
// WMP_OBJ2         → WMPObj2Policy                     (uses LAMBDA_TILDE, BETA_TILDE)
// LRS_QUEUE_LEN    → LeastRelativeSlackQueueLenPolicy   (no extra parameters)
// LRS_DELAY_TIME   → LeastRelativeSlackDelayTimePolicy  (no extra parameters)
// LRS_REMAIN_DELAY → LeastRelativeSlackRemainDelayPolicy(no extra parameters)
// Python: scheduling_type = 'wmp_obj2' / 'lrs_queue_len' / etc.

enum SchedulingPolicyMode { WMP_OBJ2, LRS_QUEUE_LEN, LRS_DELAY_TIME, LRS_REMAIN_DELAY };
constexpr SchedulingPolicyMode SCHEDULING_MODE = WMP_OBJ2;

// ---------------------------------------------------------------------------
// State space boundaries (Z-space) for ThresholdBasedPolicy and hardstop
// ---------------------------------------------------------------------------
// b_vec[r] = delay_bound[r] / sqrt(n) * lambda[r]
// delay_bound = 400/3 for both routes.
// b1 = b3 = (400/3) / 20 * (0.95/3) = 19/9
// Python: b1 = delay_bound1/sqrt(n)*lambda1
constexpr double B_VEC[N_ROUTES] = {19.0/9.0, 19.0/9.0};

// ---------------------------------------------------------------------------
// W-space face geometry for SinglePushingPerFacePolicy
// ---------------------------------------------------------------------------
// The state space boundary is defined by 4 half-spaces in workload (W) space.
// Face index: 0=F4, 1=F3, 2=F2, 3=F5  (same order as Python condition_matrix).
//
// Python: self.condition_matrix = np.array([[1,0],[0,1],[-1,2],[2,-1]])
// Face f is "violated" when: W @ condition_matrix[f] > FACE_BOUNDARY[f] * rho
constexpr double CONDITION_MATRIX[4][2] = {
    { 1.0,  0.0},   // F4: W[0] > FACE_BOUNDARY[0] * rho
    { 0.0,  1.0},   // F3: W[1] > FACE_BOUNDARY[1] * rho
    {-1.0,  2.0},   // F2: -W[0] + 2*W[1] > FACE_BOUNDARY[2] * rho
    { 2.0, -1.0}    // F5: 2*W[0] - W[1] > FACE_BOUNDARY[3] * rho
};

// The boundary values for each face (W-space).
// Python: state_space_boundary = [20/3, 20/3, 80/9, 80/9]
constexpr double FACE_BOUNDARY[4] = {20.0/3.0, 20.0/3.0, 80.0/9.0, 80.0/9.0};

// Epsilon band around the boundary: how far outside counts as "in the band".
// Python: epsilon_band = 4.0 / sqrt(n) = 4.0 / 20.0 = 0.2  (original default)
// Face f is "within epsilon" when: W @ condition_matrix[f] < FACE_BOUNDARY[f]*rho + EPSILON_BAND*EPSILON_SCALING[f]
constexpr double EPSILON_BAND = 0.25;

// Scaling per face: accounts for the norm of each face's normal vector.
// F4=[1,0]: norm=1; F3=[0,1]: norm=1; F2=[-1,2]: norm=sqrt(5); F5=[2,-1]: norm=sqrt(5)
// Python: epsilon_band * np.array([1, 1, sqrt(5), sqrt(5)])
constexpr double SQRT5 = 2.2360679774997896;
constexpr double EPSILON_SCALING[4] = {1.0, 1.0, SQRT5, SQRT5};

// ---------------------------------------------------------------------------
// Cost parameters
// ---------------------------------------------------------------------------

// Interest/discount rate.
// Python: interest_rate = 0.01/400
constexpr double INTEREST_RATE = 0.01 / 400.0;

// Outsourcing cost rate per unit of effective arrival rate.
// Python: outsourcing_cost_rate = np.array([0.05, 0.125])
constexpr double OUTSOURCE_COST_RATE[N_ROUTES] = {0.05, 0.125};

// H matrix: quadratic penalty matrix for pricing control.
// H is diagonal: H = diag(120/19, 120/19).
// Python: H = np.array([[120/19, 0],[0, 120/19]])
// Pricing cost = pricing_control @ H @ pricing_control
//              = (120/19) * (pc[0]^2 + pc[1]^2)   (since H is diagonal)
constexpr double H_DIAG = 120.0 / 19.0;  // diagonal element of H

// ---------------------------------------------------------------------------
// NN policy constants  (used by NNPricingPolicy and NNOutsourcingPolicy)
// ---------------------------------------------------------------------------

// Service times in the limiting (fluid-scaled) system.
// Python: m1, m2, m3, m4 = 1.0*20/19, 2.0*20/19, 1.0*20/19, 2.0*20/19
// M_LIM[k] = 1/mu_k * (1 + 1/mu_{downstream_k})  (effective service time per stage)
constexpr double M_LIM[N_BUFFERS] = {20.0/19.0, 40.0/19.0, 20.0/19.0, 40.0/19.0};

// Outsourcing cost rate in the limiting scale = outsourcing_cost_rate * sqrt(n)
// Python: outsourcing_cost_rate_limiting = outsourcing_cost_rate * sqrt_n
//       = [0.05 * 20, 0.125 * 20] = [1.0, 2.5]
constexpr double OUTSOURCE_COST_RATE_LIM[N_ROUTES] = {1.0, 2.5};

// Face-specific sigma vectors (inward pushing directions in W-space).
// Indexed as SIGMA_F[0]=F4, SIGMA_F[1]=F3, SIGMA_F[2]=F2, SIGMA_F[3]=F5
// to match CONDITION_MATRIX ordering.
// Python (OutsourcingPolicy.__init__):
//   self.sigma_F4 = [-1,   0  ]
//   self.sigma_F3 = [ 0,  -1  ]
//   self.sigma_F2 = [1/2, -1  ]
//   self.sigma_F5 = [-1,  1/2 ]
constexpr double SIGMA_F[4][2] = {
    {-1.0,  0.0 },   // F4: sigma = [-1, 0]
    { 0.0, -1.0 },   // F3: sigma = [ 0, -1]
    { 0.5, -1.0 },   // F2: sigma = [1/2, -1]
    {-1.0,  0.5 }    // F5: sigma = [-1, 1/2]
};

// NN pricing matrix: inv(H) @ M_tilde
//   H = diag(120/19, 120/19),  inv(H) = diag(19/120, 19/120)
//   M_tilde = [[m1, m4], [m2, m3]] = [[20/19, 40/19], [40/19, 20/19]]
//   NN_pricing_matrix = (19/120) * [[20/19, 40/19], [40/19, 20/19]]
//                     = [[1/6, 1/3], [1/3, 1/6]]
// Python: NN_pricing_matrix = np.linalg.inv(H) @ M_tilde_matrix
// Used by NNPricingPolicy: pricing_control = -0.5 * G @ NN_pricing_matrix.T * inv_sqrt_n
constexpr double NN_PRICING_MATRIX[N_ROUTES][N_ROUTES] = {
    {20.0/120.0, 40.0/120.0},  // row 0: [1/6, 1/3]
    {40.0/120.0, 20.0/120.0}   // row 1: [1/3, 1/6]
};

// Path to the NN gradient model (.weights.bin) used by NN policies.
// Model is selected for rho=1.01 (OUTSOURCE_THRESHOLD), theta=-0.8 (from eta_theta_mapping
// in nn_model_paths.h). Converted from _grad.keras by export_keras_weights.py.
// Python: "models/" + suffix + "_grad.keras"  (solver.py)
//
// Path relative to codes/ (where the binaries are run). Change it if you run elsewhere.
#include <string>
inline const std::string NN_MODEL_PATH =
    "../../../1_NN_training/3_RS_saved_NNs/"
    "RybkoStolyar_none_100_-0.8_0.0_0.01_100_0.1_64_6000"
    "_dim=2_TD1_elu_x0=1.0_v1=1.0_v2=2.5_addcost=0.05_0.2"
    "_lr0.0005_eta=1.01_grad.weights.bin";

// ---------------------------------------------------------------------------
// State transition table (from QueueInfo.transitions in Python)
// ---------------------------------------------------------------------------

// TRANSITIONS[event_idx][buf_idx] = change in buf[buf_idx] when event fires.
// Python: self.transitions = np.array([...])  in QueueInfo.__init__
//
//   event 0: +1 to buf[0]             (route 1 arrives)
//   event 1: +1 to buf[2]             (route 2 arrives)
//   event 2: -1 buf[0], +1 buf[1]     (station1 moves job from buf0 to buf1)
//   event 3: -1 buf[1]                (station2 completes buf1, job departs)
//   event 4: -1 buf[2], +1 buf[3]     (station2 moves job from buf2 to buf3)
//   event 5: -1 buf[3]                (station1 completes buf3, job departs)
constexpr int TRANSITIONS[N_EVENTS][N_BUFFERS] = {
    { 1,  0,  0,  0},  // event 0
    { 0,  0,  1,  0},  // event 1
    {-1,  1,  0,  0},  // event 2
    { 0, -1,  0,  0},  // event 3
    { 0,  0, -1,  1},  // event 4
    { 0,  0,  0, -1}   // event 5
};

// ---------------------------------------------------------------------------
// Fixed policy parameters for the minimal test
// ---------------------------------------------------------------------------
// These correspond to ONE specific parameter setting from the Python grid search.
// In Python these are passed as function arguments to simulation().
// Here we hardcode them for the minimal single-experiment version.

// --- BCPLinearPricingPolicy ---
// Compact representation: exactly 2 scalars per route (phi_own, phi_other),
// corresponding to the simplification Φ_{k,(k',l')} = φ_k if k'=k else φ_{-k}
// (see linear pricing policy definition, eq. (98) of the paper). I.e., the kth
// pricing depends equally on all own-route buffers (coefficient phi_own) and
// equally on all non-route-k buffers (coefficient phi_other).
//
// Runtime formula (see bcp_linear_pricing in policy.cpp):
//   z_own[j]   = sum of Z[k] for k belonging to route j    (uses SUM_OP[j])
//   z_other[j] = z_total - z_own[j]                        (z_total = sum of all Z)
//   pricing_control[j] = (phi_own * z_own[j] + phi_other * z_other[j]) * INV_SQRT_N
//
// PRICING_PARA[j][0] = phi_own for route j
// PRICING_PARA[j][1] = phi_other for route j
//
// TODO: Replace with the actual optimal (phi_own, phi_other) values from your experiments.
constexpr double PRICING_PARA[N_ROUTES][2] = {
    {-0.0215, 0.001},   // route 1: (phi_own_1, phi_other_1)
    {-0.0215, 0.0   }   // route 2: (phi_own_2, phi_other_2)
};

// --- HighLowPricingPolicy ---
// HL_PARA[r][0] = threshold in Z-space (real-valued)
//                 if z_total_route > threshold → use high drift; else low drift
// HL_PARA[r][1] = low drift value  (used when z_total_route <= threshold)
// HL_PARA[r][2] = high drift value (used when z_total_route >  threshold)
// pricing_control[r] = HL_PARA[r][1 or 2] * INV_SQRT_N
//
// Python: pricing_para.reshape((2, 3)) where col0=threshold (integer queue count),
//         col1=low, col2=high.  In C++ the col0 is stored directly in Z-space so
//         the per-step comparison in policy.cpp stays in Z-space (no division per
//         step).  benchmark_search_main.cpp::highlow_pricing_search computes
//         z_thresh = Q_loop × Q_BOUND_MAX × rho × INV_SQRT_N (no integer rounding).
//   pricing_control = where(z_total_route > threshold_z, high, low) * inv_sqrt_n
//
// IMPORTANT — HOW TO SCALE high/low VALUES (unchanged):
//   Keep |HL_PARA[r][2]| small so pricing cost stays the same order as outsourcing cost.
//   Rule of thumb: |high| < 0.1  (otherwise quadratic cost dominates everything).
//   For comparison, BCPLinear at Z≈1 gives pc ≈ -0.002, equivalent to |high| ≈ 0.04.
//
// Rough correspondence (for intuition; the search grid now is real-valued):
//   threshold_jobs / sqrt(n) ≈ HL_PARA[r][0]   (e.g. 16 jobs → 0.8; 24 jobs → 1.2)
//
// TODO: Replace with values from your experiments (grid search over threshold and high).
constexpr double HL_PARA[N_ROUTES][3] = {
    {0.8, -0.0046, -0.0215},  // route 1: threshold_z=0.8 (~16 jobs/20), low=-0.0046, high=-0.0215
    {1.2, -0.01,   -0.0464}   // route 2: threshold_z=1.2 (~24 jobs/20), low=-0.01,   high=-0.0464
};

// --- WMPObj2Policy ---
// scheduling_para = [lambda_tilde, beta_tilde_1, beta_tilde_2]
// Python: scheduling_init = [4.0, 0.75*beta_const, 0.75*beta_const]
//         beta_const = 19/9,  so beta_tilde = 0.75 * 19/9 = 19/12
//
// These are the initial (default) scheduling parameters from parameters_model.py.
// TODO: Replace with optimal values found from parameter search.
constexpr double LAMBDA_TILDE  = 4.0;
constexpr double BETA_TILDE_1  = 1.279;   // = 0.75 * beta_const = 0.75 * 19/9
constexpr double BETA_TILDE_2  = 1.279;

// --- Safety stock (used in idling_policy) ---
// Python: safety_stock = 0  (in parameters_model.py)
// When apply_safety_stock = 'Always':  ss = safety_stock * inv_sqrt_n + 1e-4
// With safety_stock = 0: ss = 1e-4, which equals the "buffer empty" threshold.
// Change this value if you want a positive safety stock in the future.
constexpr double SAFETY_STOCK = 0.0;

// --- ThresholdBasedPolicy ---
// Outsource route r when: Z_groups[r] >= OUTSOURCE_THRESHOLD * B_VEC[r]
// Python: outsourcing_para = rho  (a scalar from outsourcing_search_rho)
//
// TODO: Replace with the value from your experiments (typical range: 0.5 to 1.4).
constexpr double OUTSOURCE_THRESHOLD = 1.01;

// ---------------------------------------------------------------------------
// Delay tracking (for computing delay probability — not yet implemented)
// ---------------------------------------------------------------------------

// Delay bound for each route (in time units).
// Python: delay_bound1 = delay_bound3 = 400/3
// TODO: Used when delay tracking (QueueInfo timestamps) is implemented.
constexpr double DELAY_BOUND[N_ROUTES] = {400.0/3.0, 400.0/3.0};

// ---------------------------------------------------------------------------
// Delay tracking parameters
// ---------------------------------------------------------------------------

// Only record delays after this many iterations (let system reach steady state first).
// Python: delay_prob_record_start = 20000
constexpr int DELAY_PROB_RECORD_START = 20000;

// q_bound_max = int(delay_bound * lambda) = int((400/3) * (0.95/3)) = int(42.22) = 42
// Python: q_bound_max = max(int(delay_bound1*lambda1), int(delay_bound3*lambda3))
constexpr int    Q_BOUND_MAX       = 42;
constexpr double HARDSTOP_MULTIPLE = 10.0;

// Circular buffer size for storing job entry timestamps (for delay computation).
// Must be large enough to hold all jobs currently in one buffer at peak load.
// Python: buffer_size = int(hardstop_multiple * q_bound_max + 1) = int(10*42+1) = 421
constexpr int BUFFER_SIZE = 421;
