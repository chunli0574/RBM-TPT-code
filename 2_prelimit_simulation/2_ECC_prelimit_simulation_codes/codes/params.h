/*
 * params.h
 *
 * All system constants for the Extended Criss-Cross queueing simulation.
 * Supports dim_K = 3 (CC3) and dim_K = 10 (CC10).
 *
 * Python equivalent: parameters_model.py
 *   CC3:  CC3_python_code/codes/parameters_model.py
 *   CC10: CC10_python_code/codes/parameters_model.py
 *
 * Select dim_K at compile time via DIM_K_SELECT (set in Makefile):
 *   make                   → DIM_K_SELECT=3  (CC3, default)
 *   make DIM_K_SELECT=10   → DIM_K_SELECT=10 (CC10)
 *
 * HOW THE NETWORK LOOKS (general dim_K stations):
 *
 *   Routes 0..dim_K-1 (vertical): arrivals → buf[r] → Station_r → depart
 *   Route  dim_K      (horizontal): arrivals → buf[dim_K] → Station0 → buf[dim_K+1]
 *                                   → Station1 → ... → buf[2*dim_K-1] → Station_{dim_K-1} → depart
 */

#pragma once

#include <cmath>   // sqrt(), exp()
#include <string>

// ---------------------------------------------------------------------------
// Compile-time dim_K selection
// ---------------------------------------------------------------------------
#ifndef DIM_K_SELECT
#  define DIM_K_SELECT 3
#endif

#if (DIM_K_SELECT != 3) && (DIM_K_SELECT != 10)
#  error "Unsupported DIM_K_SELECT. Valid values: 3 (CC3) or 10 (CC10)."
#endif

#if DIM_K_SELECT == 3
#  define SYSTEM_NAME "CC3"
#elif DIM_K_SELECT == 10
#  define SYSTEM_NAME "CC10"
#endif

// ---------------------------------------------------------------------------
// Simulation scale  (same for CC3 and CC10)
// ---------------------------------------------------------------------------
// Python: n = 400, inv_sqrt_n = 1/20
constexpr int    N_SCALE    = 400;
constexpr double INV_SQRT_N = 1.0 / 20.0;   // = 1/sqrt(400)

// Python: nJumps = 100000
#if DIM_K_SELECT == 3
constexpr int N_JUMPS = 100000;
#elif DIM_K_SELECT == 10
constexpr int N_JUMPS = 100000;
#endif

// Python: M = 1000  (set to 100 here for testing; restore to 1000 for production)
constexpr int N_PATHS = 5000;

// ---------------------------------------------------------------------------
// Arrival and service rates  (same for CC3 and CC10)
// ---------------------------------------------------------------------------
// Python: lambda_ast = 0.95/2 = 0.475
constexpr double LAMBDA_AST = 0.95 / 2.0;   // = 0.475

// Mean service times in the pre-limit and limiting scales
// Python: mu_arr = ones(*) * 1.0;  m_arr = ones(*) * 20/19
constexpr double M_ENTRY = 20.0 / 19.0;     // = m = 1/mu_arr in limiting scale

// ---------------------------------------------------------------------------
// Delay bound (vertical, shared across CC3 and CC10)
// ---------------------------------------------------------------------------
constexpr double DELAY_BOUND_V = 60.0;
constexpr double B_V = DELAY_BOUND_V * LAMBDA_AST / 20.0;   // = 1.425 = 57/40

// ---------------------------------------------------------------------------
// Face geometry: off-diagonal constant (same for CC3 and CC10)
// ---------------------------------------------------------------------------
// OFFDIAG_FACE_CONST = b_v * m = 1.425 * 20/19 = 1.5 (exact)
constexpr double OFFDIAG_FACE_CONST = 1.5;
constexpr double SQRT2 = 1.4142135623730951;

// ---------------------------------------------------------------------------
// Cost parameters  (same for CC3 and CC10)
// ---------------------------------------------------------------------------
// Python: interest_rate = 0.01/400
constexpr double INTEREST_RATE = 0.01 / 400.0;

// H matrix: diagonal pricing penalty. H = diag([80/19]*(dim_K+1))
// Pricing cost = H_DIAG * sum(pc[r]^2)
constexpr double H_DIAG = 80.0 / 19.0;

// C_V_OVER_M: c_v/m = OUTSOURCE_COST_RATE_LIM[vertical] / M_LIM[vertical]
// = (0.05*20) / (20/19) = 1.0 * 19/20 = 0.95  (same for CC3 and CC10)
constexpr double C_V_OVER_M = 0.95;

// ---------------------------------------------------------------------------
// Policy mode selectors  (same for CC3 and CC10)
// ---------------------------------------------------------------------------
// The proposed policy uses NN pricing + NN outsourcing + WMP scheduling.
// These defaults are used by main.cpp; run_config.h provides runtime overrides.

enum OutsourcingPolicyMode { THRESHOLD_BASED, SINGLE_PUSHING, NN_OUTSOURCING };
constexpr OutsourcingPolicyMode OUTSOURCING_MODE = NN_OUTSOURCING;

enum NNOutsourcingVariant { NN_INWARD, NN_HOLD_LAST_NO_TAKEOVER };
constexpr NNOutsourcingVariant NN_OUTSOURCING_VARIANT = NN_HOLD_LAST_NO_TAKEOVER;

enum PricingPolicyMode { BCP_LINEAR, HIGH_LOW, NN_PRICING };
constexpr PricingPolicyMode PRICING_MODE = NN_PRICING;

enum SchedulingPolicyMode { WMP_OBJ2, LRS_QUEUE_LEN, LRS_DELAY_TIME, LRS_REMAIN_DELAY };
constexpr SchedulingPolicyMode SCHEDULING_MODE = WMP_OBJ2;

// ---------------------------------------------------------------------------
// Fixed scheduling / policy defaults  (same for CC3 and CC10)
// ---------------------------------------------------------------------------
constexpr double LAMBDA_TILDE   = 4.0;
constexpr double SAFETY_STOCK   = 0.0;  // 0 = no safety stock override
constexpr double OUTSOURCE_THRESHOLD = 1.0;

// ---------------------------------------------------------------------------
// Delay tracking parameters
// ---------------------------------------------------------------------------
#if DIM_K_SELECT == 3
constexpr int DELAY_PROB_RECORD_START = 20000;
#elif DIM_K_SELECT == 10
constexpr int DELAY_PROB_RECORD_START = 20000;
#endif
constexpr double HARDSTOP_MULTIPLE     = 5.0;

// ===========================================================================
// DIM_K-SPECIFIC CONSTANTS
// ===========================================================================

#if DIM_K_SELECT == 3
// ---------------------------------------------------------------------------
// CC3: dim_K = 3 stations
// ---------------------------------------------------------------------------

constexpr int DIM_K     = 3;
constexpr int N_BUFFERS = DIM_K * 2;             // = 6
constexpr int N_ROUTES  = DIM_K + 1;             // = 4
constexpr int N_EVENTS  = N_ROUTES + N_BUFFERS;  // = 10
constexpr int N_OUTSOURCE_CTRL = N_ROUTES + DIM_K;  // = 7
constexpr int N_FACES = DIM_K + DIM_K * (DIM_K - 1) / 2;  // = 6

// Initial state: (dim_K+1)=4 jobs, (dim_K-1)=2 zeros
constexpr int INITIAL_STATE[N_BUFFERS] = {10, 10, 10, 10, 0, 0};

// Arrival rates (all routes same)
constexpr double LAMBDA[N_ROUTES] = {LAMBDA_AST, LAMBDA_AST, LAMBDA_AST, LAMBDA_AST};

// Service rates and weights
constexpr double MUS[N_BUFFERS]   = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
constexpr double W_ARR[N_BUFFERS] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
constexpr double M_LIM[N_BUFFERS] = {
    20.0/19.0, 20.0/19.0, 20.0/19.0,
    20.0/19.0, 20.0/19.0, 20.0/19.0
};

// Delay bounds
// delay_bound_h = 90 = 30 * DIM_K
constexpr double DELAY_BOUND_H = 90.0;
constexpr double DELAY_BOUND[N_ROUTES] = {DELAY_BOUND_V, DELAY_BOUND_V, DELAY_BOUND_V, DELAY_BOUND_H};

// State space boundaries B_VEC
// b_h = delay_bound_h * lambda_ast * inv_sqrt_n = 90 * 0.475 / 20 = 2.1375
constexpr double B_H = DELAY_BOUND_H * LAMBDA_AST / 20.0;   // = 2.1375
constexpr double B_VEC[N_ROUTES] = {B_V, B_V, B_V, B_H};

// Face geometry
// DIAG_FACE_CONST = (b_v + b_h) * m = 3.5625 * 20/19 = 3.75
constexpr double DIAG_FACE_CONST = 3.75;

// Epsilon band width
// Python: epsilon_band = 4.0 / sqrt(n) = 4.0 / 20.0 = 0.2
constexpr double EPSILON_BAND = 0.2;

// Queue count bounds for hardstop and buffer sizing
// q_bound_max = int(delay_bound_h * lambda_ast) = int(90*0.475) = 42
constexpr int Q_BOUND_MAX  = 42;
constexpr int BUFFER_SIZE  = 211;   // = int(5*42+1)

// Outsourcing cost rates
// Python: outsourcing_cost_rate = [0.05, 0.05, 0.05, 0.075]
// outsourcing_cost_rate_limiting = rate * sqrt_n = rate * 20
constexpr double OUTSOURCE_COST_RATE[N_ROUTES]     = {0.05,  0.05,  0.05,  0.075};
constexpr double OUTSOURCE_COST_RATE_LIM[N_ROUTES] = {1.0,   1.0,   1.0,   1.5  };

// C_H_OVER_M = c_h / m = 1.5 / (20/19) = 1.425
constexpr double C_H_OVER_M = 1.425;

// Workload matrix  M_MATRIX[DIM_K][N_BUFFERS]
// Python: M_matrix = 20/19 * hstack(eye(3), tril(ones(3,3)))
//   Station 0: buf[0] + buf[3]
//   Station 1: buf[1] + buf[3] + buf[4]
//   Station 2: buf[2] + buf[3] + buf[4] + buf[5]
constexpr double M_MATRIX[DIM_K][N_BUFFERS] = {
    {M_ENTRY, 0.0,     0.0,     M_ENTRY, 0.0,     0.0    },
    {0.0,     M_ENTRY, 0.0,     M_ENTRY, M_ENTRY, 0.0    },
    {0.0,     0.0,     M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY}
};

// Queue aggregation operator  SUM_OP[N_ROUTES][N_BUFFERS]
// Route 0: Z[0]; Route 1: Z[1]; Route 2: Z[2]; Route 3: Z[3]+Z[4]+Z[5]
constexpr int SUM_OP[N_ROUTES][N_BUFFERS] = {
    {1, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 0, 1, 1, 1}
};

// NN pricing matrix  NN_PRICING_MATRIX[N_ROUTES][DIM_K]
// = (1/4) * [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
constexpr double NN_PRICING_MATRIX[N_ROUTES][DIM_K] = {
    {0.25, 0.0,  0.0 },
    {0.0,  0.25, 0.0 },
    {0.0,  0.0,  0.25},
    {0.25, 0.25, 0.25}
};

// State transition table  TRANSITIONS[N_EVENTS][N_BUFFERS]
// Events 0..3: arrivals;  Events 4..9: services
constexpr int TRANSITIONS[N_EVENTS][N_BUFFERS] = {
    { 1,  0,  0,  0,  0,  0},  // event 0: arrival buf[0]  (route 0)
    { 0,  1,  0,  0,  0,  0},  // event 1: arrival buf[1]  (route 1)
    { 0,  0,  1,  0,  0,  0},  // event 2: arrival buf[2]  (route 2)
    { 0,  0,  0,  1,  0,  0},  // event 3: arrival buf[3]  (route 3)
    {-1,  0,  0,  0,  0,  0},  // event 4: buf[0]→depart   (route 0)
    { 0, -1,  0,  0,  0,  0},  // event 5: buf[1]→depart   (route 1)
    { 0,  0, -1,  0,  0,  0},  // event 6: buf[2]→depart   (route 2)
    { 0,  0,  0, -1,  1,  0},  // event 7: buf[3]→buf[4]   (route 3 stage 1→2)
    { 0,  0,  0,  0, -1,  1},  // event 8: buf[4]→buf[5]   (route 3 stage 2→3)
    { 0,  0,  0,  0,  0, -1}   // event 9: buf[5]→depart   (route 3)
};

// WMP scheduling default betas
constexpr double BETA_TILDE_V = 0.75 * B_V;  // = 1.06875
constexpr double BETA_TILDE_H = 0.75 * B_H;  // = 1.603125

// Default NN model path (used by single-run main.cpp only)
// Update NN_MODEL_BASE_DIR if models are moved.
inline const std::string NN_MODEL_BASE_DIR =
    "../../../1_NN_training/4_ECC_saved_NNs";

inline const std::string NN_MODEL_PATH =
    NN_MODEL_BASE_DIR + "/"
    "RBMSteadyState_extendedCrissCross_none_100_100_100_D3_-0.8_0.0_0.01_200_0.1_64_6000"
    "_dim=3_TD1_elu_50_addcost=0.05_0.2_0.3_lr140000_eta=1.0_pricing=Y_delay=2.25_1.5_grad";

#elif DIM_K_SELECT == 10
// ---------------------------------------------------------------------------
// CC10: dim_K = 10 stations
// ---------------------------------------------------------------------------

constexpr int DIM_K     = 10;
constexpr int N_BUFFERS = DIM_K * 2;             // = 20
constexpr int N_ROUTES  = DIM_K + 1;             // = 11
constexpr int N_EVENTS  = N_ROUTES + N_BUFFERS;  // = 31
constexpr int N_OUTSOURCE_CTRL = N_ROUTES + DIM_K;  // = 21
constexpr int N_FACES = DIM_K + DIM_K * (DIM_K - 1) / 2;  // = 55

// Initial state: (dim_K+1)=11 jobs, (dim_K-1)=9 zeros
constexpr int INITIAL_STATE[N_BUFFERS] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
// (remaining 9 entries are zero-initialized by C++)

// Arrival rates (all routes same)
constexpr double LAMBDA[N_ROUTES] = {
    LAMBDA_AST, LAMBDA_AST, LAMBDA_AST, LAMBDA_AST, LAMBDA_AST,
    LAMBDA_AST, LAMBDA_AST, LAMBDA_AST, LAMBDA_AST, LAMBDA_AST,
    LAMBDA_AST
};

// Service rates and weights (all 1.0)
constexpr double MUS[N_BUFFERS] = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
};
constexpr double W_ARR[N_BUFFERS] = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
};
constexpr double M_LIM[N_BUFFERS] = {
    20.0/19.0, 20.0/19.0, 20.0/19.0, 20.0/19.0, 20.0/19.0,
    20.0/19.0, 20.0/19.0, 20.0/19.0, 20.0/19.0, 20.0/19.0,
    20.0/19.0, 20.0/19.0, 20.0/19.0, 20.0/19.0, 20.0/19.0,
    20.0/19.0, 20.0/19.0, 20.0/19.0, 20.0/19.0, 20.0/19.0
};

// Delay bounds
// delay_bound_h = 240 = 24 * DIM_K
constexpr double DELAY_BOUND_H = 240.0;
constexpr double DELAY_BOUND[N_ROUTES] = {
    DELAY_BOUND_V, DELAY_BOUND_V, DELAY_BOUND_V, DELAY_BOUND_V, DELAY_BOUND_V,
    DELAY_BOUND_V, DELAY_BOUND_V, DELAY_BOUND_V, DELAY_BOUND_V, DELAY_BOUND_V,
    DELAY_BOUND_H
};

// State space boundaries B_VEC
// b_h = delay_bound_h * lambda_ast * inv_sqrt_n = 240 * 0.475 / 20 = 5.7
constexpr double B_H = DELAY_BOUND_H * LAMBDA_AST / 20.0;   // = 5.7
constexpr double B_VEC[N_ROUTES] = {
    B_V, B_V, B_V, B_V, B_V, B_V, B_V, B_V, B_V, B_V,
    B_H
};

// Face geometry
// DIAG_FACE_CONST = (b_v + b_h) * m = (1.425+5.7) * 20/19 = 7.125*20/19 = 7.5
constexpr double DIAG_FACE_CONST = 7.5;

// Epsilon band width
// Python: epsilon_band = 0.125 / sqrt(n) = 0.125 / 20 = 0.00625
constexpr double EPSILON_BAND = 0.00625;

// Queue count bounds for hardstop and buffer sizing
// q_bound_max = int(delay_bound_h * lambda_ast) = int(240*0.475) = 114
constexpr int Q_BOUND_MAX  = 114;
constexpr int BUFFER_SIZE  = 571;   // = int(5*114+1)

// Outsourcing cost rates
// Python: outsourcing_cost_rate = [0.05]*10 + [0.25]  (0.05*dim_K/2 = 0.25)
// outsourcing_cost_rate_limiting = rate * sqrt_n = rate * 20
constexpr double OUTSOURCE_COST_RATE[N_ROUTES] = {
    0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05, 0.05,
    0.25
};
constexpr double OUTSOURCE_COST_RATE_LIM[N_ROUTES] = {
    1.0,  1.0,  1.0,  1.0,  1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,
    5.0
};

// C_H_OVER_M = c_h / m = 5.0 / (20/19) = 5.0 * 19/20 = 4.75
constexpr double C_H_OVER_M = 4.75;

// Workload matrix  M_MATRIX[DIM_K][N_BUFFERS]
// Python: M_matrix = 20/19 * hstack(eye(10), tril(ones(10,10)))
// Station i: buf[i] (vertical) + buf[10..10+i] (all horizontal pipeline stages up to i)
//   M_MATRIX[i][k] = M_ENTRY if k==i (k<DIM_K) or (k-DIM_K)<=i (k>=DIM_K), else 0
constexpr double M_MATRIX[DIM_K][N_BUFFERS] = {
    {M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,
     M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0    },
    {0.0,     M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,
     M_ENTRY, M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0    },
    {0.0,     0.0,     M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,
     M_ENTRY, M_ENTRY, M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0    },
    {0.0,     0.0,     0.0,     M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0,     0.0,
     M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0,     0.0    },
    {0.0,     0.0,     0.0,     0.0,     M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0,
     M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, 0.0,     0.0,     0.0,     0.0,     0.0    },
    {0.0,     0.0,     0.0,     0.0,     0.0,     M_ENTRY, 0.0,     0.0,     0.0,     0.0,
     M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, 0.0,     0.0,     0.0,     0.0    },
    {0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     M_ENTRY, 0.0,     0.0,     0.0,
     M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, 0.0,     0.0,     0.0    },
    {0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     M_ENTRY, 0.0,     0.0,
     M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, 0.0,     0.0    },
    {0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     M_ENTRY, 0.0,
     M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, 0.0    },
    {0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     M_ENTRY,
     M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY, M_ENTRY}
};

// Queue aggregation operator  SUM_OP[N_ROUTES][N_BUFFERS]
// Routes 0..9 (vertical): SUM_OP[r][k] = 1 iff k==r
// Route 10 (horizontal): SUM_OP[10][k] = 1 iff k>=10
constexpr int SUM_OP[N_ROUTES][N_BUFFERS] = {
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 0
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 1
    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 2
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 3
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 4
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 5
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 6
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 7
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 8
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // route 9
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1}   // route 10 (horizontal)
};

// NN pricing matrix  NN_PRICING_MATRIX[N_ROUTES][DIM_K]
// = (1/4) * [[eye(10)], [ones(1,10)]]
// Routes 0..9: 0.25 on diagonal, 0 elsewhere.  Route 10: all 0.25.
constexpr double NN_PRICING_MATRIX[N_ROUTES][DIM_K] = {
    {0.25, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },  // route 0
    {0.0,  0.25, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },  // route 1
    {0.0,  0.0,  0.25, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },  // route 2
    {0.0,  0.0,  0.0,  0.25, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },  // route 3
    {0.0,  0.0,  0.0,  0.0,  0.25, 0.0,  0.0,  0.0,  0.0,  0.0 },  // route 4
    {0.0,  0.0,  0.0,  0.0,  0.0,  0.25, 0.0,  0.0,  0.0,  0.0 },  // route 5
    {0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.25, 0.0,  0.0,  0.0 },  // route 6
    {0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.25, 0.0,  0.0 },  // route 7
    {0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.25, 0.0 },  // route 8
    {0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.25},  // route 9
    {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25}   // route 10 (horizontal)
};

// State transition table  TRANSITIONS[N_EVENTS][N_BUFFERS]
// Events 0..10: arrivals (11 routes)
// Events 11..20: vertical departures (buf[0..9] → depart)
// Events 21..30: horizontal pipeline (buf[10..19])
constexpr int TRANSITIONS[N_EVENTS][N_BUFFERS] = {
    // Arrivals (events 0..10):
    { 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev0: arr buf0
    { 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev1: arr buf1
    { 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev2
    { 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev3
    { 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev4
    { 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev5
    { 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev6
    { 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev7
    { 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev8
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev9
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev10: arr buf10 (route 10)
    // Vertical departures (events 11..20): buf[k] → depart
    {-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev11: buf0→depart
    { 0, -1,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev12
    { 0,  0, -1,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev13
    { 0,  0,  0, -1,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev14
    { 0,  0,  0,  0, -1,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev15
    { 0,  0,  0,  0,  0, -1,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev16
    { 0,  0,  0,  0,  0,  0, -1,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev17
    { 0,  0,  0,  0,  0,  0,  0, -1,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev18
    { 0,  0,  0,  0,  0,  0,  0,  0, -1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev19
    { 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0},  // ev20: buf9→depart
    // Horizontal pipeline (events 21..30):
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -1,  1,  0,  0,  0,  0,  0,  0,  0,  0},  // ev21: buf10→buf11
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0, -1,  1,  0,  0,  0,  0,  0,  0,  0},  // ev22: buf11→buf12
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0, -1,  1,  0,  0,  0,  0,  0,  0},  // ev23
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0, -1,  1,  0,  0,  0,  0,  0},  // ev24
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0, -1,  1,  0,  0,  0,  0},  // ev25
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0, -1,  1,  0,  0,  0},  // ev26
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0, -1,  1,  0,  0},  // ev27
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0, -1,  1,  0},  // ev28
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0, -1,  1},  // ev29: buf18→buf19
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0, -1}   // ev30: buf19→depart
};

// WMP scheduling default betas
constexpr double BETA_TILDE_V = 0.75 * B_V;  // = 1.06875
constexpr double BETA_TILDE_H = 0.75 * B_H;  // = 4.2750

// Default NN model path (used by single-run main.cpp only)
// Update NN_MODEL_BASE_DIR to point to CC10 models directory.
inline const std::string NN_MODEL_BASE_DIR =
    "../../../1_NN_training/4_ECC_saved_NNs";

inline const std::string NN_MODEL_PATH =
    NN_MODEL_BASE_DIR + "/"
    "RBMSteadyState_extendedCrissCross_none_200_200_200_D4_-0.5_0.0_0.01_100_0.1_64_9000"
    "_dim=10_TD1_elu_100_addcost=0.1_0.2_0.3_lr270000_eta=1.0_pricing=Y_delay=6.0_1.5_grad";

#endif  // DIM_K_SELECT
