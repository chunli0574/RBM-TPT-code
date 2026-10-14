/*
 * policy.cpp
 *
 * Implementations of the CC3 policy functions.
 *
 * Python equivalents in CC3_python_code/codes/functions_general.py:
 *   compute_Z_and_W                       → calculate_scaled_queues_and_workload() (lines 610-616)
 *   nn_pricing                            → NNPricingPolicy.calculate_pricing_control() (lines 58-64)
 *   wmp_scheduling                        → WMPObj2Policy.calculate_scheduling_control() (lines 156-178)
 *                                           + SchedulingPolicy._assign_scheduling_control() (lines 135-152)
 *   nn_outsourcing_hold_last_no_takeover  → NNOutsourcingPolicy (lines 489-553)
 *   hardstop_finite_buffer                → hardstop_finite_buffer() (lines 621-638)
 *   idling_policy                         → idling_policy() (lines 640-691)
 *   compute_policy                        → policy() (lines 738-781)
 */

#include "policy.h"
#include "nn_model.h"
#include "run_config.h"
#include <cmath>
#include <algorithm>
#include <numeric>  // for std::accumulate

// ==========================================================================
// Optional Level-2 profiling for compute_policy (compile with
// -DENABLE_PROFILING=1). Subdivides the compute_policy body into five
// sections so we can see, e.g., how much time BCPlinear pricing vs WMP
// scheduling vs SinglePushing outsourcing take in a benchmark run.
// Independent of simulation.cpp's Level-1 profiling (different file
// scope, separate atexit dump). Both may be active simultaneously.
// ==========================================================================
#ifdef ENABLE_PROFILING
  #include <chrono>
  #include <cstdint>
  #include <cstdio>
  #include <cstdlib>   // std::atexit
  #define L2_PROFILE_TIC(name) \
      auto name##_t0 = std::chrono::steady_clock::now()
  #define L2_PROFILE_TOC(name, counter) \
      counter += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>( \
          std::chrono::steady_clock::now() - name##_t0).count()

  namespace {
      uint64_t g_prof_p1_zw_prep      = 0;  // Z/W compute + delay_time copy + NN-guard block
      uint64_t g_prof_p2_pricing      = 0;  // pricing policy dispatch
      uint64_t g_prof_p3_scheduling   = 0;  // scheduling policy dispatch
      uint64_t g_prof_p4_outsourcing  = 0;  // outsourcing policy dispatch
      uint64_t g_prof_p5_guard        = 0;  // hardstop + idling

      void l2_prof_dump_atexit() {
          uint64_t total = g_prof_p1_zw_prep + g_prof_p2_pricing
                         + g_prof_p3_scheduling + g_prof_p4_outsourcing
                         + g_prof_p5_guard;
          if (total == 0) return;
          auto pct = [&](uint64_t v){ return 100.0 * v / total; };
          std::printf("\n========== PROFILING (compute_policy, cumulative) ==========\n");
          std::printf("  %-22s %18s  %8s\n", "section", "ns", "%sum");
          auto pp = [&](const char* name, uint64_t v){
              std::printf("  %-22s %18llu  %7.2f%%\n",
                          name, (unsigned long long)v, pct(v));
          };
          pp("P1_Z_W_prep",        g_prof_p1_zw_prep);
          pp("P2_pricing",         g_prof_p2_pricing);
          pp("P3_scheduling",      g_prof_p3_scheduling);
          pp("P4_outsourcing",     g_prof_p4_outsourcing);
          pp("P5_hardstop_idling", g_prof_p5_guard);
          std::printf("  %-22s %18llu  %7.2f%%\n", "sum",
                      (unsigned long long)total, 100.00);
          std::printf("============================================================\n");
      }
      struct L2ProfInit { L2ProfInit() { std::atexit(l2_prof_dump_atexit); } };
      L2ProfInit g_l2_prof_init;
  }
#else
  #define L2_PROFILE_TIC(name)        do {} while(0)
  #define L2_PROFILE_TOC(name, counter) do {} while(0)
#endif

// ---------------------------------------------------------------------------
// compute_Z_and_W
// ---------------------------------------------------------------------------
// Python: Z = states * inv_sqrt_n; W = Z @ M_matrix.T
//
// W exploits the sparse structure of M_MATRIX (all non-zero entries equal
// M_ENTRY = 20/19):
//   Row i: M_MATRIX[i][k] = M_ENTRY iff k == i  (vertical buf i)
//                        or k >= DIM_K and (k - DIM_K) <= i  (horizontal
//                        pipeline accumulated up to position i)
// Equivalent formula:
//   W[i] = M_ENTRY * (Z[i] + Σ_{j=DIM_K..DIM_K+i} Z[j])
// Computed via a running horizontal prefix sum so each row costs only
// 2 adds + 1 mult (3 FLOPs) instead of the full 2*DIM_K muladds.
//
// Operation count (muladds → FLOPs):
//   CC3  (DIM_K=3) :  18 → 9
//   CC10 (DIM_K=10):  200 → 30
//
// Note: accumulation order differs from the original loop, so W may differ
// by ULP-level (~1e-15) amounts. Downstream stats (cost, delay) are
// statistically unaffected.
void compute_Z_and_W(const int states[N_BUFFERS], PolicyInputs& inp) {
    for (int k = 0; k < N_BUFFERS; ++k) {
        inp.Z[k] = states[k] * INV_SQRT_N;
    }
    // Running sum over horizontal buffers Z[DIM_K..DIM_K+i].
    double sum_h = 0.0;
    for (int i = 0; i < DIM_K; ++i) {
        sum_h += inp.Z[DIM_K + i];
        inp.W[i] = M_ENTRY * (inp.Z[i] + sum_h);
    }
    // After the loop sum_h == Σ Z[DIM_K..2*DIM_K-1]. Cache for downstream
    // policies (BCP / HighLow pricing, WMP scheduling, threshold/hardstop
    // outsourcing) so they don't each recompute it.
    inp.sum_Z_h = sum_h;
}

// ---------------------------------------------------------------------------
// nn_pricing
// ---------------------------------------------------------------------------
// Python (NNPricingPolicy with pricing_outside='hold_last_price'):
//   G = model(W_last).numpy()                    shape (DIM_K,)
//   pricing_control = -0.5 * G @ NN_PRICING_MATRIX.T * inv_sqrt_n
//
// For CC3: NN_PRICING_MATRIX[r][k] = (inv(H) @ M_tilde.T)[r][k]
//   pricing_control[r] = -0.5 * sum_k(G[k] * NN_PRICING_MATRIX[r][k]) * INV_SQRT_N
void nn_pricing(const PolicyInputs& inp, double pricing_control[N_ROUTES]) {
    // Use cached G = NN(W_last), populated by compute_policy.
    // No NN call here — shared with nn_outsourcing in the same step.
    const double* G = inp.G_cached;

    for (int r = 0; r < N_ROUTES; ++r) {
        double val = 0.0;
        for (int k = 0; k < DIM_K; ++k) {
            val += G[k] * NN_PRICING_MATRIX[r][k];
        }
        pricing_control[r] = -0.5 * val * INV_SQRT_N;
    }
}

// ---------------------------------------------------------------------------
// bcp_linear_pricing
// ---------------------------------------------------------------------------
// Evaluates the compact (phi_r, phi_{-r}) parameterisation of Φ directly,
// avoiding the full N_ROUTES × N_BUFFERS matrix multiply. Mathematically
// identical to the original matrix form (may differ at ULP level due to
// different floating-point accumulation order).
//
//   Vertical route r = 0..DIM_K-1   (own = {Z[r]})
//     pc[r] = (phi_r * Z[r] + phi_{-r} * (sum_Z - Z[r])) * INV_SQRT_N
//
//   Horizontal route r = DIM_K      (own = Z[DIM_K..2*DIM_K-1])
//     pc[r] = (phi_r * sum_Z_h + phi_{-r} * (sum_Z - sum_Z_h)) * INV_SQRT_N
//
// where sum_Z   = Σ_k Z[k]  and  sum_Z_h = Σ_{k=DIM_K..2*DIM_K-1} Z[k].
// RC_PRICING_PHI[r][0] = phi_r, RC_PRICING_PHI[r][1] = phi_{-r}.
void bcp_linear_pricing(const PolicyInputs& inputs, double pricing_control[N_ROUTES]) {
    // sum over all buffers (reused across routes)
    double sum_Z = 0.0;
    for (int k = 0; k < N_BUFFERS; ++k) sum_Z += inputs.Z[k];

    // Vertical routes: own buffer is Z[r]
    for (int r = 0; r < DIM_K; ++r) {
        const double z_own   = inputs.Z[r];
        const double z_other = sum_Z - z_own;
        pricing_control[r] =
            (RC_PRICING_PHI[r][0] * z_own
           + RC_PRICING_PHI[r][1] * z_other) * INV_SQRT_N;
    }

    // Horizontal route (index DIM_K): own buffer set is Z[DIM_K..2*DIM_K-1];
    // use the horizontal sum already computed by compute_Z_and_W.
    const double sum_Z_h = inputs.sum_Z_h;
    const double sum_Z_v = sum_Z - sum_Z_h;
    pricing_control[DIM_K] =
        (RC_PRICING_PHI[DIM_K][0] * sum_Z_h
       + RC_PRICING_PHI[DIM_K][1] * sum_Z_v) * INV_SQRT_N;
}

// ---------------------------------------------------------------------------
// high_low_pricing
// ---------------------------------------------------------------------------
// Python HighLowPricingPolicy.calculate_pricing_control():
//   point = Z * sqrt_n  (integer queue lengths)
//   pricing_control[r] = where(point @ SUM_OP[r] > RC_HL_PARA[r][0],
//                              RC_HL_PARA[r][2], RC_HL_PARA[r][1]) * inv_sqrt_n
//
// Implementation exploits the sparse structure of SUM_OP (identical to the
// one used in hardstop_finite_buffer and in the compact BCP Φ layout):
//   Vertical route r = 0..DIM_K-1  : per-route sum reduces to Z[r]
//   Horizontal route r = DIM_K     : per-route sum is Σ Z[DIM_K..2*DIM_K-1]
//
// RC_HL_PARA[r][0] is stored in Z-scale (already multiplied by INV_SQRT_N at
// search time), so the runtime comparison is a direct Z > threshold check —
// no per-step divide. RC_HL_PARA[r][1] / [2] are the low / high drifts.
void high_low_pricing(const PolicyInputs& inputs, double pricing_control[N_ROUTES]) {
    // Snapshot the thread_local HL table pointer once to avoid repeated TLS
    // lookups inside the hot loop.
    const double (*hl)[3] = RC_HL_PARA;

    // Vertical routes: per-route scaled total is just Z[r].
    for (int r = 0; r < DIM_K; ++r) {
        double drift = (inputs.Z[r] > hl[r][0]) ? hl[r][2] : hl[r][1];
        pricing_control[r] = drift * INV_SQRT_N;
    }
    // Horizontal route: scaled total is Σ Z[DIM_K..2*DIM_K-1]
    // (pre-computed by compute_Z_and_W and cached in inputs.sum_Z_h).
    double drift_h = (inputs.sum_Z_h > hl[DIM_K][0]) ? hl[DIM_K][2] : hl[DIM_K][1];
    pricing_control[DIM_K] = drift_h * INV_SQRT_N;
}

// ---------------------------------------------------------------------------
// wmp_scheduling
// ---------------------------------------------------------------------------
// Python: WMPObj2Policy.calculate_scheduling_control() + _assign_scheduling_control()
//
// For each station kk:
//   RC_WMP_NO_QUEUE=0 (v1, full pressure):
//     wmp_extra_relative=0 (plain v1):
//       front_value[kk]  = (Z[kk] + lambda_tilde*(Z[kk] - beta_tilde_v)) * w_eff[kk]
//       back_value[K-1]  = (Z[2K-1] + lambda_tilde*(sum(Z[K:2K]) - beta_tilde_h)) * w_eff[2K-1]
//     wmp_extra_relative=1 (pressure term normalized by B*rho):
//       front_value[kk]  = (Z[kk]   + lambda_tilde*(Z[kk]          - beta_tilde_v)/(B_V*rho)) * w_eff[kk]
//       back_value[K-1]  = (Z[2K-1] + lambda_tilde*(sum(Z[K:2K])   - beta_tilde_h)/(B_H*rho)) * w_eff[2K-1]
//   RC_WMP_NO_QUEUE=1 (v2, no-queue pressure) [wmp_extra_relative ignored]:
//     front_value[kk]  = (lambda_tilde*(Z[kk] - beta_tilde_v)) * w_eff[kk]
//     back_value[K-1]  = (lambda_tilde*(sum(Z[K:2K]) - beta_tilde_h)) * w_eff[2K-1]
//   RC_WMP_NO_QUEUE=2 (v3, normalized pressure) [wmp_extra_relative ignored]:
//     front_value[kk]  = lambda_tilde*((Z[kk] - beta_tilde_v)/(B_V*rho)) * w_eff[kk]
//     back_value[K-1]  = lambda_tilde*((sum(Z[K:2K]) - beta_tilde_h)/(B_H*rho)) * w_eff[2K-1]
//   back_value[kk<K-1] = (Z[K+kk] - Z[K+kk+1]) * w_eff[K+kk]  (same for all)
//   pos[kk] = (front_value[kk] > back_value[kk])
//
// w_eff weights (RC_WMP_RELATIVE):
//   0 = absolute: w_eff = W_ARR          (Python: w_arr = mu_arr.copy())
//   1 = relative: w_eff = W_ARR / b_arr  (Python: w_arr = mu_arr / b_arr)
//                  vertical buffers [0..DIM_K-1]:       b = BETA_CONST_V
//                  horizontal buffers [DIM_K..2*DIM_K-1]: b = BETA_CONST_H
//
// Then work-conserving overrides:
//   if Z[kk] < 1e-4  (vertical empty): serve horizontal → sched[kk] = DIM_K+kk+1
//   if Z[DIM_K+kk] < 1e-4 (horizontal empty): serve vertical → sched[kk] = kk+1
//   if both empty: idle → sched[kk] = 0
void wmp_scheduling(const PolicyInputs& inp, int scheduling_control[DIM_K]) {
    const double lambda_tilde  = RC_LAMBDA_TILDE;
    const double beta_tilde_v  = RC_BETA_TILDE_V;
    const double beta_tilde_h  = RC_BETA_TILDE_H;
    const double rho           = RC_OUTSOURCE_THRESHOLD;  // needed for v3 / extra_relative normalization

    // Effective w weights: absolute (W_ARR as-is) or relative (W_ARR / b_arr).
    // W_ARR / BETA_CONST_V / BETA_CONST_H are all compile-time constants and
    // RC_WMP_RELATIVE is a run-time flag that does not change during a single
    // simulation, so we cache w_eff across calls, rebuilding only when the
    // flag actually changes (once per process in practice).
    //
    // thread_local: each OpenMP worker thread (sim_benchmark_omp) keeps its
    // own copy, so there is no race on init.
    static thread_local double w_eff[2 * DIM_K];
    static thread_local int    s_w_eff_last_relative = -1;   // sentinel: "not yet initialized"
    if (RC_WMP_RELATIVE != s_w_eff_last_relative) {
        if (RC_WMP_RELATIVE) {
            for (int k = 0; k < DIM_K; ++k)
                w_eff[k]         = W_ARR[k]         / BETA_CONST_V;
            for (int k = 0; k < DIM_K; ++k)
                w_eff[DIM_K + k] = W_ARR[DIM_K + k] / BETA_CONST_H;
        } else {
            for (int k = 0; k < 2 * DIM_K; ++k)
                w_eff[k] = W_ARR[k];
        }
        s_w_eff_last_relative = RC_WMP_RELATIVE;
    }

    // Horizontal-buffer sum Σ Z[DIM_K..2*DIM_K-1]: already computed in
    // compute_Z_and_W and cached in inp.sum_Z_h.
    const double sum_Z_h = inp.sum_Z_h;

    // Pre-compute reciprocals of (B_V * rho) and (B_H * rho) so the per-kk
    // loop can use a multiplication instead of a division. Only the two
    // branches (wmp_no_queue == 2 and wmp_extra_relative == 1) actually use
    // these, so the computation is free when neither is active.
    const double inv_denom_v = 1.0 / (BETA_CONST_V * rho);
    const double inv_denom_h = 1.0 / (BETA_CONST_H * rho);

    bool pos[DIM_K];
    for (int kk = 0; kk < DIM_K; ++kk) {
        double front_val;
        if (RC_WMP_NO_QUEUE == 2) {
            // v3: normalized pressure, divided by (B_V * rho)
            front_val = (lambda_tilde * (inp.Z[kk] - beta_tilde_v)
                         * inv_denom_v) * w_eff[kk];
        } else if (RC_WMP_NO_QUEUE == 1) {
            // v2: no-queue pressure
            front_val = (lambda_tilde * (inp.Z[kk] - beta_tilde_v)) * w_eff[kk];
        } else {
            // v1: full pressure (includes Z[kk])
            // wmp_extra_relative=1: normalize the pressure term by (B_V * rho)
            //   front = (Z[kk] + λ̃ * (Z[kk] - β̃_v) / (B_V * rho)) * w_eff[kk]
            // wmp_extra_relative=0: plain v1
            //   front = (Z[kk] + λ̃ * (Z[kk] - β̃_v)) * w_eff[kk]
            double pressure_v = lambda_tilde * (inp.Z[kk] - beta_tilde_v);
            if (RC_WMP_EXTRA_RELATIVE)
                pressure_v *= inv_denom_v;
            front_val = (inp.Z[kk] + pressure_v) * w_eff[kk];
        }

        double back_val;
        if (kk < DIM_K - 1) {
            // middle back: same for all variants
            back_val = (inp.Z[DIM_K + kk] - inp.Z[DIM_K + kk + 1]) * w_eff[DIM_K + kk];
        } else {
            // kk == DIM_K-1, last back
            double pressure_h = lambda_tilde * (sum_Z_h - beta_tilde_h);
            if (RC_WMP_NO_QUEUE == 2) {
                // v3: normalized by (B_H * rho)
                pressure_h *= inv_denom_h;
                back_val = pressure_h * w_eff[2 * DIM_K - 1];
            } else if (RC_WMP_NO_QUEUE == 1) {
                // v2: no-queue pressure
                back_val = pressure_h * w_eff[2 * DIM_K - 1];
            } else {
                // v1: full pressure (includes Z[2K-1])
                // wmp_extra_relative=1: normalize the pressure term by (B_H * rho)
                //   back[K-1] = (Z[2K-1] + λ̃ * (ΣZ_h - β̃_h) / (B_H * rho)) * w_eff[2K-1]
                // wmp_extra_relative=0: plain v1
                //   back[K-1] = (Z[2K-1] + λ̃ * (ΣZ_h - β̃_h)) * w_eff[2K-1]
                if (RC_WMP_EXTRA_RELATIVE)
                    pressure_h *= inv_denom_h;
                back_val = (inp.Z[2*DIM_K - 1] + pressure_h) * w_eff[2 * DIM_K - 1];
            }
        }
        pos[kk] = (front_val > back_val);
    }

    // Assign scheduling: pos[kk] → serve vertical (kk+1); else → serve horizontal (DIM_K+kk+1)
    for (int kk = 0; kk < DIM_K; ++kk) {
        scheduling_control[kk] = pos[kk] ? (kk + 1) : (DIM_K + kk + 1);
    }

    // Work-conserving overrides (Python: _assign_scheduling_control)
    for (int kk = 0; kk < DIM_K; ++kk) {
        bool vert_empty  = (inp.Z[kk]         < 1e-4);
        bool horiz_empty = (inp.Z[DIM_K + kk] < 1e-4);

        if (vert_empty && horiz_empty) {
            scheduling_control[kk] = 0;           // both empty → idle
        } else if (vert_empty) {
            scheduling_control[kk] = DIM_K + kk + 1;  // vertical empty → serve horizontal
        } else if (horiz_empty) {
            scheduling_control[kk] = kk + 1;          // horizontal empty → serve vertical
        }
        // else: keep pos-based assignment
    }
}

// ---------------------------------------------------------------------------
// _assign_scheduling_control_wc  (work-conserving override, shared helper)
// ---------------------------------------------------------------------------
// After any scheduling policy sets pos[kk], apply idle overrides:
//   if both vertical buf[kk] and horizontal buf[DIM_K+kk] are empty → idle
//   if vertical empty only → serve horizontal
//   if horizontal empty only → serve vertical
// This matches SchedulingPolicy._assign_scheduling_control() in Python.
static void assign_scheduling_wc(const bool pos[DIM_K], const PolicyInputs& inp,
                                 int scheduling_control[DIM_K]) {
    for (int kk = 0; kk < DIM_K; ++kk)
        scheduling_control[kk] = pos[kk] ? (kk + 1) : (DIM_K + kk + 1);

    for (int kk = 0; kk < DIM_K; ++kk) {
        bool vert_empty  = (inp.Z[kk]         < 1e-4);
        bool horiz_empty = (inp.Z[DIM_K + kk] < 1e-4);
        if (vert_empty && horiz_empty)
            scheduling_control[kk] = 0;
        else if (vert_empty)
            scheduling_control[kk] = DIM_K + kk + 1;
        else if (horiz_empty)
            scheduling_control[kk] = kk + 1;
    }
}

// ---------------------------------------------------------------------------
// lrs_queue_len_scheduling
// ---------------------------------------------------------------------------
// Python: LeastRelativeSlackQueueLenPolicy.calculate_scheduling_control()
//   self.pos = (Z[:, :dim_K] / b_arr[:dim_K]) > (cumsum(Z[:,dim_K:2*dim_K], axis=1) / b_arr[dim_K])
//
// For station kk:
//   pos[kk] = Z[kk] / B_VEC[kk] > cumsum(Z[DIM_K..DIM_K+kk]) / B_VEC[DIM_K]
//
// b_arr[:dim_K] = [B_V, B_V, B_V],  b_arr[dim_K] = B_H
void lrs_queue_len_scheduling(const PolicyInputs& inp, int scheduling_control[DIM_K]) {
    bool pos[DIM_K];
    for (int kk = 0; kk < DIM_K; ++kk) {
        double cumsum_h = 0.0;
        for (int i = 0; i <= kk; ++i)
            cumsum_h += inp.Z[DIM_K + i];
        pos[kk] = (inp.Z[kk] / B_VEC[kk]) > (cumsum_h / B_VEC[DIM_K]);
    }
    assign_scheduling_wc(pos, inp, scheduling_control);
}

// ---------------------------------------------------------------------------
// lrs_delay_time_scheduling
// ---------------------------------------------------------------------------
// Python: LeastRelativeSlackDelayTimePolicy.calculate_scheduling_control()
//   self.pos = (delay_time[:,:dim_K] / delay_bound[:dim_K]) > (delay_time[:,dim_K:2*dim_K] / delay_bound[dim_K])
//
// For station kk:
//   pos[kk] = delay_time[kk] / DELAY_BOUND[kk] > delay_time[DIM_K+kk] / DELAY_BOUND[DIM_K]
//
// DELAY_BOUND[:DIM_K] = [DELAY_BOUND_V, DELAY_BOUND_V, DELAY_BOUND_V]
// DELAY_BOUND[DIM_K] = DELAY_BOUND_H = 90.0
void lrs_delay_time_scheduling(const PolicyInputs& inp, int scheduling_control[DIM_K]) {
    bool pos[DIM_K];
    for (int kk = 0; kk < DIM_K; ++kk) {
        pos[kk] = (inp.delay_time[kk]        / DELAY_BOUND[kk])
                > (inp.delay_time[DIM_K + kk] / DELAY_BOUND[DIM_K]);
    }
    assign_scheduling_wc(pos, inp, scheduling_control);
}

// ---------------------------------------------------------------------------
// eps_diag / eps_offdiag  —  epsilon-band threshold helpers
// ---------------------------------------------------------------------------
// RC_EPSILON_RELATIVE == 0 (absolute, default):
//   diagonal:      threshold = DIAG_FACE_CONST * rho  + sign * epsilon_band
//   off-diagonal:  constant  = OFFDIAG_FACE_CONST * rho + sign * sqrt(2) * epsilon_band
//
// RC_EPSILON_RELATIVE == 1 (relative):
//   diagonal:      threshold = DIAG_FACE_CONST  * (rho + sign * epsilon_band)
//   off-diagonal:  constant  = OFFDIAG_FACE_CONST * (rho + sign * epsilon_band)
//
// sign = +1.0 for outward band (hold_last_no_takeover)
// sign = -1.0 for inward band  (inward)
// ---------------------------------------------------------------------------
static inline double eps_diag(double rho, double sign) {
    return RC_EPSILON_RELATIVE
        ? DIAG_FACE_CONST  * (rho + sign * RC_EPSILON_BAND)
        : DIAG_FACE_CONST  *  rho + sign * RC_EPSILON_BAND;
}
static inline double eps_offdiag(double rho, double sign) {
    return RC_EPSILON_RELATIVE
        ? OFFDIAG_FACE_CONST * (rho + sign * RC_EPSILON_BAND)
        : OFFDIAG_FACE_CONST *  rho + sign * 1.41421356 * RC_EPSILON_BAND;
}

// ---------------------------------------------------------------------------
// nn_outsourcing_hold_last_no_takeover
// ---------------------------------------------------------------------------
// Python: NNOutsourcingPolicy with idle_new_reflection =
//   'epsilon_band_hold_last_outsource_simultaneously_no_takeover'
//   (lines 489-553 of functions_general.py)
//
// N_FACES = 6: faces 0..2 (diagonal), faces 3..5 (off-diagonal)
//   face oo=kk  (kk=0..2): diagonal, W[kk] > DIAG_FACE_CONST * rho
//   face oo=3:  off-diagonal (kk_U=0, kk_I=1): W[0] > W[1] + OFFDIAG_FACE_CONST * rho
//   face oo=4:  off-diagonal (kk_U=0, kk_I=2): W[0] > W[2] + OFFDIAG_FACE_CONST * rho
//   face oo=5:  off-diagonal (kk_U=1, kk_I=2): W[1] > W[2] + OFFDIAG_FACE_CONST * rho
//
// pos_matrix[oo]: face oo violated
// pos_matrix_eps[oo]: within epsilon-band above face oo threshold
//
// mask_one: exactly 1 face violated
// mask_band: all eps-conditions satisfied AND at least 1 face violated
// mask_outside: some eps-condition NOT satisfied
//
// For mask_workload_solu (mask_one & mask_band):
//   NN-guided decision per violated face using W_last
//   Diagonal face kk:
//     pos_NN = G[kk] - C_V_OVER_M <= (G[0]+G[1]+G[2]) - C_H_OVER_M
//     ~pos_NN: outsource route kk (U_{kk+1})
//      pos_NN: idle station DIM_K (U_{dim_K+1}) → outsource_ctrl[N_ROUTES] = 1
//   Off-diagonal face (kk_U, kk_I):
//     Safety-stock guard first (mirrors idling_policy): if kk_I has a downstream
//     buffer (kk_I < DIM_K-1) that would fall below safety stock
//     (Z[DIM_K+kk_I+1] < ss) while Z[DIM_K+kk_I] is non-empty, force outsource
//     route kk_U (idling kk_I is unsafe).  Otherwise fall through to the NN test:
//     pos_NN = G[kk_U] - C_V_OVER_M <= -G[kk_I]
//     ~pos_NN: outsource route kk_U (U_{kk_U+1})
//      pos_NN: idle station kk_I → outsource_ctrl[N_ROUTES + kk_I] = 1
//
// For mask_forced_outsourcing ((~mask_one & mask_band) | mask_outside):
//   Like the workload solution, choose per violated face by NN cost
//   comparison, but only among outsourcing options (idling excluded):
//     Diagonal face kk: both candidates are outsourcing → same NN test as
//       above (outsource route kk vs horizontal route DIM_K).
//     Off-diagonal face (kk_U, kk_I): the idle-station option is excluded,
//       so directly outsource route kk_U (no comparison).
//
// pos_matrix_eps uses eps_diag/eps_offdiag helpers to support RC_EPSILON_RELATIVE:
//   0=absolute: threshold ± eps_band (×sqrt2 for off-diagonal)
//   1=relative: FACE_CONST × (rho ± eps_band)
void nn_outsourcing_hold_last_no_takeover(const PolicyInputs& inp,
                                          int outsourcing_control[N_OUTSOURCE_CTRL]) {
    const double rho = RC_OUTSOURCE_THRESHOLD;

    // Initialize to zero
    for (int i = 0; i < N_OUTSOURCE_CTRL; ++i) outsourcing_control[i] = 0;

    // Hoist kk-independent face thresholds.
    const double diag_thresh = DIAG_FACE_CONST    * rho;
    const double offdiag_add = OFFDIAG_FACE_CONST * rho;

    // ---- Pass 1: build pos_matrix + accumulate n_violated in one loop ----
    bool pos_matrix[N_FACES];
    int  n_violated = 0;

    for (int kk = 0; kk < DIM_K; ++kk) {
        bool v = (inp.W[kk] > diag_thresh);
        pos_matrix[kk] = v;
        if (v) ++n_violated;
    }
    int oo = DIM_K;
    for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
        for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
            bool v = (inp.W[kk_U] > inp.W[kk_I] + offdiag_add);
            pos_matrix[oo] = v;
            if (v) ++n_violated;
            ++oo;
        }
    }

    // Early exit when no face is violated — skip eps-band construction
    // and the NN gradient read entirely.
    if (n_violated == 0) return;  // inside state space: no outsourcing

    // ---- Pass 2: compute mask_state_space_eps (only reached if any violation) ----
    const double diag_eps_thresh = eps_diag(rho, +1.0);
    const double offdiag_eps_add = eps_offdiag(rho, +1.0);

    bool mask_state_space_eps = true;
    for (int kk = 0; kk < DIM_K; ++kk) {
        if (!(inp.W[kk] < diag_eps_thresh)) { mask_state_space_eps = false; }
    }
    for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
        for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
            if (!(inp.W[kk_U] < inp.W[kk_I] + offdiag_eps_add)) {
                mask_state_space_eps = false;
            }
        }
    }

    bool mask_one = (n_violated == 1);
    bool any_violated = (n_violated > 0);
    bool mask_band    = mask_state_space_eps && any_violated;
    bool mask_outside = !mask_state_space_eps;
    bool mask_workload_solu  = mask_one && mask_band;
    bool mask_forced         = (!mask_one && mask_band) || mask_outside;

    // Reuse cached G = NN(W_last), populated by compute_policy — no NN call.
    const double* G = inp.G_cached;

    double sum_G = 0.0;
    for (int k = 0; k < DIM_K; ++k) sum_G += G[k];

    if (mask_workload_solu) {
        // Exactly one face violated → NN-guided decision

        // Diagonal faces 0..DIM_K-1
        for (int kk = 0; kk < DIM_K; ++kk) {
            if (!pos_matrix[kk]) continue;

            // pos_NN: G[kk] - C_V_OVER_M <= sum_G - C_H_OVER_M
            bool pos_NN = (G[kk] - C_V_OVER_M <= sum_G - C_H_OVER_M);

            if (!pos_NN) {
                // Use U_{kk+1}: outsource vertical route kk
                outsourcing_control[kk] = 1;
            } else {
                // Use U_{DIM_K+1}: outsource horizontal route (index DIM_K = N_ROUTES-1 = 3)
                // Python: outsourcing_control[..., dim_K] = 1
                //   dim_K = 3 = column 3 = horizontal route outsourcing flag
                outsourcing_control[DIM_K] = 1;
            }
        }

        // Off-diagonal faces DIM_K..N_FACES-1
        // Safety-stock guard (mirrors idling_policy): before idling station
        // kk_I, check whether its downstream horizontal buffer would fall below
        // the safety stock.  If so, idling is unsafe → force outsourcing instead.
        const double ss = RC_SAFETY_STOCK * INV_SQRT_N - 1e-4;
        oo = DIM_K;
        for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
            for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
                if (pos_matrix[oo]) {
                    // Station kk_I = DIM_K-1 has no downstream buffer (idling_policy
                    // skips it), so treat it as "no safety stock needed".
                    bool need_ss = (kk_I < DIM_K - 1)
                                   && (inp.Z[DIM_K + kk_I + 1] < ss)
                                   && (inp.Z[DIM_K + kk_I] >= 1e-4);

                    if (need_ss) {
                        // Idling kk_I would starve its downstream → outsource kk_U
                        outsourcing_control[kk_U] = 1;
                    } else {
                        // pos_NN: G[kk_U] - C_V_OVER_M <= -G[kk_I]
                        bool pos_NN = (G[kk_U] - C_V_OVER_M <= -G[kk_I]);

                        if (!pos_NN) {
                            // Use U_{kk_U+1}: outsource route kk_U
                            outsourcing_control[kk_U] = 1;
                        } else {
                            // Use I_{kk_I}: idle station kk_I
                            outsourcing_control[N_ROUTES + kk_I] = 1;
                        }
                    }
                }
                ++oo;
            }
        }
    } else if (mask_forced) {
        // Multiple faces violated or outside eps band → forced outsourcing.
        // Like mask_workload_solu, choose per violated face by NN cost
        // comparison, BUT only among outsourcing options (idling excluded).

        // Diagonal faces: both candidates are outsourcing (route kk vs
        // horizontal route DIM_K) → keep the NN comparison.
        for (int kk = 0; kk < DIM_K; ++kk) {
            if (!pos_matrix[kk]) continue;

            // pos_NN: G[kk] - C_V_OVER_M <= sum_G - C_H_OVER_M
            bool pos_NN = (G[kk] - C_V_OVER_M <= sum_G - C_H_OVER_M);

            if (!pos_NN) {
                // Use U_{kk+1}: outsource vertical route kk
                outsourcing_control[kk] = 1;
            } else {
                // Use U_{DIM_K+1}: outsource horizontal route (index DIM_K)
                outsourcing_control[DIM_K] = 1;
            }
        }

        // Off-diagonal faces: the two candidates are outsource route kk_U vs
        // idle station kk_I.  Idling is excluded here, so no comparison —
        // directly take the outsourcing option (route kk_U).
        oo = DIM_K;
        for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
            for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
                if (pos_matrix[oo]) {
                    outsourcing_control[kk_U] = 1;
                }
                ++oo;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// threshold_outsourcing
// ---------------------------------------------------------------------------
// Python: ThresholdBasedPolicy.calculate_outsourcing_control()
//   Z_groups = Z @ sum_queue_len_operator.T
//   pos_matrix = Z_groups >= outsourcing_ratio_threshold * b_arr
//   outsourcing_control = where(pos_matrix, 1, 0)
//
// outsourcing_ratio_threshold = [rho, rho, rho, rho] (same rho for all routes)
// → outsource route r when Z_groups[r] >= RC_OUTSOURCE_THRESHOLD * B_VEC[r]
//
// Implementation uses the sparse structure of SUM_OP (identical to the pattern
// used in hardstop_finite_buffer and high_low_pricing):
//   Vertical routes r = 0..DIM_K-1  : Z_group = Z[r],       threshold rho * B_V
//   Horizontal route r = DIM_K      : Z_group = Σ Z[DIM_K..2*DIM_K-1], threshold rho * B_H
// Bit-identical to the original dense Z @ SUM_OP[r] formulation (zero entries
// contribute zero exactly under IEEE 754).
void threshold_outsourcing(const PolicyInputs& inputs,
                           int outsourcing_control[N_OUTSOURCE_CTRL]) {
    for (int i = 0; i < N_OUTSOURCE_CTRL; ++i) outsourcing_control[i] = 0;

    const double rho = RC_OUTSOURCE_THRESHOLD;

    // Vertical routes: Z_group reduces to Z[r]; shared threshold rho * B_V.
    const double thresh_v = rho * B_V;
    for (int r = 0; r < DIM_K; ++r) {
        if (inputs.Z[r] >= thresh_v) outsourcing_control[r] = 1;
    }
    // Horizontal route (index DIM_K): Z_group = Σ Z[DIM_K..2*DIM_K-1]
    // (already computed by compute_Z_and_W).
    if (inputs.sum_Z_h >= rho * B_H) outsourcing_control[DIM_K] = 1;
}

// ---------------------------------------------------------------------------
// single_pushing_outsourcing
// ---------------------------------------------------------------------------
// Python: SinglePushingPerFacePolicy.calculate_outsourcing_control()
//   with idle_new_reflection = 'epsilon_band_hold_last_outsource_simultaneously_no_takeover'
//
// Same face geometry as nn_outsourcing_hold_last_no_takeover but no NN:
//   mask_workload_solu (exactly 1 face, within epsilon band):
//     diagonal face kk → outsource route kk (U_{kk+1})
//     off-diagonal face (kk_U, kk_I) → idle station kk_I (I_{kk_I}), unless a
//       safety-stock guard (mirroring idling_policy) fires: if kk_I has a
//       downstream buffer (kk_I < DIM_K-1) that would drop below safety stock
//       (Z[DIM_K+kk_I+1] < ss) while Z[DIM_K+kk_I] is non-empty, outsource
//       route kk_U instead (kk_I = DIM_K-1 has no downstream → safe to idle)
//   mask_forced (multiple faces or outside band):
//     diagonal face kk → outsource route kk
//     off-diagonal face (kk_U, kk_I) → outsource route kk_U
void single_pushing_outsourcing(const PolicyInputs& inp,
                                int outsourcing_control[N_OUTSOURCE_CTRL]) {
    const double rho = RC_OUTSOURCE_THRESHOLD;

    for (int i = 0; i < N_OUTSOURCE_CTRL; ++i) outsourcing_control[i] = 0;

    // Hoist kk-independent face thresholds out of the build loops.
    const double diag_thresh     = DIAG_FACE_CONST    * rho;
    const double offdiag_add     = OFFDIAG_FACE_CONST * rho;

    // ---- Pass 1: build pos_matrix and accumulate n_violated in one loop ----
    bool pos_matrix[N_FACES];
    int  n_violated = 0;

    for (int kk = 0; kk < DIM_K; ++kk) {
        bool v = (inp.W[kk] > diag_thresh);
        pos_matrix[kk] = v;
        if (v) ++n_violated;
    }
    int oo = DIM_K;
    for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
        for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
            bool v = (inp.W[kk_U] > inp.W[kk_I] + offdiag_add);
            pos_matrix[oo] = v;
            if (v) ++n_violated;
            ++oo;
        }
    }

    // Early exit when no face is violated — skip the more expensive eps-band
    // construction entirely.
    if (n_violated == 0) return;  // inside state space: no outsourcing

    // ---- Pass 2: compute mask_state_space_eps (only reached when n_violated > 0) ----
    // eps_diag / eps_offdiag are also independent of kk / kk_U / kk_I.
    // Note: pos_matrix_eps array itself is not needed — only the reduction
    // mask_state_space_eps = ALL pos_matrix_eps[f] is consumed downstream.
    const double diag_eps_thresh = eps_diag(rho, +1.0);
    const double offdiag_eps_add = eps_offdiag(rho, +1.0);

    bool mask_state_space_eps = true;

    for (int kk = 0; kk < DIM_K; ++kk) {
        if (!(inp.W[kk] < diag_eps_thresh)) { mask_state_space_eps = false; }
    }
    for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
        for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
            if (!(inp.W[kk_U] < inp.W[kk_I] + offdiag_eps_add)) {
                mask_state_space_eps = false;
            }
        }
    }

    bool mask_one = (n_violated == 1);
    bool mask_band    = mask_state_space_eps && (n_violated > 0);
    bool mask_outside = !mask_state_space_eps;
    bool mask_workload_solu = mask_one && mask_band;
    bool mask_forced        = (!mask_one && mask_band) || mask_outside;

    if (mask_workload_solu) {
        // Exactly one face violated → workload solution (no NN needed)
        // Diagonal faces: outsource route kk
        for (int kk = 0; kk < DIM_K; ++kk) {
            if (pos_matrix[kk])
                outsourcing_control[kk] = 1;
        }
        // Off-diagonal faces: idle station kk_I, unless a safety-stock guard
        // (mirroring idling_policy) says idling kk_I would starve its downstream
        // buffer — in which case force outsourcing route kk_U instead.
        const double ss = RC_SAFETY_STOCK * INV_SQRT_N - 1e-4;
        oo = DIM_K;
        for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
            for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
                if (pos_matrix[oo]) {
                    // Station kk_I = DIM_K-1 has no downstream buffer
                    // (idling_policy skips it) → treated as safe to idle.
                    bool need_ss = (kk_I < DIM_K - 1)
                                   && (inp.Z[DIM_K + kk_I + 1] < ss)
                                   && (inp.Z[DIM_K + kk_I] >= 1e-4);
                    if (need_ss)
                        outsourcing_control[kk_U] = 1;            // force outsource
                    else
                        outsourcing_control[N_ROUTES + kk_I] = 1; // idle station kk_I
                }
                ++oo;
            }
        }
    } else if (mask_forced) {
        // Multiple violations or outside band → forced outsourcing
        // Diagonal: outsource route kk
        for (int kk = 0; kk < DIM_K; ++kk) {
            if (pos_matrix[kk])
                outsourcing_control[kk] = 1;
        }
        // Off-diagonal: outsource route kk_U
        oo = DIM_K;
        for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
            for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
                if (pos_matrix[oo])
                    outsourcing_control[kk_U] = 1;
                ++oo;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// nn_outsourcing_inward
// ---------------------------------------------------------------------------
// Python: NNOutsourcingPolicy with idle_new_reflection =
//   'epsilon_band_inward_outsource_simultaneously'  (lines 368-429)
//
// Same face geometry as nn_outsourcing_hold_last_no_takeover, but:
//   - Band is INWARD: pos_matrix checks W > threshold - eps
//                     pos_matrix_eps checks W < threshold (still inside)
//   - mask_outside fires when W >= threshold (outside state space)
//   - NN gradient is computed from current inp.W (not W_last)
void nn_outsourcing_inward(const PolicyInputs& inp,
                           int outsourcing_control[N_OUTSOURCE_CTRL]) {
    const double rho = RC_OUTSOURCE_THRESHOLD;

    for (int i = 0; i < N_OUTSOURCE_CTRL; ++i) outsourcing_control[i] = 0;

    // Inward-band constants (sign=-1): these define the "approaching-boundary"
    // region, used in Pass 1 to build pos_matrix.
    const double diag_eps_thresh = eps_diag(rho, -1.0);
    const double offdiag_eps_add = eps_offdiag(rho, -1.0);

    // ---- Pass 1: build pos_matrix + accumulate n_violated ----
    bool pos_matrix[N_FACES];
    int  n_violated = 0;

    for (int kk = 0; kk < DIM_K; ++kk) {
        bool v = (inp.W[kk] > diag_eps_thresh);
        pos_matrix[kk] = v;
        if (v) ++n_violated;
    }
    int oo = DIM_K;
    for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
        for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
            bool v = (inp.W[kk_U] > inp.W[kk_I] + offdiag_eps_add);
            pos_matrix[oo] = v;
            if (v) ++n_violated;
            ++oo;
        }
    }

    if (n_violated == 0) return;  // far inside: no outsourcing

    // ---- Pass 2: compute mask_state_space_eps (only if any face approaching) ----
    // State-space thresholds (not eps band) used in Pass 2 only.
    const double diag_thresh = DIAG_FACE_CONST    * rho;
    const double offdiag_add = OFFDIAG_FACE_CONST * rho;

    bool mask_state_space_eps = true;
    for (int kk = 0; kk < DIM_K; ++kk) {
        if (!(inp.W[kk] < diag_thresh)) { mask_state_space_eps = false; }
    }
    for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
        for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
            if (!(inp.W[kk_U] < inp.W[kk_I] + offdiag_add)) {
                mask_state_space_eps = false;
            }
        }
    }

    bool mask_one = (n_violated == 1);
    bool mask_band    = mask_state_space_eps && (n_violated > 0);  // in inward band
    bool mask_outside = !mask_state_space_eps;                      // past boundary

    bool mask_workload_solu = mask_one && mask_band;
    bool mask_forced        = (!mask_one && mask_band) || mask_outside;

    // Query NN using current W (not W_last).
    // When inside the state space (G_at_W_valid=true), W_last == W so the
    // cached G_cached is already NN(W); reuse it and skip the NN call.
    double G_local[DIM_K];
    const double* G;
    if (inp.G_at_W_valid) {
        G = inp.G_cached;
    } else {
        nn_predict(inp.W, G_local);
        G = G_local;
    }

    double sum_G = 0.0;
    for (int k = 0; k < DIM_K; ++k) sum_G += G[k];

    if (mask_workload_solu) {
        // Exactly one face approaching → NN-guided decision

        // Diagonal faces 0..DIM_K-1
        for (int kk = 0; kk < DIM_K; ++kk) {
            if (!pos_matrix[kk]) continue;
            bool pos_NN = (G[kk] - C_V_OVER_M <= sum_G - C_H_OVER_M);
            if (!pos_NN)
                outsourcing_control[kk] = 1;        // U_{kk+1}: outsource route kk
            else
                outsourcing_control[DIM_K] = 1;     // U_{DIM_K+1}: outsource horizontal
        }

        // Off-diagonal faces DIM_K..N_FACES-1
        oo = DIM_K;
        for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
            for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
                if (pos_matrix[oo]) {
                    bool pos_NN = (G[kk_U] - C_V_OVER_M <= -G[kk_I]);
                    if (!pos_NN)
                        outsourcing_control[kk_U] = 1;            // U_{kk_U+1}
                    else
                        outsourcing_control[N_ROUTES + kk_I] = 1; // I_{kk_I}
                }
                ++oo;
            }
        }
    } else if (mask_forced) {
        // Multiple faces approaching or already past boundary → forced outsourcing

        // Diagonal: outsource route kk
        for (int kk = 0; kk < DIM_K; ++kk)
            if (pos_matrix[kk]) outsourcing_control[kk] = 1;

        // Off-diagonal: outsource route kk_U
        oo = DIM_K;
        for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
            for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
                if (pos_matrix[oo]) outsourcing_control[kk_U] = 1;
                ++oo;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// single_pushing_outsourcing_inward
// ---------------------------------------------------------------------------
// Python: SinglePushingPerFacePolicy with idle_new_reflection =
//   'epsilon_band_inward_outsource_simultaneously'  (lines 577-629)
//
// Same inward band geometry as nn_outsourcing_inward, but no NN (no G needed).
// mask_workload_solu: diagonal face kk → outsource route kk (U_{kk+1})
//                     off-diagonal face (kk_U,kk_I) → idle station kk_I (I_{kk_I})
// mask_forced:        diagonal → outsource kk; off-diagonal → outsource kk_U
void single_pushing_outsourcing_inward(const PolicyInputs& inp,
                                       int outsourcing_control[N_OUTSOURCE_CTRL]) {
    const double rho = RC_OUTSOURCE_THRESHOLD;

    for (int i = 0; i < N_OUTSOURCE_CTRL; ++i) outsourcing_control[i] = 0;

    // Inward-band constants (sign=-1) used in Pass 1 (pos_matrix).
    const double diag_eps_thresh = eps_diag(rho, -1.0);
    const double offdiag_eps_add = eps_offdiag(rho, -1.0);

    // ---- Pass 1: build pos_matrix + accumulate n_violated ----
    bool pos_matrix[N_FACES];
    int  n_violated = 0;

    for (int kk = 0; kk < DIM_K; ++kk) {
        bool v = (inp.W[kk] > diag_eps_thresh);
        pos_matrix[kk] = v;
        if (v) ++n_violated;
    }
    int oo = DIM_K;
    for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
        for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
            bool v = (inp.W[kk_U] > inp.W[kk_I] + offdiag_eps_add);
            pos_matrix[oo] = v;
            if (v) ++n_violated;
            ++oo;
        }
    }

    if (n_violated == 0) return;  // far inside: no outsourcing

    // ---- Pass 2: compute mask_state_space_eps (only if any face approaching) ----
    const double diag_thresh = DIAG_FACE_CONST    * rho;
    const double offdiag_add = OFFDIAG_FACE_CONST * rho;

    bool mask_state_space_eps = true;
    for (int kk = 0; kk < DIM_K; ++kk) {
        if (!(inp.W[kk] < diag_thresh)) { mask_state_space_eps = false; }
    }
    for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
        for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
            if (!(inp.W[kk_U] < inp.W[kk_I] + offdiag_add)) {
                mask_state_space_eps = false;
            }
        }
    }

    bool mask_one = (n_violated == 1);
    bool mask_band    = mask_state_space_eps && (n_violated > 0);
    bool mask_outside = !mask_state_space_eps;
    bool mask_workload_solu = mask_one && mask_band;
    bool mask_forced        = (!mask_one && mask_band) || mask_outside;

    if (mask_workload_solu) {
        // Diagonal: outsource route kk
        for (int kk = 0; kk < DIM_K; ++kk)
            if (pos_matrix[kk]) outsourcing_control[kk] = 1;
        // Off-diagonal: idle station kk_I
        oo = DIM_K;
        for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
            for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
                if (pos_matrix[oo]) outsourcing_control[N_ROUTES + kk_I] = 1;
                ++oo;
            }
        }
    } else if (mask_forced) {
        // Diagonal: outsource route kk
        for (int kk = 0; kk < DIM_K; ++kk)
            if (pos_matrix[kk]) outsourcing_control[kk] = 1;
        // Off-diagonal: outsource route kk_U
        oo = DIM_K;
        for (int kk_U = 0; kk_U < DIM_K - 1; ++kk_U) {
            for (int kk_I = kk_U + 1; kk_I < DIM_K; ++kk_I) {
                if (pos_matrix[oo]) outsourcing_control[kk_U] = 1;
                ++oo;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// hardstop_finite_buffer
// ---------------------------------------------------------------------------
// Python: hardstop_finite_buffer() (lines 621-638)
// For each route r: if SUM_OP[r] @ Z >= HARDSTOP_MULTIPLE * B_VEC[r] → set ctrl[r] = 2
// Raises error in Python if triggered; here we just set value to 2 and let caller handle it.
//
// SUM_OP has a sparse structure we exploit directly:
//   - Vertical routes r = 0..DIM_K-1: SUM_OP[r] is one-hot at position r
//       → sum_Z reduces to Z[r]
//       → threshold HARDSTOP_MULTIPLE * B_VEC[r] is identical for all vertical
//         routes (B_VEC[0..DIM_K-1] = B_V), so we hoist the threshold out.
//   - Horizontal route r = DIM_K: SUM_OP[r] is 1 on k = DIM_K..2*DIM_K-1,
//                                  0 elsewhere
//       → sum_Z = Z[DIM_K] + Z[DIM_K+1] + ... + Z[2*DIM_K-1]
//
// This replaces N_ROUTES × N_BUFFERS muladds with DIM_K - 1 adds (the vertical
// multiplies all collapse to identity). Bit-identical to the original.
void hardstop_finite_buffer(const PolicyInputs& inp,
                            int outsourcing_control[N_OUTSOURCE_CTRL]) {
    // Vertical routes: each only reads Z[r], shared threshold.
    const double thresh_v = HARDSTOP_MULTIPLE * B_V;
    for (int r = 0; r < DIM_K; ++r) {
        if (inp.Z[r] >= thresh_v)
            outsourcing_control[r] = 2;
    }
    // Horizontal route (index DIM_K): sum of Z[DIM_K..2*DIM_K-1]
    // (already computed by compute_Z_and_W).
    if (inp.sum_Z_h >= HARDSTOP_MULTIPLE * B_H)
        outsourcing_control[DIM_K] = 2;
}

// ---------------------------------------------------------------------------
// idling_policy
// ---------------------------------------------------------------------------
// Python: idling_policy() (lines 640-691)
//
// CC3 specifics:
//   apply_exit_buffer = 'N':
//     For each station kk: if outsourcing_control[N_ROUTES + kk] == 1 → sched[kk] = 0
//
//   apply_safety_stock = 'Always', safety_stock = -1:
//     ss = -1 * INV_SQRT_N + 1e-4 < 0
//     pos_Z_safety = (Z < ss) → always false (Z >= 0)
//     → safety stock override never fires
//
//   Trim: outsourcing_control columns N_ROUTES..N_ROUTES+DIM_K-1 are internal;
//         after idling_policy they are no longer read by simulation.
void idling_policy(const PolicyInputs& inp, PolicyOutputs& out) {
    // exit_buffer = 'N': if idle station kk flagged → set scheduling to idle
    for (int kk = 0; kk < DIM_K; ++kk) {
        if (out.outsourcing_control[N_ROUTES + kk] == 1) {
            out.scheduling_control[kk] = 0;
        }
    }

    // safety_stock = 0: ss = 0*INV_SQRT_N - 1e-4 < 0 → pos_Z_safety always false (no-op)
    // safety_stock = 1: ss = 1/20 - 1e-4 ≈ 0.0499 → triggers when 20*Z < 1 (i.e. Z=0)
    double ss = RC_SAFETY_STOCK * INV_SQRT_N - 1e-4;
    if (ss > 0.0) {
        for (int kk = 0; kk < DIM_K - 1; ++kk) {
            bool pos_Z_safety_next  = (inp.Z[DIM_K + kk + 1] < ss);
            bool horiz_not_empty    = (inp.Z[DIM_K + kk] >= 1e-4);
            if (pos_Z_safety_next && horiz_not_empty) {
                out.scheduling_control[kk] = DIM_K + kk + 1;
            }
        }
    }

    // Trim: columns N_ROUTES..N_ROUTES+DIM_K-1 are zeroed (no longer meaningful)
    for (int kk = 0; kk < DIM_K; ++kk) {
        out.outsourcing_control[N_ROUTES + kk] = 0;
    }
}

// ---------------------------------------------------------------------------
// compute_policy
// ---------------------------------------------------------------------------
// Python: policy() function (lines 738-781)
//
// Steps:
//   1. compute_Z_and_W
//   2. identify_mode: check if W is inside state space
//      Inside = all faces NOT violated
//   3. Update W_last if inside state space
//   4. Dispatch to sub-policies
//   5. hardstop_finite_buffer, idling_policy
void compute_policy(const int states[N_BUFFERS],
                    const double delay_time[N_BUFFERS],
                    double W_last[DIM_K],
                    double G_last[DIM_K],
                    PolicyOutputs& outputs) {
    PolicyInputs inp;

    L2_PROFILE_TIC(p1);
    // Step 1: compute Z, W
    compute_Z_and_W(states, inp);

    // Copy delay_time into inp only when LRS_DelayTime scheduling will read it.
    // For WMP / LRS_QueueLen the downstream sub-policies never touch
    // inp.delay_time, so we skip the copy (caller may not even fill the source).
    if (RC_SCHEDULING_MODE == (int)LRS_DELAY_TIME) {
        for (int k = 0; k < N_BUFFERS; ++k) {
            inp.delay_time[k] = delay_time[k];
        }
    }

    // Only NN-based policies read W_last / G_cached / G_at_W_valid.
    // All non-NN policies (BCPlinear, HighLow, LRS, WMP, SinglePushing,
    // ThresholdBased) read only inp.Z / inp.W / inp.delay_time, so we skip
    // the inside-check + W_last + G_last maintenance when NN is not in use.
    const bool uses_nn = (RC_PRICING_MODE == (int)NN_PRICING)
                      || (RC_OUTSOURCING_MODE == (int)NN_OUTSOURCING);

    if (uses_nn) {
        // Copy W_last snapshot into inp
        for (int i = 0; i < DIM_K; ++i) {
            inp.W_last[i] = W_last[i];
        }

        // Step 2: identify_mode — check if current W is inside state space
        // Inside = no face violated (same logic as outsourcing, but without eps band)
        bool inside = true;
        const double rho = RC_OUTSOURCE_THRESHOLD;

        for (int kk = 0; kk < DIM_K && inside; ++kk) {
            if (inp.W[kk] > DIAG_FACE_CONST * rho) inside = false;
        }
        for (int kk_U = 0; kk_U < DIM_K - 1 && inside; ++kk_U) {
            for (int kk_I = kk_U + 1; kk_I < DIM_K && inside; ++kk_I) {
                if (inp.W[kk_U] > inp.W[kk_I] + OFFDIAG_FACE_CONST * rho) inside = false;
            }
        }

        // Step 3: update W_last and refresh G_last if inside state space.
        //   When inside: W_last := W, and we call NN once to refresh G_last.
        //                That single NN output serves as NN(W_last) for pricing and
        //                outsourcing (hold_last), and as NN(W) for outsourcing (inward),
        //                since W_last == W in this branch.
        //   When outside: neither W_last nor G_last change; sub-policies reuse G_last.
        if (inside) {
            for (int i = 0; i < DIM_K; ++i) {
                W_last[i]     = inp.W[i];
                inp.W_last[i] = inp.W[i];
            }
            nn_predict(inp.W, G_last);
        }

        // Populate inp.G_cached from G_last for the sub-policies.
        // G_at_W_valid: true when inside, because W_last == W in that case.
        for (int i = 0; i < DIM_K; ++i) {
            inp.G_cached[i] = G_last[i];
        }
        inp.G_at_W_valid = inside;
    }
    L2_PROFILE_TOC(p1, g_prof_p1_zw_prep);

    // Step 4: pricing policy
    L2_PROFILE_TIC(p2);
    switch (RC_PRICING_MODE) {
        case (int)NN_PRICING:
            nn_pricing(inp, outputs.pricing_control);
            break;
        case (int)BCP_LINEAR:
            bcp_linear_pricing(inp, outputs.pricing_control);
            break;
        case (int)HIGH_LOW:
            high_low_pricing(inp, outputs.pricing_control);
            break;
        default:
            nn_pricing(inp, outputs.pricing_control);
            break;
    }
    L2_PROFILE_TOC(p2, g_prof_p2_pricing);

    // Step 4: scheduling policy
    L2_PROFILE_TIC(p3);
    switch (RC_SCHEDULING_MODE) {
        case (int)WMP_OBJ2:
            wmp_scheduling(inp, outputs.scheduling_control);
            break;
        case (int)LRS_QUEUE_LEN:
            lrs_queue_len_scheduling(inp, outputs.scheduling_control);
            break;
        case (int)LRS_DELAY_TIME:
            lrs_delay_time_scheduling(inp, outputs.scheduling_control);
            break;
        default:
            wmp_scheduling(inp, outputs.scheduling_control);
            break;
    }
    L2_PROFILE_TOC(p3, g_prof_p3_scheduling);

    // Step 4: outsourcing policy
    // RC_INWARD_REFLECTION: 0 = outward band (hold_last), 1 = inward band
    L2_PROFILE_TIC(p4);
    switch (RC_OUTSOURCING_MODE) {
        case (int)NN_OUTSOURCING:
            if (RC_INWARD_REFLECTION)
                nn_outsourcing_inward(inp, outputs.outsourcing_control);
            else
                nn_outsourcing_hold_last_no_takeover(inp, outputs.outsourcing_control);
            break;
        case (int)THRESHOLD_BASED:
            threshold_outsourcing(inp, outputs.outsourcing_control);
            break;
        case (int)SINGLE_PUSHING:
            if (RC_INWARD_REFLECTION)
                single_pushing_outsourcing_inward(inp, outputs.outsourcing_control);
            else
                single_pushing_outsourcing(inp, outputs.outsourcing_control);
            break;
        default:
            if (RC_INWARD_REFLECTION)
                nn_outsourcing_inward(inp, outputs.outsourcing_control);
            else
                nn_outsourcing_hold_last_no_takeover(inp, outputs.outsourcing_control);
            break;
    }
    L2_PROFILE_TOC(p4, g_prof_p4_outsourcing);

    // Step 5: hardstop and idling
    L2_PROFILE_TIC(p5);
    hardstop_finite_buffer(inp, outputs.outsourcing_control);
    idling_policy(inp, outputs);
    L2_PROFILE_TOC(p5, g_prof_p5_guard);
}
