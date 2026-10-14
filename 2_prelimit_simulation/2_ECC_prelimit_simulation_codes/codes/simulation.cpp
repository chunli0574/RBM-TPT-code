/*
 * simulation.cpp
 *
 * Implementation of the CC3 simulation event loop and all helper functions.
 *
 * =========================================================================
 * OVERALL STRUCTURE (mirrors Python simulation(), functions_general.py):
 *
 *   run_simulation()
 *     for m in 0 .. N_PATHS-1:
 *       run_one_path(seed=123+m)
 *     compute mean / std over all paths
 *
 *   run_one_path(seed):
 *     for iter in 0 .. N_JUMPS-1:
 *       1. compute_policy()              → pricing, outsourcing, scheduling
 *       2. generate_next_event()         → which event fires + how long (dt)
 *       3. accumulate_cost()             → discounted cost for [t, t+dt]
 *       4. steady-state accumulation     → if iter > 80% of N_JUMPS
 *       5. state.t += dt                 → advance clock
 *       6. apply_transition()            → update integer queue counts
 *       7. qi_update()                   → update delay tracking queues
 *     finalize_steady_state_cost()       → add infinite-tail correction
 *
 * =========================================================================
 * CC3 EVENT MAPPING (N_EVENTS = 10):
 *   Events 0..3: arrivals to routes 0..3
 *     event 0: arrive → buf[0]  (route 0, vertical)
 *     event 1: arrive → buf[1]  (route 1, vertical)
 *     event 2: arrive → buf[2]  (route 2, vertical)
 *     event 3: arrive → buf[3]  (route 3, horizontal start)
 *   Events 4..6: vertical service departures
 *     event 4: serve buf[0] → departure (route 0 final)
 *     event 5: serve buf[1] → departure (route 1 final)
 *     event 6: serve buf[2] → departure (route 2 final)
 *   Events 7..9: horizontal pipeline
 *     event 7: serve buf[3] → transfer to buf[4]
 *     event 8: serve buf[4] → transfer to buf[5]
 *     event 9: serve buf[5] → departure (route 3 final)
 * =========================================================================
 */

#include "simulation.h"
#include "run_config.h"
#include "nn_model.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <vector>

// ==========================================================================
// Optional profiling (compile with -DENABLE_PROFILING=1).
// When ENABLE_PROFILING is defined, the batch run_simulation accumulates
// wall-clock nanoseconds into per-section counters and prints a summary
// at the end of the run. When undefined, all macros expand to nothing and
// the binary is identical to the un-instrumented build.
// ==========================================================================
#ifdef ENABLE_PROFILING
  #include <chrono>
  #include <cstdint>
  #include <cstdlib>   // std::atexit
  #define PROFILE_TIC(name) \
      auto name##_t0 = std::chrono::steady_clock::now()
  #define PROFILE_TOC(name, counter) \
      counter += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>( \
          std::chrono::steady_clock::now() - name##_t0).count()

  // ----- Cumulative counters for serial run_one_path (used by all non-batch
  //       binaries: sim_benchmark, sim_proposed serial path, sim_denser).
  //       Accumulate across ALL paths and ALL run_simulation calls in the
  //       process; dumped at program exit via atexit handler. -----
  namespace {
      uint64_t g_prof_s1_delay     = 0;   // Step 1: head-of-line delay_time construction
      uint64_t g_prof_s2_policy    = 0;   // Step 2: compute_policy (pricing + scheduling + outsourcing + hardstop + idling)
      uint64_t g_prof_s3_event     = 0;   // Step 3: generate_next_event
      uint64_t g_prof_s4_cost      = 0;   // Step 4+5: accumulate_cost + steady-state
      uint64_t g_prof_s5_update    = 0;   // Step 6+7+8: state.t += dt + apply_transition + qi_update
      uint64_t g_prof_path_init    = 0;   // Per-path init (seed rng, initialize queue/state)
      uint64_t g_prof_aggregate    = 0;   // run_simulation: mean/std passes

      void serial_prof_dump_atexit() {
          uint64_t total = g_prof_s1_delay + g_prof_s2_policy + g_prof_s3_event
                         + g_prof_s4_cost  + g_prof_s5_update
                         + g_prof_path_init + g_prof_aggregate;
          if (total == 0) return;  // nothing recorded (e.g., only batch path ran)
          auto pct = [&](uint64_t v){ return 100.0 * v / total; };
          std::printf("\n========== PROFILING (serial run_one_path, cumulative) ==========\n");
          std::printf("  %-22s %18s  %8s\n", "section", "ns", "%sum");
          auto pp = [&](const char* name, uint64_t v){
              std::printf("  %-22s %18llu  %7.2f%%\n",
                          name, (unsigned long long)v, pct(v));
          };
          pp("S1_delay_time",       g_prof_s1_delay);
          pp("S2_compute_policy",   g_prof_s2_policy);
          pp("S3_next_event",       g_prof_s3_event);
          pp("S4_accumulate_cost",  g_prof_s4_cost);
          pp("S5_state_and_queue",  g_prof_s5_update);
          pp("path_init",           g_prof_path_init);
          pp("aggregate_mean_std",  g_prof_aggregate);
          std::printf("  %-22s %18llu  %7.2f%%\n", "sum",
                      (unsigned long long)total, 100.00);
          std::printf("==================================================================\n");
      }
      struct SerialProfInit { SerialProfInit() { std::atexit(serial_prof_dump_atexit); } };
      SerialProfInit g_serial_prof_init;
  }
#else
  #define PROFILE_TIC(name)        do {} while(0)
  #define PROFILE_TOC(name, counter) do {} while(0)
#endif

// ==========================================================================
// SECTION 1: Initialization
// ==========================================================================

void initialize_sim_state(SimState& state) {
    for (int k = 0; k < N_BUFFERS; ++k) {
        state.states[k] = INITIAL_STATE[k];
    }
    state.t = 0.0;
}

void initialize_cost_accum(CostAccum& accum) {
    accum.rec_cost          = 0.0;
    accum.rec_cost_pricing  = 0.0;
    accum.rec_cost_boundary = 0.0;
    for (int r = 0; r < N_ROUTES; ++r) {
        accum.t_outsourced[r] = 0.0;
    }
}

// Pre-load initial jobs into each buffer (all arrived at t=0).
// Python: QueueInfo.__init__(begin=zeros, end=states, enter_system_time=zeros)
void initialize_queue_info(QueueInfo& qi, const int states[N_BUFFERS]) {
    for (int k = 0; k < N_BUFFERS; ++k) {
        while (!qi.job_times[k].empty()) qi.job_times[k].pop();
        for (int j = 0; j < states[k]; ++j) {
            qi.job_times[k].push(0.0);
        }
    }
}

void initialize_finished_info(FinishedInfo& fi) {
    for (int r = 0; r < N_ROUTES; ++r) {
        fi.n_completed[r] = 0;
        fi.n_late[r]      = 0;
    }
}

// ==========================================================================
// SECTION 2: Queue operations
// ==========================================================================

void qi_process_arrival(QueueInfo& qi, int i_buf, double t) {
    qi.job_times[i_buf].push(t);
}

void qi_process_transfer(QueueInfo& qi, int i_buf, int i_buf_next) {
    double entry_t = qi.job_times[i_buf].front();
    qi.job_times[i_buf].pop();
    qi.job_times[i_buf_next].push(entry_t);
}

void qi_process_departure(QueueInfo& qi, FinishedInfo& fi,
                          int i_buf, int route, double t, int iter) {
    double entry_t = qi.job_times[i_buf].front();
    qi.job_times[i_buf].pop();

    if (iter > DELAY_PROB_RECORD_START) {
        double delay = t - entry_t;
        fi.n_completed[route]++;
        if (delay > DELAY_BOUND[route]) {
            fi.n_late[route]++;
        }
        // Optional: bucket this delay into the per-route histogram (sim_one_combo).
        // The pointer being null short-circuits all histogram work — zero overhead
        // for sim/sim_proposed/sim_benchmark/sim_denser, which never set it.
        if (fi.delay_hist) {
            int bucket = (int)std::floor(delay / fi.delay_hist->bucket_width);
            fi.delay_hist->hist[route][bucket]++;
        }
    }
}

// ---------------------------------------------------------------------------
// qi_update: dispatch queue operations based on event index.
//
// Python: QueueInfo.update_queue_info_with_routes() loops over routes.
// For M=1, exactly one event fires per step, so we branch on event_idx.
//
// Generic event layout (works for any DIM_K):
//   Events 0..N_ROUTES-1          : arrivals  — event r → push to buf[r]
//                                    (vertical routes r=0..DIM_K-1 arrive to buf[r];
//                                     horizontal route r=DIM_K arrives to buf[DIM_K])
//   Events N_ROUTES..N_ROUTES+DIM_K-1 : vertical departures
//                                    — event N_ROUTES+kk → depart buf[kk], route kk
//   Events N_ROUTES+DIM_K..N_EVENTS-1 : horizontal pipeline
//                                    — event N_ROUTES+DIM_K+kk:
//                                        kk < DIM_K-1 → transfer buf[DIM_K+kk] → buf[DIM_K+kk+1]
//                                        kk = DIM_K-1 → depart buf[2*DIM_K-1], route DIM_K
//
// CC3 (DIM_K=3, N_ROUTES=4, N_EVENTS=10): events 0-3 arrivals, 4-6 vert, 7-9 horiz
// CC10 (DIM_K=10, N_ROUTES=11, N_EVENTS=31): events 0-10 arrivals, 11-20 vert, 21-30 horiz
// ---------------------------------------------------------------------------
void qi_update(QueueInfo& qi, FinishedInfo& fi, int event_idx, double t, int iter) {
    if (event_idx < N_ROUTES) {
        // Arrivals: event r → push to buf[r]
        qi_process_arrival(qi, event_idx, t);
    } else if (event_idx < N_ROUTES + DIM_K) {
        // Vertical departure: event N_ROUTES+kk → depart from buf[kk], route kk
        int kk = event_idx - N_ROUTES;
        qi_process_departure(qi, fi, kk, kk, t, iter);
    } else {
        // Horizontal pipeline: event N_ROUTES+DIM_K+kk
        int kk = event_idx - N_ROUTES - DIM_K;
        if (kk < DIM_K - 1) {
            qi_process_transfer(qi, DIM_K + kk, DIM_K + kk + 1);
        } else {
            // kk == DIM_K-1: final departure from buf[2*DIM_K-1], route DIM_K
            qi_process_departure(qi, fi, 2 * DIM_K - 1, DIM_K, t, iter);
        }
    }
}

// ==========================================================================
// SECTION 3: Event generation
// ==========================================================================

// ---------------------------------------------------------------------------
// generate_next_event
//
// Python: SimulationInfo.generate_random_variables() (lines 1007–1040)
//
// Uses the standard property of competing exponentials:
//   If event i has rate r[i], then
//     min_i Exp(r[i]) ~ Exp(Λ)            with Λ = Σ_i r[i]
//     argmin_i         ~ Categorical(r[i] / Λ)
//
// → Per step we draw only TWO RNG values (down from N_EVENTS):
//     1 × Exp(1)       → dt = exp1 / Λ
//     1 × Uniform(0,1) → idx via inverse-CDF scan over r[i]/Λ
//
// Rate construction (disabling = rate 0):
//   Arrivals r[0..N_ROUTES-1]:
//     r[r] = max(LAMBDA[r] + pricing_control[r], 0) if outsourcing_control[r]==0 else 0
//   Vertical services r[N_ROUTES+kk]:
//     r[...] = MUS[kk] if scheduling_control[kk] == kk+1 else 0
//   Horizontal services r[N_ROUTES+DIM_K+kk]:
//     r[...] = MUS[DIM_K+kk] if scheduling_control[kk] == DIM_K+kk+1 else 0
//
// Statistical equivalence: the new sampling produces events from the same
// joint distribution as the old "N exponentials, pick min" method, but the
// RNG consumption per step changes (N → 2), so per-path realizations are
// NOT bitwise identical to the previous implementation — only the aggregate
// statistics (mean cost, delay probabilities) match in distribution.
// ---------------------------------------------------------------------------
NextEvent generate_next_event(const PolicyOutputs& pol, std::mt19937& rng) {
    // Thread-local distribution objects: cheap to construct but even cheaper
    // when reused. (They hold no state for these simple distributions.)
    std::exponential_distribution<double>    exp1(1.0);
    std::uniform_real_distribution<double>   unif01(0.0, 1.0);

    // Build per-event rate array r[i] (0 = disabled)
    double r[N_EVENTS];

    // Arrivals: events 0..N_ROUTES-1
    for (int rr = 0; rr < N_ROUTES; ++rr) {
        double eff_lambda = LAMBDA[rr] + pol.pricing_control[rr];
        bool   outsourced = (pol.outsourcing_control[rr] >= 1);
        r[rr] = (!outsourced && eff_lambda > 0.0) ? eff_lambda : 0.0;
    }

    // Services: events N_ROUTES..N_ROUTES+N_BUFFERS-1
    // Vertical service kk enabled iff scheduling_control[kk] == kk+1.
    // Horizontal service kk enabled iff scheduling_control[kk] == DIM_K+kk+1.
    for (int kk = 0; kk < DIM_K; ++kk) {
        r[N_ROUTES + kk] =
            (pol.scheduling_control[kk] == (kk + 1))       ? MUS[kk]         : 0.0;
        r[N_ROUTES + DIM_K + kk] =
            (pol.scheduling_control[kk] == (DIM_K+kk+1))   ? MUS[DIM_K + kk] : 0.0;
    }

    // Total rate Λ
    double Lambda = 0.0;
    for (int k = 0; k < N_EVENTS; ++k) Lambda += r[k];

    assert(Lambda > 0.0 && "All events disabled — check scheduling/outsourcing policy");

    // Draw dt ~ Exp(Λ) via one Exp(1)
    double dt = exp1(rng) / Lambda;

    // Draw idx ~ Categorical(r[i]/Λ) via one Uniform(0,1)
    double target = unif01(rng) * Lambda;
    int    idx    = N_EVENTS - 1;   // fallback for floating-point edge case
    double cum    = 0.0;
    for (int k = 0; k < N_EVENTS; ++k) {
        cum += r[k];
        if (cum >= target) { idx = k; break; }
    }

    NextEvent ev;
    ev.idx = idx;
    ev.dt  = dt;
    return ev;
}

// ==========================================================================
// SECTION 4: Cost computation
// ==========================================================================

// ---------------------------------------------------------------------------
// compute_cost_rate
//
// Python: SimulationInfo.calculate_cost() (lines 1043–1052)
//   pricing_cost    = sum_r(pricing_control[r]^2 * H_DIAG)
//   outsourcing_cost= sum_r(outsource_cost_rate[r] * max(lambda[r]+pc[r], 0))
//                     only when outsourcing_control[r] >= 1
// ---------------------------------------------------------------------------
double compute_cost_rate(const PolicyOutputs& pol,
                         double* out_pricing,
                         double* out_outsource) {
    double pricing = 0.0;
    for (int r = 0; r < N_ROUTES; ++r) {
        pricing += pol.pricing_control[r] * pol.pricing_control[r];
    }
    pricing *= H_DIAG;

    double outsource = 0.0;
    for (int r = 0; r < N_ROUTES; ++r) {
        if (pol.outsourcing_control[r] >= 1) {
            double eff_lambda = std::max(LAMBDA[r] + pol.pricing_control[r], 0.0);
            outsource += OUTSOURCE_COST_RATE[r] * eff_lambda;
        }
    }

    if (out_pricing)   *out_pricing   = pricing;
    if (out_outsource) *out_outsource = outsource;
    return pricing + outsource;
}

// ---------------------------------------------------------------------------
// accumulate_cost
//
// Python: update_cost() + update_info() (lines 1055–1081)
//   exp_factor = (exp(-r*t) - exp(-r*(t+dt))) / r
//   rec_cost += cost * exp_factor
// ---------------------------------------------------------------------------
void accumulate_cost(const PolicyOutputs& pol,
                     double t, double dt,
                     CostAccum& accum,
                     double* out_pricing,
                     double* out_outsource,
                     double* out_total,
                     double* e_neg_rt_io) {
    double pricing, outsource;
    double total = compute_cost_rate(pol, &pricing, &outsource);

    double exp_factor;
    if (e_neg_rt_io) {
        // Incremental mode: *e_neg_rt_io == exp(-r*t).
        // One std::exp per step instead of two.
        const double e_dt      = std::exp(-INTEREST_RATE * dt);
        const double e_new     = (*e_neg_rt_io) * e_dt;
        exp_factor             = (*e_neg_rt_io - e_new) / INTEREST_RATE;
        *e_neg_rt_io           = e_new;
        (void)t;  // t no longer needed in incremental mode
    } else {
        // Legacy mode: compute both endpoints from scratch.
        exp_factor = (std::exp(-INTEREST_RATE * t)
                    - std::exp(-INTEREST_RATE * (t + dt)))
                    / INTEREST_RATE;
    }

    accum.rec_cost += total * exp_factor;

    // Detailed-only accumulators: pricing/outsourcing cost splits and
    // per-route outsourcing time. Gated by RC_PRINT_DETAILED_INFO because
    // none of these feed into cost_mean / delay_prob_mean selection.
    if (RC_PRINT_DETAILED_INFO) {
        accum.rec_cost_pricing  += pricing   * exp_factor;
        accum.rec_cost_boundary += outsource * exp_factor;
        for (int r = 0; r < N_ROUTES; ++r) {
            if (pol.outsourcing_control[r] >= 1) {
                accum.t_outsourced[r] += dt;
            }
        }
    }

    if (out_pricing)   *out_pricing   = pricing;
    if (out_outsource) *out_outsource = outsource;
    if (out_total)     *out_total     = total;
}

// ==========================================================================
// SECTION 5: State transition
// ==========================================================================

// apply_transition: states[k] += TRANSITIONS[event_idx][k]
// Python: next_states = states + mask @ transitions
void apply_transition(int states[N_BUFFERS], int event_idx) {
    for (int k = 0; k < N_BUFFERS; ++k) {
        states[k] += TRANSITIONS[event_idx][k];
    }
    for (int k = 0; k < N_BUFFERS; ++k) {
        assert(states[k] >= 0 && "Buffer went negative — check scheduling policy");
    }
}

// ==========================================================================
// SECTION 6: Single-path simulation loop
// ==========================================================================

PathResult run_one_path(unsigned int seed, DelayHistogram* out_hist) {
    std::mt19937 rng(seed);

    SimState    state;
    CostAccum   accum;
    QueueInfo   qi;
    FinishedInfo fi;

    PROFILE_TIC(p_init);
    initialize_sim_state   (state);
    initialize_cost_accum  (accum);
    initialize_queue_info  (qi, state.states);
    initialize_finished_info(fi);
    // Wire up the optional delay histogram. Null is a no-op (existing callers
    // pass nothing); non-null is shared across all paths in a run_simulation
    // call (sequential paths → safe, no race).
    fi.delay_hist = out_hist;

    // Initialize W_last from initial state
    double W_last[DIM_K];
    {
        PolicyInputs tmp;
        compute_Z_and_W(state.states, tmp);
        for (int i = 0; i < DIM_K; ++i) W_last[i] = tmp.W[i];
    }

    // Initialize G_last = NN(W_last) once, so compute_policy can reuse it across
    // any "outside" steps (when W_last doesn't get refreshed).
    // Only meaningful when any NN-based policy is active; otherwise the value is
    // never read by sub-policies and we skip the call.
    double G_last[DIM_K];
    const bool uses_nn = (RC_PRICING_MODE == (int)NN_PRICING)
                      || (RC_OUTSOURCING_MODE == (int)NN_OUTSOURCING);
    if (uses_nn) {
        nn_predict(W_last, G_last);
    } else {
        for (int i = 0; i < DIM_K; ++i) G_last[i] = 0.0;
    }

    // delay_time is only read by LRS_DelayTime scheduling; skip its computation
    // entirely for other scheduling policies (WMP / LRS_QueueLen) where the
    // downstream policies never look at inp.delay_time.
    const bool uses_delay_time = (RC_SCHEDULING_MODE == (int)LRS_DELAY_TIME);

    // Detailed-output flag (snapshot once per path for the hot loop).
    // When false: skip pricing/outsourcing cost splits and outsource-fraction
    // tracking; cost_mean/delay_prob_mean are unaffected.
    const bool print_detailed = (RC_PRINT_DETAILED_INFO != 0);

    // Steady-state tail accumulators (last 20% of iterations)
    double rec_cost_ss      = 0.0;
    double rec_pricing_ss   = 0.0;
    double rec_outsource_ss = 0.0;
    double t_ss             = 0.0;

    // Incremental discount factor: e_neg_rt = exp(-r * state.t); at t=0 this is 1.
    // Maintained across steps by accumulate_cost (one exp(-r*dt) per step).
    double e_neg_rt = 1.0;

    PolicyOutputs pol;
    PROFILE_TOC(p_init, g_prof_path_init);

    for (int iter = 0; iter < N_JUMPS; ++iter) {
        // Step 1: compute delay_time per buffer (head-of-line sojourn).
        // Only needed when the scheduling policy is LRS_DelayTime; otherwise
        // the downstream sub-policies never read inp.delay_time.
        PROFILE_TIC(p_s1);
        double delay_time[N_BUFFERS];
        if (uses_delay_time) {
            for (int k = 0; k < N_BUFFERS; ++k) {
                delay_time[k] = qi.job_times[k].empty()
                                ? 0.0
                                : (state.t - qi.job_times[k].front());
            }
        }
        PROFILE_TOC(p_s1, g_prof_s1_delay);

        // Step 2: policy
        PROFILE_TIC(p_s2);
        compute_policy(state.states, delay_time, W_last, G_last, pol);
        PROFILE_TOC(p_s2, g_prof_s2_policy);

        // Step 3: next event
        PROFILE_TIC(p_s3);
        NextEvent ev = generate_next_event(pol, rng);
        PROFILE_TOC(p_s3, g_prof_s3_event);

        // Step 4: discounted cost.
        // accumulate_cost also returns the just-computed rates via out params;
        // the steady-state branch reuses them instead of calling compute_cost_rate again.
        // Incremental discount mode: exp(-r*t) maintained in e_neg_rt, only
        // exp(-r*dt) is computed inside accumulate_cost.
        PROFILE_TIC(p_s4);
        double pr_rate = 0.0, out_rate = 0.0, cost_rate = 0.0;
        accumulate_cost(pol, state.t, ev.dt, accum,
                        &pr_rate, &out_rate, &cost_rate,
                        &e_neg_rt);

        // Step 5: steady-state accumulation (last 20%)
        if (iter > (int)(N_JUMPS * 0.8)) {
            rec_cost_ss += cost_rate * ev.dt;
            if (print_detailed) {
                rec_pricing_ss   += pr_rate  * ev.dt;
                rec_outsource_ss += out_rate * ev.dt;
            }
            t_ss += ev.dt;
        }
        PROFILE_TOC(p_s4, g_prof_s4_cost);

        PROFILE_TIC(p_s5);
        // Step 6: advance clock
        state.t += ev.dt;

        // Step 7: state transition
        apply_transition(state.states, ev.idx);

        // Step 8: queue bookkeeping
        qi_update(qi, fi, ev.idx, state.t, iter);
        PROFILE_TOC(p_s5, g_prof_s5_update);
    }

    // Steady-state tail correction
    // Python: finalize_steady_state_cost()
    //   exp_factor_ss = exp(-r*t) / r / t_steady_state
    //   rec_cost += rec_cost_steady_state * exp_factor_ss
    // e_neg_rt already equals exp(-r * state.t) after the main loop, so we
    // save another std::exp call here.
    if (t_ss > 0.0) {
        double exp_factor_ss = e_neg_rt / INTEREST_RATE;
        accum.rec_cost += (rec_cost_ss / t_ss) * exp_factor_ss;
        if (print_detailed) {
            accum.rec_cost_pricing  += (rec_pricing_ss   / t_ss) * exp_factor_ss;
            accum.rec_cost_boundary += (rec_outsource_ss / t_ss) * exp_factor_ss;
        }
    }

    PathResult res;
    res.cost             = accum.rec_cost;
    // In lean mode rec_cost_pricing/rec_cost_boundary/t_outsourced were never
    // touched, so these remain at their initial zero values.
    res.pricing_cost     = accum.rec_cost_pricing;
    res.outsourcing_cost = accum.rec_cost_boundary;
    res.final_time       = state.t;

    for (int r = 0; r < N_ROUTES; ++r) {
        res.delay_prob[r] = (fi.n_completed[r] > 0)
                            ? 100.0 * (double)fi.n_late[r] / (double)fi.n_completed[r]
                            : 0.0;
        res.outsource_frac[r] = (print_detailed && state.t > 0.0)
                                ? accum.t_outsourced[r] / state.t
                                : 0.0;
    }

    return res;
}

// ==========================================================================
// SECTION 7: Aggregate over N_PATHS
// ==========================================================================

#if DIM_K_SELECT == 10 && defined(ENABLE_BATCH_NN)
// -------------------- Batch-NN variant (CC10 proposed only) ----------------
// Step-major loop: each iteration processes all paths together, batching
// the NN forward pass into a single cblas_sgemm per layer. Strategy A —
// NN is called every step for every path (no "inside-only" skipping).
//
// Policy semantics are preserved exactly: pricing reads NN(W_last);
// outsourcing (outward) reads NN(W_last); outsourcing (inward) reads NN(W).
// For inward mode we issue two batch calls per step.
//
// Per-path RNG state matches the serial version bit-for-bit (seed 123+m),
// but BLAS accumulation order may produce ULP-level numerical differences
// vs. a serial sgemv run.
// ---------------------------------------------------------------------------

namespace {

// Per-path persistent state for the batch build.
struct PathStateBatch {
    std::mt19937  rng;
    SimState      sim;
    CostAccum     accum;
    QueueInfo     qi;
    FinishedInfo  fi;
    double        W_last[DIM_K];
    // Steady-state tail accumulators
    double        rec_cost_ss;
    double        rec_pricing_ss;
    double        rec_outsource_ss;
    double        t_ss;
    // Incremental discount factor: exp(-r * sim.t). Initialised to 1.0 at t=0
    // and updated each step by accumulate_cost (one exp per step instead of two).
    double        e_neg_rt;
};

} // anonymous namespace

SimResult run_simulation(std::vector<PathResult>* out_paths,
                         DelayHistogram* out_hist) {
    const int n_paths = RC_N_PATHS;
    const bool uses_nn = (RC_PRICING_MODE == (int)NN_PRICING)
                      || (RC_OUTSOURCING_MODE == (int)NN_OUTSOURCING);
    const bool inward  = (RC_INWARD_REFLECTION == 1);
    // delay_time is only read by LRS_DelayTime scheduling; skip its fill for
    // other scheduling policies (WMP / LRS_QueueLen).
    const bool uses_delay_time = (RC_SCHEDULING_MODE == (int)LRS_DELAY_TIME);
    // Detailed-output gate: when 0, skip pricing/outsourcing splits and
    // outsource-fraction tracking (cost_mean / delay_prob_mean still computed).
    const bool print_detailed  = (RC_PRINT_DETAILED_INFO != 0);

    printf("--- Running %d paths (batch NN) ---\n", n_paths);

    std::vector<PathStateBatch> paths(n_paths);

    // Batch scratch buffers: W input(s) and G output(s) per NN call per step.
    // outward: one call (NN(W_last))
    // inward:  two calls (NN(W_last) for pricing, NN(W) for outsourcing)
    std::vector<double> W_last_batch(n_paths * DIM_K);
    std::vector<double> G_last_batch(n_paths * DIM_K);
    std::vector<double> W_curr_batch;
    std::vector<double> G_curr_batch;
    if (inward) {
        W_curr_batch.assign(n_paths * DIM_K, 0.0);
        G_curr_batch.assign(n_paths * DIM_K, 0.0);
    }

    // Per-path scratch used during Phase 1 / Phase 3 (re-used each step).
    std::vector<PolicyInputs>  inps(n_paths);
    std::vector<PolicyOutputs> pols(n_paths);
    std::vector<double>        W_curr_per_path(n_paths * DIM_K);  // per-path current W (copied into W_curr_batch if inward)

#ifdef ENABLE_PROFILING
    // Per-section wall-clock accumulators (nanoseconds).
    uint64_t prof_phase1     = 0;   // Phase 1: Z/W/inside/W_last update/pack batch inputs
    uint64_t prof_phase2_nn  = 0;   // Phase 2: nn_predict_batch
    uint64_t prof_policy     = 0;   // Phase 3: pricing + scheduling + outsourcing + hardstop + idling
    uint64_t prof_event      = 0;   // Phase 3: generate_next_event
    uint64_t prof_cost       = 0;   // Phase 3: accumulate_cost + steady-state accumulators
    uint64_t prof_update     = 0;   // Phase 3: apply_transition + qi_update
    uint64_t prof_aggregate  = 0;   // post-loop: PathResult build + mean/std passes
    auto prof_total_t0       = std::chrono::steady_clock::now();
#endif

    // ---- Per-path initialization (matches run_one_path setup exactly) ----
    for (int m = 0; m < n_paths; ++m) {
        PathStateBatch& p = paths[m];
        p.rng.seed(123u + (unsigned)m);
        initialize_sim_state(p.sim);
        initialize_cost_accum(p.accum);
        p.e_neg_rt = 1.0;
        initialize_queue_info(p.qi, p.sim.states);
        initialize_finished_info(p.fi);
        // Wire optional delay histogram (sim_one_combo only). Sequential per-path
        // updates of the same shared map are safe in this serial-by-path loop.
        p.fi.delay_hist = out_hist;
        p.rec_cost_ss = p.rec_pricing_ss = p.rec_outsource_ss = p.t_ss = 0.0;

        PolicyInputs tmp;
        compute_Z_and_W(p.sim.states, tmp);
        for (int i = 0; i < DIM_K; ++i) p.W_last[i] = tmp.W[i];
    }

    // Main loop: step-major
    for (int iter = 0; iter < N_JUMPS; ++iter) {
        // ---------- Phase 1: compute Z, W, delay_time, W_last update, pack batch inputs ----------
        PROFILE_TIC(p1);
        for (int m = 0; m < n_paths; ++m) {
            PathStateBatch& p   = paths[m];
            PolicyInputs&   inp = inps[m];

            compute_Z_and_W(p.sim.states, inp);

            // delay_time per buffer (head-of-line sojourn): only fill when
            // LRS_DelayTime scheduling will read it.
            if (uses_delay_time) {
                for (int k = 0; k < N_BUFFERS; ++k) {
                    inp.delay_time[k] = p.qi.job_times[k].empty()
                                        ? 0.0
                                        : (p.sim.t - p.qi.job_times[k].front());
                }
            }

            // Copy W_last snapshot into inp.W_last (may be updated below if inside)
            for (int i = 0; i < DIM_K; ++i) inp.W_last[i] = p.W_last[i];

            // Determine inside/outside (face check on current W)
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
            if (inside) {
                for (int i = 0; i < DIM_K; ++i) {
                    p.W_last[i]   = inp.W[i];
                    inp.W_last[i] = inp.W[i];
                }
            }

            // Pack batch inputs (always pack W_last; inward also packs current W)
            if (uses_nn) {
                for (int i = 0; i < DIM_K; ++i)
                    W_last_batch[m * DIM_K + i] = p.W_last[i];
                if (inward) {
                    for (int i = 0; i < DIM_K; ++i)
                        W_curr_batch[m * DIM_K + i] = inp.W[i];
                }
            }
            // Save inp.W for Phase 3 (in case we need it later; actually inp is kept)
            (void)W_curr_per_path;
        }
        PROFILE_TOC(p1, prof_phase1);

        // ---------- Phase 2: batch NN ----------
        PROFILE_TIC(p2);
        if (uses_nn) {
            nn_predict_batch(W_last_batch.data(), G_last_batch.data(), n_paths);
            if (inward) {
                nn_predict_batch(W_curr_batch.data(), G_curr_batch.data(), n_paths);
            }
        }
        PROFILE_TOC(p2, prof_phase2_nn);

        // ---------- Phase 3: per-path policy, event, update ----------
        for (int m = 0; m < n_paths; ++m) {
            PathStateBatch& p   = paths[m];
            PolicyInputs&   inp = inps[m];
            PolicyOutputs&  pol = pols[m];

            PROFILE_TIC(p3pol);
            // -- Pricing: uses NN(W_last) --
            if (uses_nn) {
                for (int i = 0; i < DIM_K; ++i)
                    inp.G_cached[i] = G_last_batch[m * DIM_K + i];
                // For pricing we don't read G_at_W_valid; safe to set true since G_cached == NN(W_last)
                inp.G_at_W_valid = true;
            }

            switch (RC_PRICING_MODE) {
                case (int)NN_PRICING:
                    nn_pricing(inp, pol.pricing_control);
                    break;
                case (int)BCP_LINEAR:
                    bcp_linear_pricing(inp, pol.pricing_control);
                    break;
                case (int)HIGH_LOW:
                    high_low_pricing(inp, pol.pricing_control);
                    break;
                default:
                    nn_pricing(inp, pol.pricing_control);
                    break;
            }

            // -- Scheduling --
            switch (RC_SCHEDULING_MODE) {
                case (int)WMP_OBJ2:
                    wmp_scheduling(inp, pol.scheduling_control);
                    break;
                case (int)LRS_QUEUE_LEN:
                    lrs_queue_len_scheduling(inp, pol.scheduling_control);
                    break;
                case (int)LRS_DELAY_TIME:
                    lrs_delay_time_scheduling(inp, pol.scheduling_control);
                    break;
                default:
                    wmp_scheduling(inp, pol.scheduling_control);
                    break;
            }

            // -- Outsourcing --
            // outward: NN(W_last); G_at_W_valid = (W_last == W) = inside (recomputed from W_last == inp.W)
            // inward:  NN(W);      G_at_W_valid = true
            if (uses_nn) {
                if (inward) {
                    for (int i = 0; i < DIM_K; ++i)
                        inp.G_cached[i] = G_curr_batch[m * DIM_K + i];
                    inp.G_at_W_valid = true;
                } else {
                    // outward: G_cached already holds NN(W_last) from pricing phase
                    bool inside_now = true;
                    for (int i = 0; i < DIM_K; ++i) {
                        if (inp.W_last[i] != inp.W[i]) { inside_now = false; break; }
                    }
                    inp.G_at_W_valid = inside_now;
                }
            }

            switch (RC_OUTSOURCING_MODE) {
                case (int)NN_OUTSOURCING:
                    if (inward) nn_outsourcing_inward(inp, pol.outsourcing_control);
                    else        nn_outsourcing_hold_last_no_takeover(inp, pol.outsourcing_control);
                    break;
                case (int)THRESHOLD_BASED:
                    threshold_outsourcing(inp, pol.outsourcing_control);
                    break;
                case (int)SINGLE_PUSHING:
                    if (inward) single_pushing_outsourcing_inward(inp, pol.outsourcing_control);
                    else        single_pushing_outsourcing(inp, pol.outsourcing_control);
                    break;
                default:
                    if (inward) nn_outsourcing_inward(inp, pol.outsourcing_control);
                    else        nn_outsourcing_hold_last_no_takeover(inp, pol.outsourcing_control);
                    break;
            }

            // -- hardstop + idling --
            hardstop_finite_buffer(inp, pol.outsourcing_control);
            idling_policy(inp, pol);
            PROFILE_TOC(p3pol, prof_policy);

            // -- Event generation and state update --
            PROFILE_TIC(p3ev);
            NextEvent ev = generate_next_event(pol, p.rng);
            PROFILE_TOC(p3ev, prof_event);

            // accumulate_cost returns the just-computed rates so the
            // steady-state branch does not have to recompute them.
            // Incremental discount mode: p.e_neg_rt tracks exp(-r*t); only
            // exp(-r*dt) is computed each step.
            PROFILE_TIC(p3co);
            double pr_rate = 0.0, out_rate = 0.0, cost_rate = 0.0;
            accumulate_cost(pol, p.sim.t, ev.dt, p.accum,
                            &pr_rate, &out_rate, &cost_rate,
                            &p.e_neg_rt);

            if (iter > (int)(N_JUMPS * 0.8)) {
                p.rec_cost_ss += cost_rate * ev.dt;
                if (print_detailed) {
                    p.rec_pricing_ss   += pr_rate  * ev.dt;
                    p.rec_outsource_ss += out_rate * ev.dt;
                }
                p.t_ss += ev.dt;
            }
            PROFILE_TOC(p3co, prof_cost);

            PROFILE_TIC(p3up);
            p.sim.t += ev.dt;
            apply_transition(p.sim.states, ev.idx);
            qi_update(p.qi, p.fi, ev.idx, p.sim.t, iter);
            PROFILE_TOC(p3up, prof_update);
        }
    }

    // ---- Build PathResult[] and print progress summary ----
    PROFILE_TIC(pagg);
    std::vector<PathResult> all_results(n_paths);
    for (int m = 0; m < n_paths; ++m) {
        PathStateBatch& p = paths[m];
        if (p.t_ss > 0.0) {
            // p.e_neg_rt already equals exp(-r * p.sim.t) after the main loop.
            double exp_factor_ss = p.e_neg_rt / INTEREST_RATE;
            p.accum.rec_cost += (p.rec_cost_ss / p.t_ss) * exp_factor_ss;
            if (print_detailed) {
                p.accum.rec_cost_pricing  += (p.rec_pricing_ss   / p.t_ss) * exp_factor_ss;
                p.accum.rec_cost_boundary += (p.rec_outsource_ss / p.t_ss) * exp_factor_ss;
            }
        }
        PathResult& res = all_results[m];
        res.cost             = p.accum.rec_cost;
        // In lean mode the split/t_outsourced fields were never accumulated
        // and remain zero.
        res.pricing_cost     = p.accum.rec_cost_pricing;
        res.outsourcing_cost = p.accum.rec_cost_boundary;
        res.final_time       = p.sim.t;
        for (int r = 0; r < N_ROUTES; ++r) {
            res.delay_prob[r] = (p.fi.n_completed[r] > 0)
                                ? 100.0 * (double)p.fi.n_late[r] / (double)p.fi.n_completed[r]
                                : 0.0;
            res.outsource_frac[r] = (print_detailed && p.sim.t > 0.0)
                                    ? p.accum.t_outsourced[r] / p.sim.t
                                    : 0.0;
        }
    }

    // Free PathStateBatch memory eagerly (queues may be large)
    paths.clear();
    paths.shrink_to_fit();

    // Pass 1: means
    SimResult result;
    result.cost_mean             = 0.0;
    result.pricing_cost_mean     = 0.0;
    result.outsourcing_cost_mean = 0.0;
    for (int r = 0; r < N_ROUTES; ++r) {
        result.delay_prob_mean[r]     = 0.0;
        result.outsource_frac_mean[r] = 0.0;
    }
    for (int m = 0; m < n_paths; ++m) {
        result.cost_mean             += all_results[m].cost;
        result.pricing_cost_mean     += all_results[m].pricing_cost;
        result.outsourcing_cost_mean += all_results[m].outsourcing_cost;
        for (int r = 0; r < N_ROUTES; ++r) {
            result.delay_prob_mean[r]     += all_results[m].delay_prob[r];
            result.outsource_frac_mean[r] += all_results[m].outsource_frac[r];
        }
    }
    result.cost_mean             /= n_paths;
    result.pricing_cost_mean     /= n_paths;
    result.outsourcing_cost_mean /= n_paths;
    for (int r = 0; r < N_ROUTES; ++r) {
        result.delay_prob_mean[r]     /= n_paths;
        result.outsource_frac_mean[r] /= n_paths;
    }

    // Pass 2: variance → stderr
    double cost_var = 0.0;
    double delay_var[N_ROUTES];
    for (int r = 0; r < N_ROUTES; ++r) delay_var[r] = 0.0;
    for (int m = 0; m < n_paths; ++m) {
        double dc = all_results[m].cost - result.cost_mean;
        cost_var += dc * dc;
        for (int r = 0; r < N_ROUTES; ++r) {
            double dd = all_results[m].delay_prob[r] - result.delay_prob_mean[r];
            delay_var[r] += dd * dd;
        }
    }
    result.cost_std = std::sqrt(cost_var / ((double)n_paths * (n_paths - 1)));
    for (int r = 0; r < N_ROUTES; ++r) {
        result.delay_prob_std[r] = std::sqrt(delay_var[r] / ((double)n_paths * (n_paths - 1)));
    }
    result.elapsed_seconds = 0.0;
    PROFILE_TOC(pagg, prof_aggregate);

#ifdef ENABLE_PROFILING
    uint64_t prof_total_wall = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - prof_total_t0).count();
    uint64_t prof_measured = prof_phase1 + prof_phase2_nn + prof_policy
                           + prof_event  + prof_cost      + prof_update
                           + prof_aggregate;
    auto pct_wall     = [&](uint64_t v){ return prof_total_wall > 0 ? 100.0 * v / prof_total_wall : 0.0; };
    auto pct_measured = [&](uint64_t v){ return prof_measured   > 0 ? 100.0 * v / prof_measured   : 0.0; };
    printf("\n========== PROFILING (batch run_simulation) ==========\n");
    printf("  %-16s %15s  %8s  %8s\n", "section", "ns", "%wall", "%meas");
    auto pp = [&](const char* name, uint64_t v){
        printf("  %-16s %15llu  %7.2f%%  %7.2f%%\n",
               name, (unsigned long long)v, pct_wall(v), pct_measured(v));
    };
    pp("phase1_Z/W/pack", prof_phase1);
    pp("phase2_nn_batch", prof_phase2_nn);
    pp("phase3_policy",   prof_policy);
    pp("phase3_event",    prof_event);
    pp("phase3_cost",     prof_cost);
    pp("phase3_update",   prof_update);
    pp("aggregate_post",  prof_aggregate);
    printf("  %-16s %15llu  %7.2f%%  %7s\n",
           "measured_sum", (unsigned long long)prof_measured,
           pct_wall(prof_measured), "--");
    printf("  %-16s %15llu  %7s  %7s\n",
           "total_wall",   (unsigned long long)prof_total_wall, "--", "--");
    printf("======================================================\n");
#endif

    if (out_paths) *out_paths = std::move(all_results);
    return result;
}

#else  // !(DIM_K_SELECT == 10 && ENABLE_BATCH_NN)
// -------------------- Default serial variant -------------------------------

SimResult run_simulation(std::vector<PathResult>* out_paths,
                         DelayHistogram* out_hist) {
    int n_paths = RC_N_PATHS;
    std::vector<PathResult> all_results(n_paths);

    printf("--- Running %d paths ---\n", n_paths);

    for (int m = 0; m < n_paths; ++m) {
        // out_hist (if non-null) is shared across paths — accumulates all delays.
        all_results[m] = run_one_path(123u + (unsigned)m, out_hist);
    }

    PROFILE_TIC(p_agg);
    // Pass 1: means
    SimResult result;
    result.cost_mean             = 0.0;
    result.pricing_cost_mean     = 0.0;
    result.outsourcing_cost_mean = 0.0;
    for (int r = 0; r < N_ROUTES; ++r) {
        result.delay_prob_mean[r]     = 0.0;
        result.outsource_frac_mean[r] = 0.0;
    }
    for (int m = 0; m < n_paths; ++m) {
        result.cost_mean             += all_results[m].cost;
        result.pricing_cost_mean     += all_results[m].pricing_cost;
        result.outsourcing_cost_mean += all_results[m].outsourcing_cost;
        for (int r = 0; r < N_ROUTES; ++r) {
            result.delay_prob_mean[r]     += all_results[m].delay_prob[r];
            result.outsource_frac_mean[r] += all_results[m].outsource_frac[r];
        }
    }
    result.cost_mean             /= n_paths;
    result.pricing_cost_mean     /= n_paths;
    result.outsourcing_cost_mean /= n_paths;
    for (int r = 0; r < N_ROUTES; ++r) {
        result.delay_prob_mean[r]     /= n_paths;
        result.outsource_frac_mean[r] /= n_paths;
    }

    // Pass 2: variance → stderr
    double cost_var = 0.0;
    double delay_var[N_ROUTES];
    for (int r = 0; r < N_ROUTES; ++r) delay_var[r] = 0.0;

    for (int m = 0; m < n_paths; ++m) {
        double dc = all_results[m].cost - result.cost_mean;
        cost_var += dc * dc;
        for (int r = 0; r < N_ROUTES; ++r) {
            double dd = all_results[m].delay_prob[r] - result.delay_prob_mean[r];
            delay_var[r] += dd * dd;
        }
    }
    result.cost_std = std::sqrt(cost_var / ((double)n_paths * (n_paths - 1)));
    for (int r = 0; r < N_ROUTES; ++r) {
        result.delay_prob_std[r] = std::sqrt(delay_var[r] / ((double)n_paths * (n_paths - 1)));
    }

    result.elapsed_seconds = 0.0;  // filled in by main.cpp
    PROFILE_TOC(p_agg, g_prof_aggregate);

    if (out_paths) *out_paths = std::move(all_results);

    return result;
}

#endif  // DIM_K_SELECT == 10 && ENABLE_BATCH_NN
