/*
 * simulation.cpp
 *
 * Implementation of the simulation event loop and all helper functions.
 *
 * =========================================================================
 * OVERALL STRUCTURE (mirrors Python simulation(), lines 847–922):
 *
 *   run_simulation()
 *     for m in 0 .. N_PATHS-1:
 *       run_one_path(seed=123+m)   ← one independent simulation path
 *     compute mean / std over all paths
 *
 *   run_one_path(seed):
 *     for iter in 0 .. N_JUMPS-1:
 *       1. compute_policy()              → pricing, outsourcing, scheduling controls
 *       2. generate_next_event()         → which event fires + how long (dt)
 *       3. accumulate_cost()             → discounted cost for interval [t, t+dt]
 *       4. steady-state accumulation     → if iter > 80% of N_JUMPS
 *       5. state.t += dt                 → advance simulation clock
 *       6. apply_transition()            → update integer queue counts
 *       7. qi_update()                   → update delay tracking circular buffers
 *     finalize_steady_state_cost()       → add tail correction to rec_cost
 *
 * =========================================================================
 * WHAT IS FAITHFULLY TRANSLATED FROM PYTHON (behavior EXACTLY the same):
 *   - Event generation logic (generate_next_event)
 *   - Discounted cost formula (accumulate_cost)
 *   - Steady-state cost correction (finalize_steady_state_cost)
 *   - State transitions (apply_transition)
 *   - Circular buffer updates (qi_process_*)
 *   - Delay probability computation (count late / count total)
 *
 * WHAT DIFFERS FROM PYTHON (behavior APPROXIMATELY the same):
 *   - RNG streams: Python draws M*6 randoms per step from one shared RNG.
 *     C++ gives each path its own independent RNG (seed = 123 + m).
 *     Distribution of results is identical; exact per-path values differ.
 *   - idling_policy() and identify_mode() are still TODO → policy decisions
 *     slightly differ from Python for paths near the state-space boundary.
 *
 * =========================================================================
 */

#include "simulation.h"
#include "run_config.h"   // RC_N_PATHS
#include "nn_model.h"     // nn_predict (pre-seed G_last in run_one_path)
#include "profiling.h"    // optional TIC/TOC (zero overhead when disabled)
#include <cassert>    // assert()
#include <cmath>      // exp(), sqrt(), fabs()
#include <algorithm>  // std::max()
#include <cstdio>     // printf()
#include <vector>     // std::vector (for dynamic N_PATHS)

// ==========================================================================
// SECTION 1: Initialization
// ==========================================================================

// ---------------------------------------------------------------------------
// initialize_sim_state
// Python: initialize_simulation() — states = np.zeros((M,4)) + initial_state
//         simulation_info.t = np.zeros(M)
// ---------------------------------------------------------------------------
void initialize_sim_state(SimState& state) {
    for (int k = 0; k < N_BUFFERS; k++) {
        state.states[k] = INITIAL_STATE[k];
    }
    state.t = 0.0;
}

// ---------------------------------------------------------------------------
// initialize_cost_accum
// Python: SimulationInfo.__init__ — self.rec_cost = np.zeros(M), etc.
// ---------------------------------------------------------------------------
void initialize_cost_accum(CostAccum& accum) {
    accum.rec_cost          = 0.0;
    accum.rec_cost_pricing  = 0.0;
    accum.rec_cost_boundary = 0.0;
    accum.rec_cost_band     = 0.0;
    accum.rec_cost_outer    = 0.0;
    for (int r = 0; r < N_ROUTES; r++) accum.rec_cost_route[r] = 0.0;
    for (int r = 0; r < N_ROUTES; r++) {
        accum.t_outsourced[r] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// initialize_queue_info
//
// Python: QueueInfo.__init__(begin=zeros, end=states, enter_system_time=zeros)
//   Pre-loads states[k] jobs into each buffer, all with entry time 0.0.
//
// The initial jobs (e.g., 10 in buf[0]) are treated as having arrived at t=0.
// Their delays are not recorded because DELAY_PROB_RECORD_START > 0 means
// we skip recording until the system has mixed thoroughly.
// ---------------------------------------------------------------------------
void initialize_queue_info(QueueInfo& qi, const int states[N_BUFFERS]) {
    for (int k = 0; k < N_BUFFERS; k++) {
        // Clear any leftover jobs from a previous path
        while (!qi.job_times[k].empty()) {
            qi.job_times[k].pop();
        }
        // Pre-load initial jobs: each "arrived" at t = 0
        // Python: self.end[k] = states[k]  (buf_end pointed past the pre-loaded slots)
        for (int j = 0; j < states[k]; j++) {
            qi.job_times[k].push(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// initialize_finished_info
// Python: FinishedInfo.__init__ — self.counts = {route: 0 for route in routes}
// ---------------------------------------------------------------------------
void initialize_finished_info(FinishedInfo& fi) {
    for (int r = 0; r < N_ROUTES; r++) {
        fi.n_completed[r] = 0;
        fi.n_late[r]      = 0;
    }
}

// ==========================================================================
// SECTION 2: Queue operations  (QueueInfo — std::queue<double>)
// ==========================================================================
// Each buffer k has a FIFO queue job_times[k] of job entry times.
//   front() = oldest job (next to be served)
//   back()  = newest job (most recently arrived)
// Replaces the old circular buffer (buf_begin/buf_end/enter_time arrays).
// Follows the arrivalTimeList pattern from Summary_by_claude_code.pdf.

// ---------------------------------------------------------------------------
// qi_process_arrival
//
// Python: QueueInfo.process_arrival(pos_arrivals, i_buffer) — lines 517–532
//   self.enter_system_time[m, i_buffer, end_pos] = t
//   self.end[m, i_buffer] = (end_pos + 1) % buffer_size
//
// C++: just push entry time onto back of queue — no size limit, no pointer math.
// ---------------------------------------------------------------------------
void qi_process_arrival(QueueInfo& qi, int i_buf, double t) {
    qi.job_times[i_buf].push(t);   // Python: enter_system_time[..., end_pos] = t; end++
}

// ---------------------------------------------------------------------------
// qi_process_transfer
//
// Python: QueueInfo.process_buffer_transfer(pos_departures, i_buffer, i_buffer_next)
//   lines 534–556
//   Copy enter_system_time from front of source to back of destination.
//   Advance source begin, advance destination end.
//
// The key point: the ORIGINAL entry time (when the job first entered the whole
// system) is preserved as it moves from stage to stage.
// ---------------------------------------------------------------------------
void qi_process_transfer(QueueInfo& qi, int i_buf, int i_buf_next) {
    double entry_t = qi.job_times[i_buf].front();   // oldest job's original entry time
    qi.job_times[i_buf].pop();                      // Python: begin = (begin+1) % buf_size
    qi.job_times[i_buf_next].push(entry_t);         // Python: enter_system_time[dest, end] = entry_t; end++
}

// ---------------------------------------------------------------------------
// qi_process_departure
//
// Python: QueueInfo.process_final_departure(finished_info, pos_departures,
//                                           i_buffer, route)  — lines 558–579
//   enter_time = self.enter_system_time[m, i_buffer, begin_pos]
//   delay = t - enter_time
//   if delay > delay_bound: count as late
//   self.begin[m, i_buffer] = (begin_pos + 1) % buffer_size
//
// C++ difference: instead of storing all delays, we just count n_completed and n_late.
// This is sufficient for computing delay_prob = n_late / n_completed * 100%.
// ---------------------------------------------------------------------------
void qi_process_departure(QueueInfo& qi, FinishedInfo& fi,
                          int i_buf, int route, double t, int iter) {
    double entry_t = qi.job_times[i_buf].front();   // Python: enter_system_time[m, k, begin_pos]
    qi.job_times[i_buf].pop();                      // Python: begin = (begin+1) % buf_size

    // Only record delay after warm-up (matches Python's iter > delay_prob_record_start)
    if (iter > DELAY_PROB_RECORD_START) {
        double delay = t - entry_t;                 // Python: delay = finish_time - enter_time
        fi.n_completed[route]++;
        if (delay > DELAY_BOUND[route]) {
            fi.n_late[route]++;
        }
        // Optional: bucket this delay into the per-route histogram (sim_one_combo only).
        // The pointer being null short-circuits all histogram work (zero overhead
        // for the existing 4 binaries).
        if (fi.delay_hist) {
            int bucket = (int)std::floor(delay / fi.delay_hist->bucket_width);
            fi.delay_hist->hist[route][bucket]++;
        }
    }
}

// ---------------------------------------------------------------------------
// qi_update: dispatch circular buffer operations based on which event fired.
//
// Python: QueueInfo.update_queue_info_with_routes() — lines 598–606
//   It loops over routes and uses mask to call process_arrival/transfer/departure.
//   Specifically, update_queue_info_with_routes calls process_route for EACH route,
//   and process_route uses:
//     mask[:, route_num]               for arrivals
//     mask[:, i_buffer + 2]            for transfers
//     mask[:, final_departure_buffer+2] for departures
//
// For M=1, mask[:, k] == True only when event k fired.
// So this is equivalent to a switch on event_idx.
//
// Event-to-operation mapping (from routes definition in parameters_model.py):
//   event 0: arrival to buf[0]     (route 1, route_num=0)
//   event 1: arrival to buf[2]     (route 2, route_num=1)
//   event 2: transfer buf[0]→buf[1] (buffer_transfer for route 1: [(0,1)])
//   event 3: departure from buf[1]  (final_departure_buffer=1, route 1 → index 0)
//   event 4: transfer buf[2]→buf[3] (buffer_transfer for route 2: [(2,3)])
//   event 5: departure from buf[3]  (final_departure_buffer=3, route 2 → index 1)
//
// Behavior: EXACTLY the same as Python.
// ---------------------------------------------------------------------------
void qi_update(QueueInfo& qi, FinishedInfo& fi, int event_idx, double t, int iter) {
    switch (event_idx) {
        case 0:  // route 1 arrival → buf[0]
            qi_process_arrival(qi, 0, t);
            break;
        case 1:  // route 2 arrival → buf[2]
            qi_process_arrival(qi, 2, t);
            break;
        case 2:  // station1 serves buf[0]: job moves buf[0] → buf[1]
            qi_process_transfer(qi, 0, 1);
            break;
        case 3:  // station2 serves buf[1]: job departs (route 1, index 0)
            qi_process_departure(qi, fi, 1, 0, t, iter);
            break;
        case 4:  // station2 serves buf[2]: job moves buf[2] → buf[3]
            qi_process_transfer(qi, 2, 3);
            break;
        case 5:  // station1 serves buf[3]: job departs (route 2, index 1)
            qi_process_departure(qi, fi, 3, 1, t, iter);
            break;
        default:
            assert(false && "Unknown event index");
    }
}

// ==========================================================================
// SECTION 3: Event generation
// ==========================================================================

// ---------------------------------------------------------------------------
// generate_next_event
//
// Python: SimulationInfo.generate_random_variables() — lines 656–706
//
// Uses the identity  min_i Exp(r_i) ~ Exp(Λ)  with  argmin ~ Categorical(r_i/Λ),
// where  Λ = Σ r_i.  One Exp(1) draw gives dt, one Uniform(0,1) draw gives idx.
//
// Logic:
//   1. Build per-event rate array r[N_EVENTS]  (disabled events: r[k] = 0)
//        events 0..1 (arrivals):   r = LAMBDA + pricing_control if !outsourced & eff_lambda>0
//        event  2 (station1 → buf[0]): r = MUS[0]  if scheduling_control[0] == 1
//        event  3 (station2 → buf[1]): r = MUS[1]  if scheduling_control[1] == 2
//        event  4 (station2 → buf[2]): r = MUS[2]  if scheduling_control[1] == 3
//        event  5 (station1 → buf[3]): r = MUS[3]  if scheduling_control[0] == 4
//   2. Λ = Σ r[k];  dt = Exp(1) / Λ
//   3. target = Uniform(0,1) · Λ;  accumulate cum += r[k] until cum >= target  → idx
//      (idx = N_EVENTS-1 as floating-point edge-case fallback)
//
// Compared to the original "draw N_EVENTS Exp(1) samples and take the min"
// implementation this is mathematically equivalent in distribution but
// consumes only 2 RNG draws per step (instead of N_EVENTS). Single-path
// trajectories given the same seed are NOT bit-identical (RNG consumption
// count changed); mean/variance over many paths are within statistical error.
// ---------------------------------------------------------------------------
NextEvent generate_next_event(const PolicyOutputs& pol, std::mt19937& rng) {
    std::exponential_distribution<double>  exp1(1.0);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    // Build per-event rate array r[i]  (0 = disabled)
    double r[N_EVENTS];

    // Arrivals: events 0..N_ROUTES-1
    for (int rr = 0; rr < N_ROUTES; ++rr) {
        double eff_lambda = LAMBDA[rr] + pol.pricing_control[rr];
        bool   outsourced = (pol.outsourcing_control[rr] >= 1);
        r[rr] = (!outsourced && eff_lambda > 0.0) ? eff_lambda : 0.0;
    }

    // Services: RS-specific station/buffer mapping.
    //   scheduling_control[0] (station1): 1 → buf[0],  4 → buf[3]
    //   scheduling_control[1] (station2): 2 → buf[1],  3 → buf[2]
    // If scheduling_control == 0 (idle), that station's rates stay 0.
    r[2] = (pol.scheduling_control[0] == 1) ? MUS[0] : 0.0;  // station1 → buf[0]
    r[3] = (pol.scheduling_control[1] == 2) ? MUS[1] : 0.0;  // station2 → buf[1]
    r[4] = (pol.scheduling_control[1] == 3) ? MUS[2] : 0.0;  // station2 → buf[2]
    r[5] = (pol.scheduling_control[0] == 4) ? MUS[3] : 0.0;  // station1 → buf[3]

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
// Python: SimulationInfo.calculate_cost() — lines 709–718
//   self.admission_costs = outsourcing_cost_rate * max(lambdas + pricing_control, 0)
//   self.pricing_cost    = np.sum(pricing_control * (pricing_control @ H), axis=1)
//   self.outsourcing_cost= np.sum(admission_costs * mask_gte_1, axis=1)
//   self.cost = self.outsourcing_cost + self.pricing_cost
//
// Cost has two components:
//   1. Pricing penalty (quadratic): pc @ H @ pc = H_DIAG * (pc[0]^2 + pc[1]^2)
//      (H is diagonal with H_DIAG on the diagonal)
//   2. Outsourcing cost: sum_r outsource_cost_rate[r] * max(lambda[r]+pc[r], 0)
//      charged only when outsourcing_control[r] >= 1
//
// Optional output pointers let callers get the breakdown without recomputing.
// Behavior: EXACTLY the same as Python.
// ---------------------------------------------------------------------------
double compute_cost_rate(const PolicyOutputs& pol,
                         double* out_pricing,
                         double* out_outsource,
                         double* out_band,
                         double* out_outer,
                         double* out_route) {
    // Pricing cost: quadratic penalty  (H diagonal → H_DIAG * sum of squares)
    // Python: pricing_cost = sum(pc * (pc @ H)) = H_DIAG * (pc[0]^2 + pc[1]^2)
    double pricing = 0.0;
    for (int j = 0; j < N_ROUTES; j++) {
        pricing += pol.pricing_control[j] * pol.pricing_control[j];
    }
    pricing *= H_DIAG;

    // Outsourcing cost: charged when control >= 1.
    // control == 1.0: band zone (epsilon band, single face)       → band cost
    // control == 1.5: forced/outer zone (multi-face or outside)   → outer cost
    // control == 2.0: hardstop (counted as outer for breakdown purposes)
    //
    // band/outer breakdown and per-route attribution are only tracked when the
    // caller wants them (out_band/out_outer/out_route non-null). Skipping this
    // detail when the caller passes nullptr saves a few branches per step.
    const bool need_band_outer = (out_band || out_outer);
    double outsource = 0.0, band = 0.0, outer = 0.0;
    for (int r = 0; r < N_ROUTES; r++) {
        double v = pol.outsourcing_control[r];
        double cr = 0.0;
        if (v >= 1.0) {
            double eff_lambda = std::max(LAMBDA[r] + pol.pricing_control[r], 0.0);
            cr = OUTSOURCE_COST_RATE[r] * eff_lambda;
            outsource += cr;
            if (need_band_outer) {
                if (v < 1.5) band  += cr;
                else         outer += cr;
            }
        }
        if (out_route) out_route[r] = cr;
    }

    if (out_pricing)   *out_pricing   = pricing;
    if (out_outsource) *out_outsource = outsource;
    if (out_band)      *out_band      = band;
    if (out_outer)     *out_outer     = outer;
    return pricing + outsource;
}

// ---------------------------------------------------------------------------
// accumulate_cost
//
// Python: update_cost() — lines 721–726
//   exp_factor = (exp(-r*t) - exp(-r*(t+dt))) / r
//   self.rec_cost += self.cost * exp_factor
//
// Also tracks t_outsourced (update_info) and detailed cost breakdown.
//
// Behavior: EXACTLY the same as Python.
// ---------------------------------------------------------------------------
void accumulate_cost(const PolicyOutputs& pol,
                     double t, double dt,
                     CostAccum& accum,
                     double* out_pricing,
                     double* out_outsource,
                     double* out_band,
                     double* out_outer,
                     double* out_route,
                     double* out_total) {
    // Discounted-cost factor for interval [t, t+dt]:
    //   integral of cost_rate * e^{-r*s} ds  from s=t to s=t+dt
    //   = cost_rate * (e^{-r*t} - e^{-r*(t+dt)}) / r
    double exp_factor = (std::exp(-INTEREST_RATE * t)
                       - std::exp(-INTEREST_RATE * (t + dt)))
                       / INTEREST_RATE;

    double pricing = 0.0, outsource = 0.0, band = 0.0, outer = 0.0;
    double route[N_ROUTES] = {0.0, 0.0};
    double total;

    if (RC_PRINT_DETAILED_INFO) {
        // Full breakdown: all 6 sub-components computed and accumulated.
        total = compute_cost_rate(pol, &pricing, &outsource, &band, &outer, route);

        accum.rec_cost          += total    * exp_factor;
        accum.rec_cost_pricing  += pricing  * exp_factor;
        accum.rec_cost_boundary += outsource* exp_factor;
        accum.rec_cost_band     += band     * exp_factor;
        accum.rec_cost_outer    += outer    * exp_factor;
        for (int r = 0; r < N_ROUTES; r++)
            accum.rec_cost_route[r] += route[r] * exp_factor;
    } else {
        // Basic mode: only the total is accumulated. compute_cost_rate skips the
        // band/outer and per-route breakdowns when passed nullptr for those.
        total = compute_cost_rate(pol, &pricing, &outsource,
                                  nullptr, nullptr, nullptr);
        accum.rec_cost += total * exp_factor;
    }

    // Track time spent outsourcing per route (always needed for t_outsourced)
    // Python: self.t_outsourced += self.minInt * self.mask_gte_1
    for (int r = 0; r < N_ROUTES; r++) {
        if (pol.outsourcing_control[r] >= 1) {
            accum.t_outsourced[r] += dt;
        }
    }

    // Expose instantaneous rates to caller (reused by run_one_path SS branch)
    // NOTE: band/outer/route are 0 when !RC_PRINT_DETAILED_INFO; callers in that
    // mode should pass nullptr for those (run_one_path does).
    if (out_pricing)   *out_pricing   = pricing;
    if (out_outsource) *out_outsource = outsource;
    if (out_band)      *out_band      = band;
    if (out_outer)     *out_outer     = outer;
    if (out_route)
        for (int r = 0; r < N_ROUTES; r++) out_route[r] = route[r];
    if (out_total)     *out_total     = total;
}

// ==========================================================================
// SECTION 5: State transition
// ==========================================================================

// ---------------------------------------------------------------------------
// apply_transition
//
// Python: next_states = states + mask @ transitions  (line 618 in QueueInfo)
//   where transitions is the 6×4 matrix defined in QueueInfo.__init__
//   and mask is a boolean (M×6) row indicating which event fired.
//   For M=1, this is: states[k] += TRANSITIONS[event_idx][k]
//
// Behavior: EXACTLY the same as Python.
// ---------------------------------------------------------------------------
void apply_transition(int states[N_BUFFERS], int event_idx) {
    for (int k = 0; k < N_BUFFERS; k++) {
        states[k] += TRANSITIONS[event_idx][k];
    }
    // Sanity check: queue lengths must remain non-negative.
    for (int k = 0; k < N_BUFFERS; k++) {
        assert(states[k] >= 0 && "Buffer went negative — check idle override in wmp_scheduling()");
    }
}

// ==========================================================================
// SECTION 6: Single-path simulation loop
// ==========================================================================

// ---------------------------------------------------------------------------
// run_one_path
//
// Python: simulation() function (lines 847–922), but for exactly ONE path.
//
// Iteration-by-iteration mapping:
//   Python line 875: policy(...)                                 → compute_policy()
//   Python line 881: simulation_info.generate_random_variables() → generate_next_event()
//   Python line 884: simulation_info.calculate_cost()            → accumulate_cost() (internally)
//   Python line 885: simulation_info.update_cost()               → accumulate_cost()
//   Python line 888: simulation_info.update_info()               → accumulate_cost() (t_outsourced)
//   Python line 894: simulation_info.update_t()                  → state.t += ev.dt
//   Python line 898: queue_info.update_states_and_queues()       → apply_transition() + qi_update()
//   Python line 908: simulation_info.update_steady_state()       → rec_cost_ss / t_ss accumulators
//
// After the loop:
//   Python line 915: finalize_steady_state_cost()  → add SS correction to rec_cost
//   Python line 918: calculate_delay_probabilities()→ compute delay_prob from fi
//
// RNG seed: each path gets a unique seed so paths are independent.
// Python uses one shared global RNG (np.random.seed(123)) with vectorized draws.
// This difference is acceptable: the distribution of rec_cost is the same.
// ---------------------------------------------------------------------------
PathResult run_one_path(unsigned int seed, DelayHistogram* out_hist) {
    // Declarations first (so they outlive the PROF_L1(PATH_INIT) block below).
    std::mt19937 rng(seed);
    SimState    state;
    CostAccum   accum;
    QueueInfo   qi;
    FinishedInfo fi;
    double W_last[N_ROUTES];
    double G_last[N_ROUTES];

    // ---- Steady-state tail accumulators ----
    double rec_cost_ss          = 0.0;
    double rec_cost_pricing_ss  = 0.0;
    double rec_cost_boundary_ss = 0.0;
    double rec_cost_band_ss     = 0.0;
    double rec_cost_outer_ss    = 0.0;
    double rec_cost_route_ss[N_ROUTES] = {0.0, 0.0};
    double t_ss                 = 0.0;

    PolicyOutputs pol;

    // ---- Runtime gates (constant across this path) ----
    // uses_delay_time: only LRS_DELAY_TIME / LRS_REMAIN_DELAY scheduling modes
    // read inp.delay_time. For all other modes (WMP, LRS_QUEUE_LEN) the per-step
    // delay_time computation + copy into PolicyInputs is pure waste, so we skip
    // the whole S1 segment when uses_delay_time == false.
    const bool uses_delay_time = (RC_SCHEDULING_MODE == LRS_DELAY_TIME)
                              || (RC_SCHEDULING_MODE == LRS_REMAIN_DELAY);

    // ---- path_init: all initialization work, measured as one section ----
    {
        PROF_L1(PATH_INIT);
        initialize_sim_state   (state);
        initialize_cost_accum  (accum);
        initialize_queue_info  (qi, state.states);
        initialize_finished_info(fi);
        // Wire up the optional histogram so qi_process_departure pushes delays.
        // Null is a no-op (existing callers pass nothing); non-null is shared
        // across all paths in a run_simulation call (sequential paths → safe).
        fi.delay_hist = out_hist;

        // Initialize W_last from initial state
        {
            PolicyInputs tmp;
            compute_Z_and_W(state.states, tmp);
            W_last[0] = tmp.W[0];
            W_last[1] = tmp.W[1];
        }

        // Pre-seed G_last = NN(W_last) once (only if any NN policy is active)
        const bool uses_nn = (RC_PRICING_MODE == (int)NN_PRICING)
                          || (RC_OUTSOURCING_MODE == (int)NN_OUTSOURCING);
        if (uses_nn) {
            nn_predict(W_last[0], W_last[1], G_last);
        } else {
            G_last[0] = 0.0;
            G_last[1] = 0.0;
        }
    }

    // ---- Main event loop ----
    // Python: for i in range(nJumps):
    for (int iter = 0; iter < N_JUMPS; iter++) {

        // ---- S1: Compute head-of-line delay per buffer ----
        // delay_time[k] = time the oldest job in buf[k] has been in the system (0.0 if empty).
        // Only LRS_DELAY_TIME / LRS_REMAIN_DELAY use this; other modes skip both the
        // computation here and the copy inside compute_policy's Step 1b.
        double delay_time[N_BUFFERS];
        if (uses_delay_time) {
            PROF_L1(S1_DELAY_TIME);
            for (int k = 0; k < N_BUFFERS; k++) {
                delay_time[k] = qi.job_times[k].empty()
                                ? 0.0
                                : (state.t - qi.job_times[k].front());
            }
        }

        // ---- S2: Compute policy controls (L2 breakdown inside) ----
        {
            PROF_L1(S2_COMPUTE_POLICY);
            compute_policy(state.states, delay_time, W_last, G_last, pol);
        }

        // ---- S3: Sample next event (which one fires and how long until it) ----
        NextEvent ev;
        {
            PROF_L1(S3_NEXT_EVENT);
            ev = generate_next_event(pol, rng);
        }

        // ---- S4: Accumulate discounted cost + steady-state tail accumulation ----
        // accumulate_cost also returns the instantaneous rates it computed internally,
        // so the steady-state branch below reuses them (no second compute_cost_rate call).
        {
            PROF_L1(S4_ACCUMULATE_COST);
            // In basic mode (RC_PRINT_DETAILED_INFO=0) we only need the total rate;
            // accumulate_cost will skip the 5 sub-accumulator updates and pass nullptr
            // to compute_cost_rate for band/outer/route so that those branches are
            // skipped too.
            double cost_rate = 0.0;
            if (RC_PRINT_DETAILED_INFO) {
                double pr_rate = 0.0, ou_rate = 0.0, bd_rate = 0.0, ot_rate = 0.0;
                double rt_rate[N_ROUTES] = {0.0};
                accumulate_cost(pol, state.t, ev.dt, accum,
                                &pr_rate, &ou_rate, &bd_rate, &ot_rate, rt_rate, &cost_rate);
                if (iter > (int)(N_JUMPS * 0.8)) {
                    rec_cost_ss          += cost_rate * ev.dt;
                    rec_cost_pricing_ss  += pr_rate   * ev.dt;
                    rec_cost_boundary_ss += ou_rate   * ev.dt;
                    rec_cost_band_ss     += bd_rate   * ev.dt;
                    rec_cost_outer_ss    += ot_rate   * ev.dt;
                    for (int r = 0; r < N_ROUTES; r++)
                        rec_cost_route_ss[r] += rt_rate[r] * ev.dt;
                    t_ss                 += ev.dt;
                }
            } else {
                accumulate_cost(pol, state.t, ev.dt, accum,
                                nullptr, nullptr, nullptr, nullptr, nullptr, &cost_rate);
                if (iter > (int)(N_JUMPS * 0.8)) {
                    rec_cost_ss += cost_rate * ev.dt;
                    t_ss        += ev.dt;
                }
            }
        }

        // ---- S5: Advance clock + apply state/queue transition ----
        {
            PROF_L1(S5_STATE_AND_QUEUE);
            state.t += ev.dt;
            apply_transition(state.states, ev.idx);
            // qi_update uses state.t which is AFTER the advance (matches Python order).
            qi_update(qi, fi, ev.idx, state.t, iter);
        }
    }

    // ---- Steady-state tail correction ----
    // Python: simulation_info.finalize_steady_state_cost()
    //   exp_factor_steady_state = exp(-r * t) / r / t_steady_state
    //   self.rec_cost += self.rec_cost_steady_state * exp_factor_steady_state
    //
    // Interpretation: the steady-state cost rate (rec_cost_ss / t_ss) is assumed
    // to continue forever after time state.t. Its discounted value from state.t to ∞
    // is (cost_rate * e^{-r*T}) / r.  Adding this corrects for the infinite tail.
    if (t_ss > 0.0) {
        double exp_factor_ss = std::exp(-INTEREST_RATE * state.t) / INTEREST_RATE;
        accum.rec_cost += (rec_cost_ss / t_ss) * exp_factor_ss;
        if (RC_PRINT_DETAILED_INFO) {
            accum.rec_cost_pricing  += (rec_cost_pricing_ss  / t_ss) * exp_factor_ss;
            accum.rec_cost_boundary += (rec_cost_boundary_ss / t_ss) * exp_factor_ss;
            accum.rec_cost_band     += (rec_cost_band_ss     / t_ss) * exp_factor_ss;
            accum.rec_cost_outer    += (rec_cost_outer_ss    / t_ss) * exp_factor_ss;
            for (int r = 0; r < N_ROUTES; r++)
                accum.rec_cost_route[r] += (rec_cost_route_ss[r] / t_ss) * exp_factor_ss;
        }
        // Python: rec_cost += rec_cost_ss * exp(-r*t) / r / t_ss
        //       = (rec_cost_ss / t_ss) * exp(-r*t) / r
    }

    // ---- Pack results ----
    // Python: return states, simulation_info.rec_cost, simulation_info.delay_prob, ...
    PathResult res;
    res.cost                   = accum.rec_cost;
    res.cost_pricing           = accum.rec_cost_pricing;
    res.cost_outsourcing       = accum.rec_cost_boundary;
    res.cost_outsourcing_band  = accum.rec_cost_band;
    res.cost_outsourcing_outer = accum.rec_cost_outer;
    for (int r = 0; r < N_ROUTES; r++)
        res.cost_outsourcing_route[r] = accum.rec_cost_route[r];
    res.final_time             = state.t;

    for (int r = 0; r < N_ROUTES; r++) {
        // Delay probability: fraction of jobs that exceeded the delay bound, in percent
        // Python: delay_prob[i] = np.mean(delay_time_q > delay_bounds[i]) * 100
        if (fi.n_completed[r] > 0) {
            res.delay_prob[r] = 100.0 * (double)fi.n_late[r] / (double)fi.n_completed[r];
        } else {
            res.delay_prob[r] = 0.0;  // no departures recorded yet (very short run)
        }

        // Outsourcing fraction: fraction of total time spent outsourcing
        // Python: t_outsourced[m, r] / t[m]
        res.outsource_frac[r] = (state.t > 0.0)
                                ? accum.t_outsourced[r] / state.t
                                : 0.0;
    }

    return res;
}

// ==========================================================================
// SECTION 7: Aggregate over M paths
// ==========================================================================

// ---------------------------------------------------------------------------
// run_simulation
//
// Python equivalent: calling simulation() + calculate_mean_std() + print_simulation_results()
//
// Runs N_PATHS independent paths, each with its own RNG seed.
// Computes mean and stderr (= std / sqrt(M)) for cost and delay probability.
//
// Statistics are computed in two passes (store all PathResults first):
//   Pass 1: accumulate sum → compute mean
//   Pass 2: accumulate (x - mean)^2 → compute variance → stderr
//
// Memory: PathResult array = N_PATHS * ~48 bytes = ~48 KB for N_PATHS=1000 (fine on stack).
// ---------------------------------------------------------------------------
SimResult run_simulation(std::vector<PathResult>* out_paths,
                         DelayHistogram* out_hist) {

    // Use RC_N_PATHS so callers can override N_PATHS at runtime
    // (e.g. denser_sim_main sets RC_N_PATHS = 10000 before calling).
    // Default: RC_N_PATHS == N_PATHS == 1000 (set in run_config.cpp).
    const int n_paths = RC_N_PATHS;

    // Store per-path results (needed for two-pass std computation).
    // Use vector to support arbitrary n_paths at runtime.
    std::vector<PathResult> all_results(n_paths);

    printf("--- Running %d paths ---\n", n_paths);

    for (int m = 0; m < n_paths; m++) {
        // Each path gets a unique seed to be independent.
        // Python uses one shared global RNG for all paths simultaneously.
        // Different seeds → different draws, same distribution.
        // out_hist (if non-null) is shared across paths — accumulates all delays.
        all_results[m] = run_one_path(123u + (unsigned)m, out_hist);
    }

    // ---- aggregate_mean_std: Pass 1 (means) + Pass 2 (variance → stderr) ----
    SimResult result;
    {
        PROF_L1(AGGREGATE_MEAN_STD);
        // Pass 1: compute means
        result.cost_mean                  = 0.0;
        result.pricing_cost_mean          = 0.0;
        result.outsourcing_cost_mean      = 0.0;
        result.outsourcing_band_cost_mean = 0.0;
        result.outsourcing_outer_cost_mean= 0.0;
        for (int r = 0; r < N_ROUTES; r++) {
            result.delay_prob_mean[r]          = 0.0;
            result.outsource_frac_mean[r]      = 0.0;
            result.outsourcing_route_cost_mean[r] = 0.0;
        }
        // Core sums: always needed (cost_mean, delay_prob_mean, outsource_frac_mean).
        // Sub-cost sums: only needed when RC_PRINT_DETAILED_INFO=1.
        if (RC_PRINT_DETAILED_INFO) {
            for (int m = 0; m < n_paths; m++) {
                result.cost_mean                   += all_results[m].cost;
                result.pricing_cost_mean           += all_results[m].cost_pricing;
                result.outsourcing_cost_mean       += all_results[m].cost_outsourcing;
                result.outsourcing_band_cost_mean  += all_results[m].cost_outsourcing_band;
                result.outsourcing_outer_cost_mean += all_results[m].cost_outsourcing_outer;
                for (int r = 0; r < N_ROUTES; r++) {
                    result.delay_prob_mean[r]             += all_results[m].delay_prob[r];
                    result.outsource_frac_mean[r]         += all_results[m].outsource_frac[r];
                    result.outsourcing_route_cost_mean[r] += all_results[m].cost_outsourcing_route[r];
                }
            }
            result.pricing_cost_mean           /= n_paths;
            result.outsourcing_cost_mean       /= n_paths;
            result.outsourcing_band_cost_mean  /= n_paths;
            result.outsourcing_outer_cost_mean /= n_paths;
            for (int r = 0; r < N_ROUTES; r++)
                result.outsourcing_route_cost_mean[r] /= n_paths;
        } else {
            for (int m = 0; m < n_paths; m++) {
                result.cost_mean += all_results[m].cost;
                for (int r = 0; r < N_ROUTES; r++) {
                    result.delay_prob_mean[r]     += all_results[m].delay_prob[r];
                    result.outsource_frac_mean[r] += all_results[m].outsource_frac[r];
                }
            }
        }
        result.cost_mean /= n_paths;
        for (int r = 0; r < N_ROUTES; r++) {
            result.delay_prob_mean[r]     /= n_paths;
            result.outsource_frac_mean[r] /= n_paths;
        }

        // Pass 2: compute variance, then stderr = std / sqrt(M)
        double cost_var = 0.0;
        double delay_var[N_ROUTES] = {0.0, 0.0};
        for (int m = 0; m < n_paths; m++) {
            double dc = all_results[m].cost - result.cost_mean;
            cost_var += dc * dc;
            for (int r = 0; r < N_ROUTES; r++) {
                double dd = all_results[m].delay_prob[r] - result.delay_prob_mean[r];
                delay_var[r] += dd * dd;
            }
        }
        result.cost_std = std::sqrt(cost_var / ((double)n_paths * (n_paths - 1)));
        for (int r = 0; r < N_ROUTES; r++) {
            result.delay_prob_std[r] = std::sqrt(delay_var[r] / ((double)n_paths * (n_paths - 1)));
        }
    }

    result.elapsed_seconds = 0.0;  // filled in by main.cpp

    // If the caller wants per-path results, move the vector out.
    if (out_paths != nullptr) {
        *out_paths = std::move(all_results);
    }

    return result;
}
