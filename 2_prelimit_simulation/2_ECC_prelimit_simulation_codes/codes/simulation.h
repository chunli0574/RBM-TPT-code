/*
 * simulation.h
 *
 * Data structures and function declarations for the CC3 simulation loop.
 *
 * Python equivalents (CC3_python_code/codes/functions_general.py):
 *   QueueInfo     → QueueInfo class (lines 822–972)
 *   FinishedInfo  → FinishedInfo class (lines 784–819)
 *   CostAccum     → SimulationInfo attributes (lines 976–1148)
 *   PathResult    → return values of simulation() for ONE path
 *   SimResult     → aggregate over N_PATHS paths (mean / std)
 *   run_one_path  → simulation() for one path (one sequential path)
 *   run_simulation→ outer loop over N_PATHS paths + mean/std computation
 *
 * KEY DESIGN DIFFERENCE vs. Python:
 *   Python runs all M=1000 paths simultaneously at every step (vectorized NumPy).
 *   C++ runs one path at a time, sequentially (simple loop).
 *   The final statistics (mean cost, mean delay probability) are identical in
 *   distribution; only the RNG order differs.
 *
 * CC3 network:
 *   N_BUFFERS = 6, N_ROUTES = 4, DIM_K = 3, N_EVENTS = 10
 *   Events 0..3: arrivals (routes 0..3)
 *   Events 4..6: vertical departures (buffers 0..2)
 *   Events 7..9: horizontal transfers/final departure (buffers 3..5)
 */

#pragma once

#include "params.h"
#include "policy.h"
#include <random>   // std::mt19937, std::exponential_distribution
#include <queue>    // std::queue
#include <vector>   // std::vector (used by run_simulation out_paths)
#include <map>      // std::map (DelayHistogram, sparse buckets)
#include <cstdint>  // uint64_t

// ---------------------------------------------------------------------------
// QueueInfo: per-buffer FIFO queues tracking when each job entered the system
// ---------------------------------------------------------------------------
// Python: QueueInfo.enter_system_time[m, k, :]
//
// job_times[k] is a FIFO queue of entry times for buffer k.
//   front() = entry time of oldest job (next to be served)
//   back()  = entry time of newest job
//   size()  = current number of jobs in buffer k
//
// On arrival:   push entry time onto back of job_times[arrival_buffer]
// On transfer:  pop front from source buffer, push onto back of next buffer
//               (preserves original system-entry time across pipeline stages)
// On departure: read front for delay computation, then pop
struct QueueInfo {
    std::queue<double> job_times[N_BUFFERS];
};

// ---------------------------------------------------------------------------
// DelayHistogram: optional per-route bucketed histogram of per-job delays.
// Used only by sim_one_combo to draw delay-time histograms; default-off
// elsewhere via the FinishedInfo::delay_hist nullptr gate (zero overhead for
// existing binaries — no extra global read in the hot path).
//
//   bucket_width: bin size in raw simulation time units (e.g. 0.1).
//   hist[r][k]  : number of post-warm-up departures in route r whose
//                 delay = (finish_time - enter_time) falls in
//                 bucket k = floor(delay / bucket_width).
// Open-ended (sparse): keys auto-extend; no upper bound or overflow bucket.
// ---------------------------------------------------------------------------
struct DelayHistogram {
    double bucket_width = 1.0;
    std::map<int, uint64_t> hist[N_ROUTES];
};

// ---------------------------------------------------------------------------
// FinishedInfo: delay statistics per route (after warm-up)
// ---------------------------------------------------------------------------
// Python: FinishedInfo class (lines 784–819) + calculate_delay_probabilities
//
// Instead of storing all delay times (Python does), we store only counts.
// delay_prob = n_late / n_completed * 100%
struct FinishedInfo {
    long long n_completed[N_ROUTES];  // total departures recorded after warm-up
    long long n_late     [N_ROUTES];  // jobs where delay > DELAY_BOUND[route]
    // Optional histogram pointer: when non-null, qi_process_departure pushes
    // every post-warm-up delay into delay_hist->hist[route]. Default nullptr →
    // no histogram collection (zero overhead for sim/sim_proposed/sim_benchmark/
    // sim_denser). Used only by sim_one_combo.
    DelayHistogram* delay_hist = nullptr;
};

// ---------------------------------------------------------------------------
// CostAccum: accumulated cost statistics for one path
// ---------------------------------------------------------------------------
// Python: SimulationInfo attributes rec_cost, t_outsourced, etc.
struct CostAccum {
    double rec_cost;                // total discounted cost (pricing + outsourcing)
    double rec_cost_pricing;        // pricing penalty component
    double rec_cost_boundary;       // outsourcing cost component
    double t_outsourced[N_ROUTES];  // total time outsourcing per route
};

// ---------------------------------------------------------------------------
// SimState: queue lengths and clock for one path
// ---------------------------------------------------------------------------
// Python: local variables states[m, :] and simulation_info.t[m]
struct SimState {
    int    states[N_BUFFERS];  // integer queue lengths
    double t;                  // current simulation time
};

// ---------------------------------------------------------------------------
// NextEvent: result of sampling the next event
// ---------------------------------------------------------------------------
// Python: simulation_info.minInt[m] and simulation_info.mask[m, :] (M=1 here)
struct NextEvent {
    int    idx;  // which event fires (0..N_EVENTS-1)
    double dt;   // time until that event
};

// ---------------------------------------------------------------------------
// PathResult: output of one complete simulation run (one path)
// ---------------------------------------------------------------------------
// Python: return values of simulation() for one path
struct PathResult {
    double cost;                     // discounted total cost (after steady-state correction)
    double pricing_cost;             // discounted pricing penalty component
    double outsourcing_cost;         // discounted outsourcing cost component
    double delay_prob[N_ROUTES];     // % of jobs exceeding delay bound, per route
    double outsource_frac[N_ROUTES]; // fraction of time spent outsourcing, per route
    double final_time;               // total simulated clock time
};

// ---------------------------------------------------------------------------
// SimResult: aggregate statistics over all N_PATHS paths
// ---------------------------------------------------------------------------
// Python: print_simulation_results() output + calculate_mean_std()
struct SimResult {
    double cost_mean;                     // mean discounted cost across paths
    double cost_std;                      // stderr = std / sqrt(N_PATHS)
    double pricing_cost_mean;             // mean discounted pricing cost
    double outsourcing_cost_mean;         // mean discounted outsourcing cost
    double delay_prob_mean[N_ROUTES];     // mean delay probability (%) per route
    double delay_prob_std [N_ROUTES];     // stderr of delay probability
    double outsource_frac_mean[N_ROUTES]; // mean outsourcing fraction per route
    double elapsed_seconds;
};

// ---------------------------------------------------------------------------
// Function declarations
// ---------------------------------------------------------------------------

// initialize_sim_state: set queue lengths to INITIAL_STATE and time to 0.
void initialize_sim_state(SimState& state);

// initialize_cost_accum: zero out all cost accumulators.
void initialize_cost_accum(CostAccum& accum);

// initialize_queue_info: clear all queues; pre-load INITIAL_STATE[k] jobs
// into job_times[k], each with entry time 0.0 (warm-up will skip their delays).
void initialize_queue_info(QueueInfo& qi, const int states[N_BUFFERS]);

// initialize_finished_info: zero out delay counters.
void initialize_finished_info(FinishedInfo& fi);

// qi_process_arrival: push job entry time t onto back of job_times[i_buf].
// Python: QueueInfo.process_arrival()
void qi_process_arrival(QueueInfo& qi, int i_buf, double t);

// qi_process_transfer: move front job from i_buf to i_buf_next, preserving entry time.
// Python: QueueInfo.process_buffer_transfer()
void qi_process_transfer(QueueInfo& qi, int i_buf, int i_buf_next);

// qi_process_departure: job leaves system from buffer i_buf at time t.
// Compute delay = t - entry_time; if delay > DELAY_BOUND[route] → count as late.
// Python: QueueInfo.process_final_departure()
void qi_process_departure(QueueInfo& qi, FinishedInfo& fi,
                          int i_buf, int route, double t, int iter);

// qi_update: dispatch arrival/transfer/departure based on event_idx.
// Python: QueueInfo.update_queue_info_with_routes()
//
// CC3 event mapping (matches TRANSITIONS[10][6]):
//   event 0: arrive route 0 → push to buf 0
//   event 1: arrive route 1 → push to buf 1
//   event 2: arrive route 2 → push to buf 2
//   event 3: arrive route 3 → push to buf 3
//   event 4: depart buf 0 (route 0 final departure)
//   event 5: depart buf 1 (route 1 final departure)
//   event 6: depart buf 2 (route 2 final departure)
//   event 7: transfer buf 3 → buf 4 (route 3 stage 1)
//   event 8: transfer buf 4 → buf 5 (route 3 stage 2)
//   event 9: depart buf 5  (route 3 final departure)
void qi_update(QueueInfo& qi, FinishedInfo& fi, int event_idx, double t, int iter);

// generate_next_event: draw N_EVENTS Exp(1) times, disable based on policy, return minimum.
// Python: SimulationInfo.generate_random_variables() (lines 1007–1040)
//
// Disabling rules (CC3):
//   - Arrival event r (0..3): disabled if outsourcing_control[r] >= 1
//   - Vertical service event (4+kk, kk=0..2): disabled if scheduling_control[kk] != kk+1
//   - Horizontal service event (7+kk, kk=0..2): disabled if scheduling_control[kk] != DIM_K+kk+1
NextEvent generate_next_event(const PolicyOutputs& pol, std::mt19937& rng);

// compute_cost_rate: compute instantaneous cost rate.
// pricing_cost_rate = sum_r(pricing_control[r]^2 * H_DIAG)  [quadratic pricing penalty]
// outsourcing_cost_rate = sum_r(outsourcing_cost_rate[r] * max(LAMBDA[r]+pc[r],0))
//                         if outsourcing_control[r] >= 1
// Python: SimulationInfo.calculate_cost() (lines 1043–1052)
double compute_cost_rate(const PolicyOutputs& pol,
                         double* out_pricing   = nullptr,
                         double* out_outsource = nullptr);

// accumulate_cost: add discounted cost for interval dt at time t to accum.
// Python: calculate_cost() + update_cost() + update_info() (lines 1043–1081)
//
// Optional output parameters let callers reuse the already-computed cost rates
// (pricing, outsource, total) without a second compute_cost_rate call.
// Pass nullptr to skip the corresponding output.
//
// Incremental discount-factor maintenance:
//   e_neg_rt_io != nullptr  → incremental mode: *e_neg_rt_io is treated as
//       exp(-r*t) (the current discount state). Only exp(-r*dt) is computed
//       this call, saving one std::exp per step. On return *e_neg_rt_io is
//       updated to exp(-r*(t+dt)). The caller must initialise it to 1.0 at
//       t=0 and maintain it across steps. Parameter `t` is ignored.
//   e_neg_rt_io == nullptr  → legacy mode: two std::exp calls per step.
void accumulate_cost(const PolicyOutputs& pol,
                     double t, double dt,
                     CostAccum& accum,
                     double* out_pricing   = nullptr,
                     double* out_outsource = nullptr,
                     double* out_total     = nullptr,
                     double* e_neg_rt_io   = nullptr);

// apply_transition: update integer queue lengths based on which event fired.
// Python: states += mask @ transitions  in update_states_and_queues()
void apply_transition(int states[N_BUFFERS], int event_idx);

// run_one_path: execute the full N_JUMPS event loop for a single simulation path.
// Python: simulation() function for one path.
//
// Optional out_hist: if non-null, every post-warm-up departure inside this path
// pushes its delay = (finish_time - enter_time) into out_hist->hist[route].
// Default nullptr → no histogram collection (zero overhead for existing callers).
PathResult run_one_path(unsigned int seed, DelayHistogram* out_hist = nullptr);

// run_simulation: run all RC_N_PATHS paths and aggregate results.
// Python: outer loop over M paths + calculate_mean_std() + print_simulation_results()
//
// Optional: pass a non-null pointer to receive per-path results.
// If out_paths != nullptr, it is resized to RC_N_PATHS and filled with each
// path's individual cost, delay_prob, and outsource_frac.
// Used by denser_sim_main to save the full path-level distribution.
//
// Optional out_hist: if non-null, all post-warm-up per-job delays from every
// path are accumulated into the same DelayHistogram (sequential paths → safe).
// Caller pre-fills out_hist->bucket_width before calling. Default nullptr.
// Used by sim_one_combo.
SimResult run_simulation(std::vector<PathResult>* out_paths = nullptr,
                         DelayHistogram* out_hist = nullptr);
