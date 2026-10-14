/*
 * simulation.h
 *
 * Data structures and function declarations for the simulation loop.
 *
 * Python equivalents:
 *   QueueInfo     → QueueInfo class (lines 470–623)
 *   FinishedInfo  → FinishedInfo class (lines 433–468)
 *   CostAccum     → SimulationInfo class attributes (lines 626–824)
 *   PathResult    → return tuple of simulation() for ONE path
 *   SimResult     → aggregate over M paths (mean / std)
 *   run_one_path  → simulation() for one path (lines 847–922)
 *   run_simulation→ outer loop over M paths + mean/std computation
 *
 * KEY DESIGN DIFFERENCE vs. Python:
 *   Python runs all M=1000 paths simultaneously at every step (vectorized NumPy).
 *   C++ runs one path at a time, sequentially (simple loop).
 *   The final statistics (mean cost, mean delay probability) are identical in
 *   distribution; only the RNG order differs.
 */

#pragma once

#include "params.h"
#include "policy.h"
#include <random>   // std::mt19937, std::exponential_distribution
#include <queue>    // std::queue
#include <vector>   // std::vector (used by run_simulation out_paths)
#include <map>      // std::map (DelayHistogram)
#include <cstdint>  // uint64_t

// ---------------------------------------------------------------------------
// QueueInfo: per-buffer FIFO queues tracking when each job entered the system
// ---------------------------------------------------------------------------
// Python equivalent: QueueInfo class (to_cpp_single_parameter.py, line 470)
//
// HOW IT WORKS:
//   For each buffer k ∈ {0,1,2,3}, job_times[k] is a FIFO queue of entry times.
//     job_times[k].front() = entry time of the OLDEST job (next to be served)
//     job_times[k].back()  = entry time of the NEWEST job (most recently arrived)
//     job_times[k].size()  = current number of jobs in buffer k
//
//   On arrival:   push current time onto back of job_times[k]
//   On transfer:  pop front from source, push that time onto back of destination
//                 (preserves original entry time across buffer stages)
//   On departure: read front for delay computation, then pop
//
// Replaces the old circular buffer (buf_begin/buf_end/enter_time arrays).
// Advantages: no fixed BUFFER_SIZE limit, no modulo arithmetic, simpler code.
// Follows the arrivalTimeList pattern from Summary_by_claude_code.pdf.
//
// Note: BUFFER_SIZE in params.h is kept for reference but no longer used here.

struct QueueInfo {
    std::queue<double> job_times[N_BUFFERS];  // job_times[k] = FIFO queue of entry times for buffer k
                                               // Python: self.enter_system_time[m, k, :]
};

// ---------------------------------------------------------------------------
// FinishedInfo: counts completed jobs and late jobs per route (for delay probability)
// ---------------------------------------------------------------------------
// Python equivalent: FinishedInfo class (line 433) + calculate_delay_probabilities (line 768)
//
// Python stores ALL individual delay times in a large array (up to max_capacity).
// C++ stores only counts: n_completed and n_late per route.
// This is sufficient because delay_prob = n_late / n_completed * 100%.
// Assumption: this matches Python's np.mean(delay_time_q > delay_bounds[i]) * 100.

// DelayHistogram (forward decl): per-route bucketed histogram of per-job sojourn times.
//   bucket_width controls the bin size (e.g. 0.1 → 0.1-second-wide bins).
//   hist[r][k] = number of completed (post-warm-up) jobs in route r whose
//                delay = (finish_time - enter_time) falls in bucket k = floor(delay / bucket_width).
//   Open-ended: keys auto-extend as needed; no upper bound, no overflow bucket.
struct DelayHistogram {
    double bucket_width = 1.0;
    std::map<int, uint64_t> hist[N_ROUTES];
};

struct FinishedInfo {
    long long n_completed[N_ROUTES];  // total departures recorded after warm-up
                                      // Python: self.counts[route]  (up to max_capacity)
    long long n_late     [N_ROUTES];  // jobs where delay > DELAY_BOUND[route]
                                      // Python: np.sum(delay_time_q > delay_bounds[i])
    // Optional histogram pointer: when non-null, qi_process_departure pushes
    // every post-warm-up delay into delay_hist->hist[route]. Default nullptr →
    // no histogram collection (zero overhead for sim/sim_proposed/sim_benchmark/sim_denser).
    // Used only by sim_one_combo to draw delay-time histograms for a single combo.
    DelayHistogram* delay_hist = nullptr;
};

// ---------------------------------------------------------------------------
// CostAccum: accumulated cost statistics for one path
// ---------------------------------------------------------------------------
// Python equivalent: SimulationInfo attributes rec_cost, t_outsourced, etc.

struct CostAccum {
    double rec_cost;               // total discounted cost
                                   // Python: self.rec_cost[m]
    double rec_cost_pricing;       // pricing penalty component
                                   // Python: self.rec_cost_pricing[m]
    double rec_cost_boundary;      // outsourcing cost component (band + outer)
                                   // Python: self.rec_cost_boundary[m]
    double rec_cost_band;          // outsourcing cost: band zone (control == 1.0)
    double rec_cost_outer;         // outsourcing cost: forced/outer zone (control == 1.5)
    double rec_cost_route[N_ROUTES]; // outsourcing cost per route (route 0 = route 1, route 1 = route 2)
    double t_outsourced[N_ROUTES]; // total time spent outsourcing per route
                                   // Python: self.t_outsourced[m, r]
};

// ---------------------------------------------------------------------------
// SimState: queue lengths and clock for one path
// ---------------------------------------------------------------------------
// Python equivalent: local variables states[m, :] and simulation_info.t[m]

struct SimState {
    int    states[N_BUFFERS];  // integer queue lengths
    double t;                  // current simulation time
};

// ---------------------------------------------------------------------------
// NextEvent: result of sampling the next event
// ---------------------------------------------------------------------------
// Python: self.minInt[m] and self.mask[m, :] from generate_random_variables()

struct NextEvent {
    int    idx;  // which event fires (0..5)
    double dt;   // time until that event
};

// ---------------------------------------------------------------------------
// PathResult: output of one complete simulation run (one path)
// ---------------------------------------------------------------------------
// Python: return values of simulation() for one path:
//   rec_cost[m], delay_prob[0], delay_prob[1], outsource_pct[0], outsource_pct[1]

struct PathResult {
    double cost;                    // discounted total cost (after steady-state correction)
                                    // Python: simulation_info.rec_cost[m]
    double cost_pricing;            // discounted pricing cost component (rec_cost_pricing + SS tail)
    double cost_outsourcing;        // discounted outsourcing cost component (band + outer)
    double cost_outsourcing_band;        // outsourcing cost: band zone (control == 1.0)
    double cost_outsourcing_outer;       // outsourcing cost: forced/outer zone (control == 1.5)
    double cost_outsourcing_route[N_ROUTES]; // outsourcing cost per route
    double delay_prob[N_ROUTES];    // % of jobs exceeding delay bound, per route
                                    // Python: simulation_info.delay_prob[r]  (in %)
    double outsource_frac[N_ROUTES];// fraction of time spent outsourcing, per route
                                    // Python: t_outsourced[m,r] / t[m]
    double final_time;              // total simulated clock time
};

// ---------------------------------------------------------------------------
// SimResult: aggregate statistics over all N_PATHS paths
// ---------------------------------------------------------------------------
// Python: print_simulation_results() output + calculate_mean_std()

struct SimResult {
    double cost_mean;                       // mean discounted total cost across paths
    double cost_std;                        // stderr = std / sqrt(M)
                                            // Python: calculate_mean_std(rec_cost, M)
    double pricing_cost_mean;              // mean discounted pricing cost component
    double outsourcing_cost_mean;          // mean discounted outsourcing cost component (band + outer)
    double outsourcing_band_cost_mean;        // mean outsourcing cost: band zone
    double outsourcing_outer_cost_mean;       // mean outsourcing cost: forced/outer zone
    double outsourcing_route_cost_mean[N_ROUTES]; // mean outsourcing cost per route
    double delay_prob_mean[N_ROUTES];    // mean delay probability (%) per route
    double delay_prob_std [N_ROUTES];    // stderr of delay probability
    double outsource_frac_mean[N_ROUTES];// mean outsourcing fraction per route
    double elapsed_seconds;
};

// ---------------------------------------------------------------------------
// Function declarations
// ---------------------------------------------------------------------------

// initialize_sim_state: set queue lengths to INITIAL_STATE and time to 0.
// Python: states = np.zeros((M,4)) + initial_state  in initialize_simulation()
void initialize_sim_state(SimState& state);

// initialize_cost_accum: zero out all cost accumulators.
void initialize_cost_accum(CostAccum& accum);

// initialize_queue_info: clear queues and pre-load initial jobs (entry time = 0).
// Python: QueueInfo(begin=zeros, end=states, enter_system_time=zeros)
//   Pre-loads states[k] jobs into job_times[k], each with entry time 0.0.
//   Their delays are not recorded because warm-up skips the first DELAY_PROB_RECORD_START iters.
void initialize_queue_info(QueueInfo& qi, const int states[N_BUFFERS]);

// initialize_finished_info: zero out delay counters.
void initialize_finished_info(FinishedInfo& fi);

// ------- Queue operations (std::queue<double>) -------

// qi_process_arrival: push job entry time t onto back of job_times[i_buf].
// Python: QueueInfo.process_arrival(pos_arrivals, i_buffer)  — for M=1, pos_arrivals=[True]
// Behavior: EXACTLY the same as Python.
void qi_process_arrival(QueueInfo& qi, int i_buf, double t);

// qi_process_transfer: move front job from i_buf to i_buf_next, preserving original entry time.
// Python: QueueInfo.process_buffer_transfer(pos_departures, i_buffer, i_buffer_next)
// Behavior: EXACTLY the same as Python.
void qi_process_transfer(QueueInfo& qi, int i_buf, int i_buf_next);

// qi_process_departure: a job leaves the system from buffer i_buf at time t.
// Compute delay = t - enter_time; if delay > DELAY_BOUND[route], count as late.
// Python: QueueInfo.process_final_departure(finished_info, pos_departures, i_buffer, route)
// Behavior: EXACTLY the same as Python (only count, not store full delay array).
void qi_process_departure(QueueInfo& qi, FinishedInfo& fi,
                          int i_buf, int route, double t, int iter);

// qi_update: dispatch to the correct operation based on which event fired.
// Python: QueueInfo.update_queue_info_with_routes()
//   It loops over all routes and uses mask to call process_arrival/transfer/departure.
//   For M=1 in C++, we just switch on event_idx directly.
// Behavior: EXACTLY the same as Python.
void qi_update(QueueInfo& qi, FinishedInfo& fi, int event_idx, double t, int iter);

// ------- Event generation -------

// generate_next_event: draw 6 Exp(1) times, apply policy controls, return minimum.
// Python: SimulationInfo.generate_random_variables() (line 656)
NextEvent generate_next_event(const PolicyOutputs& pol, std::mt19937& rng);

// ------- Cost computation -------

// compute_cost_rate: compute instantaneous cost rate from current policy controls.
// Returns total cost rate = pricing_cost_rate + outsourcing_cost_rate.
// Used by both accumulate_cost() and the steady-state accumulator.
// Python: SimulationInfo.calculate_cost() (line 709) — split into rate computation
// out_route: if non-null, must point to N_ROUTES doubles; filled with per-route outsourcing cost.
double compute_cost_rate(const PolicyOutputs& pol,
                         double* out_pricing   = nullptr,
                         double* out_outsource = nullptr,
                         double* out_band      = nullptr,
                         double* out_outer     = nullptr,
                         double* out_route     = nullptr);

// accumulate_cost: add discounted cost for interval [t, t+dt] to accum.
// Python: calculate_cost() + update_cost() + update_info() (lines 709–735)
//
// Optional out parameters: if non-null, receive the instantaneous rates
// computed internally (pricing / outsource / band / outer / total) and the
// per-route outsourcing rate vector (out_route must point to an array of
// length N_ROUTES when non-null).  Callers (e.g. run_one_path's steady-state
// branch) can reuse these to avoid a redundant compute_cost_rate call.
void accumulate_cost(const PolicyOutputs& pol,
                     double t, double dt,
                     CostAccum& accum,
                     double* out_pricing   = nullptr,
                     double* out_outsource = nullptr,
                     double* out_band      = nullptr,
                     double* out_outer     = nullptr,
                     double* out_route     = nullptr,
                     double* out_total     = nullptr);

// apply_transition: update integer queue lengths based on which event fired.
// Python: states += mask @ transitions  in update_states_and_queues() (line 618)
void apply_transition(int states[N_BUFFERS], int event_idx);

// ------- Top-level runners -------

// run_one_path: execute the full N_JUMPS event loop for a single simulation path.
// Python: simulation() function (lines 847–922) — but for one path, not M paths.
// 'seed' is the RNG seed for this path (each path gets a different seed).
//
// Optional out_hist: if non-null, every post-warm-up departure inside this path
// pushes its delay = (finish_time - enter_time) into out_hist->hist[route].
// Default nullptr → no histogram (zero overhead).
PathResult run_one_path(unsigned int seed, DelayHistogram* out_hist = nullptr);

// run_simulation: run all RC_N_PATHS paths and aggregate results.
// Python: outer call to simulation() + calculate_mean_std() + print_simulation_results()
//
// Optional: pass a non-null pointer to receive the per-path results.
// If out_paths != nullptr, it is resized to RC_N_PATHS and filled with each
// path's individual cost, delay_prob, and outsource_frac.
// Used by denser_sim_main to save the full path-level distribution.
//
// Optional out_hist: if non-null, all post-warm-up per-job delays from every
// path are accumulated into the same DelayHistogram (sequential paths → safe).
// Caller pre-fills out_hist->bucket_width before calling. Default nullptr.
SimResult run_simulation(std::vector<PathResult>* out_paths = nullptr,
                         DelayHistogram* out_hist = nullptr);
