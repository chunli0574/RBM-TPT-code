/*
 * benchmark_search_main.cpp
 *
 * Parameter search for 12 benchmark policies (A1–A4, B1–B4, C1–C2, D1–D2).
 *
 * Benchmark taxonomy:
 *   Pricing:      BCPlinear (A1,A2,B1,B2,C1,D1)  |  High_low (A3,A4,B3,B4,C2,D2)
 *   Outsourcing:  single_pushing (A1-A4, C1-C2)   |  threshold_based (B1-B4, D1-D2)
 *   Scheduling:   LRS_DelayTime (A1,A3,B1,B3)     |  LRS_QueueLen (A2,A4,B2,B4)
 *                 WMPobj2 (C1,C2,D1,D2)
 *   Group:        A/B = no scheduling search; C/D = pricing then scheduling search
 *
 * Pricing search: two rounds (matching Python search_pricing_and_rho_parameters_logic):
 *   Round 0: sweep (beta11, beta12) or (Q_threshold, zeta1, zeta2) for route 0 row.
 *   Round 1: fix round-0 optimal for route 0 row; sweep for route 1 row.
 *   Best row = min-cost row (NO delay constraint on pricing rounds).
 *
 * Scheduling search (Group C/D only, after pricing):
 *   Sweep (lambda_tilde, beta_rate_1, beta_rate_2).
 *   Best row = min-cost row satisfying delay1 < delay_constraint AND delay2 < delay_constraint.
 *
 * CSV output layout (matches Python save_final_results):
 *   param_1, param_2, ..., cost, std, delay1, delay2, hardstop1, hardstop2
 *   Number of params = (pricing_params) + [scheduling_params] + (outsourcing_params)
 *
 * Global optimal (across all rho): saved to {output_dir}/optimal_parameter.csv.
 *   For Group A: min cost satisfying delay constraint.
 *   For Group C: same (using scheduling results which already enforce delay constraint).
 *
 * Configuration file: benchmark_search_config.txt
 *
 * TO BUILD:
 *   make sim_benchmark
 * TO RUN:
 *   make run_benchmark
 */

#include "params.h"
#include "run_config.h"
#include "simulation.h"
#include "nn_model.h"   // only included so link works; not called for benchmarks
#include "profiling.h"  // optional TIC/TOC (zero overhead when disabled)
#include "benchmark_kinds.h"  // BmkKind, bmk_kind_from_name, set_policy_modes, set_bcp/hl_pricing_para

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <sys/stat.h>

#ifdef USE_OMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Benchmark definitions
// ---------------------------------------------------------------------------

// BmkKind / enum class BmkPricing/Outsource/Scheduling / bmk_kind_from_name
// have been moved to benchmark_kinds.h/.cpp so sim_one_combo can reuse them.

// ---------------------------------------------------------------------------
// Config file structure and parser
// ---------------------------------------------------------------------------

struct BmkConfig {
    std::string benchmark_name;
    // Outer parameter search
    std::vector<double> safety_list  = {0.0};   // all benchmarks search safety
    std::vector<double> epsilon_list = {0.25};  // A/C single_pushing search epsilon
    std::vector<double> rho_list;
    // BCPlinear pricing search
    std::vector<double> beta11_list;
    std::vector<double> beta12_list;
    // High_low pricing search
    std::vector<double> Q_threshold_list;
    std::vector<double> zeta_list;
    // WMP scheduling search (Group C/D)
    std::vector<double> lambda_tilde_list;
    std::vector<double> beta_rate_1_list;
    std::vector<double> beta_rate_2_list;
    // Constraints / output
    double delay_constraint  = 1.0;
    int    inward_reflection = 0;   // 0=hold_last_no_takeover, 1=inward (A/C only)
    int    wmp_limiting      = 0;   // 0='N' full WMP formula, 1='Y' simplified limiting (C/D only)
    // wmp_extra_relative: only used when wmp_limiting == 0 (full WMP formula).
    //   0 = original extra term; 1 = extra term scaled by 1/(BETA_CONST*rho)
    int    wmp_extra_relative = 0;
    // epsilon_relative: 0=absolute (× EPSILON_SCALING={1,1,√5,√5}), 1=relative (× FACE_BOUNDARY)
    int    epsilon_relative  = 0;
    // beta_rate_symmetric: 0 = use separate beta_rate_1 and beta_rate_2 lists (default)
    //                       1 = beta_rate_2 always equals beta_rate_1 (beta_rate_2_list ignored)
    int    beta_rate_symmetric = 0;
    // highlow_fix_low_zero (HL pricing only):
    //   0 = search (zeta_low, zeta_high) with iz2 >= iz1 (default)
    //   1 = fix zeta_low = 0, search zeta_high over the full zeta_list
    int    highlow_fix_low_zero = 0;
    // pricing_magnitude_scale_by_mc_eq (BCP / HighLow pricing only):
    //   0 = default — both rounds use the same beta11/beta12 lists (BCP) or
    //                  same zeta_list (HighLow).
    //   1 = round 0 uses the original lists; round 1's lists are scaled by the
    //       outsourcing-cost ratio s = OUTSOURCE_COST_RATE[1]/OUTSOURCE_COST_RATE[0]
    //       (= 0.125/0.05 = 2.5 with the default params.h values).
    //         BCP:     beta11_list and beta12_list both ×s in round 1
    //         HighLow: zeta_list ×s in round 1; Q_threshold_list unchanged
    //       Sequential search procedure is unchanged — only the per-route lists differ.
    //       Mutually exclusive with use_shared_parameter=1 (no Round 1 to scale).
    int    pricing_magnitude_scale_by_mc_eq = 0;
    // iterative_pricing_scheduling (C/D benchmarks with WMP scheduling only):
    //   0 = default — single pass: pricing search (with sched init from cfg.sched_init_*)
    //                  followed by scheduling search.
    //   1 = iterative — inside each (safety, epsilon, rho) outer combo, repeat the
    //                   pricing→scheduling sequence; iteration k's sched init comes
    //                   from iteration k-1's scheduling optimum. Terminates when the
    //                   scheduling optimum matches the previous iteration (within
    //                   floating-point tolerance) OR iterative_pricing_scheduling_num
    //                   iterations have been run. The terminating iteration is
    //                   renamed to {se_rho_folder}/iteration_last/.
    // A/B benchmarks (no scheduling search) silently fall back to non-iterative.
    int    iterative_pricing_scheduling = 0;
    // Maximum number of pricing-scheduling iterations (must be >= 1).
    int    iterative_pricing_scheduling_num = 5;
    // use_shared_parameter (BCP / HighLow pricing only):
    //   0 = default — sequential two-round search (route 0 first, then route 1).
    //   1 = single-round shared-parameter search: route 1's params are forced to
    //       equal route 0's params (mirror), so the search effectively iterates
    //       over only the per-route parameter set.
    //         BCP:     route 0 = route 1 = (beta11, beta12)
    //         HighLow: both routes share (z_thresh, zeta_L, zeta_H)
    //       Output goes to {rho}/pricing/shared_common/.
    //   2 = shared with marginal-cost-equality scaling: route 1 equals route 0
    //       multiplied by s = OUTSOURCE_COST_RATE[1]/OUTSOURCE_COST_RATE[0]
    //       (= 0.125/0.05 = 2.5 with the default params.h values), except
    //       z_thresh in HighLow which stays equal to route 0 unscaled.
    //         BCP:     route 0 = (beta11, beta12); route 1 = (beta11×s, beta12×s)
    //         HighLow: route 0 = (z_thresh, zeta_L, zeta_H);
    //                  route 1 = (z_thresh, zeta_L×s, zeta_H×s)
    //       Output goes to {rho}/pricing/shared_mc_eq/.
    //   3 = pooled: union of the search lists from =1 and =2. Each route-0 combo
    //       is tried twice — once with route1_scale=1.0 (common) and once with
    //       route1_scale=s (mc_eq). Total combos doubles. Best is picked across
    //       all 2N rows. Output goes to {rho}/pricing/shared_pooled/.
    // Mutually exclusive with pricing_magnitude_scale_by_mc_eq=1 (any non-zero
    // value of use_shared_parameter conflicts; program errors out).
    int    use_shared_parameter = 0;
    // WMP scheduling init for pricing phase (C/D only, single values not lists)
    // Applied inside the rho loop: beta_tilde_i = sched_init_beta_rate_i * BETA_CONST * rho
    double sched_init_lambda_tilde = LAMBDA_TILDE;                // default = 4.0
    double sched_init_beta_rate_1  = BETA_TILDE_1 / BETA_CONST;  // default ≈ 0.606
    double sched_init_beta_rate_2  = BETA_TILDE_2 / BETA_CONST;  // default ≈ 0.606
    // Parallel threads for benchmark search inner loops (USE_OMP build only).
    // 0 or 1 = serial; N > 1 uses min(N, SLURM_CPUS_PER_TASK) threads.
    int    parallel_threads  = 1;
    // print_detailed_info: 0 = basic CSV only; 1 = full cost breakdown columns
    int    print_detailed_info = 0;
    std::string output_dir   = "benchmark_results";
};

static std::vector<double> parse_double_list(const std::string& s) {
    std::vector<double> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t\r\n");
        size_t b = tok.find_last_not_of(" \t\r\n");
        if (a != std::string::npos)
            v.push_back(std::stod(tok.substr(a, b - a + 1)));
    }
    return v;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

static BmkConfig load_config(const std::string& path) {
    BmkConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open config file: %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if      (key == "benchmark")          cfg.benchmark_name     = val;
        else if (key == "safety_list")        cfg.safety_list        = parse_double_list(val);
        else if (key == "epsilon_list")       cfg.epsilon_list       = parse_double_list(val);
        else if (key == "rho_list")           cfg.rho_list           = parse_double_list(val);
        else if (key == "beta11_list")        cfg.beta11_list        = parse_double_list(val);
        else if (key == "beta12_list")        cfg.beta12_list        = parse_double_list(val);
        else if (key == "Q_threshold_list")   cfg.Q_threshold_list   = parse_double_list(val);
        else if (key == "zeta_list")          cfg.zeta_list          = parse_double_list(val);
        else if (key == "lambda_tilde_list")  cfg.lambda_tilde_list  = parse_double_list(val);
        else if (key == "beta_rate_1_list")   cfg.beta_rate_1_list   = parse_double_list(val);
        else if (key == "beta_rate_2_list")   cfg.beta_rate_2_list   = parse_double_list(val);
        else if (key == "delay_constraint")   cfg.delay_constraint   = std::stod(val);
        else if (key == "inward_reflection")   cfg.inward_reflection   = std::stoi(val);
        else if (key == "wmp_limiting")        cfg.wmp_limiting        = std::stoi(val);
        else if (key == "wmp_extra_relative")  cfg.wmp_extra_relative  = std::stoi(val);
        else if (key == "epsilon_relative")    cfg.epsilon_relative    = std::stoi(val);
        else if (key == "beta_rate_symmetric")    cfg.beta_rate_symmetric    = std::stoi(val);
        else if (key == "highlow_fix_low_zero")   cfg.highlow_fix_low_zero   = std::stoi(val);
        else if (key == "pricing_magnitude_scale_by_mc_eq") cfg.pricing_magnitude_scale_by_mc_eq = std::stoi(val);
        else if (key == "use_shared_parameter")            cfg.use_shared_parameter            = std::stoi(val);
        else if (key == "iterative_pricing_scheduling")     cfg.iterative_pricing_scheduling     = std::stoi(val);
        else if (key == "iterative_pricing_scheduling_num") cfg.iterative_pricing_scheduling_num = std::stoi(val);
        else if (key == "sched_init_lambda_tilde") cfg.sched_init_lambda_tilde = std::stod(val);
        else if (key == "sched_init_beta_rate_1")  cfg.sched_init_beta_rate_1  = std::stod(val);
        else if (key == "sched_init_beta_rate_2")  cfg.sched_init_beta_rate_2  = std::stod(val);
        else if (key == "parallel_threads")        cfg.parallel_threads        = std::stoi(val);
        else if (key == "print_detailed_info")     cfg.print_detailed_info     = std::stoi(val);
        else if (key == "output_dir")              cfg.output_dir              = val;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// resolve_omp_threads
// Returns the actual number of OpenMP threads to use, capped by both the
// user-supplied cfg.parallel_threads and (if set) the SLURM_CPUS_PER_TASK
// environment variable. Also calls omp_set_num_threads + omp_set_dynamic(0)
// so the library cannot expand beyond the returned count.
// When USE_OMP is not defined, returns 1 (serial) regardless of cfg.
// ---------------------------------------------------------------------------
static int resolve_omp_threads(int cfg_threads) {
    if (cfg_threads < 1) cfg_threads = 1;
#ifdef USE_OMP
    int slurm_cap = -1;
    if (const char* s = std::getenv("SLURM_CPUS_PER_TASK")) {
        slurm_cap = std::atoi(s);
    }
    int actual = cfg_threads;
    if (slurm_cap > 0 && slurm_cap < actual) actual = slurm_cap;
    if (actual < 1) actual = 1;
    omp_set_num_threads(actual);
    omp_set_dynamic(0);
    return actual;
#else
    (void)cfg_threads;
    return 1;
#endif
}

// ---------------------------------------------------------------------------
// mkdir_p (same as in proposed_policy_main.cpp)
// ---------------------------------------------------------------------------
static void mkdir_p(const std::string& dir) {
    std::string path;
    for (char c : dir) {
        path += c;
        if (c == '/') mkdir(path.c_str(), 0755);
    }
    mkdir(dir.c_str(), 0755);
}

// ---------------------------------------------------------------------------
// fmt_param: format double for file/folder names
// ---------------------------------------------------------------------------
static std::string fmt_param(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%g", v);
    std::string s(buf);
    if (!s.empty() && s[0] == '-') s[0] = 'n';
    return s;
}

// ---------------------------------------------------------------------------
// kind_uses_wmp: returns true iff this benchmark uses WMP scheduling.
// (C1, C2, D1, D2 use WMP; A1–A4, B1–B4 use LRS_DelayTime / LRS_QueueLen.)
// ---------------------------------------------------------------------------
static inline bool kind_uses_wmp(const BmkKind& kind) {
    return kind.sched == BmkScheduling::WMP;
}

// ---------------------------------------------------------------------------
// wmp_prefix: file-name prefix that distinguishes WMP scheduling variants.
// Returns "" for non-WMP benchmarks (A/B) and for wmp_limiting=1.
//   wmp_limiting == 1     → ""       (limiting formula: extra term absent;
//                                     wmp_extra_relative is irrelevant)
//   wmp_limiting == 0  +
//     wmp_extra_relative=0 → "abs_"  (full WMP, absolute extra term)
//     wmp_extra_relative=1 → "rel_"  (full WMP, scaled / relative extra term)
// ---------------------------------------------------------------------------
static std::string wmp_prefix(const BmkKind& kind,
                              int wmp_limiting,
                              int wmp_extra_relative) {
    if (!kind_uses_wmp(kind)) return "";
    if (wmp_limiting != 0)    return "";
    return (wmp_extra_relative == 0) ? "abs_" : "rel_";
}

// ---------------------------------------------------------------------------
// BmkRecord: one simulation result row (variable-length params)
// ---------------------------------------------------------------------------
struct BmkRecord {
    std::vector<double> params;  // all model parameters for this combo
    double cost_mean, cost_std;
    double delay1, delay2;
    double hardstop1 = 0.0, hardstop2 = 0.0;
    double delay1_std = 0.0;
    double delay2_std = 0.0;
    double pricing_cost_mean           = 0.0;
    double outsourcing_cost_mean       = 0.0;
    double outsourcing_band_cost_mean  = 0.0;
    double outsourcing_outer_cost_mean = 0.0;
    double outsourcing_cost_1          = 0.0;
    double outsourcing_cost_2          = 0.0;
};

// ---------------------------------------------------------------------------
// Save a vector of BmkRecord to CSV
// ---------------------------------------------------------------------------
static void save_bmk_csv(const std::string& filepath,
                          const std::vector<BmkRecord>& recs) {
    PROF_L3(SAVE_BMK_CSV);
    if (recs.empty()) return;
    FILE* fp = fopen(filepath.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot write CSV: %s\n", filepath.c_str());
        return;
    }
    int np = (int)recs[0].params.size();
    for (int i = 0; i < np; i++)
        fprintf(fp, "param_%d,", i + 1);
    if (RC_PRINT_DETAILED_INFO) {
        fprintf(fp, "cost,std,delay1,delay2,delay1_std,delay2_std,hardstop1,hardstop2,pricing_cost,outsourcing_cost,outsourcing_cost_band,outsourcing_cost_outer,outsourcing_cost_1,outsourcing_cost_2\n");
        for (const auto& r : recs) {
            for (double p : r.params) fprintf(fp, "%.6g,", p);
            fprintf(fp, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    r.cost_mean, r.cost_std, r.delay1, r.delay2, r.delay1_std, r.delay2_std,
                    r.hardstop1, r.hardstop2,
                    r.pricing_cost_mean, r.outsourcing_cost_mean,
                    r.outsourcing_band_cost_mean, r.outsourcing_outer_cost_mean,
                    r.outsourcing_cost_1, r.outsourcing_cost_2);
        }
    } else {
        fprintf(fp, "cost,std,delay1,delay2,delay1_std,delay2_std\n");
        for (const auto& r : recs) {
            for (double p : r.params) fprintf(fp, "%.6g,", p);
            fprintf(fp, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    r.cost_mean, r.cost_std, r.delay1, r.delay2, r.delay1_std, r.delay2_std);
        }
    }
    fclose(fp);
    printf("  Saved: %s\n", filepath.c_str());
}

// ---------------------------------------------------------------------------
// Find the min-cost record satisfying the delay constraint (or without
// constraint when apply_delay_constraint==false).  Returns nullptr if none.
// Python: find_min_cost_proposed_scheduling
// ---------------------------------------------------------------------------
static const BmkRecord* find_min_cost(const std::vector<BmkRecord>& recs,
                                       double delay_constraint,
                                       bool apply_delay_constraint) {
    PROF_L3(FIND_MIN_COST);
    const BmkRecord* best = nullptr;
    for (const auto& r : recs) {
        if (apply_delay_constraint &&
            !(r.delay1 < delay_constraint && r.delay2 < delay_constraint))
            continue;
        if (best == nullptr || r.cost_mean < best->cost_mean)
            best = &r;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Print and optionally save a single optimal record
// ---------------------------------------------------------------------------
static void print_and_save_optimal(const BmkRecord* best,
                                    double delay_constraint,
                                    const std::string& opt_path,
                                    const std::string& label) {
    printf("\n  [optimal %s]", label.c_str());
    if (best == nullptr) {
        printf(" No feasible row (delay < %.2f%%) found.\n", delay_constraint);
        return;
    }
    printf(" min-cost feasible row:\n");
    printf("    params:");
    for (double p : best->params) printf(" %.4g", p);
    printf("\n    cost=%.4f  std=%.4f  delay1=%.3f%%  delay2=%.3f%%\n",
           best->cost_mean, best->cost_std, best->delay1, best->delay2);

    FILE* fp = fopen(opt_path.c_str(), "w");
    if (fp) {
        int np = (int)best->params.size();
        for (int i = 0; i < np; i++)
            fprintf(fp, "param_%d,", i + 1);
        if (RC_PRINT_DETAILED_INFO) {
            fprintf(fp, "cost,std,delay1,delay2,delay1_std,delay2_std,hardstop1,hardstop2,pricing_cost,outsourcing_cost,outsourcing_cost_band,outsourcing_cost_outer,outsourcing_cost_1,outsourcing_cost_2\n");
            for (double p : best->params) fprintf(fp, "%.6g,", p);
            fprintf(fp, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    best->cost_mean, best->cost_std, best->delay1, best->delay2, best->delay1_std, best->delay2_std,
                    best->hardstop1, best->hardstop2,
                    best->pricing_cost_mean, best->outsourcing_cost_mean,
                    best->outsourcing_band_cost_mean, best->outsourcing_outer_cost_mean,
                    best->outsourcing_cost_1, best->outsourcing_cost_2);
        } else {
            fprintf(fp, "cost,std,delay1,delay2,delay1_std,delay2_std\n");
            for (double p : best->params) fprintf(fp, "%.6g,", p);
            fprintf(fp, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    best->cost_mean, best->cost_std, best->delay1, best->delay2,
                    best->delay1_std, best->delay2_std);
        }
        fclose(fp);
        printf("  Saved optimal to: %s\n", opt_path.c_str());
    }
}

// set_policy_modes / set_bcp_pricing_para / set_hl_pricing_para
// are now defined in benchmark_kinds.cpp.

// ---------------------------------------------------------------------------
// Run simulation and return a BmkRecord with the given param vector
// ---------------------------------------------------------------------------
static BmkRecord run_one_bmk(std::vector<double> params) {
    PROF_L3(RUN_ONE_BMK);
    SimResult res = run_simulation();
    BmkRecord rec;
    rec.params               = std::move(params);
    rec.cost_mean            = res.cost_mean;
    rec.cost_std             = res.cost_std;
    rec.delay1               = res.delay_prob_mean[0];
    rec.delay2               = res.delay_prob_mean[1];
    // hardstop: tracked as outsource fraction spikes; benchmarks don't trigger it
    rec.hardstop1            = 0.0;
    rec.hardstop2            = 0.0;
    rec.delay1_std                  = res.delay_prob_std[0];
    rec.delay2_std                  = res.delay_prob_std[1];
    rec.pricing_cost_mean           = res.pricing_cost_mean;
    rec.outsourcing_cost_mean       = res.outsourcing_cost_mean;
    rec.outsourcing_band_cost_mean  = res.outsourcing_band_cost_mean;
    rec.outsourcing_outer_cost_mean = res.outsourcing_outer_cost_mean;
    rec.outsourcing_cost_1          = res.outsourcing_route_cost_mean[0];
    rec.outsourcing_cost_2          = res.outsourcing_route_cost_mean[1];
    return rec;
}

// ---------------------------------------------------------------------------
// BCPlinear pricing parameter search: two rounds
//
// Round 0: row0=[beta11,beta11,beta12,beta12], row1=[0,0,0,0]
//          Enumerate beta11_list × beta12_list.
// Round 1: row0 = pricing_opt_row0 (fixed from round 0)
//          row1 = [beta12,beta12,beta11,beta11]  (swapped roles for route-2 search)
//          Enumerate beta11_list × beta12_list.
//
// Returns the optimal flat(2×4) pricing params found after round 1,
// or all-zeros if no records collected.
// ---------------------------------------------------------------------------
static std::vector<double> bcplinear_pricing_search(
    const BmkKind& kind,
    double rho,
    const BmkConfig& cfg,
    const std::string& rho_folder,
    std::vector<BmkRecord>& all_pricing_records,  // accumulate for global opt
    int n_threads)
{
    // outsourcing_para: [rho] or [rho, rho]
    std::vector<double> out_para = (kind.outsource == BmkOutsource::SinglePushing)
        ? std::vector<double>{rho}
        : std::vector<double>{rho, rho};

    // Current optimal pricing params: 4 scalars total, 2 per route,
    // ordered as [phi_own_0, phi_other_0, phi_own_1, phi_other_1].
    // Round 0 searches route 0's (phi_own, phi_other) with route 1's params fixed at 0;
    // Round 1 searches route 1's (phi_own, phi_other) with route 0's params fixed at round-0 optimum.
    double pricing_opt[4] = {0, 0, 0, 0};
    bool found_opt = false;

    std::vector<BmkRecord> best_records;  // best per round (for printing)

    // use_shared_parameter: collapse to a single shared-parameter round.
    // - n_rounds = 1 (no Round 1)
    // - flat assignment mirrors route 0 into route 1's slots
    // - =1 (shared_common): route 1 = route 0 directly                       (scales = {1.0})
    // - =2 (shared_mc_eq):  route 1 = route 0 × s (BCP); HighLow zeta scaled (scales = {s})
    // - =3 (shared_pooled): union of =1 and =2 search lists                  (scales = {1.0, s})
    // - results go to {rho}/pricing/{shared_common|shared_mc_eq|shared_pooled}/
    const int n_rounds = cfg.use_shared_parameter ? 1 : 2;
    const double mc_eq_ratio = OUTSOURCE_COST_RATE[1] / OUTSOURCE_COST_RATE[0];
    std::vector<double> shared_scales;
    if (cfg.use_shared_parameter == 1)      shared_scales = { 1.0 };
    else if (cfg.use_shared_parameter == 2) shared_scales = { mc_eq_ratio };
    else if (cfg.use_shared_parameter == 3) shared_scales = { 1.0, mc_eq_ratio };

    for (int i_round = 0; i_round < n_rounds; i_round++) {
        std::string round_subdir;
        if (cfg.use_shared_parameter == 1)      round_subdir = "shared_common";
        else if (cfg.use_shared_parameter == 2) round_subdir = "shared_mc_eq";
        else if (cfg.use_shared_parameter == 3) round_subdir = "shared_pooled";
        else                                    round_subdir = std::string("round") + std::to_string(i_round);
        std::string round_folder = rho_folder + "/pricing/" + round_subdir;
        mkdir_p(round_folder);
        std::string csv_path = round_folder + "/RS_final_results.csv";

        // pricing_magnitude_scale_by_mc_eq: in round 1 only, scale beta11/beta12
        // by OUTSOURCE_COST_RATE[1]/OUTSOURCE_COST_RATE[0] (= 2.5 with default
        // params.h values). round 0 always uses scale = 1 (original lists).
        // Note: use_shared_parameter > 0 always gives scale=1 (i_round only reaches 0,
        // and the shared-mode scaling is handled separately via per-combo route1_scale).
        const double scale = (cfg.pricing_magnitude_scale_by_mc_eq && i_round == 1)
            ? (OUTSOURCE_COST_RATE[1] / OUTSOURCE_COST_RATE[0])
            : 1.0;

        // Pre-expand combos so the parallel region can index into a flat vector.
        // BcpCombo carries route1_scale so the shared/pooled branch can mirror
        // route 0 → route 1 with either scale 1.0 or s on a per-combo basis.
        struct BcpCombo { double beta11, beta12, route1_scale; };
        std::vector<BcpCombo> combos;
        if (cfg.use_shared_parameter) {
            combos.reserve(cfg.beta11_list.size() * cfg.beta12_list.size() * shared_scales.size());
            for (double sc : shared_scales)
                for (double b11 : cfg.beta11_list)
                    for (double b12 : cfg.beta12_list)
                        combos.push_back({b11, b12, sc});
        } else {
            combos.reserve(cfg.beta11_list.size() * cfg.beta12_list.size());
            for (double b11 : cfg.beta11_list)
                for (double b12 : cfg.beta12_list)
                    combos.push_back({b11, b12, 1.0});  // route1_scale unused in non-shared mode
        }
        const int total_combos = (int)combos.size();
        if (cfg.use_shared_parameter == 1) {
            printf("  [shared_common] %d combos (threads=%d)\n", total_combos, n_threads);
        } else if (cfg.use_shared_parameter == 2) {
            printf("  [shared_mc_eq] %d combos (threads=%d, route1_scale=%.4g)\n",
                   total_combos, n_threads, mc_eq_ratio);
        } else if (cfg.use_shared_parameter == 3) {
            printf("  [shared_pooled] %d combos (threads=%d, route1_scale ∈ {1.0, %.4g})\n",
                   total_combos, n_threads, mc_eq_ratio);
        } else if (scale != 1.0) {
            printf("  [Round %d] %d combos (threads=%d, scale=%.4g)\n",
                   i_round, total_combos, n_threads, scale);
        } else {
            printf("  [Round %d] %d combos (threads=%d)\n", i_round, total_combos, n_threads);
        }

        std::vector<BmkRecord> round_recs(total_combos);

#ifdef USE_OMP
        // The WMP scheduling controls are thread_local.  The caller fixes them
        // for this pricing round on the encountering thread, so explicitly
        // copy that state into every OpenMP worker before evaluating combos.
        const double fixed_lambda_tilde = RC_LAMBDA_TILDE;
        const double fixed_beta_tilde_1 = RC_BETA_TILDE_1;
        const double fixed_beta_tilde_2 = RC_BETA_TILDE_2;
        #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
        {
            RC_LAMBDA_TILDE = fixed_lambda_tilde;
            RC_BETA_TILDE_1 = fixed_beta_tilde_1;
            RC_BETA_TILDE_2 = fixed_beta_tilde_2;

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < total_combos; ++i) {
                double beta11 = combos[i].beta11 * scale;
                double beta12 = combos[i].beta12 * scale;
                double flat[4];
                if (cfg.use_shared_parameter) {
                    // route 1 mirrors route 0 with per-combo route1_scale
                    // (=1: scale=1.0; =2: scale=s; =3: scale ∈ {1.0, s} per combo)
                    double sc = combos[i].route1_scale;
                    flat[0]=beta11;       flat[1]=beta12;
                    flat[2]=beta11*sc;    flat[3]=beta12*sc;
                } else if (i_round == 0) {
                    // route 0 active: (phi_own_0, phi_other_0) = (beta11, beta12); route 1 = 0
                    flat[0]=beta11; flat[1]=beta12; flat[2]=0; flat[3]=0;
                } else {
                    // route 1 active: (phi_own_1, phi_other_1) = (beta11, beta12);
                    // route 0 fixed at round-0 optimum
                    flat[0]=pricing_opt[0]; flat[1]=pricing_opt[1];
                    flat[2]=beta11;         flat[3]=beta12;
                }
                // RC_PRICING_PARA is thread_local — each thread sets its own copy.
                set_bcp_pricing_para(flat);

                std::vector<double> model_para(flat, flat + 4);
                for (double v : out_para) model_para.push_back(v);

                round_recs[i] = run_one_bmk(model_para);
            }
        }
#else
        for (int i = 0; i < total_combos; ++i) {
            double beta11 = combos[i].beta11 * scale;
            double beta12 = combos[i].beta12 * scale;
            double flat[4];
            if (cfg.use_shared_parameter) {
                double sc = combos[i].route1_scale;
                flat[0]=beta11;       flat[1]=beta12;
                flat[2]=beta11*sc;    flat[3]=beta12*sc;
            } else if (i_round == 0) {
                flat[0]=beta11; flat[1]=beta12; flat[2]=0; flat[3]=0;
            } else {
                flat[0]=pricing_opt[0]; flat[1]=pricing_opt[1];
                flat[2]=beta11;         flat[3]=beta12;
            }
            set_bcp_pricing_para(flat);

            std::vector<double> model_para(flat, flat + 4);
            for (double v : out_para) model_para.push_back(v);

            printf("    [%d/%d] beta11=%.4g beta12=%.4g rho=%.4g ",
                   i + 1, total_combos, beta11, beta12, rho);
            round_recs[i] = run_one_bmk(model_para);
            printf("cost=%.4f delay1=%.3f%% delay2=%.3f%%\n",
                   round_recs[i].cost_mean, round_recs[i].delay1, round_recs[i].delay2);
        }
#endif

        for (const auto& r : round_recs) all_pricing_records.push_back(r);

        save_bmk_csv(csv_path, round_recs);

        // Find min-cost (no delay constraint) in this round
        const BmkRecord* best = find_min_cost(round_recs, cfg.delay_constraint, false);
        if (best == nullptr) {
            const char* label_no_rec = (cfg.use_shared_parameter == 1) ? "shared_common"  :
                                       (cfg.use_shared_parameter == 2) ? "shared_mc_eq"   :
                                       (cfg.use_shared_parameter == 3) ? "shared_pooled"  : nullptr;
            if (label_no_rec)
                printf("  [%s] No records — stopping pricing search.\n", label_no_rec);
            else
                printf("  [Round %d] No records — stopping pricing search.\n", i_round);
            break;
        }
        // Extract optimal pricing params (first 4 entries of params)
        int n_out = (int)out_para.size();
        int n_pr  = (int)best->params.size() - n_out;  // = 4
        for (int k = 0; k < 4 && k < n_pr; k++)
            pricing_opt[k] = best->params[k];
        found_opt = true;

        std::string opt_path = rho_folder + "/pricing/optimal_parameter.csv";
        std::string label;
        if (cfg.use_shared_parameter == 1)
            label = "rho=" + fmt_param(rho) + " shared_common";
        else if (cfg.use_shared_parameter == 2)
            label = "rho=" + fmt_param(rho) + " shared_mc_eq";
        else if (cfg.use_shared_parameter == 3)
            label = "rho=" + fmt_param(rho) + " shared_pooled";
        else
            label = "rho=" + fmt_param(rho) + " round" + std::to_string(i_round);
        print_and_save_optimal(best, cfg.delay_constraint, opt_path, label);
    }

    if (!found_opt) {
        // Return zeros (4 scalars: phi_own_0, phi_other_0, phi_own_1, phi_other_1)
        return std::vector<double>(4, 0.0);
    }
    return std::vector<double>(pricing_opt, pricing_opt + 4);
}

// ---------------------------------------------------------------------------
// High_low pricing parameter search: two rounds
//
// Round i_round: modify pricing_para[i_round,:] = [int(Q*42*rho), zeta1, zeta2]
//                Keep other row from previous round's optimal.
//
// zeta1 iterates over zeta_list; zeta2 iterates over zeta_list[idx_zeta1:]
// (combinations with repetition: zeta2 >= zeta1 by index)
//
// Returns optimal flat(2×3) pricing params.
// ---------------------------------------------------------------------------
static std::vector<double> highlow_pricing_search(
    const BmkKind& kind,
    double rho,
    const BmkConfig& cfg,
    const std::string& rho_folder,
    std::vector<BmkRecord>& all_pricing_records,
    int n_threads)
{
    std::vector<double> out_para = (kind.outsource == BmkOutsource::SinglePushing)
        ? std::vector<double>{rho}
        : std::vector<double>{rho, rho};

    // q_bound_max = 42 (= int(delay_bound * lambda) = int((400/3)*(0.95/3)))
    const int Q_BOUND = Q_BOUND_MAX;  // 42 from params.h

    double pricing_opt[6] = {0,0,0, 0,0,0};
    bool found_opt = false;

    // use_shared_parameter: collapse to a single shared-parameter round.
    // - n_rounds = 1
    // - =1 (shared_common): route 1 = route 0 (mirror)          (scales = {1.0})
    // - =2 (shared_mc_eq):  zeta scaled by s; z_thresh unchanged (scales = {s})
    // - =3 (shared_pooled): union of =1 and =2                  (scales = {1.0, s})
    const int n_rounds = cfg.use_shared_parameter ? 1 : 2;
    const double mc_eq_ratio = OUTSOURCE_COST_RATE[1] / OUTSOURCE_COST_RATE[0];
    std::vector<double> shared_scales;
    if (cfg.use_shared_parameter == 1)      shared_scales = { 1.0 };
    else if (cfg.use_shared_parameter == 2) shared_scales = { mc_eq_ratio };
    else if (cfg.use_shared_parameter == 3) shared_scales = { 1.0, mc_eq_ratio };

    for (int i_round = 0; i_round < n_rounds; i_round++) {
        std::string round_subdir;
        if (cfg.use_shared_parameter == 1)      round_subdir = "shared_common";
        else if (cfg.use_shared_parameter == 2) round_subdir = "shared_mc_eq";
        else if (cfg.use_shared_parameter == 3) round_subdir = "shared_pooled";
        else                                    round_subdir = std::string("round") + std::to_string(i_round);
        std::string round_folder = rho_folder + "/pricing/" + round_subdir;
        mkdir_p(round_folder);
        std::string csv_path = round_folder + "/RS_final_results.csv";

        // pricing_magnitude_scale_by_mc_eq: in round 1 only, scale zeta_list
        // by OUTSOURCE_COST_RATE[1]/OUTSOURCE_COST_RATE[0] (= 2.5 with default
        // params.h). Q_threshold_list (and thus z_thresh) is NOT scaled.
        // Note: use_shared_parameter > 0 always gives scale=1 (i_round only reaches 0,
        // and shared-mode scaling is handled separately via per-combo route1_scale).
        const double scale = (cfg.pricing_magnitude_scale_by_mc_eq && i_round == 1)
            ? (OUTSOURCE_COST_RATE[1] / OUTSOURCE_COST_RATE[0])
            : 1.0;

        // Pre-expand combos so the parallel region can index into a flat vector.
        // Two grid modes (cfg.highlow_fix_low_zero):
        //   0 = default: zeta1 iterates zeta_list, zeta2 from iz1..end
        //                (combinations with repetition, |Q| * |zeta|*(|zeta|+1)/2 combos)
        //   1 = zeta1 fixed to 0, zeta2 iterates the full zeta_list
        //                (|Q| * |zeta| combos)
        // HlCombo carries route1_scale so the shared/pooled branch can mirror
        // route 0 → route 1 with either scale 1.0 or s on a per-combo basis.
        // z_thresh is NEVER scaled (Q_threshold preserved as-is on route 1).
        // z_thresh formula: Q_loop * Q_BOUND_MAX * rho * INV_SQRT_N
        struct HlCombo { double Q_loop; double z_thresh; double zeta1, zeta2; double route1_scale; };
        const int nZ = (int)cfg.zeta_list.size();
        std::vector<HlCombo> combos;
        // Helper: build base (Q, z_thresh, zeta1, zeta2) tuples (apply optional
        // per-round 'scale' from pricing_magnitude_scale_by_mc_eq to zeta only).
        auto build_base = [&](std::vector<HlCombo>& out, double sc_for_route1) {
            if (cfg.highlow_fix_low_zero) {
                for (double Q_loop : cfg.Q_threshold_list) {
                    double z_thresh = Q_loop * Q_BOUND * rho * INV_SQRT_N;
                    for (int iz = 0; iz < nZ; ++iz) {
                        double zeta2 = cfg.zeta_list[iz] * scale;  // 0 * scale = 0 still 0
                        out.push_back({Q_loop, z_thresh, 0.0, zeta2, sc_for_route1});
                    }
                }
            } else {
                for (double Q_loop : cfg.Q_threshold_list) {
                    double z_thresh = Q_loop * Q_BOUND * rho * INV_SQRT_N;
                    for (int iz1 = 0; iz1 < nZ; ++iz1) {
                        double zeta1 = cfg.zeta_list[iz1] * scale;
                        for (int iz2 = iz1; iz2 < nZ; ++iz2) {
                            double zeta2 = cfg.zeta_list[iz2] * scale;
                            out.push_back({Q_loop, z_thresh, zeta1, zeta2, sc_for_route1});
                        }
                    }
                }
            }
        };
        if (cfg.use_shared_parameter) {
            // Build one combo set per shared_scale; route1_scale is per-combo
            for (double sc : shared_scales) build_base(combos, sc);
        } else {
            // Non-shared mode: route1_scale unused, fill with 1.0
            build_base(combos, 1.0);
        }
        const int total_combos = (int)combos.size();
        if (cfg.use_shared_parameter == 1) {
            printf("  [shared_common] %d combos (threads=%d, fix_low_zero=%d)\n",
                   total_combos, n_threads, cfg.highlow_fix_low_zero);
        } else if (cfg.use_shared_parameter == 2) {
            printf("  [shared_mc_eq] %d combos (threads=%d, fix_low_zero=%d, "
                   "route1_zeta_scale=%.4g, z_thresh_unchanged)\n",
                   total_combos, n_threads, cfg.highlow_fix_low_zero, mc_eq_ratio);
        } else if (cfg.use_shared_parameter == 3) {
            printf("  [shared_pooled] %d combos (threads=%d, fix_low_zero=%d, "
                   "route1_zeta_scale ∈ {1.0, %.4g}, z_thresh_unchanged)\n",
                   total_combos, n_threads, cfg.highlow_fix_low_zero, mc_eq_ratio);
        } else if (scale != 1.0) {
            printf("  [Round %d] %d combos (threads=%d, fix_low_zero=%d, zeta_scale=%.4g)\n",
                   i_round, total_combos, n_threads, cfg.highlow_fix_low_zero, scale);
        } else {
            printf("  [Round %d] %d combos (threads=%d, fix_low_zero=%d)\n",
                   i_round, total_combos, n_threads, cfg.highlow_fix_low_zero);
        }

        std::vector<BmkRecord> round_recs(total_combos);

#ifdef USE_OMP
        // Keep every worker on the scheduling policy fixed by the caller for
        // this pricing round.  Workers may otherwise retain defaults or the
        // last candidate from a preceding scheduling search.
        const double fixed_lambda_tilde = RC_LAMBDA_TILDE;
        const double fixed_beta_tilde_1 = RC_BETA_TILDE_1;
        const double fixed_beta_tilde_2 = RC_BETA_TILDE_2;
        #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
        {
            RC_LAMBDA_TILDE = fixed_lambda_tilde;
            RC_BETA_TILDE_1 = fixed_beta_tilde_1;
            RC_BETA_TILDE_2 = fixed_beta_tilde_2;

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < total_combos; ++i) {
                const HlCombo& c = combos[i];
                double flat[6];
                if (cfg.use_shared_parameter) {
                    // route 1 mirrors route 0 with per-combo route1_scale.
                    // z_thresh is never scaled (Q_threshold preserved as-is).
                    // (=1: route1_scale=1.0; =2: route1_scale=s; =3: route1_scale ∈ {1.0, s} per combo)
                    double sc = c.route1_scale;
                    flat[0] = c.z_thresh; flat[1] = c.zeta1;        flat[2] = c.zeta2;
                    flat[3] = c.z_thresh; flat[4] = c.zeta1*sc;     flat[5] = c.zeta2*sc;
                } else {
                    for (int k = 0; k < 6; k++) flat[k] = pricing_opt[k];
                    flat[i_round * 3 + 0] = c.z_thresh;
                    flat[i_round * 3 + 1] = c.zeta1;
                    flat[i_round * 3 + 2] = c.zeta2;
                }
                // RC_HL_PARA is thread_local — each thread sets its own copy.
                set_hl_pricing_para(flat);

                std::vector<double> model_para(flat, flat + 6);
                for (double v : out_para) model_para.push_back(v);

                round_recs[i] = run_one_bmk(model_para);
            }
        }
#else
        for (int i = 0; i < total_combos; ++i) {
            const HlCombo& c = combos[i];
            double flat[6];
            if (cfg.use_shared_parameter) {
                double sc = c.route1_scale;
                flat[0] = c.z_thresh; flat[1] = c.zeta1;        flat[2] = c.zeta2;
                flat[3] = c.z_thresh; flat[4] = c.zeta1*sc;     flat[5] = c.zeta2*sc;
            } else {
                for (int k = 0; k < 6; k++) flat[k] = pricing_opt[k];
                flat[i_round * 3 + 0] = c.z_thresh;
                flat[i_round * 3 + 1] = c.zeta1;
                flat[i_round * 3 + 2] = c.zeta2;
            }
            set_hl_pricing_para(flat);

            std::vector<double> model_para(flat, flat + 6);
            for (double v : out_para) model_para.push_back(v);

            printf("    [%d/%d] Q_loop=%.2f z_thresh=%.4f zeta1=%.4g zeta2=%.4g rho=%.4g ",
                   i + 1, total_combos, c.Q_loop, c.z_thresh, c.zeta1, c.zeta2, rho);
            round_recs[i] = run_one_bmk(model_para);
            printf("cost=%.4f delay1=%.3f%% delay2=%.3f%%\n",
                   round_recs[i].cost_mean, round_recs[i].delay1, round_recs[i].delay2);
        }
#endif

        for (const auto& r : round_recs) all_pricing_records.push_back(r);

        save_bmk_csv(csv_path, round_recs);

        const BmkRecord* best = find_min_cost(round_recs, cfg.delay_constraint, false);
        if (best == nullptr) {
            const char* label_no_rec = (cfg.use_shared_parameter == 1) ? "shared_common"  :
                                       (cfg.use_shared_parameter == 2) ? "shared_mc_eq"   :
                                       (cfg.use_shared_parameter == 3) ? "shared_pooled"  : nullptr;
            if (label_no_rec)
                printf("  [%s] No records — stopping pricing search.\n", label_no_rec);
            else
                printf("  [Round %d] No records — stopping pricing search.\n", i_round);
            break;
        }
        int n_out = (int)out_para.size();
        int n_pr  = (int)best->params.size() - n_out;  // = 6
        for (int k = 0; k < 6 && k < n_pr; k++)
            pricing_opt[k] = best->params[k];
        found_opt = true;

        std::string opt_path = rho_folder + "/pricing/optimal_parameter.csv";
        std::string label;
        if (cfg.use_shared_parameter == 1)
            label = "rho=" + fmt_param(rho) + " shared_common";
        else if (cfg.use_shared_parameter == 2)
            label = "rho=" + fmt_param(rho) + " shared_mc_eq";
        else if (cfg.use_shared_parameter == 3)
            label = "rho=" + fmt_param(rho) + " shared_pooled";
        else
            label = "rho=" + fmt_param(rho) + " round" + std::to_string(i_round);
        print_and_save_optimal(best, cfg.delay_constraint, opt_path, label);
    }

    if (!found_opt)
        return std::vector<double>(6, 0.0);
    return std::vector<double>(pricing_opt, pricing_opt + 6);
}

// ---------------------------------------------------------------------------
// WMP scheduling parameter search (Group C/D)
//
// Uses the optimal pricing params found above.
// Sweeps lambda_tilde × beta_rate_1 × beta_rate_2.
// Applies delay constraint when finding optimal.
// ---------------------------------------------------------------------------
static std::vector<BmkRecord> wmp_scheduling_search(
    const BmkKind& kind,
    double rho,
    const std::vector<double>& pricing_flat,  // flattened pricing params (8 or 6)
    const BmkConfig& cfg,
    const std::string& rho_folder,
    int n_threads)
{
    std::vector<double> out_para = (kind.outsource == BmkOutsource::SinglePushing)
        ? std::vector<double>{rho}
        : std::vector<double>{rho, rho};

    // Set WMP scheduling mode (already set globally; but make explicit)
    RC_SCHEDULING_MODE = WMP_OBJ2;

    std::string sched_folder = rho_folder + "/scheduling";
    mkdir_p(sched_folder);
    std::string csv_path = sched_folder + "/RS_final_results.csv";

    // Pre-expand all combos into a flat vector so the parallel region can
    // index by i and write to a pre-allocated results vector.
    struct SchedCombo { double lam, br1, br2, bt1, bt2; };
    std::vector<SchedCombo> combos;
    for (double lam : cfg.lambda_tilde_list) {
        for (double br1 : cfg.beta_rate_1_list) {
            const std::vector<double>& br2_list = cfg.beta_rate_symmetric
                ? std::vector<double>{br1}
                : cfg.beta_rate_2_list;
            for (double br2 : br2_list) {
                SchedCombo c;
                c.lam = lam; c.br1 = br1; c.br2 = br2;
                c.bt1 = br1 * BETA_CONST * rho;
                c.bt2 = br2 * BETA_CONST * rho;
                combos.push_back(c);
            }
        }
    }
    const int total_combos = (int)combos.size();
    printf("  [Scheduling search] %d combos  (beta_rate_symmetric=%d, threads=%d)\n",
           total_combos, cfg.beta_rate_symmetric, n_threads);

    std::vector<BmkRecord> sched_recs(total_combos);

#ifdef USE_OMP
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
    {
        // Seed this thread's thread_local RC_PRICING_PARA / RC_HL_PARA with
        // the optimal pricing params (already set in the master before this
        // function was called; thread_local copies start at the defaults).
        if (kind.pricing == BmkPricing::BCPlinear) {
            double flat[4] = {0};
            for (int k = 0; k < 4 && k < (int)pricing_flat.size(); ++k)
                flat[k] = pricing_flat[k];
            set_bcp_pricing_para(flat);
        } else {
            double flat[6] = {0};
            for (int k = 0; k < 6 && k < (int)pricing_flat.size(); ++k)
                flat[k] = pricing_flat[k];
            set_hl_pricing_para(flat);
        }

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < total_combos; ++i) {
            const SchedCombo& c = combos[i];
            RC_LAMBDA_TILDE = c.lam;
            RC_BETA_TILDE_1 = c.bt1;
            RC_BETA_TILDE_2 = c.bt2;

            std::vector<double> model_para = pricing_flat;
            model_para.push_back(c.lam);
            model_para.push_back(c.bt1);
            model_para.push_back(c.bt2);
            for (double v : out_para) model_para.push_back(v);

            sched_recs[i] = run_one_bmk(model_para);
        }
    }
#else
    for (int i = 0; i < total_combos; ++i) {
        const SchedCombo& c = combos[i];
        RC_LAMBDA_TILDE = c.lam;
        RC_BETA_TILDE_1 = c.bt1;
        RC_BETA_TILDE_2 = c.bt2;

        std::vector<double> model_para = pricing_flat;
        model_para.push_back(c.lam);
        model_para.push_back(c.bt1);
        model_para.push_back(c.bt2);
        for (double v : out_para) model_para.push_back(v);

        printf("    [%d/%d] lam=%.4g br1=%.4g br2=%.4g rho=%.4g ",
               i + 1, total_combos, c.lam, c.br1, c.br2, rho);
        sched_recs[i] = run_one_bmk(model_para);
        printf("cost=%.4f delay1=%.3f%% delay2=%.3f%%\n",
               sched_recs[i].cost_mean, sched_recs[i].delay1, sched_recs[i].delay2);
    }
#endif

    save_bmk_csv(csv_path, sched_recs);

    const BmkRecord* best = find_min_cost(sched_recs, cfg.delay_constraint, true);
    std::string opt_path = sched_folder + "/optimal_parameter.csv";
    print_and_save_optimal(best, cfg.delay_constraint, opt_path,
                           "rho=" + fmt_param(rho) + " scheduling");

    return sched_recs;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Bracket the whole main() body as total_search (L3 denominator).
    // ProfScope destructor runs at main() scope exit, before atexit handlers fire.
    PROF_L3(TOTAL_SEARCH);

    // CLI usage:
    //   ./sim_benchmark [config] [benchmark] [safety] [epsilon] [rho]
    //   argv[1] = config path         (optional, default: benchmark_search_config.txt)
    //   argv[2] = benchmark name      (optional, overrides config)
    //   argv[3] = safety value        (optional, overrides safety_list → single value)
    //   argv[4] = epsilon value       (optional, overrides epsilon_list → single value)
    //   argv[5] = rho value           (optional, overrides rho_list → single value)
    // argv[3]-[5] are used by server array jobs to parallelize over safety/epsilon/rho.
    std::string cfg_path = "benchmark_search_config.txt";
    if (argc >= 2) cfg_path = argv[1];

    printf("=== RS-Network Benchmark Search (%d paths per combo) ===\n\n", N_PATHS);
    printf("Reading config from: %s\n\n", cfg_path.c_str());

    BmkConfig cfg = load_config(cfg_path);

    if (argc >= 3) cfg.benchmark_name  = argv[2];
    if (argc >= 4) cfg.safety_list     = {std::stod(argv[3])};
    if (argc >= 5) cfg.epsilon_list    = {std::stod(argv[4])};
    if (argc >= 6) cfg.rho_list        = {std::stod(argv[5])};

    if (cfg.benchmark_name.empty()) {
        fprintf(stderr, "ERROR: 'benchmark' not set in config file.\n");
        return 1;
    }

    // Validate use_shared_parameter ∈ {0, 1, 2, 3}.
    if (cfg.use_shared_parameter < 0 || cfg.use_shared_parameter > 3) {
        fprintf(stderr,
            "ERROR: use_shared_parameter must be 0, 1, 2, or 3 (got %d).\n"
            "  0 = sequential two-round search (default)\n"
            "  1 = shared single-round (route 1 = route 0)\n"
            "  2 = shared single-round with marginal-cost-equality scaling\n"
            "  3 = pooled (=1 and =2 search lists combined into one search)\n",
            cfg.use_shared_parameter);
        return 1;
    }

    // Validate iterative_pricing_scheduling ∈ {0, 1}.
    if (cfg.iterative_pricing_scheduling != 0 && cfg.iterative_pricing_scheduling != 1) {
        fprintf(stderr,
            "ERROR: iterative_pricing_scheduling must be 0 or 1 (got %d).\n",
            cfg.iterative_pricing_scheduling);
        return 1;
    }
    // Validate iterative_pricing_scheduling_num ≥ 1.
    if (cfg.iterative_pricing_scheduling_num < 1) {
        fprintf(stderr,
            "ERROR: iterative_pricing_scheduling_num must be >= 1 (got %d).\n",
            cfg.iterative_pricing_scheduling_num);
        return 1;
    }

    // Fail-fast: any non-zero use_shared_parameter and pricing_magnitude_scale_by_mc_eq
    // are mutually exclusive. The latter scales Round 1's lists, but
    // use_shared_parameter collapses the search to a single round, so there is
    // no Round 1 to scale.
    if (cfg.use_shared_parameter && cfg.pricing_magnitude_scale_by_mc_eq) {
        fprintf(stderr,
            "ERROR: use_shared_parameter=%d and pricing_magnitude_scale_by_mc_eq=1 "
            "cannot both be enabled.\n"
            "  use_shared_parameter collapses pricing search to a single round, so\n"
            "  pricing_magnitude_scale_by_mc_eq (which scales Round 1) has no effect.\n"
            "Set one of them back to 0 in the config file.\n",
            cfg.use_shared_parameter);
        return 1;
    }

    BmkKind kind = bmk_kind_from_name(cfg.benchmark_name);

    // Print config summary
    printf("Benchmark: %s\n", cfg.benchmark_name.c_str());
    printf("  pricing:    %s\n", kind.pricing == BmkPricing::BCPlinear ? "BCPlinear" : "High_low");
    printf("  outsource:  %s\n", kind.outsource == BmkOutsource::SinglePushing ? "single_pushing" : "threshold_based");
    printf("  scheduling: %s\n", kind.sched == BmkScheduling::WMP ? "WMP" :
                                 kind.sched == BmkScheduling::LRS_QueueLen ? "LRS_QueueLen" : "LRS_DelayTime");
    printf("  has_sched_search: %s\n", kind.has_sched_search ? "yes" : "no");
    printf("  safety_list: ");
    for (double v : cfg.safety_list) printf("%.4g ", v);
    printf("\n");
    printf("  epsilon_list (searched for SinglePushing only): ");
    for (double v : cfg.epsilon_list) printf("%.4g ", v);
    printf("\n");
    printf("  rho_list: ");
    for (double v : cfg.rho_list) printf("%.4g ", v);
    printf("\n");

    if (kind.pricing == BmkPricing::BCPlinear) {
        printf("  beta11_list: ");
        for (double v : cfg.beta11_list) printf("%.4g ", v);
        printf("\n  beta12_list: ");
        for (double v : cfg.beta12_list) printf("%.4g ", v);
        printf("\n");
    } else {
        printf("  Q_threshold_list: ");
        for (double v : cfg.Q_threshold_list) printf("%.4g ", v);
        printf("\n  zeta_list: ");
        for (double v : cfg.zeta_list) printf("%.4g ", v);
        printf("\n  highlow_fix_low_zero: %d\n", cfg.highlow_fix_low_zero);
    }
    if (kind.has_sched_search) {
        printf("  lambda_tilde_list: ");
        for (double v : cfg.lambda_tilde_list) printf("%.4g ", v);
        printf("\n  beta_rate_1_list: ");
        for (double v : cfg.beta_rate_1_list) printf("%.4g ", v);
        if (cfg.beta_rate_symmetric) {
            printf("\n  beta_rate_2_list: [symmetric — equals beta_rate_1]");
        } else {
            printf("\n  beta_rate_2_list: ");
            for (double v : cfg.beta_rate_2_list) printf("%.4g ", v);
        }
        printf("\n  beta_rate_symmetric: %d\n", cfg.beta_rate_symmetric);
        printf("  sched_init (pricing phase): lambda=%.4g  beta_rate_1=%.4g  beta_rate_2=%.4g\n",
               cfg.sched_init_lambda_tilde,
               cfg.sched_init_beta_rate_1,
               cfg.sched_init_beta_rate_2);
    }
    printf("  delay_constraint: %.2f%%\n", cfg.delay_constraint);
    if (cfg.pricing_magnitude_scale_by_mc_eq) {
        double s = OUTSOURCE_COST_RATE[1] / OUTSOURCE_COST_RATE[0];
        printf("  pricing_magnitude_scale_by_mc_eq: 1  (round 1 lists scaled by %.4g = OC[1]/OC[0])\n", s);
    } else {
        printf("  pricing_magnitude_scale_by_mc_eq: 0  (both rounds use original lists)\n");
    }
    if (cfg.use_shared_parameter == 1) {
        printf("  use_shared_parameter: 1  (shared single-round; route 1 = route 0; folder: shared_common)\n");
    } else if (cfg.use_shared_parameter == 2) {
        double s = OUTSOURCE_COST_RATE[1] / OUTSOURCE_COST_RATE[0];
        printf("  use_shared_parameter: 2  (shared single-round; route 1 = route 0 × %.4g; "
               "z_thresh unchanged in HighLow; folder: shared_mc_eq)\n", s);
    } else if (cfg.use_shared_parameter == 3) {
        double s = OUTSOURCE_COST_RATE[1] / OUTSOURCE_COST_RATE[0];
        printf("  use_shared_parameter: 3  (pooled: route 1 ∈ {route 0, route 0 × %.4g}; "
               "z_thresh unchanged in HighLow; folder: shared_pooled)\n", s);
    } else {
        printf("  use_shared_parameter: 0  (sequential two-round search)\n");
    }

    // Iterative mode is only meaningful for C/D (has_sched_search).
    // A/B + iterative=1 silently falls back to non-iterative with a warning here.
    const bool effective_iterative = cfg.iterative_pricing_scheduling && kind.has_sched_search;
    if (cfg.iterative_pricing_scheduling && !kind.has_sched_search) {
        printf("  NOTE: iterative_pricing_scheduling=1 but benchmark %s has no scheduling search\n"
               "        (only C1/C2/D1/D2 use WMP scheduling). Falling back to non-iterative.\n",
               cfg.benchmark_name.c_str());
    } else if (effective_iterative) {
        printf("  iterative_pricing_scheduling: 1  (max %d iters; folder: iteration1, .., iteration_last)\n",
               cfg.iterative_pricing_scheduling_num);
    } else {
        printf("  iterative_pricing_scheduling: 0  (single pricing→scheduling pass)\n");
    }
    printf("  output_dir: %s\n\n", cfg.output_dir.c_str());

    // Resolve OpenMP thread count (capped at SLURM allocation).
    // Returns 1 when USE_OMP is not defined or cfg.parallel_threads <= 1.
    const int n_threads = resolve_omp_threads(cfg.parallel_threads);
#ifdef USE_OMP
    printf("OpenMP: parallel_threads=%d (config=%d, SLURM_CPUS_PER_TASK=%s) → using %d threads\n\n",
           n_threads, cfg.parallel_threads,
           std::getenv("SLURM_CPUS_PER_TASK") ? std::getenv("SLURM_CPUS_PER_TASK") : "unset",
           n_threads);
#else
    if (cfg.parallel_threads > 1) {
        printf("NOTE: parallel_threads=%d in config but binary built without USE_OMP — running serial\n\n",
               cfg.parallel_threads);
    }
#endif

    // Create output directory
    mkdir_p(cfg.output_dir);

    // Set policy modes (fixed for this benchmark)
    set_policy_modes(kind);

    // Folder for this benchmark under output_dir
    std::string bmk_folder = cfg.output_dir + "/benchmark" + cfg.benchmark_name;
    mkdir_p(bmk_folder);

    // Epsilon is only searched for SinglePushing benchmarks (A1-A4, C1-C2).
    // ThresholdBased benchmarks (B1-B4, D1-D2) use a fixed epsilon=0.25.
    bool has_epsilon_search = (kind.outsource == BmkOutsource::SinglePushing);
    std::vector<double> effective_epsilon_list = has_epsilon_search
        ? cfg.epsilon_list
        : std::vector<double>{0.25};

    clock_t t_wall_start = clock();

    // Collect all results for global optimal across safety × epsilon × rho
    std::vector<BmkRecord> global_records;

    RC_INWARD_REFLECTION    = cfg.inward_reflection;
    RC_WMP_LIMITING         = cfg.wmp_limiting;
    RC_WMP_EXTRA_RELATIVE   = cfg.wmp_extra_relative;
    RC_EPSILON_RELATIVE     = cfg.epsilon_relative;
    RC_PRINT_DETAILED_INFO  = cfg.print_detailed_info;

    for (double safety : cfg.safety_list) {
        RC_SAFETY_STOCK = safety;

    for (double epsilon : effective_epsilon_list) {
        RC_EPSILON_BAND = epsilon;

    for (double rho : cfg.rho_list) {
        RC_OUTSOURCE_THRESHOLD = rho;

        // For WMP benchmarks (C/D), prefix folder with abs_/rel_ to distinguish
        // wmp_extra_relative variants. A/B benchmarks (LRS scheduling) get "".
        std::string wmp_pfx = wmp_prefix(kind, cfg.wmp_limiting, cfg.wmp_extra_relative);
        std::string se_rho_folder = bmk_folder
            + "/" + wmp_pfx + "rho" + fmt_param(rho)
            + "_safety" + fmt_param(safety)
            + (has_epsilon_search ? "_epsilon" + fmt_param(epsilon) : "");
        mkdir_p(se_rho_folder);

        printf("============================================================\n");
        if (has_epsilon_search)
            printf("[rho=%.4g  safety=%.4g  epsilon=%.4g] benchmark %s\n",
                   rho, safety, epsilon, cfg.benchmark_name.c_str());
        else
            printf("[rho=%.4g  safety=%.4g] benchmark %s\n",
                   rho, safety, cfg.benchmark_name.c_str());
        printf("============================================================\n");

        if (effective_iterative) {
            // ───────────── Iterative pricing-scheduling loop (C/D only) ─────────────
            // iteration 1: sched_init from cfg.sched_init_*
            // iteration k≥2: sched_init from iteration k-1's scheduling optimum
            // Terminates on (i) convergence or (ii) max iterations.
            // The terminating iteration's folder is renamed to {se_rho_folder}/iteration_last.
            const int n_max = cfg.iterative_pricing_scheduling_num;
            const double SCHED_CONVERGE_TOL = 1e-9;

            double prev_lam = 0.0, prev_bt1 = 0.0, prev_bt2 = 0.0;
            bool   have_prev = false;

            // Track the sched_init used for each iteration (in beta_tilde units, no
            // round-trip through beta_rate). iteration 1 derives from cfg.sched_init_*.
            double cur_init_lam = cfg.sched_init_lambda_tilde;
            double cur_init_bt1 = cfg.sched_init_beta_rate_1 * BETA_CONST * rho;
            double cur_init_bt2 = cfg.sched_init_beta_rate_2 * BETA_CONST * rho;

            std::vector<BmkRecord> last_sched_recs;  // accumulated to global only from final iter

            for (int i_iter = 1; i_iter <= n_max; ++i_iter) {
                std::string iter_folder = se_rho_folder + "/iteration" + std::to_string(i_iter);
                mkdir_p(iter_folder);

                // Apply this iteration's sched init
                RC_LAMBDA_TILDE = cur_init_lam;
                RC_BETA_TILDE_1 = cur_init_bt1;
                RC_BETA_TILDE_2 = cur_init_bt2;

                printf("\n--- [Iteration %d / %d]  sched_init: lambda_tilde=%.4g  beta_tilde_1=%.4g  beta_tilde_2=%.4g ---\n",
                       i_iter, n_max, cur_init_lam, cur_init_bt1, cur_init_bt2);

                // Pricing search → iter_folder/pricing/...
                std::vector<BmkRecord> pricing_recs_this_iter;
                std::vector<double> pricing_flat_opt;
                if (kind.pricing == BmkPricing::BCPlinear) {
                    pricing_flat_opt = bcplinear_pricing_search(
                        kind, rho, cfg, iter_folder, pricing_recs_this_iter, n_threads);
                } else {
                    pricing_flat_opt = highlow_pricing_search(
                        kind, rho, cfg, iter_folder, pricing_recs_this_iter, n_threads);
                }

                // Apply best pricing, then run scheduling search → iter_folder/scheduling/...
                if (kind.pricing == BmkPricing::BCPlinear) {
                    double flat[4];
                    for (int k = 0; k < 4; k++) flat[k] = pricing_flat_opt[k];
                    set_bcp_pricing_para(flat);
                } else {
                    double flat[6];
                    for (int k = 0; k < 6; k++) flat[k] = pricing_flat_opt[k];
                    set_hl_pricing_para(flat);
                }
                std::vector<BmkRecord> sched_recs = wmp_scheduling_search(
                    kind, rho, pricing_flat_opt, cfg, iter_folder, n_threads);

                // Extract this iteration's scheduling optimum
                const BmkRecord* best_sched = find_min_cost(sched_recs, cfg.delay_constraint, true);
                bool feasible = (best_sched != nullptr);
                double cur_lam = 0.0, cur_bt1 = 0.0, cur_bt2 = 0.0;
                if (feasible) {
                    int n_pr = (int)pricing_flat_opt.size();   // 4 or 6
                    cur_lam = best_sched->params[n_pr + 0];
                    cur_bt1 = best_sched->params[n_pr + 1];
                    cur_bt2 = best_sched->params[n_pr + 2];
                }

                // Termination conditions
                bool converged = feasible && have_prev
                              && std::fabs(cur_lam - prev_lam) < SCHED_CONVERGE_TOL
                              && std::fabs(cur_bt1 - prev_bt1) < SCHED_CONVERGE_TOL
                              && std::fabs(cur_bt2 - prev_bt2) < SCHED_CONVERGE_TOL;
                bool reached_max = (i_iter == n_max);
                bool no_feasible = !feasible;
                bool terminated  = converged || reached_max || no_feasible;

                if (terminated) {
                    if (converged) {
                        printf("[Iteration %d] CONVERGED: scheduling optimum matches iteration %d.\n",
                               i_iter, i_iter - 1);
                    } else if (no_feasible) {
                        printf("[Iteration %d] No feasible scheduling row — terminating iteration loop.\n", i_iter);
                    } else {
                        printf("[Iteration %d] Reached max iterations (%d) without convergence.\n", i_iter, n_max);
                    }
                    // Rename iter_folder → iteration_last (remove existing iteration_last if any).
                    std::string last_path = se_rho_folder + "/iteration_last";
                    std::string cmd_rm = "rm -rf '" + last_path + "'";
                    int rc_rm = system(cmd_rm.c_str()); (void)rc_rm;
                    if (std::rename(iter_folder.c_str(), last_path.c_str()) != 0) {
                        fprintf(stderr, "WARNING: failed to rename '%s' to '%s'\n",
                                iter_folder.c_str(), last_path.c_str());
                    }
                    last_sched_recs = std::move(sched_recs);
                    break;
                }

                // Otherwise continue: use this iteration's optimum as next sched init
                prev_lam = cur_lam;  prev_bt1 = cur_bt1;  prev_bt2 = cur_bt2;
                have_prev = true;
                cur_init_lam = cur_lam;
                cur_init_bt1 = cur_bt1;
                cur_init_bt2 = cur_bt2;
            }

            // Accumulate only the iteration_last sched records to global.
            for (auto& r : last_sched_recs) global_records.push_back(r);
        } else {
            // ───────────── Non-iterative path (existing behavior) ─────────────
            // Set WMP scheduling init for pricing phase (C/D only).
            // beta_tilde_i = sched_init_beta_rate_i * BETA_CONST * rho.
            // For A/B benchmarks (no sched search) these values are unused, but set them anyway for safety.
            RC_LAMBDA_TILDE = cfg.sched_init_lambda_tilde;
            RC_BETA_TILDE_1 = cfg.sched_init_beta_rate_1 * BETA_CONST * rho;
            RC_BETA_TILDE_2 = cfg.sched_init_beta_rate_2 * BETA_CONST * rho;

            // Pricing search (2 rounds, or shared single-round depending on use_shared_parameter)
            std::vector<BmkRecord> pricing_recs_this_combo;
            std::vector<double> pricing_flat_opt;

            if (kind.pricing == BmkPricing::BCPlinear) {
                pricing_flat_opt = bcplinear_pricing_search(
                    kind, rho, cfg, se_rho_folder, pricing_recs_this_combo, n_threads);
            } else {
                pricing_flat_opt = highlow_pricing_search(
                    kind, rho, cfg, se_rho_folder, pricing_recs_this_combo, n_threads);
            }

            // Scheduling search (Group C/D only)
            if (kind.has_sched_search) {
                if (kind.pricing == BmkPricing::BCPlinear) {
                    double flat[4];
                    for (int k = 0; k < 4; k++) flat[k] = pricing_flat_opt[k];
                    set_bcp_pricing_para(flat);
                } else {
                    double flat[6];
                    for (int k = 0; k < 6; k++) flat[k] = pricing_flat_opt[k];
                    set_hl_pricing_para(flat);
                }
                std::vector<BmkRecord> sched_recs = wmp_scheduling_search(
                    kind, rho, pricing_flat_opt, cfg, se_rho_folder, n_threads);
                for (auto& r : sched_recs) global_records.push_back(r);
            } else {
                // Group A/B: accumulate pricing records for global optimal
                for (auto& r : pricing_recs_this_combo) global_records.push_back(r);
            }
        }
    } // rho
    } // epsilon
    } // safety

    // -----------------------------------------------------------------------
    // Global optimal across all rho
    // (Prefix abs_/rel_ on WMP benchmarks so abs/rel runs don't overwrite each other.)
    // -----------------------------------------------------------------------
    // ---- Top-level optimal_parameter.csv writing is disabled (uncomment to re-enable). ----
    // std::string global_opt_path = bmk_folder + "/"
    //     + wmp_prefix(kind, cfg.wmp_limiting, cfg.wmp_extra_relative)
    //     + "optimal_parameter.csv";

    printf("\n======================================================\n");
    printf("Global optimal across all safety × epsilon × rho (benchmark %s):\n",
           cfg.benchmark_name.c_str());
    printf("======================================================\n");

    // For Group A: apply delay constraint to pricing records.
    // For Group C: delay constraint already enforced during scheduling search;
    //   but we still filter here for consistency.
    const BmkRecord* global_best = find_min_cost(global_records, cfg.delay_constraint, true);

    // Replaced print_and_save_optimal(...) with stdout-only inline print to avoid
    // writing the top-level CSV. The original call (commented below) writes both.
    // print_and_save_optimal(global_best, cfg.delay_constraint, global_opt_path, "global");
    printf("\n  [optimal global]");
    if (global_best == nullptr) {
        printf(" No feasible row (delay < %.2f%%) found.\n", cfg.delay_constraint);
    } else {
        printf(" min-cost feasible row:\n");
        printf("    params:");
        for (double p : global_best->params) printf(" %.4g", p);
        printf("\n    cost=%.4f  std=%.4f  delay1=%.3f%%  delay2=%.3f%%\n",
               global_best->cost_mean, global_best->cost_std,
               global_best->delay1, global_best->delay2);
    }

    clock_t t_wall_end = clock();
    double total_elapsed = (double)(t_wall_end - t_wall_start) / CLOCKS_PER_SEC;
    printf("\n=== Benchmark search complete ===\n");
    printf("Total wall-clock time: %.2f seconds\n", total_elapsed);
    return 0;
}
