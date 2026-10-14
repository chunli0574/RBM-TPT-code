/*
 * benchmark_search_main.cpp
 *
 * Parameter search for 12 benchmark policies (A1–A4, B1–B4, C1–C2, D1–D2).
 * Extended Criss-Cross — supports CC3 (dim_K=3) and CC10 (dim_K=10) via DIM_K_SELECT.
 *
 * Benchmark taxonomy:
 *   Pricing:      BCPlinear (A1,A2,B1,B2,C1,D1)  |  High_low (A3,A4,B3,B4,C2,D2)
 *   Outsourcing:  single_pushing (A1-A4, C1-C2)   |  threshold_based (B1-B4, D1-D2)
 *   Scheduling:   LRS_DelayTime (A1,A3,B1,B3)     |  LRS_QueueLen (A2,A4,B2,B4)
 *                 WMPobj2 (C1,C2,D1,D2)
 *   Group:        A/B = no scheduling search; C/D = pricing then scheduling search
 *
 * Pricing search: FOUR rounds (CC3 has N_ROUTES=4):
 *   Round i_round (0..3): sweep (beta11, beta12) or (Q_threshold, zeta1, zeta2) for row i_round.
 *   Best row = min-cost (NO delay constraint on pricing rounds).
 *
 *   BCPlinear pricing_para CC3 (4×6):
 *     i_round < DIM_K (0,1,2): row[i_round] = all beta12 except position i_round → beta11
 *     i_round == DIM_K  (3):   row[3] = [beta12,beta12,beta12, beta11,beta11,beta11]
 *
 *   High_low pricing_para CC3 (4×3):
 *     row[i_round] = [int(Q_loop * QUEUE_LEN_BOUND[i_round] * rho), zeta1, zeta2]
 *
 * Scheduling search (Group C/D only, after pricing):
 *   Sweep (lambda_tilde, beta_rate_v, beta_rate_h).
 *   RC_BETA_TILDE_V = beta_rate_v * BETA_CONST_V * rho
 *   RC_BETA_TILDE_H = beta_rate_h * BETA_CONST_H * rho
 *   Best = min-cost row satisfying ALL 4 delay_prob_mean < delay_constraint.
 *
 * CC3 outsourcing_para:
 *   SinglePushing:   [rho]               (1 element — epsilon and rho separate)
 *   ThresholdBased:  [rho, rho, rho, rho] (4 elements, one per route)
 *
 * CSV output: CC3_final_results.csv
 *   param_1,..,param_N, cost, std, delay1, delay2, delay3, delay4,
 *   hardstop1, hardstop2, hardstop3, hardstop4
 *
 * Global optimal: {bmk_folder}/optimal_parameter.csv
 *   Min-cost row satisfying all 4 delay constraints.
 *
 * Configuration file: benchmark_search_config.txt
 *
 * TO BUILD:  make sim_benchmark
 * TO RUN:    make run_benchmark
 */

#include "params.h"
#include "run_config.h"
#include "simulation.h"
#include "nn_model.h"   // included so link works; not called for benchmarks
#include "benchmark_kinds.h"

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
#include <array>
#include <limits>
#include <sys/stat.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef ENABLE_PROFILING
#include <chrono>
#include <cstdint>
#include <cstdlib>   // std::atexit
// ----- Cumulative counters for benchmark search outer loops.
//       All timings aggregate across the whole program run and are dumped at
//       exit (after the serial run_one_path profiling block). -----
namespace {
    uint64_t g_bmk_prof_sim      = 0;   // time inside run_one_bmk (i.e., run_simulation per combo)
    uint64_t g_bmk_prof_save_csv = 0;   // save_bmk_csv calls
    uint64_t g_bmk_prof_findmin  = 0;   // find_min_cost calls
    uint64_t g_bmk_prof_io_resume = 0;  // load_bmk_csv (resume checks)
    uint64_t g_bmk_prof_other    = 0;   // setup, model_para build, print, etc.
    uint64_t g_bmk_prof_total    = 0;   // full benchmark search loop

    void bmk_prof_dump_atexit() {
        if (g_bmk_prof_total == 0) return;
        auto pct_tot = [&](uint64_t v){ return 100.0 * v / g_bmk_prof_total; };
        std::printf("\n========== PROFILING (benchmark search outer) ==========\n");
        std::printf("  %-22s %18s  %8s\n", "section", "ns", "%total");
        auto pp = [&](const char* name, uint64_t v){
            std::printf("  %-22s %18llu  %7.2f%%\n",
                        name, (unsigned long long)v, pct_tot(v));
        };
        pp("run_one_bmk (sim)",    g_bmk_prof_sim);
        pp("save_bmk_csv",         g_bmk_prof_save_csv);
        pp("find_min_cost",        g_bmk_prof_findmin);
        pp("load_bmk_csv (resume)",g_bmk_prof_io_resume);
        pp("other (setup/loop)",   g_bmk_prof_other);
        std::printf("  %-22s %18llu  %7.2f%%\n", "total_search",
                    (unsigned long long)g_bmk_prof_total, 100.00);
        std::printf("========================================================\n");
    }
    struct BmkProfInit { BmkProfInit() { std::atexit(bmk_prof_dump_atexit); } };
    BmkProfInit g_bmk_prof_init;
}

#define BMK_PROF_TIC(name) \
    auto name##_t0 = std::chrono::steady_clock::now()
#define BMK_PROF_TOC(name, counter) \
    counter += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>( \
        std::chrono::steady_clock::now() - name##_t0).count()
#else
#define BMK_PROF_TIC(name)        do {} while(0)
#define BMK_PROF_TOC(name, counter) do {} while(0)
#endif

// ---------------------------------------------------------------------------
// Benchmark kind taxonomy + dispatch helpers — moved to benchmark_kinds.{h,cpp}
// so sim_one_combo can share the exact same definitions without duplication.
// (See "#include \"benchmark_kinds.h\"" near the top of this file.)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Config file structure and parser
// ---------------------------------------------------------------------------

struct BmkConfig {
    std::string benchmark_name;
    // Outer parameter search
    std::vector<double> safety_list  = {0.0};
    std::vector<double> epsilon_list = {0.25};  // A/C single_pushing: search epsilon
    std::vector<double> rho_list;
    // BCPlinear pricing search
    std::vector<double> beta11_list;
    std::vector<double> beta12_list;
    // High_low pricing search
    std::vector<double> Q_threshold_list;
    std::vector<double> zeta_list;
    // WMP scheduling search (Group C/D).
    // Lists are auto-selected by wmp_extra_relative:
    //   wmp_extra_relative=0 → *_abs lists
    //   wmp_extra_relative=1 → *_rel lists
    // After parsing, resolve_wmp_lists() copies the chosen variant into the
    // unsuffixed effective fields below. C/D benchmarks error out if the
    // resulting effective list is empty.
    std::vector<double> lambda_tilde_list;     // effective list (resolved)
    std::vector<double> beta_rate_1_list;      // effective list (resolved) — vertical → RC_BETA_TILDE_V
    std::vector<double> beta_rate_2_list;      // effective list (resolved) — horizontal → RC_BETA_TILDE_H
    std::vector<double> lambda_tilde_list_abs; // parsed config source (empty = unset)
    std::vector<double> beta_rate_1_list_abs;
    std::vector<double> beta_rate_2_list_abs;
    std::vector<double> lambda_tilde_list_rel;
    std::vector<double> beta_rate_1_list_rel;
    std::vector<double> beta_rate_2_list_rel;
    // beta_rate_symmetric: 0 = use separate beta_rate_1 and beta_rate_2 lists (default)
    //                       1 = beta_rate_2 always equals beta_rate_1 (beta_rate_2_list ignored)
    //                       2 = beta_rate_2 = 1-(1-beta_rate_1)*B_V/B_H (beta_rate_2_list ignored)
    int    beta_rate_symmetric = 0;
    // Scheduling init: fixed WMP params used during pricing search phase (C/D benchmarks).
    // Converted at runtime per rho:
    //   RC_BETA_TILDE_V = sched_init_beta_rate_1 * BETA_CONST_V * rho
    //   RC_BETA_TILDE_H = br2_eff                * BETA_CONST_H * rho
    // where br2_eff honors beta_rate_symmetric (same formula as the scheduling
    // grid expansion):
    //   0 → cfg.sched_init_beta_rate_2
    //   1 → cfg.sched_init_beta_rate_1
    //   2 → 1 - (1 - cfg.sched_init_beta_rate_1) * B_V / B_H
    // When beta_rate_symmetric != 0 the config value of sched_init_beta_rate_2
    // is ignored; see effective_sched_init_br2().
    // Effective (resolved) values; resolve_sched_init() copies the chosen
    // variant's _abs/_rel scalar into these per wmp_extra_relative. If the
    // chosen variant's scalar is NaN (key not provided), the effective field
    // keeps its default below.
    double sched_init_lambda_tilde = 4.0;
    double sched_init_beta_rate_1  = 0.75;
    double sched_init_beta_rate_2  = 0.75;
    // Per-variant config source (NaN = "not provided in config").
    double sched_init_lambda_tilde_abs = std::numeric_limits<double>::quiet_NaN();
    double sched_init_beta_rate_1_abs  = std::numeric_limits<double>::quiet_NaN();
    double sched_init_beta_rate_2_abs  = std::numeric_limits<double>::quiet_NaN();
    double sched_init_lambda_tilde_rel = std::numeric_limits<double>::quiet_NaN();
    double sched_init_beta_rate_1_rel  = std::numeric_limits<double>::quiet_NaN();
    double sched_init_beta_rate_2_rel  = std::numeric_limits<double>::quiet_NaN();
    // epsilon_relative: 0=absolute band (±eps×scaling), 1=relative band (FACE_CONST×(rho±eps))
    int    epsilon_relative  = 0;
    // wmp_relative: 0=absolute w=mu_arr (old), 1=relative w=mu_arr/b_arr (new)
    int    wmp_relative      = 0;
    // wmp_no_queue: 0=v1 full pressure (Z+λ*(Z-β)), 1=v2 no-queue pressure (λ*(Z-β) only)
    int    wmp_no_queue      = 0;
    // wmp_extra_relative: 0=plain v1; 1=normalize v1 pressure term by (B_V*rho) / (B_H*rho)
    int    wmp_extra_relative = 0;
    // Constraints / output
    double delay_constraint  = 1.0;
    int    inward_reflection = 0;    // 0 = hold_last (outward band), 1 = inward band
    std::string output_dir   = "benchmark_results";
    // Parallel threads for benchmark search inner loops (USE_OMP build only).
    // 0 or 1 = serial; N > 1 uses min(N, SLURM_CPUS_PER_TASK) threads.
    int    parallel_threads  = 1;
    // print_detailed_info:
    //   1           = full CSV: hardstop1..N, pricing_cost, outsourcing_cost
    //   0 (default) = lean CSV: those 6 columns are omitted AND the simulation
    //                 skips accumulating them.
    int    print_detailed_info = 0;
    // pricing_magnitude_scale_by_mc_eq:
    //   0 (default) = every round uses the same pricing search list.
    //   1           = the LAST round (the horizontal route, i_round = N_ROUTES-1)
    //                 uses a scaled list so the pricing magnitude matches the
    //                 horizontal route's larger outsourcing cost.  Scale factors
    //                 are derived from the constexpr arrays in params.h /
    //                 run_config.h, i.e. they auto-adapt to CC3 / CC10:
    //                   BCP beta11/12  *= (mc_H / mc_V) / (qb_H / qb_V)
    //                   HL  zeta_list  *=  mc_H / mc_V
    //                   Q_threshold_list never scales (z_thresh already uses qb[r]).
    //                 CC3:  BCP scale = 1.5/1.5 = 1.0 (no-op for BCP),
    //                       HL  scale = 1.5
    //                 CC10: BCP scale = 5/4    = 1.25,
    //                       HL  scale = 5.0
    // Resume is NOT folder-name-encoded — if you flip the flag between runs that
    // share the same rho_folder, delete `pricing/` first or the stale CSVs will
    // be picked up as already-completed rounds.
    int    pricing_magnitude_scale_by_mc_eq = 0;
    // use_shared_parameter: pricing search variant selector.
    //   0 (default) = old per-route N_ROUTES-round coordinate-descent search.
    //   1           = single-round shared-parameter search WITHOUT mc-ratio
    //                 scaling on the last route. All routes share the same
    //                 (beta11, beta12) for BCP, or (Q_threshold, zeta1, zeta2)
    //                 for HL. Last-route adjustments:
    //                   BCP last route: (beta11, beta12) *=  1 / (qb_H/qb_V)
    //                   HL  last route: (zeta1, zeta2)   *= 1.0  (i.e. unchanged)
    //                   Q_threshold same as other routes (z_thresh uses qb[r]).
    //                 Output goes under {rho_folder}/pricing/shared_common/.
    //   2           = single-round shared-parameter search WITH mc-ratio
    //                 scaling on the last route (otherwise same shape as =1):
    //                   BCP last route: (beta11, beta12) *= (mc_H/mc_V) / (qb_H/qb_V)
    //                   HL  last route: (zeta1, zeta2)   *=  mc_H/mc_V
    //                   Q_threshold same as other routes.
    //                 Output goes under {rho_folder}/pricing/shared_mc_eq/.
    //
    // In modes 1 and 2 the flag pricing_magnitude_scale_by_mc_eq is IGNORED
    // (last-route scaling is inherent to the variant).
    int    use_shared_parameter = 0;
    // iterative_pricing_scheduling: only affects C1/C2/D1/D2 (WMP scheduling).
    //   0 (default) = single pricing→scheduling pass per (safety, eps, rho).
    //   1           = iterate pricing↔scheduling within each (safety, eps, rho)
    //                 combo. Iteration 1 uses cfg.sched_init_* as the WMP init
    //                 (same as today). Iteration k > 1 uses iteration (k-1)'s
    //                 optimal scheduling (lambda, beta_rate_1, beta_rate_2) as
    //                 the init. Terminates when:
    //                   (a) the current iteration's scheduling search finds no
    //                       feasible solution, OR
    //                   (b) the current iteration's optimal scheduling equals
    //                       the previous iteration's optimal scheduling, OR
    //                   (c) reached iterative_pricing_scheduling_num iterations.
    //                 Per-iteration output lives under
    //                 {rho_folder}/iteration{N}/{pricing,scheduling}/; the final
    //                 iteration's folder is renamed to iteration_last/.
    // A/B benchmarks (no WMP scheduling stage) silently ignore this flag.
    int    iterative_pricing_scheduling     = 0;
    int    iterative_pricing_scheduling_num = 5;
};

// Derive the effective sched_init_beta_rate_2 used to seed RC_BETA_TILDE_H
// during the pricing-search phase. Honors beta_rate_symmetric with the same
// formula used when the scheduling search grid expands beta_rate_2 (see
// wmp_scheduling_search):
//   0 → cfg.sched_init_beta_rate_2 (config value, backward-compatible)
//   1 → cfg.sched_init_beta_rate_1
//   2 → 1 - (1 - cfg.sched_init_beta_rate_1) * B_V / B_H
// When beta_rate_symmetric != 0 the config's sched_init_beta_rate_2 is ignored.
static inline double effective_sched_init_br2(const BmkConfig& cfg) {
    if (cfg.beta_rate_symmetric == 1) return cfg.sched_init_beta_rate_1;
    if (cfg.beta_rate_symmetric == 2)
        return 1.0 - (1.0 - cfg.sched_init_beta_rate_1) * B_V / B_H;
    return cfg.sched_init_beta_rate_2;
}

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

// ---------------------------------------------------------------------------
// Resolve WMP scheduling lists based on wmp_extra_relative.
// Copies the chosen variant's *_abs / *_rel list into the effective list.
// If the chosen variant's list is empty (not provided in config), the
// effective list stays empty — caught by C/D validation in main().
// ---------------------------------------------------------------------------
static void resolve_wmp_lists(BmkConfig& cfg) {
    const bool is_rel = (cfg.wmp_extra_relative == 1);
    auto pick = [is_rel](std::vector<double>& effective,
                         const std::vector<double>& abs_v,
                         const std::vector<double>& rel_v) {
        const auto& src = is_rel ? rel_v : abs_v;
        if (!src.empty()) effective = src;
    };
    pick(cfg.lambda_tilde_list, cfg.lambda_tilde_list_abs, cfg.lambda_tilde_list_rel);
    pick(cfg.beta_rate_1_list,  cfg.beta_rate_1_list_abs,  cfg.beta_rate_1_list_rel);
    pick(cfg.beta_rate_2_list,  cfg.beta_rate_2_list_abs,  cfg.beta_rate_2_list_rel);
}

// ---------------------------------------------------------------------------
// Resolve sched_init scalars based on wmp_extra_relative.
// Copies the chosen variant's *_abs / *_rel value into the effective scalar.
// If the chosen variant's scalar is NaN (key not provided), the effective
// scalar keeps its hard-coded default (4.0 / 0.75 / -1.0).
// ---------------------------------------------------------------------------
static void resolve_sched_init(BmkConfig& cfg) {
    const bool is_rel = (cfg.wmp_extra_relative == 1);
    auto pick = [is_rel](double& effective, double abs_v, double rel_v) {
        const double src = is_rel ? rel_v : abs_v;
        if (!std::isnan(src)) effective = src;
    };
    pick(cfg.sched_init_lambda_tilde,
         cfg.sched_init_lambda_tilde_abs, cfg.sched_init_lambda_tilde_rel);
    pick(cfg.sched_init_beta_rate_1,
         cfg.sched_init_beta_rate_1_abs,  cfg.sched_init_beta_rate_1_rel);
    pick(cfg.sched_init_beta_rate_2,
         cfg.sched_init_beta_rate_2_abs,  cfg.sched_init_beta_rate_2_rel);
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
        else if (key == "lambda_tilde_list_abs")  cfg.lambda_tilde_list_abs  = parse_double_list(val);
        else if (key == "beta_rate_1_list_abs")   cfg.beta_rate_1_list_abs   = parse_double_list(val);
        else if (key == "beta_rate_2_list_abs")   cfg.beta_rate_2_list_abs   = parse_double_list(val);
        else if (key == "lambda_tilde_list_rel")  cfg.lambda_tilde_list_rel  = parse_double_list(val);
        else if (key == "beta_rate_1_list_rel")   cfg.beta_rate_1_list_rel   = parse_double_list(val);
        else if (key == "beta_rate_2_list_rel")   cfg.beta_rate_2_list_rel   = parse_double_list(val);
        else if (key == "beta_rate_symmetric")      cfg.beta_rate_symmetric      = std::stoi(val);
        else if (key == "sched_init_lambda_tilde_abs")  cfg.sched_init_lambda_tilde_abs  = std::stod(val);
        else if (key == "sched_init_beta_rate_1_abs")   cfg.sched_init_beta_rate_1_abs   = std::stod(val);
        else if (key == "sched_init_beta_rate_2_abs")   cfg.sched_init_beta_rate_2_abs   = std::stod(val);
        else if (key == "sched_init_lambda_tilde_rel")  cfg.sched_init_lambda_tilde_rel  = std::stod(val);
        else if (key == "sched_init_beta_rate_1_rel")   cfg.sched_init_beta_rate_1_rel   = std::stod(val);
        else if (key == "sched_init_beta_rate_2_rel")   cfg.sched_init_beta_rate_2_rel   = std::stod(val);
        else if (key == "wmp_relative")             cfg.wmp_relative             = std::stoi(val);
        else if (key == "wmp_no_queue")        cfg.wmp_no_queue        = std::stoi(val);
        else if (key == "wmp_extra_relative") cfg.wmp_extra_relative  = std::stoi(val);
        else if (key == "epsilon_relative")    cfg.epsilon_relative    = std::stoi(val);
        else if (key == "delay_constraint")    cfg.delay_constraint    = std::stod(val);
        else if (key == "output_dir")         cfg.output_dir         = val;
        else if (key == "inward_reflection")  cfg.inward_reflection  = std::stoi(val);
        else if (key == "parallel_threads")   cfg.parallel_threads   = std::stoi(val);
        else if (key == "print_detailed_info") cfg.print_detailed_info = std::stoi(val);
        else if (key == "pricing_magnitude_scale_by_mc_eq")
            cfg.pricing_magnitude_scale_by_mc_eq = std::stoi(val);
        else if (key == "use_shared_parameter")
            cfg.use_shared_parameter = std::stoi(val);
        else if (key == "iterative_pricing_scheduling")
            cfg.iterative_pricing_scheduling = std::stoi(val);
        else if (key == "iterative_pricing_scheduling_num")
            cfg.iterative_pricing_scheduling_num = std::stoi(val);
    }
    // Resolve _abs / _rel overrides into the effective fields per
    // wmp_extra_relative. Must happen AFTER full parse so wmp_extra_relative is
    // known.
    resolve_wmp_lists(cfg);
    resolve_sched_init(cfg);
    return cfg;
}

// ---------------------------------------------------------------------------
// mkdir_p
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
// resolve_omp_threads
//   Cap the config-requested thread count at SLURM_CPUS_PER_TASK (if set) so
//   we never oversubscribe the cores the scheduler allocated. Returns 1 when
//   OMP is disabled or cfg <= 1.
//   Also calls omp_set_num_threads + omp_set_dynamic(0) at program startup.
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
// BmkRecord: one simulation result row (variable-length params + 4 delays/hardstops)
// ---------------------------------------------------------------------------
struct BmkRecord {
    std::vector<double> params;      // all model parameters for this combo
    double cost_mean, cost_std;
    double pricing_cost_mean;
    double outsourcing_cost_mean;
    double delay[N_ROUTES];          // delay_prob_mean[0..N_ROUTES-1]
    double delay_std[N_ROUTES];      // delay_prob_std[0..N_ROUTES-1]
    double hardstop[N_ROUTES];       // outsource_frac_mean[0..N_ROUTES-1]
};

// ---------------------------------------------------------------------------
// Save a vector of BmkRecord to CSV (N_ROUTES-generic)
// ---------------------------------------------------------------------------
static void save_bmk_csv(const std::string& filepath,
                          const std::vector<BmkRecord>& recs) {
    BMK_PROF_TIC(csv);
    if (recs.empty()) { BMK_PROF_TOC(csv, g_bmk_prof_save_csv); return; }
    FILE* fp = fopen(filepath.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot write CSV: %s\n", filepath.c_str());
        return;
    }
    const bool detailed = (RC_PRINT_DETAILED_INFO != 0);
    int np = (int)recs[0].params.size();
    for (int i = 0; i < np; i++) fprintf(fp, "param_%d,", i + 1);
    // Header: cost, std, delay1..N, delay1_std..delayN_std are always written.
    // Trailing columns (hardstop*, pricing_cost, outsourcing_cost) only in
    // detailed mode.
    fprintf(fp, "cost,std");
    for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",delay%d", r + 1);
    for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",delay%d_std", r + 1);
    if (detailed) {
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",hardstop%d", r + 1);
        fprintf(fp, ",pricing_cost,outsourcing_cost");
    }
    fprintf(fp, "\n");
    for (const auto& rec : recs) {
        for (double p : rec.params) fprintf(fp, "%.6g,", p);
        fprintf(fp, "%.4f,%.4f", rec.cost_mean, rec.cost_std);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.4f", rec.delay[r]);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.4f", rec.delay_std[r]);
        if (detailed) {
            for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.6g", rec.hardstop[r]);
            fprintf(fp, ",%.4f,%.4f", rec.pricing_cost_mean, rec.outsourcing_cost_mean);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("  Saved: %s\n", filepath.c_str());
    BMK_PROF_TOC(csv, g_bmk_prof_save_csv);
}

// ---------------------------------------------------------------------------
// Find the min-cost record satisfying all 4 delay constraints
// (or without constraint when apply_delay_constraint==false).
// Returns nullptr if none.
// ---------------------------------------------------------------------------
static const BmkRecord* find_min_cost(const std::vector<BmkRecord>& recs,
                                       double delay_constraint,
                                       bool apply_delay_constraint) {
    BMK_PROF_TIC(fmc);
    const BmkRecord* best = nullptr;
    for (const auto& r : recs) {
        if (apply_delay_constraint) {
            bool feasible = true;
            for (int i = 0; i < N_ROUTES; ++i) {
                if (!(r.delay[i] < delay_constraint)) { feasible = false; break; }
            }
            if (!feasible) continue;
        }
        if (best == nullptr || r.cost_mean < best->cost_mean)
            best = &r;
    }
    BMK_PROF_TOC(fmc, g_bmk_prof_findmin);
    return best;
}

// ---------------------------------------------------------------------------
// Load BmkRecords from a previously saved CSV file (for resume support).
// Returns empty vector if file doesn't exist or can't be parsed.
// ---------------------------------------------------------------------------
static std::vector<BmkRecord> load_bmk_csv(const std::string& filepath) {
    BMK_PROF_TIC(ldr);
    std::vector<BmkRecord> recs;
    std::ifstream f(filepath);
    if (!f.is_open()) { BMK_PROF_TOC(ldr, g_bmk_prof_io_resume); return recs; }

    std::string header;
    if (!std::getline(f, header)) { BMK_PROF_TOC(ldr, g_bmk_prof_io_resume); return recs; }

    // Count number of param columns from header ("param_1,param_2,...,cost,...")
    int n_params = 0;
    {
        std::stringstream ss(header);
        std::string col;
        while (std::getline(ss, col, ',')) {
            size_t a = col.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) continue;
            col = col.substr(a);
            if (col.size() >= 6 && col.substr(0, 6) == "param_") n_params++;
            else break;
        }
    }
    if (n_params == 0) { BMK_PROF_TOC(ldr, g_bmk_prof_io_resume); return recs; }

    // Detect whether header has delay_std columns (new format) or not (old format)
    bool has_delay_std = (header.find("delay1_std") != std::string::npos);
    // Detect whether the detailed trailing columns are present in this CSV.
    // We key on "hardstop1" in the header since it is the first of the
    // detailed-only columns.
    bool file_has_detailed = (header.find("hardstop1") != std::string::npos);

    // Resume policy: if the on-disk CSV's detail level does not match the
    // current RC_PRINT_DETAILED_INFO, we refuse to resume from it — the
    // caller treats an empty vector as "no resume" and will re-run the round.
    const bool want_detailed = (RC_PRINT_DETAILED_INFO != 0);
    if (file_has_detailed != want_detailed) {
        fprintf(stderr,
            "  [resume] skipping %s: column set does not match current "
            "print_detailed_info=%d (file %s detailed columns).\n",
            filepath.c_str(),
            want_detailed ? 1 : 0,
            file_has_detailed ? "has" : "lacks");
        BMK_PROF_TOC(ldr, g_bmk_prof_io_resume);
        return recs;
    }

    // Each row: n_params, cost, std, N_ROUTES delays, [N_ROUTES delay_stds],
    //           [N_ROUTES hardstops + pricing_cost + outsourcing_cost]
    int expected_old = n_params + 2 + N_ROUTES
                       + (file_has_detailed ? (N_ROUTES + 2) : 0);
    int expected_new = expected_old + N_ROUTES;     // + delay_stds
    int expected = has_delay_std ? expected_new : expected_old;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<double> vals;
        while (std::getline(ss, tok, ',')) {
            size_t a = tok.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) continue;
            tok = tok.substr(a);
            if (tok.empty()) continue;
            vals.push_back(std::stod(tok));
        }
        if ((int)vals.size() < expected) continue;

        BmkRecord rec;
        int idx = 0;
        rec.params.resize(n_params);
        for (int i = 0; i < n_params; i++) rec.params[i] = vals[idx++];
        rec.cost_mean = vals[idx++];
        rec.cost_std  = vals[idx++];
        for (int r = 0; r < N_ROUTES; r++) rec.delay[r] = vals[idx++];
        if (has_delay_std) {
            for (int r = 0; r < N_ROUTES; r++) rec.delay_std[r] = vals[idx++];
        } else {
            for (int r = 0; r < N_ROUTES; r++) rec.delay_std[r] = 0.0;
        }
        if (file_has_detailed) {
            for (int r = 0; r < N_ROUTES; r++) rec.hardstop[r] = vals[idx++];
            rec.pricing_cost_mean     = vals[idx++];
            rec.outsourcing_cost_mean = vals[idx++];
        } else {
            for (int r = 0; r < N_ROUTES; r++) rec.hardstop[r] = 0.0;
            rec.pricing_cost_mean     = 0.0;
            rec.outsourcing_cost_mean = 0.0;
        }
        recs.push_back(rec);
    }
    BMK_PROF_TOC(ldr, g_bmk_prof_io_resume);
    return recs;
}

// ---------------------------------------------------------------------------
// Resume support: scan pricing round directories for completed rounds.
// Returns the index of the last completed round (-1 if none).
// If a completed round is found, pricing_opt is filled from the best record.
// ---------------------------------------------------------------------------
static int find_resume_round(const std::string& rho_folder,
                             double delay_constraint,
                             int flat_sz,
                             double pricing_opt[]) {
    int last_completed = -1;
    for (int i = 0; i < N_ROUTES; i++) {
        std::string csv_path = rho_folder + "/pricing/round" + std::to_string(i)
                               + "/" SYSTEM_NAME "_final_results.csv";
        auto recs = load_bmk_csv(csv_path);
        if (recs.empty()) break;

        const BmkRecord* best = find_min_cost(recs, delay_constraint, false);
        if (best == nullptr) break;

        for (int k = 0; k < flat_sz && k < (int)best->params.size(); k++)
            pricing_opt[k] = best->params[k];
        last_completed = i;
    }
    return last_completed;
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
        printf(" No feasible row (all delays < %.2f%%) found.\n", delay_constraint);
        return;
    }
    printf(" min-cost feasible row:\n");
    printf("    params:");
    for (double p : best->params) printf(" %.4g", p);
    printf("\n    cost=%.4f  std=%.4f  delay=[", best->cost_mean, best->cost_std);
    for (int r = 0; r < N_ROUTES; ++r)
        printf("%.3f%%%s", best->delay[r], r < N_ROUTES-1 ? ", " : "");
    printf("]\n");

    FILE* fp = fopen(opt_path.c_str(), "w");
    if (fp) {
        const bool detailed = (RC_PRINT_DETAILED_INFO != 0);
        int np = (int)best->params.size();
        // Header
        for (int i = 0; i < np; i++) fprintf(fp, "param_%d,", i + 1);
        fprintf(fp, "cost,std");
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",delay%d", r + 1);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",delay%d_std", r + 1);
        if (detailed) {
            for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",hardstop%d", r + 1);
            fprintf(fp, ",pricing_cost,outsourcing_cost");
        }
        fprintf(fp, "\n");
        // Single data row
        for (double p : best->params) fprintf(fp, "%.6g,", p);
        fprintf(fp, "%.4f,%.4f", best->cost_mean, best->cost_std);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.4f", best->delay[r]);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.4f", best->delay_std[r]);
        if (detailed) {
            for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.6g", best->hardstop[r]);
            fprintf(fp, ",%.4f,%.4f", best->pricing_cost_mean, best->outsourcing_cost_mean);
        }
        fprintf(fp, "\n");
        fclose(fp);
        printf("  Saved optimal to: %s\n", opt_path.c_str());
    }
}

// set_policy_modes / set_bcp_pricing_phi / set_hl_pricing_para — moved to
// benchmark_kinds.{h,cpp} (no behavioural change).

// ---------------------------------------------------------------------------
// outsourcing_para helper: returns param vector for the outsourcing mode.
//   SinglePushing:   [rho]               (epsilon set globally in RC_EPSILON_BAND)
//   ThresholdBased:  [rho, rho, rho, rho] (one per route)
// ---------------------------------------------------------------------------
static std::vector<double> out_para_vec(const BmkKind& kind, double rho) {
    if (kind.outsource == BmkOutsource::SinglePushing)
        return {rho};
    else
        return std::vector<double>(N_ROUTES, rho);  // [rho, rho, rho, rho]
}

// ---------------------------------------------------------------------------
// Run simulation and return a BmkRecord with the given param vector
// ---------------------------------------------------------------------------
static BmkRecord run_one_bmk(std::vector<double> params) {
    BMK_PROF_TIC(sim);
    SimResult res = run_simulation();
    BMK_PROF_TOC(sim, g_bmk_prof_sim);
    BmkRecord rec;
    rec.params                = std::move(params);
    rec.cost_mean             = res.cost_mean;
    rec.cost_std              = res.cost_std;
    rec.pricing_cost_mean     = res.pricing_cost_mean;
    rec.outsourcing_cost_mean = res.outsourcing_cost_mean;
    for (int i = 0; i < N_ROUTES; ++i) {
        rec.delay[i]     = res.delay_prob_mean[i];
        rec.delay_std[i] = res.delay_prob_std[i];
        rec.hardstop[i]  = 0.0;  // not tracked in C++ (Python raises error on hardstop)
    }
    return rec;
}

// ---------------------------------------------------------------------------
// BCPlinear pricing parameter search: N_ROUTES (=4) rounds
//
// Compact representation (see run_config.h RC_PRICING_PHI):
//   For each route r, two scalars (phi_r, phi_{-r}):
//     phi_r    = coefficient of route-r's "own" buffer(s)
//     phi_{-r} = coefficient of all other buffers
//   Equivalent to β11 / β12 in the paper's notation.
//
// Round i_round: sweep (beta11 = phi_{i_round}, beta12 = phi_{-i_round})
//                while holding other routes' (phi, phi_neg) at the previous
//                round's optimum (initially 0).
//
// Best = min-cost row (no delay constraint during pricing rounds).
// Returns the optimal flat (N_ROUTES × 2 = 2K) pricing params.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// B_others_vec[r] = Σ_{k != r} B_VEC[k]   (paper notation: β_{-r}).
// Computed once into a static array; used by the BCP shared-parameter pricing
// search to honor the paper's shared-parameter assumption (Eqs. D.62-D.65):
//   φ_{-k} β_{-k} = const across routes  →  φ_{-k} ∝ 1/B_others_vec[k]
// ---------------------------------------------------------------------------
static const std::array<double, N_ROUTES>& get_B_others_vec() {
    static const std::array<double, N_ROUTES> v = [](){
        std::array<double, N_ROUTES> arr{};
        double sum = 0.0;
        for (int r = 0; r < N_ROUTES; ++r) sum += B_VEC[r];
        for (int r = 0; r < N_ROUTES; ++r) arr[r] = sum - B_VEC[r];
        return arr;
    }();
    return v;
}

// ---------------------------------------------------------------------------
// Shared-parameter resume helper.
// Looks for {rho_folder}/pricing/{subfolder}/{SYSTEM_NAME}_final_results.csv.
// If it exists AND has the expected number of rows AND contains a best record,
// loads the best record's params into pricing_opt[] and returns true.
// Otherwise returns false (caller should re-run).
// ---------------------------------------------------------------------------
static bool try_resume_shared(const std::string& rho_folder,
                              const char* subfolder,
                              double delay_constraint,
                              int expected_n_rows,
                              int flat_sz,
                              double pricing_opt[])
{
    std::string csv_path = rho_folder + "/pricing/" + subfolder + "/"
                           SYSTEM_NAME "_final_results.csv";
    auto recs = load_bmk_csv(csv_path);
    if ((int)recs.size() != expected_n_rows) return false;
    const BmkRecord* best = find_min_cost(recs, delay_constraint, false);
    if (best == nullptr) return false;
    for (int k = 0; k < flat_sz; ++k) pricing_opt[k] = best->params[k];
    printf("  [Resume:%s] Loaded %d existing records, pricing_opt =",
           subfolder, (int)recs.size());
    for (int k = 0; k < flat_sz; ++k) printf(" %.4g", pricing_opt[k]);
    printf("\n");
    return true;
}

// ---------------------------------------------------------------------------
// BCPlinear shared-parameter search (use_shared_parameter == 1 or 2).
// Total combos = len(beta11_list) × len(beta12_list). For each combo (b11, b12)
// and each route r, the searched pricing coefficients are:
//   φ_r   (own)   = b11 * phi_own_scale[r]
//   φ_{-r} (other) = b12 * phi_other_scale[r]
// where the caller computes per-route scales according to the shared-parameter
// rule (paper Eqs. D.62-D.65):
//   variant 1: phi_own_scale[r]   = B_VEC[0] / B_VEC[r]
//              phi_other_scale[r] = B_VEC[0] / B_others_vec[r]
//   variant 2: above × (ν_r / ν_0)
// (caller in bcplinear_pricing_search builds these arrays before calling.)
//
// CSV written under {rho_folder}/pricing/{subfolder}/.
// (subfolder = "shared_common" for =1, "shared_mc_eq" for =2.)
// ---------------------------------------------------------------------------
static std::vector<double> bcplinear_shared_param_search(
    const BmkKind& kind,
    double rho,
    const BmkConfig& cfg,
    const std::string& rho_folder,
    std::vector<BmkRecord>& all_pricing_records,
    int n_threads,
    const double phi_own_scale[N_ROUTES],
    const double phi_other_scale[N_ROUTES],
    const char* subfolder)
{
    std::vector<double> out_para = out_para_vec(kind, rho);
    const int FLAT_SZ = N_ROUTES * 2;
    double pricing_opt[FLAT_SZ] = {};
    bool found_opt = false;

    const int expected_n_rows = (int)(cfg.beta11_list.size() * cfg.beta12_list.size());

    // Resume: just check the single {subfolder}/ CSV.
    if (try_resume_shared(rho_folder, subfolder, cfg.delay_constraint, expected_n_rows,
                          FLAT_SZ, pricing_opt)) {
        return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
    }

    std::string shared_folder = rho_folder + "/pricing/" + subfolder;
    mkdir_p(shared_folder);
    std::string csv_path = shared_folder + "/" SYSTEM_NAME "_final_results.csv";

    struct BcpCombo { double beta11, beta12; };
    std::vector<BcpCombo> combos;
    for (double b11 : cfg.beta11_list)
        for (double b12 : cfg.beta12_list)
            combos.push_back({b11, b12});
    const int total_combos = (int)combos.size();

    printf("  [Shared/%s] %d combos (threads=%d)  [BCP per-route scales:", subfolder, total_combos, n_threads);
    for (int r = 0; r < N_ROUTES; ++r)
        printf("  r=%d own=%g other=%g", r, phi_own_scale[r], phi_other_scale[r]);
    printf("]\n");

    std::vector<BmkRecord> recs(total_combos);

#ifdef USE_OMP
    // The WMP scheduling controls are thread_local.  The caller fixes them
    // for this pricing search, so copy that state into every worker before
    // any candidate is evaluated.
    const double fixed_lambda_tilde = RC_LAMBDA_TILDE;
    const double fixed_beta_tilde_v = RC_BETA_TILDE_V;
    const double fixed_beta_tilde_h = RC_BETA_TILDE_H;
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
    {
        RC_LAMBDA_TILDE = fixed_lambda_tilde;
        RC_BETA_TILDE_V = fixed_beta_tilde_v;
        RC_BETA_TILDE_H = fixed_beta_tilde_h;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < total_combos; ++i) {
            double b11 = combos[i].beta11;
            double b12 = combos[i].beta12;
            double flat[FLAT_SZ];
            for (int r = 0; r < N_ROUTES; ++r) {
                flat[r * 2 + 0] = b11 * phi_own_scale[r];
                flat[r * 2 + 1] = b12 * phi_other_scale[r];
            }
            set_bcp_pricing_phi(flat);
            std::vector<double> model_para(flat, flat + FLAT_SZ);
            for (double v : out_para) model_para.push_back(v);
            recs[i] = run_one_bmk(model_para);
        }
    }
#else
    for (int i = 0; i < total_combos; ++i) {
        double b11 = combos[i].beta11;
        double b12 = combos[i].beta12;
        double flat[FLAT_SZ];
        for (int r = 0; r < N_ROUTES; ++r) {
            flat[r * 2 + 0] = b11 * phi_own_scale[r];
            flat[r * 2 + 1] = b12 * phi_other_scale[r];
        }
        set_bcp_pricing_phi(flat);
        std::vector<double> model_para(flat, flat + FLAT_SZ);
        for (double v : out_para) model_para.push_back(v);
        recs[i] = run_one_bmk(model_para);
    }
#endif

    for (auto& r : recs) all_pricing_records.push_back(r);
    save_bmk_csv(csv_path, recs);

    const BmkRecord* best = find_min_cost(recs, cfg.delay_constraint, false);
    if (best != nullptr) {
        for (int k = 0; k < FLAT_SZ; ++k) pricing_opt[k] = best->params[k];
        found_opt = true;
        std::string opt_path = rho_folder + "/pricing/optimal_parameter.csv";
        print_and_save_optimal(best, cfg.delay_constraint, opt_path,
                               std::string("rho=") + fmt_param(rho) + " " + subfolder);
    } else {
        printf("  [Shared/%s] No records — pricing search yields no result.\n", subfolder);
    }

    if (!found_opt) return std::vector<double>(FLAT_SZ, 0.0);
    return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
}

// ---------------------------------------------------------------------------
// High-Low shared-parameter search (use_shared_parameter == 1 or 2).
// Total combos = len(Q_threshold_list) × pairs(zeta_list). For each combo,
// build flat[N_ROUTES*3] where routes 0..N-2 use (z_thresh[r], zeta1, zeta2)
// — z_thresh per-route via QUEUE_LEN_BOUND[r] — and route N-1 uses
// (z_thresh[N-1], zeta1 * hl_last_scale, zeta2 * hl_last_scale).
// Q_threshold itself is shared (same fraction across routes).
// ---------------------------------------------------------------------------
static std::vector<double> highlow_shared_param_search(
    const BmkKind& kind,
    double rho,
    const BmkConfig& cfg,
    const std::string& rho_folder,
    std::vector<BmkRecord>& all_pricing_records,
    int n_threads,
    double hl_last_scale,
    const char* subfolder)
{
    std::vector<double> out_para = out_para_vec(kind, rho);
    const int FLAT_SZ = N_ROUTES * 3;
    double pricing_opt[FLAT_SZ] = {};
    bool found_opt = false;

    const int nZ = (int)cfg.zeta_list.size();
    const int expected_n_rows = (int)cfg.Q_threshold_list.size() * (nZ * (nZ + 1) / 2);

    if (try_resume_shared(rho_folder, subfolder, cfg.delay_constraint, expected_n_rows,
                          FLAT_SZ, pricing_opt)) {
        return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
    }

    std::string shared_folder = rho_folder + "/pricing/" + subfolder;
    mkdir_p(shared_folder);
    std::string csv_path = shared_folder + "/" SYSTEM_NAME "_final_results.csv";

    struct HlSharedCombo { double Q_loop; double zeta1, zeta2; };
    std::vector<HlSharedCombo> combos;
    for (double Q_loop : cfg.Q_threshold_list) {
        for (int iz1 = 0; iz1 < nZ; ++iz1) {
            double zeta1 = cfg.zeta_list[iz1];
            for (int iz2 = iz1; iz2 < nZ; ++iz2) {
                double zeta2 = cfg.zeta_list[iz2];
                combos.push_back({Q_loop, zeta1, zeta2});
            }
        }
    }
    const int total_combos = (int)combos.size();

    printf("  [Shared/%s] %d combos (threads=%d)  [HL last-route zeta scale=%g]\n",
           subfolder, total_combos, n_threads, hl_last_scale);

    std::vector<BmkRecord> recs(total_combos);

#ifdef USE_OMP
    // Install the pricing search's fixed WMP schedule in every worker.
    const double fixed_lambda_tilde = RC_LAMBDA_TILDE;
    const double fixed_beta_tilde_v = RC_BETA_TILDE_V;
    const double fixed_beta_tilde_h = RC_BETA_TILDE_H;
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
    {
        RC_LAMBDA_TILDE = fixed_lambda_tilde;
        RC_BETA_TILDE_V = fixed_beta_tilde_v;
        RC_BETA_TILDE_H = fixed_beta_tilde_h;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < total_combos; ++i) {
            double Q_loop = combos[i].Q_loop;
            double zeta1  = combos[i].zeta1;
            double zeta2  = combos[i].zeta2;
            double flat[FLAT_SZ];
            for (int r = 0; r < N_ROUTES; ++r) {
                double s = (r == N_ROUTES - 1) ? hl_last_scale : 1.0;
                double z_thresh = Q_loop * QUEUE_LEN_BOUND[r] * rho * INV_SQRT_N;
                flat[r * 3 + 0] = z_thresh;
                flat[r * 3 + 1] = zeta1 * s;
                flat[r * 3 + 2] = zeta2 * s;
            }
            set_hl_pricing_para(flat);
            std::vector<double> model_para(flat, flat + FLAT_SZ);
            for (double v : out_para) model_para.push_back(v);
            recs[i] = run_one_bmk(model_para);
        }
    }
#else
    for (int i = 0; i < total_combos; ++i) {
        double Q_loop = combos[i].Q_loop;
        double zeta1  = combos[i].zeta1;
        double zeta2  = combos[i].zeta2;
        double flat[FLAT_SZ];
        for (int r = 0; r < N_ROUTES; ++r) {
            double s = (r == N_ROUTES - 1) ? hl_last_scale : 1.0;
            double z_thresh = Q_loop * QUEUE_LEN_BOUND[r] * rho * INV_SQRT_N;
            flat[r * 3 + 0] = z_thresh;
            flat[r * 3 + 1] = zeta1 * s;
            flat[r * 3 + 2] = zeta2 * s;
        }
        set_hl_pricing_para(flat);
        std::vector<double> model_para(flat, flat + FLAT_SZ);
        for (double v : out_para) model_para.push_back(v);
        recs[i] = run_one_bmk(model_para);
    }
#endif

    for (auto& r : recs) all_pricing_records.push_back(r);
    save_bmk_csv(csv_path, recs);

    const BmkRecord* best = find_min_cost(recs, cfg.delay_constraint, false);
    if (best != nullptr) {
        for (int k = 0; k < FLAT_SZ; ++k) pricing_opt[k] = best->params[k];
        found_opt = true;
        std::string opt_path = rho_folder + "/pricing/optimal_parameter.csv";
        print_and_save_optimal(best, cfg.delay_constraint, opt_path,
                               std::string("rho=") + fmt_param(rho) + " " + subfolder);
    } else {
        printf("  [Shared/%s] No records — pricing search yields no result.\n", subfolder);
    }

    if (!found_opt) return std::vector<double>(FLAT_SZ, 0.0);
    return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
}

// ---------------------------------------------------------------------------
// BCPlinear shared-parameter POOLED search (use_shared_parameter == 3).
// Concatenates the variant-1 (paper D.63) and variant-2 (paper D.65) per-route
// scale grids into one combined search. Combo count = 2 × |beta11_list| ×
// |beta12_list|. The first half uses (phi_*_v1, phi_*_v1) scales, the second
// half uses (phi_*_v2, phi_*_v2) scales. The best record is picked from the
// pooled records. Output: {rho_folder}/pricing/shared_pooled/.
// ---------------------------------------------------------------------------
static std::vector<double> bcplinear_shared_pooled_search(
    const BmkKind& kind,
    double rho,
    const BmkConfig& cfg,
    const std::string& rho_folder,
    std::vector<BmkRecord>& all_pricing_records,
    int n_threads,
    const double phi_own_scale_v1[N_ROUTES],
    const double phi_other_scale_v1[N_ROUTES],
    const double phi_own_scale_v2[N_ROUTES],
    const double phi_other_scale_v2[N_ROUTES])
{
    std::vector<double> out_para = out_para_vec(kind, rho);
    const int FLAT_SZ = N_ROUTES * 2;
    double pricing_opt[FLAT_SZ] = {};
    bool found_opt = false;
    const char* subfolder = "shared_pooled";

    const int half_n = (int)(cfg.beta11_list.size() * cfg.beta12_list.size());
    const int expected_n_rows = 2 * half_n;

    if (try_resume_shared(rho_folder, subfolder, cfg.delay_constraint, expected_n_rows,
                          FLAT_SZ, pricing_opt)) {
        return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
    }

    std::string shared_folder = rho_folder + "/pricing/" + subfolder;
    mkdir_p(shared_folder);
    std::string csv_path = shared_folder + "/" SYSTEM_NAME "_final_results.csv";

    // Pre-expand (b11, b12) combos in the same nesting order as the single-
    // variant helper, so variant-1 rows occupy indices [0, half_n) and
    // variant-2 rows occupy [half_n, 2*half_n) — letting callers identify the
    // winning variant from the row index alone.
    struct BcpCombo { double beta11, beta12; };
    std::vector<BcpCombo> base_combos;
    for (double b11 : cfg.beta11_list)
        for (double b12 : cfg.beta12_list)
            base_combos.push_back({b11, b12});

    printf("  [Shared/%s] %d combos (threads=%d)  [variant 1 rows 0..%d, variant 2 rows %d..%d]\n",
           subfolder, expected_n_rows, n_threads,
           half_n - 1, half_n, expected_n_rows - 1);
    auto print_scales = [&](const char* tag, const double own[], const double oth[]) {
        printf("    %s per-route scales:", tag);
        for (int r = 0; r < N_ROUTES; ++r)
            printf("  r=%d own=%g other=%g", r, own[r], oth[r]);
        printf("\n");
    };
    print_scales("v1", phi_own_scale_v1, phi_other_scale_v1);
    print_scales("v2", phi_own_scale_v2, phi_other_scale_v2);

    std::vector<BmkRecord> recs(expected_n_rows);

    auto run_one = [&](int idx, double b11, double b12,
                       const double own[N_ROUTES], const double oth[N_ROUTES]) {
        double flat[FLAT_SZ];
        for (int r = 0; r < N_ROUTES; ++r) {
            flat[r * 2 + 0] = b11 * own[r];
            flat[r * 2 + 1] = b12 * oth[r];
        }
        set_bcp_pricing_phi(flat);
        std::vector<double> model_para(flat, flat + FLAT_SZ);
        for (double v : out_para) model_para.push_back(v);
        recs[idx] = run_one_bmk(model_para);
    };

#ifdef USE_OMP
    // Workers may retain defaults or a previous scheduling candidate in
    // these thread_local fields; reset them to this pricing search's snapshot.
    const double fixed_lambda_tilde = RC_LAMBDA_TILDE;
    const double fixed_beta_tilde_v = RC_BETA_TILDE_V;
    const double fixed_beta_tilde_h = RC_BETA_TILDE_H;
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
    {
        RC_LAMBDA_TILDE = fixed_lambda_tilde;
        RC_BETA_TILDE_V = fixed_beta_tilde_v;
        RC_BETA_TILDE_H = fixed_beta_tilde_h;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < expected_n_rows; ++i) {
            int   half  = i / half_n;          // 0 → v1, 1 → v2
            int   local = i % half_n;
            double b11  = base_combos[local].beta11;
            double b12  = base_combos[local].beta12;
            if (half == 0) run_one(i, b11, b12, phi_own_scale_v1, phi_other_scale_v1);
            else           run_one(i, b11, b12, phi_own_scale_v2, phi_other_scale_v2);
        }
    }
#else
    for (int i = 0; i < expected_n_rows; ++i) {
        int   half  = i / half_n;
        int   local = i % half_n;
        double b11  = base_combos[local].beta11;
        double b12  = base_combos[local].beta12;
        if (half == 0) run_one(i, b11, b12, phi_own_scale_v1, phi_other_scale_v1);
        else           run_one(i, b11, b12, phi_own_scale_v2, phi_other_scale_v2);
    }
#endif

    for (auto& r : recs) all_pricing_records.push_back(r);
    save_bmk_csv(csv_path, recs);

    const BmkRecord* best = find_min_cost(recs, cfg.delay_constraint, false);
    if (best != nullptr) {
        for (int k = 0; k < FLAT_SZ; ++k) pricing_opt[k] = best->params[k];
        found_opt = true;
        std::string opt_path = rho_folder + "/pricing/optimal_parameter.csv";
        print_and_save_optimal(best, cfg.delay_constraint, opt_path,
                               std::string("rho=") + fmt_param(rho) + " " + subfolder);
    } else {
        printf("  [Shared/%s] No records — pricing search yields no result.\n", subfolder);
    }

    if (!found_opt) return std::vector<double>(FLAT_SZ, 0.0);
    return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
}

// ---------------------------------------------------------------------------
// High-Low shared-parameter POOLED search (use_shared_parameter == 3).
// Concatenates variant-1 (zeta scale = 1.0) and variant-2 (zeta scale =
// mc_H/mc_V) HL grids into one combined search. Combo count = 2 ×
// |Q_threshold_list| × pairs(zeta_list). First half = v1, second half = v2.
// Output: {rho_folder}/pricing/shared_pooled/.
// ---------------------------------------------------------------------------
static std::vector<double> highlow_shared_pooled_search(
    const BmkKind& kind,
    double rho,
    const BmkConfig& cfg,
    const std::string& rho_folder,
    std::vector<BmkRecord>& all_pricing_records,
    int n_threads,
    double hl_zeta_scale_v1,
    double hl_zeta_scale_v2)
{
    std::vector<double> out_para = out_para_vec(kind, rho);
    const int FLAT_SZ = N_ROUTES * 3;
    double pricing_opt[FLAT_SZ] = {};
    bool found_opt = false;
    const char* subfolder = "shared_pooled";

    const int nZ = (int)cfg.zeta_list.size();
    const int half_n = (int)cfg.Q_threshold_list.size() * (nZ * (nZ + 1) / 2);
    const int expected_n_rows = 2 * half_n;

    if (try_resume_shared(rho_folder, subfolder, cfg.delay_constraint, expected_n_rows,
                          FLAT_SZ, pricing_opt)) {
        return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
    }

    std::string shared_folder = rho_folder + "/pricing/" + subfolder;
    mkdir_p(shared_folder);
    std::string csv_path = shared_folder + "/" SYSTEM_NAME "_final_results.csv";

    struct HlSharedCombo { double Q_loop; double zeta1, zeta2; };
    std::vector<HlSharedCombo> base_combos;
    for (double Q_loop : cfg.Q_threshold_list) {
        for (int iz1 = 0; iz1 < nZ; ++iz1) {
            double zeta1 = cfg.zeta_list[iz1];
            for (int iz2 = iz1; iz2 < nZ; ++iz2) {
                double zeta2 = cfg.zeta_list[iz2];
                base_combos.push_back({Q_loop, zeta1, zeta2});
            }
        }
    }

    printf("  [Shared/%s] %d combos (threads=%d)  [v1 zeta scale=%g rows 0..%d, "
           "v2 zeta scale=%g rows %d..%d]\n",
           subfolder, expected_n_rows, n_threads,
           hl_zeta_scale_v1, half_n - 1,
           hl_zeta_scale_v2, half_n, expected_n_rows - 1);

    std::vector<BmkRecord> recs(expected_n_rows);

    auto run_one = [&](int idx, double Q_loop, double zeta1, double zeta2, double s) {
        double flat[FLAT_SZ];
        for (int r = 0; r < N_ROUTES; ++r) {
            const double s_r = (r == N_ROUTES - 1) ? s : 1.0;
            double z_thresh = Q_loop * QUEUE_LEN_BOUND[r] * rho * INV_SQRT_N;
            flat[r * 3 + 0] = z_thresh;
            flat[r * 3 + 1] = zeta1 * s_r;
            flat[r * 3 + 2] = zeta2 * s_r;
        }
        set_hl_pricing_para(flat);
        std::vector<double> model_para(flat, flat + FLAT_SZ);
        for (double v : out_para) model_para.push_back(v);
        recs[idx] = run_one_bmk(model_para);
    };

#ifdef USE_OMP
    // Install the fixed scheduling snapshot before evaluating pooled pricing
    // candidates so dynamic work assignment cannot change the policy.
    const double fixed_lambda_tilde = RC_LAMBDA_TILDE;
    const double fixed_beta_tilde_v = RC_BETA_TILDE_V;
    const double fixed_beta_tilde_h = RC_BETA_TILDE_H;
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
    {
        RC_LAMBDA_TILDE = fixed_lambda_tilde;
        RC_BETA_TILDE_V = fixed_beta_tilde_v;
        RC_BETA_TILDE_H = fixed_beta_tilde_h;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < expected_n_rows; ++i) {
            int   half  = i / half_n;
            int   local = i % half_n;
            double Q_loop = base_combos[local].Q_loop;
            double zeta1  = base_combos[local].zeta1;
            double zeta2  = base_combos[local].zeta2;
            const double s = (half == 0) ? hl_zeta_scale_v1 : hl_zeta_scale_v2;
            run_one(i, Q_loop, zeta1, zeta2, s);
        }
    }
#else
    for (int i = 0; i < expected_n_rows; ++i) {
        int   half  = i / half_n;
        int   local = i % half_n;
        double Q_loop = base_combos[local].Q_loop;
        double zeta1  = base_combos[local].zeta1;
        double zeta2  = base_combos[local].zeta2;
        const double s = (half == 0) ? hl_zeta_scale_v1 : hl_zeta_scale_v2;
        run_one(i, Q_loop, zeta1, zeta2, s);
    }
#endif

    for (auto& r : recs) all_pricing_records.push_back(r);
    save_bmk_csv(csv_path, recs);

    const BmkRecord* best = find_min_cost(recs, cfg.delay_constraint, false);
    if (best != nullptr) {
        for (int k = 0; k < FLAT_SZ; ++k) pricing_opt[k] = best->params[k];
        found_opt = true;
        std::string opt_path = rho_folder + "/pricing/optimal_parameter.csv";
        print_and_save_optimal(best, cfg.delay_constraint, opt_path,
                               std::string("rho=") + fmt_param(rho) + " " + subfolder);
    } else {
        printf("  [Shared/%s] No records — pricing search yields no result.\n", subfolder);
    }

    if (!found_opt) return std::vector<double>(FLAT_SZ, 0.0);
    return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
}

static std::vector<double> bcplinear_pricing_search(
    const BmkKind& kind,
    double rho,
    const BmkConfig& cfg,
    const std::string& rho_folder,
    std::vector<BmkRecord>& all_pricing_records,
    int n_threads)
{
    std::vector<double> out_para = out_para_vec(kind, rho);

    // Current optimal pricing phi (N_ROUTES × 2 = 2K elements, all-zero initially).
    // flat[r*2 + 0] = phi_r, flat[r*2 + 1] = phi_{-r}.
    const int FLAT_SZ = N_ROUTES * 2;
    double pricing_opt[FLAT_SZ] = {};
    bool found_opt = false;

    // Last-round (horizontal route) scaling factor when
    // pricing_magnitude_scale_by_mc_eq == 1.  See BmkConfig comment.
    const bool   scale_last     = (cfg.pricing_magnitude_scale_by_mc_eq != 0);
    const double mc_ratio       = OUTSOURCE_COST_RATE[N_ROUTES-1] / OUTSOURCE_COST_RATE[0];
    const double qb_ratio       = QUEUE_LEN_BOUND   [N_ROUTES-1] / QUEUE_LEN_BOUND   [0];
    const double bcp_last_scale = mc_ratio / qb_ratio;

    // Variant branch: use_shared_parameter == 1 or 2 collapses N_ROUTES rounds
    // into one shared-param search; the search grid honors the paper's
    // shared-parameter assumption (Eqs. D.62-D.65):
    //   variant 1 (=1, "shared_common"): φ_r β_r and φ_{-r} β_{-r} both shared,
    //                                    so φ_r ∝ 1/β_r and φ_{-r} ∝ 1/β_{-r}.
    //   variant 2 (=2, "shared_mc_eq"):  additionally multiplied by ν_r/ν_0
    //                                    (the outsourcing-marginal-cost ratio).
    // Per-route scale arrays derived from B_VEC (β_r), get_B_others_vec() (β_{-r})
    // and OUTSOURCE_COST_RATE (ν_r). HL pricing is handled separately and is
    // already consistent with D.62/D.64 (see highlow_pricing_search).
    if (cfg.use_shared_parameter == 1 || cfg.use_shared_parameter == 2) {
        const auto& B_others_vec = get_B_others_vec();
        double phi_own_scale  [N_ROUTES];
        double phi_other_scale[N_ROUTES];
        for (int r = 0; r < N_ROUTES; ++r) {
            const double mc_factor = (cfg.use_shared_parameter == 2)
                                       ? OUTSOURCE_COST_RATE[r] / OUTSOURCE_COST_RATE[0]
                                       : 1.0;
            phi_own_scale  [r] = (B_VEC[0] / B_VEC[r])         * mc_factor;
            phi_other_scale[r] = (B_VEC[0] / B_others_vec[r])  * mc_factor;
        }
        const char* subfolder = (cfg.use_shared_parameter == 2)
                                   ? "shared_mc_eq"
                                   : "shared_common";
        return bcplinear_shared_param_search(kind, rho, cfg, rho_folder,
                                             all_pricing_records, n_threads,
                                             phi_own_scale, phi_other_scale,
                                             subfolder);
    }

    // Pooled search (variant 1 + variant 2 combined). Output goes to
    // {rho_folder}/pricing/shared_pooled/. Best is selected across all
    // 2 × |beta11_list| × |beta12_list| records.
    if (cfg.use_shared_parameter == 3) {
        const auto& B_others_vec = get_B_others_vec();
        double phi_own_v1  [N_ROUTES], phi_other_v1[N_ROUTES];
        double phi_own_v2  [N_ROUTES], phi_other_v2[N_ROUTES];
        for (int r = 0; r < N_ROUTES; ++r) {
            const double mc_factor = OUTSOURCE_COST_RATE[r] / OUTSOURCE_COST_RATE[0];
            phi_own_v1  [r] = (B_VEC[0] / B_VEC[r]);
            phi_other_v1[r] = (B_VEC[0] / B_others_vec[r]);
            phi_own_v2  [r] = phi_own_v1  [r] * mc_factor;
            phi_other_v2[r] = phi_other_v1[r] * mc_factor;
        }
        return bcplinear_shared_pooled_search(
            kind, rho, cfg, rho_folder, all_pricing_records, n_threads,
            phi_own_v1, phi_other_v1, phi_own_v2, phi_other_v2);
    }

    // Resume support: check for previously completed rounds
    int resume_from = find_resume_round(rho_folder, cfg.delay_constraint, FLAT_SZ, pricing_opt);
    int start_round = 0;
    if (resume_from >= 0) {
        found_opt = true;
        start_round = resume_from + 1;
        if (start_round >= N_ROUTES) {
            printf("  [Resume] All %d pricing rounds already completed. Skipping pricing search.\n", N_ROUTES);
            return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
        }
        printf("  [Resume] Found completed rounds 0..%d. Resuming from round %d.\n",
               resume_from, start_round);
        printf("  [Resume] pricing_opt from round %d:", resume_from);
        for (int k = 0; k < FLAT_SZ; k++) printf(" %.4g", pricing_opt[k]);
        printf("\n");
    }

    for (int i_round = start_round; i_round < N_ROUTES; i_round++) {
        std::string round_folder = rho_folder + "/pricing/round" + std::to_string(i_round);
        mkdir_p(round_folder);
        std::string csv_path = round_folder + "/" SYSTEM_NAME "_final_results.csv";

        // Pre-expand (beta11, beta12) combos for this round.
        // For the last round (horizontal route) with scale_last on, multiply
        // every list element by bcp_last_scale = (mc_H/mc_V) / (qb_H/qb_V).
        const bool   is_last      = (i_round == N_ROUTES - 1);
        const double bcp_s        = (scale_last && is_last) ? bcp_last_scale : 1.0;
        struct BcpCombo { double beta11, beta12; };
        std::vector<BcpCombo> combos;
        for (double b11 : cfg.beta11_list)
            for (double b12 : cfg.beta12_list)
                combos.push_back({b11 * bcp_s, b12 * bcp_s});
        const int total_combos = (int)combos.size();

        if (scale_last && is_last)
            printf("  [Round %d] %d combos (threads=%d)  [scale_by_mc_eq=1 → beta scale=%g]\n",
                   i_round, total_combos, n_threads, bcp_s);
        else
            printf("  [Round %d] %d combos (threads=%d)\n", i_round, total_combos, n_threads);

        std::vector<BmkRecord> round_recs(total_combos);

#ifdef USE_OMP
        // Keep every worker on the scheduling policy fixed by the caller for
        // this route's pricing round.
        const double fixed_lambda_tilde = RC_LAMBDA_TILDE;
        const double fixed_beta_tilde_v = RC_BETA_TILDE_V;
        const double fixed_beta_tilde_h = RC_BETA_TILDE_H;
        #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
        {
            RC_LAMBDA_TILDE = fixed_lambda_tilde;
            RC_BETA_TILDE_V = fixed_beta_tilde_v;
            RC_BETA_TILDE_H = fixed_beta_tilde_h;

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < total_combos; ++i) {
                double beta11 = combos[i].beta11;
                double beta12 = combos[i].beta12;

                // Build phi array: copy previous-round optimum, then overwrite
                // the (phi_{i_round}, phi_{-i_round}) pair with (beta11, beta12).
                double flat[FLAT_SZ];
                for (int k = 0; k < FLAT_SZ; ++k) flat[k] = pricing_opt[k];
                flat[i_round * 2 + 0] = beta11;
                flat[i_round * 2 + 1] = beta12;
                set_bcp_pricing_phi(flat);

                std::vector<double> model_para(flat, flat + FLAT_SZ);
                for (double v : out_para) model_para.push_back(v);

                round_recs[i] = run_one_bmk(model_para);
            }
        }
#else
        for (int i = 0; i < total_combos; ++i) {
            double beta11 = combos[i].beta11;
            double beta12 = combos[i].beta12;

            double flat[FLAT_SZ];
            for (int k = 0; k < FLAT_SZ; ++k) flat[k] = pricing_opt[k];
            flat[i_round * 2 + 0] = beta11;
            flat[i_round * 2 + 1] = beta12;
            set_bcp_pricing_phi(flat);

            std::vector<double> model_para(flat, flat + FLAT_SZ);
            for (double v : out_para) model_para.push_back(v);

            round_recs[i] = run_one_bmk(model_para);
        }
#endif

        for (auto& r : round_recs) all_pricing_records.push_back(r);

        save_bmk_csv(csv_path, round_recs);

        // Find min-cost (no delay constraint) in this round
        const BmkRecord* best = find_min_cost(round_recs, cfg.delay_constraint, false);
        if (best == nullptr) {
            printf("  [Round %d] No records — stopping pricing search.\n", i_round);
            break;
        }
        // Extract pricing phi from the best record (pricing params come first)
        for (int k = 0; k < FLAT_SZ; k++)
            pricing_opt[k] = best->params[k];
        found_opt = true;

        std::string opt_path = rho_folder + "/pricing/optimal_parameter.csv";
        print_and_save_optimal(best, cfg.delay_constraint, opt_path,
                               "rho=" + fmt_param(rho) + " round" + std::to_string(i_round));
    }

    if (!found_opt)
        return std::vector<double>(FLAT_SZ, 0.0);
    return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
}

// ---------------------------------------------------------------------------
// High_low pricing parameter search: N_ROUTES (=4) rounds
//
// Round i_round: modify pricing_para[i_round,:] = [int(Q_loop * QUEUE_LEN_BOUND[i_round] * rho), zeta1, zeta2]
//                Keep other rows from previous optimal.
//
// zeta1 iterates over zeta_list; zeta2 iterates over zeta_list[idx_zeta1..]
// (combinations with repetition: zeta2 >= zeta1 by index)
//
// Returns optimal flat(N_ROUTES×3) pricing params.
// ---------------------------------------------------------------------------
static std::vector<double> highlow_pricing_search(
    const BmkKind& kind,
    double rho,
    const BmkConfig& cfg,
    const std::string& rho_folder,
    std::vector<BmkRecord>& all_pricing_records,
    int n_threads)
{
    std::vector<double> out_para = out_para_vec(kind, rho);

    const int FLAT_SZ = N_ROUTES * 3;
    double pricing_opt[FLAT_SZ] = {};
    bool found_opt = false;

    int nZ = (int)cfg.zeta_list.size();

    // Last-round (horizontal route) zeta scaling factor when
    // pricing_magnitude_scale_by_mc_eq == 1.  Q_threshold_list never scales.
    const bool   scale_last    = (cfg.pricing_magnitude_scale_by_mc_eq != 0);
    const double hl_last_scale = OUTSOURCE_COST_RATE[N_ROUTES-1] / OUTSOURCE_COST_RATE[0];

    // Variant branch: use_shared_parameter == 1 or 2 collapses N_ROUTES rounds
    // into one shared-param search.
    //   =1: HL last-route zeta scale = 1.0 (no scaling), output under shared_common/
    //   =2: HL last-route zeta scale = mc_H/mc_V,        output under shared_mc_eq/
    if (cfg.use_shared_parameter == 1 || cfg.use_shared_parameter == 2) {
        const double hl_s      = (cfg.use_shared_parameter == 2) ? hl_last_scale : 1.0;
        const char*  subfolder = (cfg.use_shared_parameter == 2)
                                    ? "shared_mc_eq" : "shared_common";
        return highlow_shared_param_search(kind, rho, cfg, rho_folder,
                                           all_pricing_records, n_threads,
                                           hl_s, subfolder);
    }

    // Pooled search (variant 1 + variant 2 combined). Output goes to
    // {rho_folder}/pricing/shared_pooled/.
    if (cfg.use_shared_parameter == 3) {
        const double hl_v1 = 1.0;
        const double hl_v2 = hl_last_scale;  // = mc_H / mc_V
        return highlow_shared_pooled_search(
            kind, rho, cfg, rho_folder, all_pricing_records, n_threads,
            hl_v1, hl_v2);
    }

    // Resume support: check for previously completed rounds
    int resume_from = find_resume_round(rho_folder, cfg.delay_constraint, FLAT_SZ, pricing_opt);
    int start_round = 0;
    if (resume_from >= 0) {
        found_opt = true;
        start_round = resume_from + 1;
        if (start_round >= N_ROUTES) {
            printf("  [Resume] All %d pricing rounds already completed. Skipping pricing search.\n", N_ROUTES);
            return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
        }
        printf("  [Resume] Found completed rounds 0..%d. Resuming from round %d.\n",
               resume_from, start_round);
        printf("  [Resume] pricing_opt from round %d:", resume_from);
        for (int k = 0; k < FLAT_SZ; k++) printf(" %.4g", pricing_opt[k]);
        printf("\n");
    }

    for (int i_round = start_round; i_round < N_ROUTES; i_round++) {
        std::string round_folder = rho_folder + "/pricing/round" + std::to_string(i_round);
        mkdir_p(round_folder);
        std::string csv_path = round_folder + "/" SYSTEM_NAME "_final_results.csv";

        // QUEUE_LEN_BOUND[i_round]: vertical routes 0..2 → 28.5, horizontal route 3 → 42.75
        double q_bound = QUEUE_LEN_BOUND[i_round];

        // Last-round zeta scaling under pricing_magnitude_scale_by_mc_eq.
        // Q_threshold_list is NEVER scaled (z_thresh already uses qb[r]).
        const bool   is_last = (i_round == N_ROUTES - 1);
        const double hl_s    = (scale_last && is_last) ? hl_last_scale : 1.0;

        // Pre-expand (Q_loop, zeta1, zeta2) combos for this round.
        // Threshold is stored in Z-scale (i.e. already multiplied by INV_SQRT_N)
        // so that high_low_pricing can compare against inp.Z[r] directly without
        // a per-step divide. No integer truncation: the full real-valued product
        // is used, giving finer search granularity than the previous (int)-cast.
        struct HlCombo { double z_thresh; double zeta1, zeta2; };
        std::vector<HlCombo> combos;
        for (double Q_loop : cfg.Q_threshold_list) {
            double z_thresh = Q_loop * q_bound * rho * INV_SQRT_N;
            for (int iz1 = 0; iz1 < nZ; iz1++) {
                double zeta1 = cfg.zeta_list[iz1] * hl_s;
                for (int iz2 = iz1; iz2 < nZ; iz2++) {
                    double zeta2 = cfg.zeta_list[iz2] * hl_s;
                    combos.push_back({z_thresh, zeta1, zeta2});
                }
            }
        }
        const int total_combos = (int)combos.size();
        if (scale_last && is_last)
            printf("  [Round %d] %d combos (threads=%d)  [scale_by_mc_eq=1 → zeta scale=%g]\n",
                   i_round, total_combos, n_threads, hl_s);
        else
            printf("  [Round %d] %d combos (threads=%d)\n", i_round, total_combos, n_threads);

        std::vector<BmkRecord> round_recs(total_combos);

#ifdef USE_OMP
        // Reinstall the fixed WMP schedule for each High-low pricing round;
        // workers may otherwise carry state from a preceding scheduling sweep.
        const double fixed_lambda_tilde = RC_LAMBDA_TILDE;
        const double fixed_beta_tilde_v = RC_BETA_TILDE_V;
        const double fixed_beta_tilde_h = RC_BETA_TILDE_H;
        #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
        {
            RC_LAMBDA_TILDE = fixed_lambda_tilde;
            RC_BETA_TILDE_V = fixed_beta_tilde_v;
            RC_BETA_TILDE_H = fixed_beta_tilde_h;

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < total_combos; ++i) {
                double flat[FLAT_SZ];
                for (int k = 0; k < FLAT_SZ; ++k) flat[k] = pricing_opt[k];
                flat[i_round * 3 + 0] = combos[i].z_thresh;
                flat[i_round * 3 + 1] = combos[i].zeta1;
                flat[i_round * 3 + 2] = combos[i].zeta2;
                set_hl_pricing_para(flat);

                std::vector<double> model_para(flat, flat + FLAT_SZ);
                for (double v : out_para) model_para.push_back(v);

                round_recs[i] = run_one_bmk(model_para);
            }
        }
#else
        for (int i = 0; i < total_combos; ++i) {
            double flat[FLAT_SZ];
            for (int k = 0; k < FLAT_SZ; ++k) flat[k] = pricing_opt[k];
            flat[i_round * 3 + 0] = combos[i].z_thresh;
            flat[i_round * 3 + 1] = combos[i].zeta1;
            flat[i_round * 3 + 2] = combos[i].zeta2;
            set_hl_pricing_para(flat);

            std::vector<double> model_para(flat, flat + FLAT_SZ);
            for (double v : out_para) model_para.push_back(v);

            round_recs[i] = run_one_bmk(model_para);
        }
#endif

        for (auto& r : round_recs) all_pricing_records.push_back(r);

        save_bmk_csv(csv_path, round_recs);

        const BmkRecord* best = find_min_cost(round_recs, cfg.delay_constraint, false);
        if (best == nullptr) {
            printf("  [Round %d] No records — stopping pricing search.\n", i_round);
            break;
        }
        for (int k = 0; k < FLAT_SZ; k++)
            pricing_opt[k] = best->params[k];
        found_opt = true;

        std::string opt_path = rho_folder + "/pricing/optimal_parameter.csv";
        print_and_save_optimal(best, cfg.delay_constraint, opt_path,
                               "rho=" + fmt_param(rho) + " round" + std::to_string(i_round));
    }

    if (!found_opt)
        return std::vector<double>(FLAT_SZ, 0.0);
    return std::vector<double>(pricing_opt, pricing_opt + FLAT_SZ);
}

// ---------------------------------------------------------------------------
// WMP scheduling parameter search (Group C/D)
//
// Uses the optimal pricing params found in pricing search.
// Sweeps lambda_tilde × beta_rate_1 × beta_rate_2.
//   RC_BETA_TILDE_V = beta_rate_1 * BETA_CONST_V * rho   (vertical routes)
//   RC_BETA_TILDE_H = beta_rate_2 * BETA_CONST_H * rho   (horizontal route)
//
// Applies delay constraint (all 4 delays) when finding optimal.
// ---------------------------------------------------------------------------
static std::vector<BmkRecord> wmp_scheduling_search(
    const BmkKind& kind,
    double rho,
    const std::vector<double>& pricing_flat,  // flattened pricing params
    const BmkConfig& cfg,
    const std::string& rho_folder,
    int n_threads)
{
    std::vector<double> out_para = out_para_vec(kind, rho);

    RC_SCHEDULING_MODE = WMP_OBJ2;

    std::string sched_folder = rho_folder + "/scheduling";
    mkdir_p(sched_folder);
    std::string csv_path = sched_folder + "/" SYSTEM_NAME "_final_results.csv";

    // Resume support: if scheduling CSV already exists, skip
    {
        auto existing = load_bmk_csv(csv_path);
        if (!existing.empty()) {
            printf("  [Resume] Scheduling search already completed (%zu records). Skipping.\n",
                   existing.size());
            const BmkRecord* best = find_min_cost(existing, cfg.delay_constraint, true);
            std::string opt_path = sched_folder + "/optimal_parameter.csv";
            print_and_save_optimal(best, cfg.delay_constraint, opt_path,
                                   "rho=" + fmt_param(rho) + " scheduling (resumed)");
            return existing;
        }
    }

    // ---- Pre-expand all combos into a flat list ----
    struct SchedCombo { double lam, br_v, br_h, bt_v, bt_h; };
    std::vector<SchedCombo> combos;
    for (double lam : cfg.lambda_tilde_list) {
        for (double br_v : cfg.beta_rate_1_list) {
            const std::vector<double> br2_list =
                (cfg.beta_rate_symmetric == 1)
                    ? std::vector<double>{br_v}
                : (cfg.beta_rate_symmetric == 2)
                    ? std::vector<double>{1.0 - (1.0 - br_v) * B_V / B_H}
                : cfg.beta_rate_2_list;
            for (double br_h : br2_list) {
                SchedCombo c;
                c.lam  = lam;
                c.br_v = br_v;
                c.br_h = br_h;
                c.bt_v = br_v * BETA_CONST_V * rho;
                c.bt_h = br_h * BETA_CONST_H * rho;
                combos.push_back(c);
            }
        }
    }
    const int total_combos = (int)combos.size();

    printf("  [Scheduling search] %d combos  (beta_rate_symmetric=%d, threads=%d)\n",
           total_combos, cfg.beta_rate_symmetric, n_threads);

    // Snapshot RC values that benchmark threads read but don't write.
    // Thread-local RC_PRICING_PHI / RC_HL_PARA need to be seeded per-thread.
    // pricing_flat already in hand; we apply it in each thread below.
    std::vector<BmkRecord> sched_recs(total_combos);

#ifdef USE_OMP
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads > 1 ? n_threads : 1)
    {
        // Seed this thread's thread_local RC_PRICING_PHI / RC_HL_PARA.
        if (kind.pricing == BmkPricing::BCPlinear) {
            double flat[N_ROUTES * 2];
            for (int k = 0; k < (int)pricing_flat.size() && k < N_ROUTES * 2; ++k)
                flat[k] = pricing_flat[k];
            set_bcp_pricing_phi(flat);
        } else {
            double flat[N_ROUTES * 3];
            for (int k = 0; k < (int)pricing_flat.size() && k < N_ROUTES * 3; ++k)
                flat[k] = pricing_flat[k];
            set_hl_pricing_para(flat);
        }

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < total_combos; ++i) {
            const SchedCombo& c = combos[i];
            RC_LAMBDA_TILDE = c.lam;
            RC_BETA_TILDE_V = c.bt_v;
            RC_BETA_TILDE_H = c.bt_h;

            std::vector<double> model_para = pricing_flat;
            model_para.push_back(c.lam);
            model_para.push_back(c.bt_v);
            model_para.push_back(c.bt_h);
            for (double v : out_para) model_para.push_back(v);

            sched_recs[i] = run_one_bmk(model_para);
        }
    }
#else
    // Serial path (USE_OMP not defined or n_threads <= 1)
    for (int i = 0; i < total_combos; ++i) {
        const SchedCombo& c = combos[i];
        RC_LAMBDA_TILDE = c.lam;
        RC_BETA_TILDE_V = c.bt_v;
        RC_BETA_TILDE_H = c.bt_h;

        std::vector<double> model_para = pricing_flat;
        model_para.push_back(c.lam);
        model_para.push_back(c.bt_v);
        model_para.push_back(c.bt_h);
        for (double v : out_para) model_para.push_back(v);

        sched_recs[i] = run_one_bmk(model_para);
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
    // CLI usage:
    //   ./sim_benchmark [config] [benchmark] [safety] [epsilon] [rho] [variant]
    //   argv[1] = config path         (optional, default: benchmark_search_config.txt)
    //   argv[2] = benchmark name      (optional, overrides config)
    //   argv[3] = safety value        (optional, overrides safety_list → single value)
    //   argv[4] = epsilon value       (optional, overrides epsilon_list → single value)
    //   argv[5] = rho value           (optional, overrides rho_list → single value)
    //   argv[6] = variant: "abs" | "rel" | ""  (optional, C/D benchmarks only:
    //               "abs" → wmp_extra_relative=0, beta_rate_symmetric=2
    //               "rel" → wmp_extra_relative=1, beta_rate_symmetric=1
    //               ""    → use config values as-is)
    std::string cfg_path = "benchmark_search_config.txt";
    if (argc >= 2) cfg_path = argv[1];

    printf("=== Extended Criss-Cross Benchmark Search (dim_K=%d, %d paths per combo) ===\n\n",
           DIM_K, N_PATHS);
    printf("Reading config from: %s\n\n", cfg_path.c_str());

    BmkConfig cfg = load_config(cfg_path);

    if (argc >= 3) cfg.benchmark_name  = argv[2];
    if (argc >= 4) cfg.safety_list     = {std::stod(argv[3])};
    if (argc >= 5) cfg.epsilon_list    = {std::stod(argv[4])};
    if (argc >= 6) cfg.rho_list        = {std::stod(argv[5])};

    // Optional 6th arg: variant selector for abs/rel WMP variants (C/D only).
    if (argc >= 7) {
        std::string variant = argv[6];
        if (variant == "abs") {
            cfg.wmp_extra_relative  = 0;
            cfg.beta_rate_symmetric = 2;
            resolve_wmp_lists(cfg);   // re-resolve _abs/_rel lists under new flag
            resolve_sched_init(cfg);
            printf("CLI override: variant=abs (wmp_extra_relative=0, beta_rate_symmetric=2)\n");
        } else if (variant == "rel") {
            cfg.wmp_extra_relative  = 1;
            cfg.beta_rate_symmetric = 1;
            resolve_wmp_lists(cfg);
            resolve_sched_init(cfg);
            printf("CLI override: variant=rel (wmp_extra_relative=1, beta_rate_symmetric=1)\n");
        } else if (!variant.empty()) {
            fprintf(stderr, "ERROR: 6th CLI arg must be 'abs', 'rel', or omitted; got '%s'\n",
                    variant.c_str());
            return 1;
        }
        // empty string → keep config defaults
    }

    if (cfg.benchmark_name.empty()) {
        fprintf(stderr, "ERROR: 'benchmark' not set in config file.\n");
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
    if (kind.outsource == BmkOutsource::SinglePushing) {
        printf("  epsilon_list: ");
        for (double v : cfg.epsilon_list) printf("%.4g ", v);
        printf("\n");
    }
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
        printf("\n");
    }
    if (kind.has_sched_search) {
        // Validate the resolved WMP lists. Errors point at which key was missing.
        const char* variant_tag = (cfg.wmp_extra_relative == 1) ? "rel" : "abs";
        auto check_list = [&](const std::vector<double>& v, const char* name) {
            if (v.empty()) {
                fprintf(stderr,
                    "ERROR: benchmark %s (wmp_extra_relative=%d → variant '%s') needs\n"
                    "       %s_%s but the key was not provided in the config.\n",
                    cfg.benchmark_name.c_str(), cfg.wmp_extra_relative, variant_tag,
                    name, variant_tag);
                exit(1);
            }
        };
        check_list(cfg.lambda_tilde_list, "lambda_tilde_list");
        check_list(cfg.beta_rate_1_list,  "beta_rate_1_list");
        check_list(cfg.beta_rate_2_list,  "beta_rate_2_list");

        // Print which source the lists / sched_init came from.
        const char* list_suffix   = (cfg.wmp_extra_relative == 1) ? "_rel" : "_abs";
        auto scalar_source = [&](double abs_v, double rel_v) -> const char* {
            const double v = (cfg.wmp_extra_relative == 1) ? rel_v : abs_v;
            return std::isnan(v) ? "default (no _abs/_rel provided)"
                                 : (cfg.wmp_extra_relative == 1 ? "_rel suffix" : "_abs suffix");
        };
        printf("  WMP scheduling lists (wmp_extra_relative=%d → variant '%s'):\n",
               cfg.wmp_extra_relative, variant_tag);
        printf("    lambda_tilde_list    ← %s suffix\n", list_suffix);
        printf("    beta_rate_1_list     ← %s suffix\n", list_suffix);
        printf("    beta_rate_2_list     ← %s suffix\n", list_suffix);
        printf("    sched_init_lambda_tilde ← %s\n",
               scalar_source(cfg.sched_init_lambda_tilde_abs, cfg.sched_init_lambda_tilde_rel));
        printf("    sched_init_beta_rate_1  ← %s\n",
               scalar_source(cfg.sched_init_beta_rate_1_abs,  cfg.sched_init_beta_rate_1_rel));
        printf("    sched_init_beta_rate_2  ← %s\n",
               scalar_source(cfg.sched_init_beta_rate_2_abs,  cfg.sched_init_beta_rate_2_rel));

        const double br2_eff = effective_sched_init_br2(cfg);
        const char* br2_src = (cfg.beta_rate_symmetric == 1)
                                  ? " (=beta_rate_1, symmetric=1)"
                            : (cfg.beta_rate_symmetric == 2)
                                  ? " (derived from beta_rate_1, symmetric=2)"
                                  : "";
        printf("  sched_init (pricing phase): lambda_tilde=%.4g  beta_rate_1=%.4g  beta_rate_2=%.4g%s\n"
               "    → beta_tilde_v = %.4g * BETA_CONST_V * rho,  beta_tilde_h = %.4g * BETA_CONST_H * rho\n",
               cfg.sched_init_lambda_tilde, cfg.sched_init_beta_rate_1, br2_eff, br2_src,
               cfg.sched_init_beta_rate_1, br2_eff);
        printf("  lambda_tilde_list: ");
        for (double v : cfg.lambda_tilde_list) printf("%.4g ", v);
        printf("\n  beta_rate_1_list (→ RC_BETA_TILDE_V): ");
        for (double v : cfg.beta_rate_1_list) printf("%.4g ", v);
        if (cfg.beta_rate_symmetric == 1) {
            printf("\n  beta_rate_2_list (→ RC_BETA_TILDE_H): [symmetric — equals beta_rate_1]");
        } else if (cfg.beta_rate_symmetric == 2) {
            printf("\n  beta_rate_2_list (→ RC_BETA_TILDE_H): [derived — 1-(1-beta_rate_1)*B_V/B_H"
                   "  (B_V=%.4g, B_H=%.4g)]", B_V, B_H);
        } else {
            printf("\n  beta_rate_2_list (→ RC_BETA_TILDE_H): ");
            for (double v : cfg.beta_rate_2_list) printf("%.4g ", v);
        }
        printf("\n");
    }
    printf("  delay_constraint: %.2f%%\n", cfg.delay_constraint);
    printf("  output_dir: %s\n\n", cfg.output_dir.c_str());

    // Create output directory
    mkdir_p(cfg.output_dir);

    // Resolve OpenMP thread count (capped at SLURM allocation).
    // Returns 1 when USE_OMP is not defined or cfg.parallel_threads <= 1.
    const int n_threads = resolve_omp_threads(cfg.parallel_threads);
#ifdef USE_OMP
    printf("OpenMP: parallel_threads=%d (config=%d, SLURM_CPUS_PER_TASK=%s) → using %d threads\n",
           n_threads, cfg.parallel_threads,
           std::getenv("SLURM_CPUS_PER_TASK") ? std::getenv("SLURM_CPUS_PER_TASK") : "unset",
           n_threads);
#else
    if (cfg.parallel_threads > 1) {
        printf("NOTE: parallel_threads=%d in config but binary built without USE_OMP — running serial\n",
               cfg.parallel_threads);
    }
#endif

    // Set policy modes (fixed for this benchmark)
    set_policy_modes(kind);
    RC_INWARD_REFLECTION = cfg.inward_reflection;
    RC_EPSILON_RELATIVE  = cfg.epsilon_relative;
    RC_WMP_RELATIVE      = cfg.wmp_relative;
    RC_WMP_NO_QUEUE      = cfg.wmp_no_queue;
    RC_WMP_EXTRA_RELATIVE = cfg.wmp_extra_relative;
    RC_PRINT_DETAILED_INFO = cfg.print_detailed_info;
    printf("  print_detailed_info: %d (%s)\n",
           cfg.print_detailed_info,
           cfg.print_detailed_info ? "full: cost splits + outsource_frac"
                                    : "lean: skip cost splits + outsource_frac");
    // ---- Validate use_shared_parameter ----
    if (cfg.use_shared_parameter != 0 &&
        cfg.use_shared_parameter != 1 &&
        cfg.use_shared_parameter != 2 &&
        cfg.use_shared_parameter != 3) {
        fprintf(stderr, "ERROR: use_shared_parameter must be 0, 1, 2, or 3, got %d\n",
                cfg.use_shared_parameter);
        exit(1);
    }
    {
        const double mc_ratio_dbg = OUTSOURCE_COST_RATE[N_ROUTES-1] / OUTSOURCE_COST_RATE[0];
        const double qb_ratio_dbg = QUEUE_LEN_BOUND   [N_ROUTES-1] / QUEUE_LEN_BOUND   [0];
        printf("  pricing_magnitude_scale_by_mc_eq: %d  (BCP last scale=%g, HL last scale=%g)\n",
               cfg.pricing_magnitude_scale_by_mc_eq,
               mc_ratio_dbg / qb_ratio_dbg,
               mc_ratio_dbg);
        const char* shared_mode_desc;
        switch (cfg.use_shared_parameter) {
            case 1: shared_mode_desc = "shared-param 1-round, NO mc-ratio scaling, under pricing/shared_common/";
                    break;
            case 2: shared_mode_desc = "shared-param 1-round, WITH mc-ratio scaling, under pricing/shared_mc_eq/";
                    break;
            case 3: shared_mode_desc = "shared-param POOLED (variants 1 + 2 combined), under pricing/shared_pooled/";
                    break;
            default: shared_mode_desc = "per-route N_ROUTES-round coordinate descent (under pricing/round{i}/)";
        }
        printf("  use_shared_parameter: %d  (%s)\n",
               cfg.use_shared_parameter, shared_mode_desc);
        if (cfg.use_shared_parameter == 1 || cfg.use_shared_parameter == 2) {
            // Print BCP per-route phi_own / phi_other scales (paper D.63/D.65).
            const auto& B_others_dbg = get_B_others_vec();
            printf("    → BCP per-route scales (paper D.63/D.65), variant %d:\n",
                   cfg.use_shared_parameter);
            for (int r = 0; r < N_ROUTES; ++r) {
                const double mc_factor = (cfg.use_shared_parameter == 2)
                                          ? OUTSOURCE_COST_RATE[r] / OUTSOURCE_COST_RATE[0]
                                          : 1.0;
                const double own_s   = (B_VEC[0] / B_VEC[r])        * mc_factor;
                const double other_s = (B_VEC[0] / B_others_dbg[r]) * mc_factor;
                printf("        r=%2d  own=%.4g  other=%.4g  (β_0/β_r=%.4g, β_0/β_{-r}=%.4g, ν_r/ν_0=%.4g)\n",
                       r, own_s, other_s,
                       B_VEC[0] / B_VEC[r],
                       B_VEC[0] / B_others_dbg[r],
                       (cfg.use_shared_parameter == 2)
                           ? OUTSOURCE_COST_RATE[r] / OUTSOURCE_COST_RATE[0] : 1.0);
            }
            if (cfg.use_shared_parameter == 1) {
                printf("    → HL  last-route zeta scale = 1.0 (no scaling)\n");
            } else {
                printf("    → HL  last-route zeta scale = mc_H/mc_V = %g\n", mc_ratio_dbg);
            }
        } else if (cfg.use_shared_parameter == 3) {
            // Pooled: print BOTH variant 1 and variant 2 per-route scale tables.
            const auto& B_others_dbg = get_B_others_vec();
            for (int v : {1, 2}) {
                printf("    → BCP per-route scales (paper D.63/D.65), variant %d:\n", v);
                for (int r = 0; r < N_ROUTES; ++r) {
                    const double mc_factor = (v == 2)
                                              ? OUTSOURCE_COST_RATE[r] / OUTSOURCE_COST_RATE[0]
                                              : 1.0;
                    const double own_s   = (B_VEC[0] / B_VEC[r])        * mc_factor;
                    const double other_s = (B_VEC[0] / B_others_dbg[r]) * mc_factor;
                    printf("        r=%2d  own=%.4g  other=%.4g\n", r, own_s, other_s);
                }
            }
            printf("    → HL  v1 last-route zeta scale = 1.0\n");
            printf("    → HL  v2 last-route zeta scale = mc_H/mc_V = %g\n", mc_ratio_dbg);
            printf("    → Pooled CSV: first half = v1 rows, second half = v2 rows;\n");
            printf("                 find_min_cost picks best across the combined pool.\n");
        }
        if (cfg.use_shared_parameter != 0 && cfg.pricing_magnitude_scale_by_mc_eq != 0) {
            printf("  WARNING: pricing_magnitude_scale_by_mc_eq=%d is IGNORED because\n",
                   cfg.pricing_magnitude_scale_by_mc_eq);
            printf("           use_shared_parameter=%d (last-route scaling is inherent to the variant).\n",
                   cfg.use_shared_parameter);
        }
        if (cfg.pricing_magnitude_scale_by_mc_eq || cfg.use_shared_parameter != 0) {
            printf("  NOTE: resume is NOT folder-name-encoded by these flags; if you flipped\n");
            printf("        either flag (or changed the per-route shared scaling rule)\n");
            printf("        since the last run, delete the relevant pricing/ subdir\n");
            printf("        (round{i}/, shared_common/, shared_mc_eq/, or shared_pooled/)\n");
            printf("        under each rho_folder or stale CSVs will be reused.\n");
        }
    }

    // ---- Validate iterative_pricing_scheduling ----
    if (cfg.iterative_pricing_scheduling != 0 && cfg.iterative_pricing_scheduling != 1) {
        fprintf(stderr, "ERROR: iterative_pricing_scheduling must be 0 or 1, got %d\n",
                cfg.iterative_pricing_scheduling);
        exit(1);
    }
    if (cfg.iterative_pricing_scheduling == 1 && cfg.iterative_pricing_scheduling_num < 1) {
        fprintf(stderr, "ERROR: iterative_pricing_scheduling_num must be >= 1 (got %d)\n",
                cfg.iterative_pricing_scheduling_num);
        exit(1);
    }
    if (cfg.iterative_pricing_scheduling == 1) {
        printf("  iterative_pricing_scheduling: 1  (max iterations = %d)\n",
               cfg.iterative_pricing_scheduling_num);
        if (kind.has_sched_search) {
            printf("    → per-iteration output: {rho_folder}/iteration{N}/{pricing,scheduling}/\n");
            printf("    → final iteration's folder renamed to iteration_last/\n");
            printf("    → terminates on: (a) no feasible solution, (b) sched optimal unchanged,\n");
            printf("                     or (c) iter == %d\n",
                   cfg.iterative_pricing_scheduling_num);
        } else {
            printf("    NOTE: benchmark %s has no WMP scheduling stage; flag has NO effect.\n",
                   cfg.benchmark_name.c_str());
        }
    } else {
        printf("  iterative_pricing_scheduling: 0  (single pricing→scheduling pass)\n");
    }
    printf("  inward_reflection: %d (%s)\n",
           cfg.inward_reflection,
           cfg.inward_reflection ? "inward band" : "hold_last outward band");
    printf("  epsilon_relative:  %d (%s)\n",
           cfg.epsilon_relative,
           cfg.epsilon_relative ? "relative FACE_CONST×(rho±eps)" : "absolute ±eps×scaling");
    printf("  wmp_relative:      %d (%s)\n",
           cfg.wmp_relative,
           cfg.wmp_relative ? "relative w=mu/b" : "absolute w=mu");
    printf("  wmp_no_queue:      %d (%s)\n",
           cfg.wmp_no_queue,
           cfg.wmp_no_queue == 2 ? "v3 normalized (Z-β)/(B*rho)" :
           cfg.wmp_no_queue == 1 ? "v2 pressure-only λ*(Z-β)" :
                                   "v1 full Z+λ*(Z-β)");
    printf("  wmp_extra_relative:%d (%s)%s\n",
           cfg.wmp_extra_relative,
           cfg.wmp_extra_relative ? "pressure normalized by (B*rho) in front & back[K-1]"
                                  : "original",
           (kind.sched == BmkScheduling::WMP)
               ? (cfg.wmp_extra_relative ? "  → output prefix: \"rel_\""
                                         : "  → output prefix: \"abs_\"")
               : "  (no prefix — non-WMP benchmark)");

    // Epsilon is only searched for SinglePushing benchmarks (A1-A4, C1-C2).
    // ThresholdBased benchmarks (B1-B4, D1-D2) use a fixed epsilon=0.25.
    bool has_epsilon_search = (kind.outsource == BmkOutsource::SinglePushing);
    std::vector<double> effective_epsilon_list = has_epsilon_search
        ? cfg.epsilon_list
        : std::vector<double>{0.25};

    // Folder for this benchmark under output_dir
    std::string bmk_folder = cfg.output_dir + "/benchmark" + cfg.benchmark_name;
    mkdir_p(bmk_folder);

    clock_t t_wall_start = clock();
    BMK_PROF_TIC(bmk_total);

    // Collect all results for global optimal across safety × epsilon × rho
    std::vector<BmkRecord> global_records;

    for (double safety : cfg.safety_list) {
        RC_SAFETY_STOCK = safety;

    for (double epsilon : effective_epsilon_list) {
        RC_EPSILON_BAND = epsilon;

    for (double rho : cfg.rho_list) {
        RC_OUTSOURCE_THRESHOLD = rho;

        // WMP-scheduling benchmarks (C1/C2/D1/D2) get an "abs_" / "rel_" prefix
        // on the rho-folder so different wmp_extra_relative settings don't mix.
        // A/B benchmarks (LRS_*) are unaffected.
        const bool is_wmp_sched = (kind.sched == BmkScheduling::WMP);
        const char* wmp_prefix = is_wmp_sched
            ? ((cfg.wmp_extra_relative == 0) ? "abs_" : "rel_")
            : "";
        std::string se_rho_folder = bmk_folder
            + "/" + wmp_prefix + "rho" + fmt_param(rho)
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

        // -------------------------------------------------------------------
        // Decide path: iterative pricing↔scheduling (C/D only) vs single-pass.
        // -------------------------------------------------------------------
        const bool use_iterative = (cfg.iterative_pricing_scheduling == 1)
                                && kind.has_sched_search;

        if (use_iterative) {
            // ---------------------------------------------------------------
            // Iterative pricing↔scheduling within this (safety, eps, rho).
            // ---------------------------------------------------------------
            //   iter 1     : sched_init = cfg.sched_init_*
            //   iter k > 1 : sched_init = iter (k-1)'s optimal sched params
            // Stop on: (a) no feasible sched solution, (b) sched optimal
            // equal to previous iter's, (c) iter == max.
            // Each iter writes under {se_rho_folder}/iteration{k}/{pricing,
            // scheduling}/; on termination, that folder is renamed to
            // {se_rho_folder}/iteration_last/.
            // Only the final iteration's sched records feed global_records.
            // ---------------------------------------------------------------
            const int max_iter = cfg.iterative_pricing_scheduling_num;
            const double br2_init = effective_sched_init_br2(cfg);
            double sched_lam = cfg.sched_init_lambda_tilde;
            double sched_br1 = cfg.sched_init_beta_rate_1;
            double sched_br2 = br2_init;
            bool   have_prev_sched = false;
            double prev_lam = 0.0, prev_br1 = 0.0, prev_br2 = 0.0;
            constexpr double SCHED_EPS = 1e-9;

            printf("\n=== Iterative pricing↔scheduling (max %d iters) ===\n", max_iter);

            std::vector<BmkRecord> sched_recs_last;   // populated on termination
            int  iter        = 1;
            int  iter_done   = 0;                     // iteration index that triggered termination
            const char* terminate_reason = nullptr;

            for (; iter <= max_iter; ++iter) {
                std::string iter_folder = se_rho_folder + "/iteration" + std::to_string(iter);
                mkdir_p(iter_folder);

                printf("[Iter %d] sched_init: lambda=%.4g  br1=%.4g  br2=%.4g  (%s)\n",
                       iter, sched_lam, sched_br1, sched_br2,
                       (iter == 1) ? "from cfg.sched_init_*"
                                   : "from previous iter's optimal");

                // Apply scheduling init for this iteration's pricing search.
                RC_LAMBDA_TILDE = sched_lam;
                RC_BETA_TILDE_V = sched_br1 * BETA_CONST_V * rho;
                RC_BETA_TILDE_H = sched_br2 * BETA_CONST_H * rho;

                // ----- Stage 1: pricing -----
                std::vector<BmkRecord> pricing_recs_iter;
                std::vector<double>    pricing_flat_opt;
                if (kind.pricing == BmkPricing::BCPlinear) {
                    pricing_flat_opt = bcplinear_pricing_search(
                        kind, rho, cfg, iter_folder, pricing_recs_iter, n_threads);
                } else {
                    pricing_flat_opt = highlow_pricing_search(
                        kind, rho, cfg, iter_folder, pricing_recs_iter, n_threads);
                }

                // ----- Stage 2: scheduling (apply pricing first) -----
                if (kind.pricing == BmkPricing::BCPlinear) {
                    double flat[N_ROUTES * 2];
                    for (int k = 0; k < N_ROUTES * 2; ++k) flat[k] = pricing_flat_opt[k];
                    set_bcp_pricing_phi(flat);
                } else {
                    double flat[N_ROUTES * 3];
                    for (int k = 0; k < N_ROUTES * 3; ++k) flat[k] = pricing_flat_opt[k];
                    set_hl_pricing_para(flat);
                }
                std::vector<BmkRecord> sched_recs = wmp_scheduling_search(
                    kind, rho, pricing_flat_opt, cfg, iter_folder, n_threads);

                // ----- Find feasible best from this iter's scheduling -----
                const BmkRecord* best = find_min_cost(sched_recs, cfg.delay_constraint, true);
                if (best == nullptr) {
                    printf("[Iter %d] No feasible scheduling combo → terminating.\n", iter);
                    sched_recs_last  = std::move(sched_recs);
                    iter_done        = iter;
                    terminate_reason = "no_feasible";
                    break;
                }

                // ----- Decode (lam, br_v, br_h) from best.params -----
                // wmp_scheduling_search packs (lam, bt_v, bt_h) right after the
                // flat pricing block (see model_para construction in that fn).
                const int pricing_flat_sz = (kind.pricing == BmkPricing::BCPlinear)
                                              ? N_ROUTES * 2 : N_ROUTES * 3;
                const double cur_lam = best->params[pricing_flat_sz + 0];
                const double cur_btv = best->params[pricing_flat_sz + 1];
                const double cur_bth = best->params[pricing_flat_sz + 2];
                const double cur_br1 = cur_btv / (BETA_CONST_V * rho);
                const double cur_br2 = cur_bth / (BETA_CONST_H * rho);
                printf("[Iter %d] sched_optimal: lambda=%.4g  br1=%.4g  br2=%.4g\n",
                       iter, cur_lam, cur_br1, cur_br2);

                // ----- Convergence check (against previous iter) -----
                if (have_prev_sched
                    && std::abs(cur_lam - prev_lam) < SCHED_EPS
                    && std::abs(cur_br1 - prev_br1) < SCHED_EPS
                    && std::abs(cur_br2 - prev_br2) < SCHED_EPS) {
                    printf("[Iter %d] sched_optimal unchanged vs iter %d → converged, terminating.\n",
                           iter, iter - 1);
                    sched_recs_last  = std::move(sched_recs);
                    iter_done        = iter;
                    terminate_reason = "converged";
                    break;
                }
                prev_lam = cur_lam; prev_br1 = cur_br1; prev_br2 = cur_br2;
                have_prev_sched = true;

                // Update sched_init for next iter.
                sched_lam = cur_lam;
                sched_br1 = cur_br1;
                sched_br2 = cur_br2;

                if (iter == max_iter) {
                    printf("[Iter %d] reached iterative_pricing_scheduling_num=%d → terminating.\n",
                           iter, max_iter);
                    sched_recs_last  = std::move(sched_recs);
                    iter_done        = iter;
                    terminate_reason = "max_iter";
                    break;
                }
            }

            // ----- Rename final iteration folder to iteration_last/ -----
            // If iteration_last/ already exists (stale from a previous run),
            // back it up to iteration_last_old/ first to avoid collision.
            {
                std::string finished = se_rho_folder + "/iteration"
                                       + std::to_string(iter_done);
                std::string last     = se_rho_folder + "/iteration_last";
                struct stat st_buf;
                if (stat(last.c_str(), &st_buf) == 0) {
                    std::string backup = se_rho_folder + "/iteration_last_old";
                    // Remove any prior backup, then rename.
                    if (stat(backup.c_str(), &st_buf) == 0) {
                        // best-effort cleanup via system() (portable enough)
                        std::string cmd = "rm -rf '" + backup + "'";
                        if (system(cmd.c_str()) != 0) {
                            fprintf(stderr,
                                "WARNING: failed to remove stale backup %s\n",
                                backup.c_str());
                        }
                    }
                    if (rename(last.c_str(), backup.c_str()) != 0) {
                        fprintf(stderr,
                            "WARNING: failed to back up existing %s → %s\n",
                            last.c_str(), backup.c_str());
                    } else {
                        printf("[Iter %d] backed up existing iteration_last/ → iteration_last_old/\n",
                               iter_done);
                    }
                }
                if (rename(finished.c_str(), last.c_str()) != 0) {
                    fprintf(stderr,
                        "WARNING: failed to rename %s → %s (reason='%s'); "
                        "iteration folder left in place.\n",
                        finished.c_str(), last.c_str(),
                        terminate_reason ? terminate_reason : "?");
                } else {
                    printf("[Iter %d] iteration%d/ → iteration_last/  (reason: %s)\n",
                           iter_done, iter_done,
                           terminate_reason ? terminate_reason : "?");
                }
            }

            // Feed only the final iteration's scheduling records into the
            // global aggregator (so the per-benchmark optimal_parameter.csv
            // reflects the converged combo, not intermediate iters).
            for (auto& r : sched_recs_last) global_records.push_back(r);

        } else if (kind.has_sched_search) {
            // ---------------------------------------------------------------
            // C/D benchmark, non-iterative path (single pricing + scheduling).
            // ---------------------------------------------------------------
            // Reset WMP scheduling params to sched_init values for this rho.
            // These are held fixed during pricing search; overwritten by scheduling search.
            // br2_eff honors beta_rate_symmetric (see effective_sched_init_br2).
            const double br2_eff = effective_sched_init_br2(cfg);
            RC_LAMBDA_TILDE = cfg.sched_init_lambda_tilde;
            RC_BETA_TILDE_V = cfg.sched_init_beta_rate_1 * BETA_CONST_V * rho;
            RC_BETA_TILDE_H = br2_eff                    * BETA_CONST_H * rho;

            std::vector<BmkRecord> pricing_recs_this_combo;
            std::vector<double> pricing_flat_opt;
            if (kind.pricing == BmkPricing::BCPlinear) {
                pricing_flat_opt = bcplinear_pricing_search(
                    kind, rho, cfg, se_rho_folder, pricing_recs_this_combo, n_threads);
            } else {
                pricing_flat_opt = highlow_pricing_search(
                    kind, rho, cfg, se_rho_folder, pricing_recs_this_combo, n_threads);
            }

            // Apply best pricing params before scheduling search
            if (kind.pricing == BmkPricing::BCPlinear) {
                double flat[N_ROUTES * 2];
                for (int k = 0; k < N_ROUTES * 2; k++) flat[k] = pricing_flat_opt[k];
                set_bcp_pricing_phi(flat);
            } else {
                double flat[N_ROUTES * 3];
                for (int k = 0; k < N_ROUTES * 3; k++) flat[k] = pricing_flat_opt[k];
                set_hl_pricing_para(flat);
            }

            std::vector<BmkRecord> sched_recs = wmp_scheduling_search(
                kind, rho, pricing_flat_opt, cfg, se_rho_folder, n_threads);

            for (auto& r : sched_recs) global_records.push_back(r);

        } else {
            // ---------------------------------------------------------------
            // A/B benchmark: no scheduling stage.
            // ---------------------------------------------------------------
            const double br2_eff = effective_sched_init_br2(cfg);
            RC_LAMBDA_TILDE = cfg.sched_init_lambda_tilde;
            RC_BETA_TILDE_V = cfg.sched_init_beta_rate_1 * BETA_CONST_V * rho;
            RC_BETA_TILDE_H = br2_eff                    * BETA_CONST_H * rho;

            std::vector<BmkRecord> pricing_recs_this_combo;
            std::vector<double> pricing_flat_opt;
            if (kind.pricing == BmkPricing::BCPlinear) {
                pricing_flat_opt = bcplinear_pricing_search(
                    kind, rho, cfg, se_rho_folder, pricing_recs_this_combo, n_threads);
            } else {
                pricing_flat_opt = highlow_pricing_search(
                    kind, rho, cfg, se_rho_folder, pricing_recs_this_combo, n_threads);
            }
            // A/B: accumulate pricing records for global optimal
            for (auto& r : pricing_recs_this_combo) global_records.push_back(r);
        }
    } // rho
    } // epsilon
    } // safety

    // -----------------------------------------------------------------------
    // Global optimal across all rho
    // -----------------------------------------------------------------------
    // ---- Top-level optimal_parameter.csv writing is disabled ----
    // (Per user request 2026-05-11: stop saving the per-benchmark global CSV
    //  at {bmk_folder}/{prefix}optimal_parameter.csv; keep stdout banner +
    //  inline print of the best row. Sub-level CSVs under rho_folder/pricing/,
    //  scheduling/, iteration_last/ are unaffected.)
    // const bool is_wmp_sched_global = (kind.sched == BmkScheduling::WMP);
    // const char* wmp_prefix_global = is_wmp_sched_global
    //     ? ((cfg.wmp_extra_relative == 0) ? "abs_" : "rel_")
    //     : "";
    // std::string global_opt_path = bmk_folder + "/" + wmp_prefix_global + "optimal_parameter.csv";

    printf("\n======================================================\n");
    printf("Global optimal across all safety × epsilon × rho (benchmark %s):\n",
           cfg.benchmark_name.c_str());
    printf("======================================================\n");

    const BmkRecord* global_best = find_min_cost(global_records, cfg.delay_constraint, true);

    // Replaced print_and_save_optimal(...) with stdout-only inline print to
    // avoid writing the file (mirrors the printf block inside the helper):
    // print_and_save_optimal(global_best, cfg.delay_constraint, global_opt_path, "global");
    printf("\n  [optimal global]");
    if (global_best == nullptr) {
        printf(" No feasible row (all delays < %.2f%%) found.\n", cfg.delay_constraint);
    } else {
        printf(" min-cost feasible row:\n");
        printf("    params:");
        for (double p : global_best->params) printf(" %.4g", p);
        printf("\n    cost=%.4f  std=%.4f  delay=[", global_best->cost_mean, global_best->cost_std);
        for (int r = 0; r < N_ROUTES; ++r)
            printf("%.3f%%%s", global_best->delay[r], r < N_ROUTES-1 ? ", " : "");
        printf("]\n");
    }

    BMK_PROF_TOC(bmk_total, g_bmk_prof_total);
#ifdef ENABLE_PROFILING
    // Derive "other" = total - (sim + csv + findmin + io_resume)
    // In case of tiny rounding we clamp to 0.
    uint64_t measured = g_bmk_prof_sim + g_bmk_prof_save_csv
                      + g_bmk_prof_findmin + g_bmk_prof_io_resume;
    g_bmk_prof_other = (g_bmk_prof_total > measured)
                      ? (g_bmk_prof_total - measured) : 0;
#endif
    clock_t t_wall_end = clock();
    double total_elapsed = (double)(t_wall_end - t_wall_start) / CLOCKS_PER_SEC;
    printf("\n=== Benchmark search complete ===\n");
    printf("Total wall-clock time: %.2f seconds\n", total_elapsed);
    return 0;
}
