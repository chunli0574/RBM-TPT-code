/*
 * proposed_policy_main.cpp
 *
 * Parameter search over 6 dimensions for the proposed NN policy
 * (supports CC3 dim_K=3 and CC10 dim_K=10 via DIM_K_SELECT compile flag):
 *   safety, epsilon, rho, lambda_tilde, beta_rate_1 (vertical), beta_rate_2 (horizontal)
 *
 * For each combo:
 *   1. Set RC_* runtime globals (used by policy.cpp).
 *   2. Reload the NN model when rho changes (rho selects the model file).
 *   3. Call run_simulation() and record cost / delay probs for all N_ROUTES routes.
 *
 * beta_tilde_v = beta_rate_1 * BETA_CONST_V * rho
 * beta_tilde_h = beta_rate_2 * BETA_CONST_H * rho
 * (Python: process_scheduling_params in functions_general.py)
 *
 * Results per (rho, safety, epsilon) block are saved to:
 *   {output_dir}/rho{r}_safety{s}_epsilon{e}/CC3_final_results.csv
 * with columns:
 *   param_1, param_2, param_3, param_4, cost, std,
 *   delay1..delayN, hardstop1..hardstopN
 * where param_1=lambda_tilde, param_2=beta_tilde_v,
 *       param_3=beta_tilde_h, param_4=rho
 *
 * After all combos, the min-cost feasible row (all N_ROUTES delay probs < constraint)
 * is printed and saved to: {output_dir}/optimal_parameter.csv
 *
 * Configuration is read from proposed_policy_config.txt (same directory as binary).
 *
 * TO BUILD (CC3):   make sim_proposed
 * TO BUILD (CC10):  make DIM_K_SELECT=10 sim_proposed
 * TO RUN:           make run_proposed
 */

#include "params.h"
#include "run_config.h"
#include "simulation.h"
#include "nn_model.h"

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
#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>   // mkdir

#include "model_path.h"   // shared NN-model path / eta_theta maps

#ifdef USE_OMP
#include <omp.h>
#endif

// NOTE on MKL nested threading:
// When running combo-level OMP parallel for, each worker thread will call
// nn_predict_batch which (in USE_BLAS builds) invokes cblas_sgemm/sgemv.
// To prevent MKL from spawning its own thread pool on top of OMP, set
// MKL_NUM_THREADS=1 in the submit script (mirrors sim_benchmark_omp's
// approach). MKL respects that env var at runtime.

// fmt_rho / fmt_rho_3dec / rho_3dec_key — moved to model_path.{h,cpp}.
// (No local callers of fmt_rho remain; helper alias removed to silence warning.)

// make_model_path — moved to model_path.{h,cpp} (CC3 & CC10 branches).

// ---------------------------------------------------------------------------
// Config file parser
// ---------------------------------------------------------------------------
struct ProposedConfig {
    std::vector<double> safety_list;
    std::vector<double> epsilon_list;
    std::vector<double> rho_list;
    // WMP scheduling search lists (auto-selected by wmp_extra_relative):
    //   wmp_extra_relative=0 → *_abs lists
    //   wmp_extra_relative=1 → *_rel lists
    // After parsing, resolve_wmp_lists() copies the chosen variant into the
    // effective fields below. Validation in main() errors out if the resulting
    // effective list is empty.
    std::vector<double> lambda_tilde_list;     // effective (resolved)
    std::vector<double> beta_rate_1_list;
    std::vector<double> beta_rate_2_list;
    std::vector<double> lambda_tilde_list_abs; // parsed config source
    std::vector<double> beta_rate_1_list_abs;
    std::vector<double> beta_rate_2_list_abs;
    std::vector<double> lambda_tilde_list_rel;
    std::vector<double> beta_rate_1_list_rel;
    std::vector<double> beta_rate_2_list_rel;
    // beta_rate_symmetric: 0 = use separate beta_rate_1 and beta_rate_2 lists (default)
    //                       1 = beta_rate_2 always equals beta_rate_1 (beta_rate_2_list ignored)
    //                       2 = beta_rate_2 = 1-(1-beta_rate_1)*B_V/B_H (beta_rate_2_list ignored)
    int    beta_rate_symmetric = 0;
    double delay_constraint  = 1.0;
    int    inward_reflection = 0;    // 0 = hold_last (outward band), 1 = inward band
    // epsilon_relative: 0=absolute band (±eps×scaling), 1=relative band (FACE_CONST×(rho±eps))
    int    epsilon_relative  = 0;
    // wmp_relative: 0=absolute w=mu_arr (old), 1=relative w=mu_arr/b_arr (new)
    int    wmp_relative      = 0;
    // wmp_no_queue: 0=v1 full pressure (Z+λ*(Z-β)), 1=v2 no-queue pressure (λ*(Z-β) only)
    int    wmp_no_queue      = 0;
    // wmp_extra_relative: 0=plain v1; 1=normalize v1 pressure term by (B_V*rho) / (B_H*rho)
    int    wmp_extra_relative = 0;
    std::string output_dir      = "proposed_policy_results";
    std::string model_base_dir;
    // new_model_base_dir: base directory for new-format (.weights.bin) NN models.
    // Required when any rho in rho_list is in NEW_FORMAT_RHOS.
    std::string new_model_base_dir;
    // print_detailed_info: 1=full CSV including hardstop/pricing_cost/
    // outsourcing_cost columns; 0=lean CSV and the simulation skips
    // accumulating those splits.
    int    print_detailed_info = 0;
    // Parallel threads for proposed-policy inner combo loops (USE_OMP build only).
    // 0 or 1 = serial; N > 1 uses min(N, SLURM_CPUS_PER_TASK) threads.
    // Used by sim_proposed_batch_omp (CC10 BATCH_NN+USE_BLAS+USE_OMP); the plain
    // serial sim_proposed / sim_proposed_batch binaries silently treat any value
    // > 1 as 1 (a notice is printed at startup).
    int    parallel_threads  = 1;
};

static std::vector<double> parse_double_list(const std::string& s) {
    std::vector<double> v;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t a = token.find_first_not_of(" \t\r\n");
        size_t b = token.find_last_not_of(" \t\r\n");
        if (a != std::string::npos) {
            v.push_back(std::stod(token.substr(a, b - a + 1)));
        }
    }
    return v;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// Resolve WMP scheduling lists based on wmp_extra_relative.
// Copies the chosen variant's *_abs / *_rel list into the effective list.
// If the chosen variant's list is empty (not provided in config), the effective
// list stays empty — caught by validation in main().
static void resolve_wmp_lists(ProposedConfig& cfg) {
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

static ProposedConfig load_config(const std::string& path) {
    ProposedConfig cfg;
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

        if      (key == "safety_list")       cfg.safety_list       = parse_double_list(val);
        else if (key == "epsilon_list")      cfg.epsilon_list      = parse_double_list(val);
        else if (key == "rho_list")          cfg.rho_list          = parse_double_list(val);
        else if (key == "lambda_tilde_list_abs") cfg.lambda_tilde_list_abs = parse_double_list(val);
        else if (key == "beta_rate_1_list_abs")  cfg.beta_rate_1_list_abs  = parse_double_list(val);
        else if (key == "beta_rate_2_list_abs")  cfg.beta_rate_2_list_abs  = parse_double_list(val);
        else if (key == "lambda_tilde_list_rel") cfg.lambda_tilde_list_rel = parse_double_list(val);
        else if (key == "beta_rate_1_list_rel")  cfg.beta_rate_1_list_rel  = parse_double_list(val);
        else if (key == "beta_rate_2_list_rel")  cfg.beta_rate_2_list_rel  = parse_double_list(val);
        else if (key == "beta_rate_symmetric") cfg.beta_rate_symmetric = std::stoi(val);
        else if (key == "delay_constraint")    cfg.delay_constraint    = std::stod(val);
        else if (key == "output_dir")         cfg.output_dir         = val;
        else if (key == "model_base_dir")     cfg.model_base_dir     = val;
        else if (key == "new_model_base_dir") cfg.new_model_base_dir = val;
        else if (key == "inward_reflection")  cfg.inward_reflection  = std::stoi(val);
        else if (key == "epsilon_relative")   cfg.epsilon_relative   = std::stoi(val);
        else if (key == "wmp_relative")       cfg.wmp_relative       = std::stoi(val);
        else if (key == "wmp_no_queue")            cfg.wmp_no_queue            = std::stoi(val);
        else if (key == "wmp_extra_relative")     cfg.wmp_extra_relative      = std::stoi(val);
        else if (key == "print_detailed_info")    cfg.print_detailed_info     = std::stoi(val);
        else if (key == "parallel_threads")        cfg.parallel_threads        = std::stoi(val);
    }
    // Resolve _abs / _rel overrides into effective fields per wmp_extra_relative.
    resolve_wmp_lists(cfg);
    return cfg;
}

// rho_in_map — moved to model_path.{h,cpp}.

// ---------------------------------------------------------------------------
// CSV result record (one row per (lambda_tilde, beta_rate_1, beta_rate_2) combo)
// ---------------------------------------------------------------------------
struct SearchRecord {
    double lambda_tilde;
    double beta_tilde_v;
    double beta_tilde_h;
    double rho;
    double cost_mean;
    double cost_std;
    double pricing_cost_mean;
    double outsourcing_cost_mean;
    double delay[N_ROUTES];     // delay prob (%) for each route
    double delay_std[N_ROUTES]; // delay prob std (%) for each route
    double hardstop[N_ROUTES];  // always 0.0 (not tracked in C++)
};

// ---------------------------------------------------------------------------
// Save results vector to CSV (N_ROUTES-generic)
// ---------------------------------------------------------------------------
static void save_csv(const std::string& filepath,
                     const std::vector<SearchRecord>& recs) {
    FILE* fp = fopen(filepath.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot write CSV: %s\n", filepath.c_str());
        return;
    }
    const bool detailed = (RC_PRINT_DETAILED_INFO != 0);
    // Header: param_1..param_4, cost, std, delay1..delayN, delay1_std..delayN_std
    // + (detailed only) hardstop1..hardstopN, pricing_cost, outsourcing_cost
    fprintf(fp, "param_1,param_2,param_3,param_4,cost,std");
    for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",delay%d", r + 1);
    for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",delay%d_std", r + 1);
    if (detailed) {
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",hardstop%d", r + 1);
        fprintf(fp, ",pricing_cost,outsourcing_cost");
    }
    fprintf(fp, "\n");
    for (const auto& rec : recs) {
        fprintf(fp, "%.6g,%.6g,%.6g,%.6g,%.4f,%.4f",
                rec.lambda_tilde, rec.beta_tilde_v, rec.beta_tilde_h, rec.rho,
                rec.cost_mean, rec.cost_std);
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
}

// ---------------------------------------------------------------------------
// Find and print/save the min-cost feasible row.
// Feasible: ALL N_ROUTES delay probs < delay_constraint.
// Python: find_min_cost_proposed_scheduling (functions_general.py)
// ---------------------------------------------------------------------------
// [[maybe_unused]] because the only call site (top-level global optimal) was
// disabled per user request 2026-05-11; helper preserved for easy re-enable.
[[maybe_unused]] static void find_and_save_optimal(const std::vector<SearchRecord>& recs,
                                  double delay_constraint,
                                  const std::string& optimal_path) {
    const SearchRecord* best = nullptr;
    for (const auto& r : recs) {
        bool feasible = true;
        for (int i = 0; i < N_ROUTES; ++i)
            if (r.delay[i] >= delay_constraint) { feasible = false; break; }
        if (feasible) {
            if (best == nullptr || r.cost_mean < best->cost_mean)
                best = &r;
        }
    }

    if (best == nullptr) {
        printf("\n  [optimal] No feasible row found (delay constraint %.2f%%).\n",
               delay_constraint);
        return;
    }

    printf("\n  [optimal] min-cost feasible row:\n");
    printf("    lambda_tilde=%.4g  beta_tilde_v=%.4g  beta_tilde_h=%.4g  rho=%.4g\n",
           best->lambda_tilde, best->beta_tilde_v, best->beta_tilde_h, best->rho);
    printf("    cost=%.4f  std=%.4f\n", best->cost_mean, best->cost_std);
    printf("    delay=[");
    for (int r = 0; r < N_ROUTES; ++r)
        printf("%.3f%%%s", best->delay[r], r < N_ROUTES-1 ? ", " : "");
    printf("]\n");
    printf("    delay_std=[");
    for (int r = 0; r < N_ROUTES; ++r)
        printf("%.3f%%%s", best->delay_std[r], r < N_ROUTES-1 ? ", " : "");
    printf("]\n");

    FILE* fp = fopen(optimal_path.c_str(), "w");
    if (fp) {
        const bool detailed = (RC_PRINT_DETAILED_INFO != 0);
        // Single data row (no header, matching previous format). Lean mode
        // drops the trailing hardstop/pricing_cost/outsourcing_cost columns.
        fprintf(fp, "%.6g,%.6g,%.6g,%.6g,%.4f,%.4f",
                best->lambda_tilde, best->beta_tilde_v, best->beta_tilde_h, best->rho,
                best->cost_mean, best->cost_std);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.4f", best->delay[r]);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.4f", best->delay_std[r]);
        if (detailed) {
            for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.6g", best->hardstop[r]);
            fprintf(fp, ",%.4f,%.4f",
                    best->pricing_cost_mean, best->outsourcing_cost_mean);
        }
        fprintf(fp, "\n");
        fclose(fp);
        printf("  Saved optimal to: %s\n", optimal_path.c_str());
    }
}

// ---------------------------------------------------------------------------
// resolve_omp_threads
//   Cap the config-requested thread count at SLURM_CPUS_PER_TASK (if set) so
//   we never oversubscribe the cores the scheduler allocated. Returns 1 when
//   OMP is disabled or cfg <= 1. Also sets omp_set_num_threads + omp_set_dynamic(0)
//   at program startup.
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
// mkdir -p equivalent (creates all missing path components)
// ---------------------------------------------------------------------------
static void mkdir_p(const std::string& dir) {
    std::string path;
    for (char c : dir) {
        path += c;
        if (c == '/') {
#ifdef _WIN32
            _mkdir(path.c_str());
#else
            mkdir(path.c_str(), 0755);
#endif
        }
    }
#ifdef _WIN32
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif
}

// ---------------------------------------------------------------------------
// Format a double for use in folder/file names.
// e.g. -1.0 → "n1.0", 0.9 → "0.9", 0.95 → "0.95"
// (negative sign replaced with 'n' for file-name safety)
// ---------------------------------------------------------------------------
static std::string fmt_param(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%g", v);
    std::string s(buf);
    if (!s.empty() && s[0] == '-') s[0] = 'n';
    return s;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string cfg_path = "proposed_policy_config.txt";
    if (argc >= 2) cfg_path = argv[1];

    printf("=== Extended Criss-Cross Grid Search (dim_K=%d, %d paths per combo) ===\n\n",
           DIM_K, N_PATHS);
    printf("Reading config from: %s\n\n", cfg_path.c_str());

    ProposedConfig cfg = load_config(cfg_path);

    // Optional CLI overrides: argv[2]=safety, argv[3]=epsilon, argv[4]=rho,
    //                          argv[5]=variant ("abs"|"rel"|"")
    if (argc >= 3) cfg.safety_list  = {std::stod(argv[2])};
    if (argc >= 4) cfg.epsilon_list = {std::stod(argv[3])};
    if (argc >= 5) cfg.rho_list     = {std::stod(argv[4])};
    if (argc >= 6) {
        std::string variant = argv[5];
        if (variant == "abs") {
            cfg.wmp_extra_relative  = 0;
            cfg.beta_rate_symmetric = 2;
            resolve_wmp_lists(cfg);
            printf("CLI override: variant=abs (wmp_extra_relative=0, beta_rate_symmetric=2)\n");
        } else if (variant == "rel") {
            cfg.wmp_extra_relative  = 1;
            cfg.beta_rate_symmetric = 1;
            resolve_wmp_lists(cfg);
            printf("CLI override: variant=rel (wmp_extra_relative=1, beta_rate_symmetric=1)\n");
        } else if (!variant.empty()) {
            fprintf(stderr, "ERROR: 5th CLI arg must be 'abs', 'rel', or omitted; got '%s'\n",
                    variant.c_str());
            return 1;
        }
    }

    // Validate effective WMP lists (after resolve).
    {
        const char* variant_tag = (cfg.wmp_extra_relative == 1) ? "rel" : "abs";
        auto check = [&](const std::vector<double>& v, const char* name) {
            if (v.empty()) {
                fprintf(stderr,
                    "ERROR: proposed policy (wmp_extra_relative=%d → variant '%s') needs %s_%s\n"
                    "       but the key was not provided in the config.\n",
                    cfg.wmp_extra_relative, variant_tag, name, variant_tag);
                exit(1);
            }
        };
        check(cfg.lambda_tilde_list, "lambda_tilde_list");
        check(cfg.beta_rate_1_list,  "beta_rate_1_list");
        check(cfg.beta_rate_2_list,  "beta_rate_2_list");
    }

    printf("Parameter lists:\n");
    printf("  safety:       ");
    for (double v : cfg.safety_list)       printf("%.4g ", v); printf("\n");
    printf("  epsilon:      ");
    for (double v : cfg.epsilon_list)      printf("%.4g ", v); printf("\n");
    printf("  rho:          ");
    for (double v : cfg.rho_list)          printf("%.4g ", v); printf("\n");
    printf("  lambda_tilde: ");
    for (double v : cfg.lambda_tilde_list) printf("%.4g ", v); printf("\n");
    printf("  beta_rate_1:  ");
    for (double v : cfg.beta_rate_1_list)  printf("%.4g ", v); printf("\n");
    if (cfg.beta_rate_symmetric == 1) {
        printf("  beta_rate_2:  [symmetric — equals beta_rate_1]\n");
    } else if (cfg.beta_rate_symmetric == 2) {
        printf("  beta_rate_2:  [derived — 1-(1-beta_rate_1)*B_V/B_H  (B_V=%.4g, B_H=%.4g)]\n",
               B_V, B_H);
    } else {
        printf("  beta_rate_2:  ");
        for (double v : cfg.beta_rate_2_list) printf("%.4g ", v); printf("\n");
    }

    int n_b2  = (cfg.beta_rate_symmetric != 0) ? 1 : (int)cfg.beta_rate_2_list.size();
    int total = (int)(cfg.safety_list.size() * cfg.epsilon_list.size()
                    * cfg.rho_list.size()    * cfg.lambda_tilde_list.size()
                    * cfg.beta_rate_1_list.size() * n_b2);
    printf("\nTotal combos: %d\n\n", total);

    // Resolve OMP threads (capped by SLURM_CPUS_PER_TASK if set).
    // Returns 1 when USE_OMP is not defined or cfg.parallel_threads <= 1.
    const int n_omp_threads = resolve_omp_threads(cfg.parallel_threads);
#ifdef USE_OMP
    printf("OpenMP: parallel_threads=%d (config=%d, SLURM_CPUS_PER_TASK=%s) → using %d threads\n\n",
           n_omp_threads, cfg.parallel_threads,
           std::getenv("SLURM_CPUS_PER_TASK") ? std::getenv("SLURM_CPUS_PER_TASK") : "unset",
           n_omp_threads);
#else
    if (cfg.parallel_threads > 1) {
        printf("NOTE: parallel_threads=%d in config but binary built without USE_OMP — running serial\n\n",
               cfg.parallel_threads);
    }
    (void)n_omp_threads;
#endif

    RC_INWARD_REFLECTION     = cfg.inward_reflection;
    RC_EPSILON_RELATIVE      = cfg.epsilon_relative;
    RC_WMP_RELATIVE          = cfg.wmp_relative;
    RC_WMP_NO_QUEUE          = cfg.wmp_no_queue;
    RC_WMP_EXTRA_RELATIVE    = cfg.wmp_extra_relative;
    RC_PRINT_DETAILED_INFO   = cfg.print_detailed_info;
    printf("  print_detailed_info=%d (%s)\n",
           cfg.print_detailed_info,
           cfg.print_detailed_info ? "full" : "lean (skip cost splits + outsource_frac)");

    printf("  inward_reflection:       %d (%s)\n",
           cfg.inward_reflection,
           cfg.inward_reflection ? "inward band" : "hold_last outward band");
    printf("  epsilon_relative:        %d (%s)\n",
           cfg.epsilon_relative,
           cfg.epsilon_relative ? "relative FACE_CONST×(rho±eps)" : "absolute ±eps×scaling");
    printf("  wmp_relative:            %d (%s)\n",
           cfg.wmp_relative,
           cfg.wmp_relative ? "relative w=mu/b" : "absolute w=mu");
    printf("  wmp_no_queue:            %d (%s)\n",
           cfg.wmp_no_queue,
           cfg.wmp_no_queue == 2 ? "v3 normalized (Z-β)/(B*rho)" :
           cfg.wmp_no_queue == 1 ? "v2 pressure-only λ*(Z-β)" :
                                   "v1 full Z+λ*(Z-β)");
    printf("  wmp_extra_relative:      %d (%s)  → output prefix: \"%s\"\n\n",
           cfg.wmp_extra_relative,
           cfg.wmp_extra_relative ? "pressure normalized by (B*rho) in front & back[K-1]"
                                  : "original",
           cfg.wmp_extra_relative ? "rel_" : "abs_");

    auto eta_theta     = build_eta_theta_map();
    auto eta_theta_b50 = build_eta_theta_b50_map();

    // -----------------------------------------------------------------------
    // Pre-flight: verify all required NN model files exist before starting.
    // -----------------------------------------------------------------------
    {
        printf("Pre-flight: checking NN model files...\n");
        bool ok = true;
        for (double rho : cfg.rho_list) {
            if (!rho_in_map(rho, eta_theta, eta_theta_b50)) {
                fprintf(stderr, "ERROR: rho=%.4g not found in eta_theta_mapping\n", rho);
                ok = false;
                continue;
            }
            std::string p = make_model_path(rho, eta_theta, eta_theta_b50,
                                            cfg.model_base_dir, cfg.new_model_base_dir);
            if (!nn_model_path_exists(p)) {
                fprintf(stderr, "ERROR: outsourcing model missing for rho=%.4g:\n  %s\n", rho, p.c_str());
                ok = false;
            }
        }
        if (!ok) {
            fprintf(stderr, "Aborting: fix missing model files before running.\n");
            exit(1);
        }
        printf("Pre-flight OK: all model files verified.\n\n");
    }

    mkdir_p(cfg.output_dir);

    clock_t t_wall_start = clock();

    int N_r = (int)cfg.rho_list.size();
    int N_s = (int)cfg.safety_list.size();
    int N_e = (int)cfg.epsilon_list.size();

    std::vector<std::vector<std::vector<SearchRecord>>> all_records(
        N_r, std::vector<std::vector<SearchRecord>>(N_s * N_e));

    double current_rho         = -1.0;   // tracks loaded outsourcing model
    int combo_idx = 0;

    // -----------------------------------------------------------------------
    // Outermost loop over rho: NN model depends only on rho.
    // The W→G cache built for any combo at a given rho is fully reusable for
    // every other combo at the same rho.
    // -----------------------------------------------------------------------
    for (int ri = 0; ri < N_r; ++ri) {
        double rho = cfg.rho_list[ri];

        double rho_key     = rho_3dec_key(rho);
        double current_key = rho_3dec_key(current_rho);
        if (rho_key != current_key) {
            std::string model_path = make_model_path(rho, eta_theta, eta_theta_b50,
                                                     cfg.model_base_dir, cfg.new_model_base_dir);
            printf("===========================================================\n");
            printf("[rho=%.4g] Loading NN model:\n  %s\n", rho, model_path.c_str());
            if (current_rho < 0) {
                nn_model_load(model_path);
            } else {
                nn_model_reload(model_path);
            }
            current_rho = rho;
            printf("NN model loaded.\n");
            printf("===========================================================\n\n");
        }

        RC_OUTSOURCE_THRESHOLD = rho;

        for (int si = 0; si < N_s; ++si) {
            double safety = cfg.safety_list[si];
            for (int ei = 0; ei < N_e; ++ei) {
                double epsilon = cfg.epsilon_list[ei];

                RC_SAFETY_STOCK = safety;
                RC_EPSILON_BAND = epsilon;

                // ----------------------------------------------------------
                // Build a flat combo list for this (rho, safety, epsilon).
                // Inner OMP parallel-for sweeps over it. We keep all outer
                // loops serial because: (a) NN model state depends on rho,
                // (b) outer-loop RC_* (safety, epsilon, outsource_threshold)
                // are NOT thread_local. Only the per-combo RC_LAMBDA_TILDE /
                // RC_BETA_TILDE_V / RC_BETA_TILDE_H are thread_local and
                // safe to write inside the parallel region.
                // ----------------------------------------------------------
                struct ComboPS { double lam, br1, br2, bt_v, bt_h; };
                std::vector<ComboPS> combos;
                combos.reserve(cfg.lambda_tilde_list.size()
                              * cfg.beta_rate_1_list.size()
                              * (cfg.beta_rate_symmetric ? 1 : cfg.beta_rate_2_list.size()));
                for (double lambda_tilde : cfg.lambda_tilde_list) {
                    for (double beta_rate_1 : cfg.beta_rate_1_list) {
                        const std::vector<double> br2_list =
                            (cfg.beta_rate_symmetric == 1)
                                ? std::vector<double>{beta_rate_1}
                            : (cfg.beta_rate_symmetric == 2)
                                ? std::vector<double>{1.0 - (1.0 - beta_rate_1) * B_V / B_H}
                            : cfg.beta_rate_2_list;
                        for (double beta_rate_2 : br2_list) {
                            ComboPS c;
                            c.lam  = lambda_tilde;
                            c.br1  = beta_rate_1;
                            c.br2  = beta_rate_2;
                            c.bt_v = beta_rate_1 * BETA_CONST_V * rho;
                            c.bt_h = beta_rate_2 * BETA_CONST_H * rho;
                            combos.push_back(c);
                        }
                    }
                }
                const int n_combos_block = (int)combos.size();
                std::vector<SearchRecord> block_recs(n_combos_block);
                const int combo_idx_base = combo_idx;
                combo_idx += n_combos_block;

#ifdef USE_OMP
                #pragma omp parallel for schedule(dynamic) \
                    num_threads(n_omp_threads > 1 ? n_omp_threads : 1) \
                    if(n_omp_threads > 1)
#endif
                for (int ci = 0; ci < n_combos_block; ++ci) {
                    const ComboPS& c = combos[ci];

                    RC_LAMBDA_TILDE = c.lam;
                    RC_BETA_TILDE_V = c.bt_v;
                    RC_BETA_TILDE_H = c.bt_h;

                    clock_t t0 = clock();
                    SimResult res = run_simulation();
                    clock_t t1 = clock();
                    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

                    // Build a small stdout block as one string so OMP threads
                    // don't interleave each other's lines. Single printf at
                    // the end keeps output readable when n_omp_threads > 1.
                    char buf[1024];
                    int  off = 0;
                    off += snprintf(buf + off, sizeof(buf) - off,
                        "[combo %d/%d] rho=%.4g  safety=%.4g  epsilon=%.4g  "
                        "lambda_tilde=%.4g  beta_rate_1=%.4g  beta_rate_2=%.4g\n"
                        "          → beta_tilde_v=%.4f  beta_tilde_h=%.4f\n"
                        "  → cost=%.4f ± %.4f  delay=[",
                        combo_idx_base + ci + 1, total,
                        rho, safety, epsilon, c.lam, c.br1, c.br2,
                        c.bt_v, c.bt_h, res.cost_mean, res.cost_std);
                    for (int r = 0; r < N_ROUTES; ++r)
                        off += snprintf(buf + off, sizeof(buf) - off,
                            "%.2f%%%s", res.delay_prob_mean[r],
                            r < N_ROUTES - 1 ? ", " : "");
                    off += snprintf(buf + off, sizeof(buf) - off, "]");
                    if (cfg.print_detailed_info) {
                        off += snprintf(buf + off, sizeof(buf) - off, "  outsource=[");
                        for (int r = 0; r < N_ROUTES; ++r)
                            off += snprintf(buf + off, sizeof(buf) - off,
                                "%.2f%%%s", res.outsource_frac_mean[r] * 100.0,
                                r < N_ROUTES - 1 ? ", " : "");
                        off += snprintf(buf + off, sizeof(buf) - off, "]");
                    }
                    snprintf(buf + off, sizeof(buf) - off,
                        "  time=%.1fs\n", elapsed);
#ifdef USE_OMP
                    #pragma omp critical(stdout_proposed)
#endif
                    { fputs(buf, stdout); fflush(stdout); }

                    SearchRecord rec;
                    rec.lambda_tilde = c.lam;
                    rec.beta_tilde_v = c.bt_v;
                    rec.beta_tilde_h = c.bt_h;
                    rec.rho          = rho;
                    rec.cost_mean             = res.cost_mean;
                    rec.cost_std              = res.cost_std;
                    rec.pricing_cost_mean     = res.pricing_cost_mean;
                    rec.outsourcing_cost_mean = res.outsourcing_cost_mean;
                    for (int r = 0; r < N_ROUTES; ++r) {
                        rec.delay[r]     = res.delay_prob_mean[r];
                        rec.delay_std[r] = res.delay_prob_std[r];
                        rec.hardstop[r]  = 0.0;
                    }
                    block_recs[ci] = rec;
                }

                all_records[ri][si * N_e + ei] = std::move(block_recs);

                // Save CSV for this (rho, safety, epsilon) block immediately.
                // Subfolder gets a "abs_" / "rel_" prefix per wmp_extra_relative
                // (0 = original/absolute pressure formula, 1 = (B*rho)-normalized).
                const char* wmp_prefix = (cfg.wmp_extra_relative == 0) ? "abs_" : "rel_";
                std::string subfolder = cfg.output_dir
                    + "/" + wmp_prefix + "rho" + fmt_param(rho)
                    + "_safety" + fmt_param(safety)
                    + "_epsilon" + fmt_param(epsilon);
                mkdir_p(subfolder);
                std::string csv_path = subfolder + "/" SYSTEM_NAME "_final_results.csv";
                printf("\n");
                save_csv(csv_path, all_records[ri][si * N_e + ei]);

            } // epsilon
        } // safety

    } // rho

    // After all rho: find global optimal across ALL (rho, safety, epsilon)
    std::vector<SearchRecord> global_merged;
    for (int ri = 0; ri < N_r; ++ri)
        for (int si = 0; si < N_s; ++si)
            for (int ei = 0; ei < N_e; ++ei) {
                const auto& v = all_records[ri][si * N_e + ei];
                global_merged.insert(global_merged.end(), v.begin(), v.end());
            }

    // ---- Top-level optimal_parameter.csv writing is disabled ----
    // (Per user request 2026-05-11: stop saving the global CSV at
    //  {output_dir}/{prefix}optimal_parameter.csv; keep stdout banner +
    //  inline print of the best row. Per-combo subfolder CSVs are unaffected.)
    // const char* wmp_prefix_global = (cfg.wmp_extra_relative == 0) ? "abs_" : "rel_";
    // std::string global_optimal_path = cfg.output_dir + "/" + wmp_prefix_global + "optimal_parameter.csv";
    printf("\n======================================================\n");
    printf("Global optimal across all rho, safety, epsilon:\n");
    printf("======================================================\n");

    // Replaced find_and_save_optimal(...) with stdout-only inline equivalent
    // (mirrors the find + printf block inside the helper, no fopen):
    // find_and_save_optimal(global_merged, cfg.delay_constraint, global_optimal_path);
    {
        const SearchRecord* best = nullptr;
        for (const auto& r : global_merged) {
            bool feasible = true;
            for (int i = 0; i < N_ROUTES; ++i)
                if (r.delay[i] >= cfg.delay_constraint) { feasible = false; break; }
            if (feasible && (best == nullptr || r.cost_mean < best->cost_mean))
                best = &r;
        }
        if (best == nullptr) {
            printf("\n  [optimal] No feasible row found (delay constraint %.2f%%).\n",
                   cfg.delay_constraint);
        } else {
            printf("\n  [optimal] min-cost feasible row:\n");
            printf("    lambda_tilde=%.4g  beta_tilde_v=%.4g  beta_tilde_h=%.4g  rho=%.4g\n",
                   best->lambda_tilde, best->beta_tilde_v, best->beta_tilde_h, best->rho);
            printf("    cost=%.4f  std=%.4f\n", best->cost_mean, best->cost_std);
            printf("    delay=[");
            for (int r = 0; r < N_ROUTES; ++r)
                printf("%.3f%%%s", best->delay[r], r < N_ROUTES-1 ? ", " : "");
            printf("]\n");
            printf("    delay_std=[");
            for (int r = 0; r < N_ROUTES; ++r)
                printf("%.3f%%%s", best->delay_std[r], r < N_ROUTES-1 ? ", " : "");
            printf("]\n");
        }
    }

    clock_t t_wall_end = clock();
    double total_elapsed = (double)(t_wall_end - t_wall_start) / CLOCKS_PER_SEC;
    printf("\n=== Grid search complete ===\n");
    printf("Total wall-clock time: %.2f seconds\n", total_elapsed);

    nn_cache_stats();
    return 0;
}
