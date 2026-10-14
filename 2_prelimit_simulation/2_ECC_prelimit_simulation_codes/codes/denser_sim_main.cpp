/*
 * denser_sim_main.cpp
 *
 * Single-combo simulation with a large number of paths (n_denser_paths).
 * Designed to be called once per parameter combination, enabling SLURM array
 * job parallelism over the rows of CC3_denser_params.txt.
 *
 * Usage:
 *   ./sim_denser <config> <safety> <rho> <epsilon> <lambda> <beta_rate_1> <beta_rate_2>
 *
 * Configuration is read from denser_sim_config.txt (same directory as binary):
 *   model_base_dir     = /path/to/nn_models
 *   new_model_base_dir = /path/to/new_format_nn_models   (required for new-format rhos)
 *   output_dir         = prelimit_simulation_results/denser_sim_results
 *   n_denser_paths     = 10000
 *
 * Output: one CSV file per combo in output_dir, named:
 *   s{safety}_r{rho}_e{epsilon}_l{lambda}_b1{beta_rate_1}_b2{beta_rate_2}.csv
 * CSV has one data row, same column format as proposed_policy_main:
 *   param_1, param_2, param_3, param_4, cost, std, delay1..N, hardstop1..N
 *   (param_1=lambda_tilde, param_2=beta_tilde_v, param_3=beta_tilde_h, param_4=rho)
 *
 * TO BUILD:  make sim_denser
 * TO RUN:    ./sim_denser denser_sim_config.txt 0.0 0.99 0.3 14 0.8 0.8
 */

#include "params.h"
#include "run_config.h"
#include "simulation.h"
#include "nn_model.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// eta_theta maps + make_model_path — moved to model_path.{h,cpp}
// (shared with proposed_policy_main.cpp / one_combo_main.cpp).
// ---------------------------------------------------------------------------

#include "model_path.h"

// ---------------------------------------------------------------------------
// Local filesystem / formatting helpers (kept in this file because
// denser_sim_main.cpp uses them only for output filename construction).
// ---------------------------------------------------------------------------
static void mkdir_p(const std::string& dir) {
    std::string acc;
    for (size_t i = 0; i <= dir.size(); ++i) {
        if (i == dir.size() || dir[i] == '/') {
            if (!acc.empty()) mkdir(acc.c_str(), 0755);
            if (i < dir.size()) acc += dir[i];
        } else {
            acc += dir[i];
        }
    }
    mkdir(dir.c_str(), 0755);
}

// Format double for file names (%g, negative sign → 'n')
static std::string fmt_param(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%g", v);
    std::string s(buf);
    if (!s.empty() && s[0] == '-') s[0] = 'n';
    return s;
}

// ---------------------------------------------------------------------------
// Config file parser
// ---------------------------------------------------------------------------
struct DenserConfig {
    std::string model_base_dir;
    // new_model_base_dir: base directory for new-format (.weights.bin) NN models.
    // Required when the rho argument is in NEW_FORMAT_RHOS.
    std::string new_model_base_dir;
    std::string output_dir   = "prelimit_simulation_results/denser_sim_results";
    int         n_denser_paths = 10000;
    // epsilon_relative: 0=absolute band (±eps×scaling), 1=relative band (FACE_CONST×(rho±eps))
    int         epsilon_relative = 0;
    // wmp_extra_relative: 0=plain v1; 1=normalize v1 pressure term by (B_V*rho) / (B_H*rho)
    int         wmp_extra_relative = 0;
    // print_detailed_info: 1=full CSVs with cost splits; 0=lean (skip hardstop
    // columns, pricing_cost, outsourcing_cost, and in _paths.csv also skip
    // pricing_cost / outsourcing_cost).
    int         print_detailed_info = 0;
};

static std::string trim2(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

static DenserConfig load_config(const std::string& path) {
    DenserConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open config file: %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim2(line);
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim2(line.substr(0, eq));
        std::string val = trim2(line.substr(eq + 1));
        if      (key == "model_base_dir")         cfg.model_base_dir         = val;
        else if (key == "new_model_base_dir")      cfg.new_model_base_dir      = val;
        else if (key == "output_dir")              cfg.output_dir              = val;
        else if (key == "n_denser_paths")          cfg.n_denser_paths          = std::stoi(val);
        else if (key == "epsilon_relative")        cfg.epsilon_relative        = std::stoi(val);
        else if (key == "wmp_extra_relative")     cfg.wmp_extra_relative      = std::stoi(val);
        else if (key == "print_detailed_info")    cfg.print_detailed_info     = std::stoi(val);
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 8) {
        fprintf(stderr,
            "Usage: %s <config> <safety> <rho> <epsilon> <lambda> <beta_rate_1> <beta_rate_2>\n",
            argv[0]);
        return 1;
    }

    std::string cfg_path = argv[1];
    double safety      = std::stod(argv[2]);
    double rho         = std::stod(argv[3]);
    double epsilon     = std::stod(argv[4]);
    double lambda_t    = std::stod(argv[5]);
    double beta_rate_1 = std::stod(argv[6]);
    double beta_rate_2 = std::stod(argv[7]);
    DenserConfig cfg = load_config(cfg_path);

    printf("=== denser_sim (dim_K=%d) ===\n", DIM_K);
    printf("  safety=%.4g  rho=%.4g  epsilon=%.4g  lambda=%.4g"
           "  beta_rate_1=%.4g  beta_rate_2=%.4g\n",
           safety, rho, epsilon, lambda_t, beta_rate_1, beta_rate_2);
    printf("  n_denser_paths=%d\n", cfg.n_denser_paths);

    // Set runtime path count and epsilon mode
    RC_N_PATHS               = cfg.n_denser_paths;
    RC_EPSILON_RELATIVE      = cfg.epsilon_relative;
    RC_WMP_EXTRA_RELATIVE    = cfg.wmp_extra_relative;
    printf("  wmp_extra_relative=%d (%s)  → output prefix: \"%s\"\n",
           cfg.wmp_extra_relative,
           cfg.wmp_extra_relative ? "pressure normalized by (B*rho)" : "original",
           cfg.wmp_extra_relative ? "rel_" : "abs_");
    RC_PRINT_DETAILED_INFO   = cfg.print_detailed_info;
    printf("  print_detailed_info=%d (%s)\n",
           cfg.print_detailed_info,
           cfg.print_detailed_info ? "full" : "lean (skip cost splits + outsource_frac)");

    auto eta_theta     = build_eta_theta_map();
    auto eta_theta_b50 = build_eta_theta_b50_map();

    // ---- Pre-flight: verify model file exists before loading ----
    {
        std::string p = make_model_path(rho, eta_theta, eta_theta_b50,
                                        cfg.model_base_dir, cfg.new_model_base_dir);
        if (!nn_model_path_exists(p)) {
            fprintf(stderr, "ERROR: NN model missing for rho=%.4g:\n  %s\n", rho, p.c_str());
            fprintf(stderr, "Aborting: fix missing model file before running.\n");
            return 1;
        }
    }

    // Load NN model (rho) — shared by pricing and outsourcing
    std::string model_path = make_model_path(rho, eta_theta, eta_theta_b50,
                                             cfg.model_base_dir, cfg.new_model_base_dir);
    printf("  NN model: %s\n", model_path.c_str());
    nn_model_load(model_path);

    // Set RC_* policy parameters
    RC_SAFETY_STOCK        = safety;
    RC_EPSILON_BAND        = epsilon;
    RC_OUTSOURCE_THRESHOLD = rho;
    RC_LAMBDA_TILDE        = lambda_t;
    RC_BETA_TILDE_V        = beta_rate_1 * BETA_CONST_V * rho;
    RC_BETA_TILDE_H        = beta_rate_2 * BETA_CONST_H * rho;

    printf("  beta_tilde_v=%.4f  beta_tilde_h=%.4f\n", RC_BETA_TILDE_V, RC_BETA_TILDE_H);

    // Run simulation — collect per-path results for distribution CSV
    std::vector<PathResult> path_results;
    clock_t t0 = clock();
    SimResult res = run_simulation(&path_results);
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("  cost=%.4f ± %.4f  delay=[", res.cost_mean, res.cost_std);
    for (int r = 0; r < N_ROUTES; ++r)
        printf("%.2f%%%s", res.delay_prob_mean[r], r < N_ROUTES - 1 ? ", " : "");
    printf("]  time=%.1fs\n", elapsed);

    // Build shared filename prefix:
    //   {output_dir}/{abs_|rel_}s{safety}_r{rho}_e{epsilon}_l{lambda}_b1{b1}_b2{b2}
    // The abs_/rel_ prefix tracks wmp_extra_relative (0=abs_, 1=rel_) so different
    // settings don't collide.
    mkdir_p(cfg.output_dir);
    const char* wmp_prefix = (cfg.wmp_extra_relative == 0) ? "abs_" : "rel_";
    std::string name_prefix = cfg.output_dir
        + "/" + wmp_prefix
        + "s"   + fmt_param(safety)
        + "_r"  + fmt_param(rho)
        + "_e"  + fmt_param(epsilon)
        + "_l"  + fmt_param(lambda_t)
        + "_b1" + fmt_param(beta_rate_1)
        + "_b2" + fmt_param(beta_rate_2);

    const bool detailed = (cfg.print_detailed_info != 0);

    // ----- 1. Aggregate summary CSV (one row, matches proposed_policy_main format) -----
    {
        std::string csv_path = name_prefix + ".csv";
        FILE* fp = fopen(csv_path.c_str(), "w");
        if (!fp) { fprintf(stderr, "ERROR: cannot write CSV: %s\n", csv_path.c_str()); return 1; }

        fprintf(fp, "param_1,param_2,param_3,param_4,cost,std");
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",delay%d", r + 1);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",delay%d_std", r + 1);
        if (detailed) {
            for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",hardstop%d", r + 1);
            fprintf(fp, ",pricing_cost,outsourcing_cost");
        }
        fprintf(fp, "\n");

        fprintf(fp, "%.6g,%.6g,%.6g,%.6g,%.4f,%.4f",
                lambda_t, RC_BETA_TILDE_V, RC_BETA_TILDE_H, rho,
                res.cost_mean, res.cost_std);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.4f", res.delay_prob_mean[r]);
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",%.4f", res.delay_prob_std[r]);
        if (detailed) {
            // hardstop columns are legacy placeholders (always 0 in C++).
            for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",0");
            fprintf(fp, ",%.4f,%.4f",
                    res.pricing_cost_mean, res.outsourcing_cost_mean);
        }
        fprintf(fp, "\n");
        fclose(fp);
        printf("  Saved summary:  %s\n", csv_path.c_str());
    }

    // ----- 2. Per-path distribution CSV (one row per path) -----
    {
        std::string paths_path = name_prefix + "_paths.csv";
        FILE* fp = fopen(paths_path.c_str(), "w");
        if (!fp) { fprintf(stderr, "ERROR: cannot write paths CSV: %s\n", paths_path.c_str()); return 1; }

        // path_id, cost always present. pricing_cost/outsourcing_cost only in
        // detailed mode (lean mode never accumulates them).
        fprintf(fp, "path_id,cost");
        if (detailed) fprintf(fp, ",pricing_cost,outsourcing_cost");
        for (int r = 0; r < N_ROUTES; ++r) fprintf(fp, ",delay%d", r + 1);
        fprintf(fp, "\n");

        for (int m = 0; m < (int)path_results.size(); ++m) {
            fprintf(fp, "%d,%.6f", m, path_results[m].cost);
            if (detailed) {
                fprintf(fp, ",%.6f,%.6f",
                        path_results[m].pricing_cost,
                        path_results[m].outsourcing_cost);
            }
            for (int r = 0; r < N_ROUTES; ++r)
                fprintf(fp, ",%.4f", path_results[m].delay_prob[r]);
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("  Saved per-path: %s\n", paths_path.c_str());
    }

    return 0;
}
