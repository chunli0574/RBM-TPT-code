/*
 * denser_sim_main.cpp
 *
 * Single-combination denser simulation for post-hoc refinement.
 *
 * Given one parameter combination (safety, rho, epsilon, lambda_tilde,
 * beta_rate_1, beta_rate_2), runs n_denser_paths paths and saves the result
 * to a CSV file.
 *
 * Intended use: after a coarse grid search (sim_proposed) identifies promising
 * parameter combinations, this binary re-runs each one at higher precision
 * (n_denser_paths = 10000 by default).
 *
 * Usage:
 *   ./sim_denser [config_file] safety rho epsilon lambda beta_rate_1 beta_rate_2
 *
 * Arguments:
 *   config_file   optional path to config file (default: ./denser_sim_config.txt)
 *   safety        RC_SAFETY_STOCK value
 *   rho           RC_OUTSOURCE_THRESHOLD / NN model selector
 *   epsilon       RC_EPSILON_BAND value
 *   lambda        RC_LAMBDA_TILDE value (WMP scheduling)
 *   beta_rate_1   beta_tilde_1 = beta_rate_1 * (19/9) * rho
 *   beta_rate_2   beta_tilde_2 = beta_rate_2 * (19/9) * rho
 *
 * Config file keys (all optional, see defaults below):
 *   n_denser_paths   = 10000     # number of simulation paths
 *   inward_reflection= 0         # 0=hold_last_no_takeover, 1=inward
 *   wmp_limiting     = 0         # 0='N' full WMP, 1='Y' simplified limiting
 *   output_dir       = ../prelimit_simulation_results/denser_sim_results
 *   model_new_base_dir = directory of *_grad.weights.bin models
 *                        (empty = NN_MODEL_DIR_DEFAULT in nn_model_paths.h)
 *
 * Output:
 *   {output_dir}/s{safety}_r{rho}_e{epsilon}_l{lambda}_b1{br1}_b2{br2}.csv
 *   One data row with columns:
 *     param_1, param_2, param_3, param_4, cost, std, delay1, delay2,
 *     hardstop1, hardstop2
 *   where param_1=lambda_tilde, param_2=beta_tilde_1, param_3=beta_tilde_2,
 *         param_4=rho  (matches Python save_final_results format)
 *
 * TO BUILD:
 *   make sim_denser
 */

#include "params.h"
#include "run_config.h"
#include "simulation.h"
#include "nn_model.h"
#include "nn_model_paths.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <sys/stat.h>   // mkdir

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct DenserConfig {
    int    n_denser_paths    = 10000;
    int    inward_reflection = 0;
    int    wmp_limiting      = 0;
    // wmp_extra_relative: only used when wmp_limiting == 0 (full WMP formula).
    //   0 = original extra term; 1 = extra term scaled by 1/(BETA_CONST*rho)
    int    wmp_extra_relative = 0;
    // print_detailed_info: 0 = basic CSV only; 1 = full cost breakdown columns
    int    print_detailed_info = 0;
    // epsilon_relative: 0=absolute (× EPSILON_SCALING), 1=relative (× FACE_BOUNDARY)
    int    epsilon_relative  = 0;
    std::string output_dir        = "../prelimit_simulation_results/denser_sim_results";
    std::string model_new_base_dir;   // .weights.bin directory (empty = NN_MODEL_DIR_DEFAULT)
};

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

static std::vector<double> parse_double_list(const std::string& s) {
    std::vector<double> v;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t a = token.find_first_not_of(" \t\r\n");
        size_t b = token.find_last_not_of(" \t\r\n");
        if (a != std::string::npos)
            v.push_back(std::stod(token.substr(a, b - a + 1)));
    }
    return v;
}

static DenserConfig load_config(const std::string& path) {
    DenserConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        // config file missing → use defaults
        fprintf(stderr, "WARNING: cannot open config file: %s — using defaults\n", path.c_str());
        return cfg;
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

        if      (key == "n_denser_paths")        cfg.n_denser_paths        = std::stoi(val);
        else if (key == "inward_reflection")     cfg.inward_reflection     = std::stoi(val);
        else if (key == "wmp_limiting")          cfg.wmp_limiting          = std::stoi(val);
        else if (key == "wmp_extra_relative")    cfg.wmp_extra_relative    = std::stoi(val);
        else if (key == "print_detailed_info")   cfg.print_detailed_info   = std::stoi(val);
        else if (key == "epsilon_relative")      cfg.epsilon_relative      = std::stoi(val);
        else if (key == "output_dir")            cfg.output_dir            = val;
        else if (key == "model_new_base_dir")    cfg.model_new_base_dir    = val;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// mkdir -p (single-level, same implementation as proposed_policy_main.cpp)
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
// Format double for file names (%g, negative → 'n')
// ---------------------------------------------------------------------------
static std::string fmt_param(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%g", v);
    std::string s(buf);
    if (!s.empty() && s[0] == '-') s[0] = 'n';
    return s;
}

// ---------------------------------------------------------------------------
// wmp_prefix: file-name prefix that distinguishes WMP scheduling variants.
//   wmp_limiting == 1     → ""       (limiting formula: extra term absent;
//                                     wmp_extra_relative is irrelevant)
//   wmp_limiting == 0  +
//     wmp_extra_relative=0 → "abs_"  (full WMP, absolute extra term)
//     wmp_extra_relative=1 → "rel_"  (full WMP, scaled / relative extra term)
// ---------------------------------------------------------------------------
static std::string wmp_prefix(int wmp_limiting, int wmp_extra_relative) {
    if (wmp_limiting != 0) return "";
    return (wmp_extra_relative == 0) ? "abs_" : "rel_";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Parse arguments:
    //   ./sim_denser [config_file] safety rho epsilon lambda beta_rate_1 beta_rate_2
    // The 6 numeric parameters are always the last 6 positional arguments.
    if (argc < 7) {
        fprintf(stderr,
            "Usage: %s [config_file] safety rho epsilon lambda beta_rate_1 beta_rate_2\n"
            "  config_file defaults to ./denser_sim_config.txt\n", argv[0]);
        return 1;
    }

    // Detect whether argv[1] is the config file or the first numeric param.
    // If argc == 7: no config file given → argv[1..6] are the 6 params.
    // If argc == 8: argv[1] is config file, argv[2..7] are the 6 params.
    std::string cfg_path = "denser_sim_config.txt";
    int param_start = 1;
    if (argc == 8) {
        cfg_path    = argv[1];
        param_start = 2;
    }

    double safety     = std::stod(argv[param_start + 0]);
    double rho        = std::stod(argv[param_start + 1]);
    double epsilon    = std::stod(argv[param_start + 2]);
    double lambda     = std::stod(argv[param_start + 3]);
    double beta_rate_1= std::stod(argv[param_start + 4]);
    double beta_rate_2= std::stod(argv[param_start + 5]);

    DenserConfig cfg = load_config(cfg_path);

    // Derived parameters
    double beta_tilde_1 = beta_rate_1 * BETA_CONST * rho;
    double beta_tilde_2 = beta_rate_2 * BETA_CONST * rho;

    printf("=== Denser Simulation (%d paths) ===\n", cfg.n_denser_paths);
    printf("  safety=%.4g  rho=%.4g  epsilon=%.4g\n", safety, rho, epsilon);
    printf("  lambda_tilde=%.4g  beta_rate_1=%.4g  beta_rate_2=%.4g\n",
           lambda, beta_rate_1, beta_rate_2);
    printf("  → beta_tilde_1=%.4f  beta_tilde_2=%.4f\n", beta_tilde_1, beta_tilde_2);
    printf("  inward_reflection=%d  wmp_limiting=%d  wmp_extra_relative=%d  epsilon_relative=%d\n",
           cfg.inward_reflection, cfg.wmp_limiting, cfg.wmp_extra_relative, cfg.epsilon_relative);

    std::string model_path = make_model_path(rho, cfg.model_new_base_dir);

    // Pre-flight: check outsourcing model (rho) exists
    if (!nn_model_path_exists(model_path)) {
        fprintf(stderr, "ERROR: outsourcing model not found for rho=%.4g:\n  %s\n",
                rho, model_path.c_str());
        return 1;
    }

    // Load outsourcing model (rho)
    printf("Loading outsourcing model:\n  %s\n", model_path.c_str());
    nn_model_load(model_path);

    printf("Model loaded.\n\n");

    // Set runtime globals
    RC_N_PATHS               = cfg.n_denser_paths;
    RC_SAFETY_STOCK          = safety;
    RC_OUTSOURCE_THRESHOLD   = rho;
    RC_EPSILON_BAND          = epsilon;
    RC_LAMBDA_TILDE          = lambda;
    RC_BETA_TILDE_1          = beta_tilde_1;
    RC_BETA_TILDE_2          = beta_tilde_2;
    RC_INWARD_REFLECTION     = cfg.inward_reflection;
    RC_WMP_LIMITING          = cfg.wmp_limiting;
    RC_WMP_EXTRA_RELATIVE    = cfg.wmp_extra_relative;
    RC_EPSILON_RELATIVE      = cfg.epsilon_relative;
    RC_PRINT_DETAILED_INFO   = cfg.print_detailed_info;

    // Run simulation — collect per-path results for distribution CSV
    std::vector<PathResult> path_results;
    clock_t t0 = clock();
    SimResult res = run_simulation(&path_results);
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("\n=== Result ===\n");
    printf("  cost=%.4f ± %.4f  delay1=%.3f%%  delay2=%.3f%%\n",
           res.cost_mean, res.cost_std,
           res.delay_prob_mean[0], res.delay_prob_mean[1]);
    printf("  outsource1=%.2f%%  outsource2=%.2f%%  time=%.1fs\n",
           res.outsource_frac_mean[0]*100.0,
           res.outsource_frac_mean[1]*100.0,
           elapsed);

    // Build base file name prefix shared by both output files
    // Pattern: {output_dir}/{abs_|rel_|}s{safety}_r{rho}_e{epsilon}_l{lambda}_b1{br1}_b2{br2}
    // Leading abs_/rel_ encodes the WMP scheduling variant (see wmp_prefix).
    mkdir_p(cfg.output_dir);
    std::string wmp_pfx = wmp_prefix(cfg.wmp_limiting, cfg.wmp_extra_relative);
    std::string name_prefix = cfg.output_dir
        + "/" + wmp_pfx + "s"  + fmt_param(safety)
        + "_r"  + fmt_param(rho)
        + "_e"  + fmt_param(epsilon)
        + "_l"  + fmt_param(lambda)
        + "_b1" + fmt_param(beta_rate_1)
        + "_b2" + fmt_param(beta_rate_2);

    // ----- 1. Aggregate summary CSV (one row, matches proposed_policy_main format) -----
    std::string out_path = name_prefix + ".csv";
    {
        FILE* fp = fopen(out_path.c_str(), "w");
        if (!fp) {
            fprintf(stderr, "ERROR: cannot write output: %s\n", out_path.c_str());
            return 1;
        }
        if (RC_PRINT_DETAILED_INFO) {
            fprintf(fp, "param_1,param_2,param_3,param_4,cost,std,delay1,delay2,delay1_std,delay2_std,hardstop1,hardstop2,pricing_cost,outsourcing_cost,outsourcing_cost_band,outsourcing_cost_outer,outsourcing_cost_1,outsourcing_cost_2\n");
            fprintf(fp, "%.6g,%.6g,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    lambda, beta_tilde_1, beta_tilde_2, rho,
                    res.cost_mean, res.cost_std,
                    res.delay_prob_mean[0], res.delay_prob_mean[1],
                    res.delay_prob_std[0],  res.delay_prob_std[1],
                    0.0, 0.0,
                    res.pricing_cost_mean, res.outsourcing_cost_mean,
                    res.outsourcing_band_cost_mean, res.outsourcing_outer_cost_mean,
                    res.outsourcing_route_cost_mean[0], res.outsourcing_route_cost_mean[1]);
        } else {
            fprintf(fp, "param_1,param_2,param_3,param_4,cost,std,delay1,delay2,delay1_std,delay2_std\n");
            fprintf(fp, "%.6g,%.6g,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    lambda, beta_tilde_1, beta_tilde_2, rho,
                    res.cost_mean, res.cost_std,
                    res.delay_prob_mean[0], res.delay_prob_mean[1],
                    res.delay_prob_std[0],  res.delay_prob_std[1]);
        }
        fclose(fp);
        printf("Saved summary:   %s\n", out_path.c_str());
    }

    // ----- 2. Per-path distribution CSV (one row per path) -----
    std::string paths_path = name_prefix + "_paths.csv";
    {
        FILE* fp = fopen(paths_path.c_str(), "w");
        if (!fp) {
            fprintf(stderr, "ERROR: cannot write paths output: %s\n", paths_path.c_str());
            return 1;
        }
        if (RC_PRINT_DETAILED_INFO) {
            fprintf(fp, "path_id,cost,pricing_cost,outsourcing_cost,outsourcing_cost_band,outsourcing_cost_outer,outsourcing_cost_1,outsourcing_cost_2,delay1,delay2\n");
            for (int m = 0; m < (int)path_results.size(); m++) {
                fprintf(fp, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f\n",
                        m,
                        path_results[m].cost,
                        path_results[m].cost_pricing,
                        path_results[m].cost_outsourcing,
                        path_results[m].cost_outsourcing_band,
                        path_results[m].cost_outsourcing_outer,
                        path_results[m].cost_outsourcing_route[0],
                        path_results[m].cost_outsourcing_route[1],
                        path_results[m].delay_prob[0],
                        path_results[m].delay_prob[1]);
            }
        } else {
            fprintf(fp, "path_id,cost,delay1,delay2\n");
            for (int m = 0; m < (int)path_results.size(); m++) {
                fprintf(fp, "%d,%.6f,%.4f,%.4f\n",
                        m,
                        path_results[m].cost,
                        path_results[m].delay_prob[0],
                        path_results[m].delay_prob[1]);
            }
        }
        fclose(fp);
        printf("Saved per-path:  %s\n", paths_path.c_str());
    }

    nn_cache_stats();
    return 0;
}
