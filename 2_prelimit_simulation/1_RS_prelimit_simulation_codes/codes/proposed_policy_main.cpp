/*
 * proposed_policy_main.cpp
 *
 * Parameter search over 6 dimensions for the proposed NN policy:
 *   safety, epsilon, rho, lambda_tilde, beta_rate_1, beta_rate_2
 *
 * For each combo:
 *   1. Set RC_* runtime globals (used by policy.cpp instead of params.h constexpr).
 *   2. Reload the NN model when rho changes (rho selects the model file).
 *   3. Call run_simulation() and record cost / delay probs.
 *
 * Results per (rho, safety, epsilon) block are saved to:
 *   {output_dir}/rho{r}_safety{s}_epsilon{e}/RS_final_results.csv
 * with columns:  param_1, param_2, param_3, param_4, cost, std,
 *                delay1, delay2, hardstop1, hardstop2
 * where param_1=lambda_tilde, param_2=beta_tilde_1,
 *       param_3=beta_tilde_2, param_4=rho
 * (matches Python save_final_results format from functions_general.py)
 *
 * After all combos, the min-cost feasible row (delay1 < constraint AND
 * delay2 < constraint) is printed and saved to:
 *   {output_dir}/optimal_parameter.csv
 *
 * Configuration is read from proposed_policy_config.txt (same directory as binary).
 *
 * TO BUILD:
 *   make sim_proposed
 * TO RUN:
 *   make run_proposed
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
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <sys/stat.h>   // mkdir

// ---------------------------------------------------------------------------
// Config file parser
// ---------------------------------------------------------------------------
struct ProposedConfig {
    std::vector<double> safety_list;
    std::vector<double> epsilon_list;
    std::vector<double> rho_list;
    std::vector<double> lambda_tilde_list;
    std::vector<double> beta_rate_1_list;
    std::vector<double> beta_rate_2_list;
    double delay_constraint  = 1.0;
    int    inward_reflection = 0;   // 0=hold_last_no_takeover, 1=inward
    int    wmp_limiting      = 0;   // 0='N' full WMP formula, 1='Y' simplified limiting
    // wmp_extra_relative: only used when wmp_limiting == 0 (full WMP formula).
    //   0 = original extra term:  lam * (Z_other - beta_tilde_other)
    //   1 = scaled extra term:    lam * (Z_other - beta_tilde_other) / (BETA_CONST * rho)
    int    wmp_extra_relative = 0;
    // print_detailed_info: 0 = basic CSV only; 1 = full cost breakdown columns
    int    print_detailed_info = 0;
    // epsilon_relative: 0=absolute (× EPSILON_SCALING={1,1,√5,√5}), 1=relative (× FACE_BOUNDARY)
    int    epsilon_relative  = 0;
    // beta_rate_symmetric: 0 = use separate beta_rate_1 and beta_rate_2 lists (default)
    //                       1 = beta_rate_2 always equals beta_rate_1 (beta_rate_2_list ignored)
    int    beta_rate_symmetric = 0;
    std::string output_dir        = "proposed_policy_results";
    std::string model_new_base_dir;  // .weights.bin directory (empty = NN_MODEL_DIR_DEFAULT)
};

static std::vector<double> parse_double_list(const std::string& s) {
    std::vector<double> v;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // trim whitespace
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

static ProposedConfig load_config(const std::string& path) {
    ProposedConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open config file: %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        // strip comments
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
        else if (key == "lambda_tilde_list") cfg.lambda_tilde_list = parse_double_list(val);
        else if (key == "beta_rate_1_list")  cfg.beta_rate_1_list  = parse_double_list(val);
        else if (key == "beta_rate_2_list")  cfg.beta_rate_2_list  = parse_double_list(val);
        else if (key == "delay_constraint")  cfg.delay_constraint  = std::stod(val);
        else if (key == "inward_reflection")      cfg.inward_reflection      = std::stoi(val);
        else if (key == "wmp_limiting")           cfg.wmp_limiting           = std::stoi(val);
        else if (key == "wmp_extra_relative")     cfg.wmp_extra_relative     = std::stoi(val);
        else if (key == "print_detailed_info")    cfg.print_detailed_info    = std::stoi(val);
        else if (key == "epsilon_relative")       cfg.epsilon_relative       = std::stoi(val);
        else if (key == "beta_rate_symmetric")    cfg.beta_rate_symmetric    = std::stoi(val);
        else if (key == "output_dir")             cfg.output_dir             = val;
        else if (key == "model_new_base_dir")     cfg.model_new_base_dir     = val;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// CSV result record
// ---------------------------------------------------------------------------
struct SearchRecord {
    double lambda_tilde;
    double beta_tilde_1;
    double beta_tilde_2;
    double rho;
    double cost_mean;
    double cost_std;
    double delay1;   // delay prob route 1 (%)
    double delay2;   // delay prob route 2 (%)
    // hardstop: not tracked in C++ — always 0 (same assumption as Python: never triggered)
    double hardstop1 = 0.0;
    double hardstop2 = 0.0;
    // cost breakdown
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
// Save results vector to CSV
// ---------------------------------------------------------------------------
static void save_csv(const std::string& filepath,
                     const std::vector<SearchRecord>& recs) {
    FILE* fp = fopen(filepath.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot write CSV: %s\n", filepath.c_str());
        return;
    }
    if (RC_PRINT_DETAILED_INFO) {
        fprintf(fp, "param_1,param_2,param_3,param_4,cost,std,delay1,delay2,delay1_std,delay2_std,hardstop1,hardstop2,pricing_cost,outsourcing_cost,outsourcing_cost_band,outsourcing_cost_outer,outsourcing_cost_1,outsourcing_cost_2\n");
        for (const auto& r : recs) {
            fprintf(fp, "%.6g,%.6g,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    r.lambda_tilde, r.beta_tilde_1, r.beta_tilde_2, r.rho,
                    r.cost_mean, r.cost_std, r.delay1, r.delay2, r.delay1_std, r.delay2_std,
                    r.hardstop1, r.hardstop2,
                    r.pricing_cost_mean, r.outsourcing_cost_mean,
                    r.outsourcing_band_cost_mean, r.outsourcing_outer_cost_mean,
                    r.outsourcing_cost_1, r.outsourcing_cost_2);
        }
    } else {
        fprintf(fp, "param_1,param_2,param_3,param_4,cost,std,delay1,delay2,delay1_std,delay2_std\n");
        for (const auto& r : recs) {
            fprintf(fp, "%.6g,%.6g,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    r.lambda_tilde, r.beta_tilde_1, r.beta_tilde_2, r.rho,
                    r.cost_mean, r.cost_std, r.delay1, r.delay2, r.delay1_std, r.delay2_std);
        }
    }
    fclose(fp);
    printf("  Saved: %s\n", filepath.c_str());
}

// ---------------------------------------------------------------------------
// Find and print/save the min-cost feasible row
// Python: find_min_cost_proposed_scheduling (functions_general.py, line 1944)
// ---------------------------------------------------------------------------
static void find_and_save_optimal(const std::vector<SearchRecord>& recs,
                                  double delay_constraint,
                                  const std::string& optimal_path) {
    const SearchRecord* best = nullptr;
    for (const auto& r : recs) {
        if (r.delay1 < delay_constraint && r.delay2 < delay_constraint) {
            if (best == nullptr || r.cost_mean < best->cost_mean) {
                best = &r;
            }
        }
    }

    if (best == nullptr) {
        printf("\n  [optimal] No feasible row found (delay constraint %.2f%%).\n",
               delay_constraint);
        return;
    }

    printf("\n  [optimal] min-cost feasible row:\n");
    printf("    lambda_tilde=%.4g  beta_tilde_1=%.4g  beta_tilde_2=%.4g  rho=%.4g\n",
           best->lambda_tilde, best->beta_tilde_1, best->beta_tilde_2, best->rho);
    printf("    cost=%.4f  std=%.4f  delay1=%.3f%%  delay2=%.3f%%\n",
           best->cost_mean, best->cost_std, best->delay1, best->delay2);

    // ---- Top-level optimal_parameter.csv writing is disabled. ----
    // Stdout banner above still prints. Uncomment the block below to re-enable.
    (void)optimal_path;  // suppress unused-parameter warning
    // FILE* fp = fopen(optimal_path.c_str(), "w");
    // if (fp) {
    //     if (RC_PRINT_DETAILED_INFO) {
    //         fprintf(fp, "param_1,param_2,param_3,param_4,cost,std,delay1,delay2,delay1_std,delay2_std,hardstop1,hardstop2,pricing_cost,outsourcing_cost,outsourcing_cost_band,outsourcing_cost_outer,outsourcing_cost_1,outsourcing_cost_2\n");
    //         fprintf(fp, "%.6g,%.6g,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
    //                 best->lambda_tilde, best->beta_tilde_1, best->beta_tilde_2, best->rho,
    //                 best->cost_mean, best->cost_std, best->delay1, best->delay2, best->delay1_std, best->delay2_std,
    //                 best->hardstop1, best->hardstop2,
    //                 best->pricing_cost_mean, best->outsourcing_cost_mean,
    //                 best->outsourcing_band_cost_mean, best->outsourcing_outer_cost_mean,
    //                 best->outsourcing_cost_1, best->outsourcing_cost_2);
    //     } else {
    //         fprintf(fp, "param_1,param_2,param_3,param_4,cost,std,delay1,delay2,delay1_std,delay2_std\n");
    //         fprintf(fp, "%.6g,%.6g,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
    //                 best->lambda_tilde, best->beta_tilde_1, best->beta_tilde_2, best->rho,
    //                 best->cost_mean, best->cost_std, best->delay1, best->delay2,
    //                 best->delay1_std, best->delay2_std);
    //     }
    //     fclose(fp);
    //     printf("  Saved optimal to: %s\n", optimal_path.c_str());
    // }
}

// ---------------------------------------------------------------------------
// mkdir -p equivalent (single level; nested dirs created step by step)
// ---------------------------------------------------------------------------
static void mkdir_p(const std::string& dir) {
    // Try to create the full path; if it fails, create parents first.
    // Simple approach: split by '/' and create each prefix.
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
// Format a double for use in file/folder names (matches Python f-string style)
// e.g. -1.0 → "n1.0", 0.0 → "0.0", 0.25 → "0.25"
// (negative sign replaced with 'n' so it's safe in file names)
// ---------------------------------------------------------------------------
static std::string fmt_param(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%g", v);
    // Replace '-' with 'n' for file name safety
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
    // Config file path: default is "./proposed_policy_config.txt"
    // Optional CLI overrides for server array jobs:
    //   argv[1] = config path
    //   argv[2] = safety   (overrides safety_list   → single value)
    //   argv[3] = epsilon  (overrides epsilon_list  → single value)
    //   argv[4] = rho      (overrides rho_list      → single value)
    // Each array job handles one (safety, epsilon, rho); lambda/beta loop runs inside.
    std::string cfg_path = "proposed_policy_config.txt";
    if (argc >= 2) cfg_path = argv[1];

    printf("=== W-Network Grid Search (%d paths per combo) ===\n\n", N_PATHS);
    printf("Reading config from: %s\n\n", cfg_path.c_str());

    ProposedConfig cfg = load_config(cfg_path);

    // Override lists with single values if CLI args provided
    if (argc >= 3) cfg.safety_list   = {std::stod(argv[2])};
    if (argc >= 4) cfg.epsilon_list  = {std::stod(argv[3])};
    if (argc >= 5) cfg.rho_list      = {std::stod(argv[4])};

    // Print config summary
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
    if (cfg.beta_rate_symmetric) {
        printf("  beta_rate_2:  [symmetric — equals beta_rate_1]\n");
    } else {
        printf("  beta_rate_2:  ");
        for (double v : cfg.beta_rate_2_list) printf("%.4g ", v); printf("\n");
    }
    printf("  beta_rate_symmetric: %d\n", cfg.beta_rate_symmetric);

    int n_b2   = cfg.beta_rate_symmetric ? 1 : (int)cfg.beta_rate_2_list.size();
    int total  = (int)(cfg.safety_list.size() * cfg.epsilon_list.size()
                     * cfg.rho_list.size()
                     * cfg.lambda_tilde_list.size()
                     * cfg.beta_rate_1_list.size() * n_b2);
    printf("\nTotal combos: %d\n\n", total);

    // -----------------------------------------------------------------------
    // Pre-flight model existence check
    // -----------------------------------------------------------------------
    // Verify the NN model exists for every rho that will be used.
    {
        bool all_ok = true;
        for (double rho : cfg.rho_list) {
            std::string path = make_model_path(rho, cfg.model_new_base_dir);
            if (!nn_model_path_exists(path)) {
                fprintf(stderr,
                    "ERROR: NN model not found for rho=%.4g:\n  %s\n",
                    rho, path.c_str());
                all_ok = false;
            }
        }
        if (!all_ok) {
            fprintf(stderr, "Aborting: missing NN model(s).\n");
            return 1;
        }
    }
    // Create output directory
    mkdir_p(cfg.output_dir);

    clock_t t_wall_start = clock();

    int N_r  = (int)cfg.rho_list.size();
    int N_s  = (int)cfg.safety_list.size();
    int N_e  = (int)cfg.epsilon_list.size();

    // all_records[ri][si * N_e + ei]: records for one (rho, safety, epsilon) block.
    // Saved to CSV immediately after each block; also merged at the end for global optimal.
    std::vector<std::vector<std::vector<SearchRecord>>> all_records(
        N_r, std::vector<std::vector<SearchRecord>>(N_s * N_e));

    double current_model_rho = -1.0;  // tracks which NN model is currently loaded
    int combo_idx = 0;

    // -----------------------------------------------------------------------
    // rho is the absolute outermost loop: the NN model depends only on rho,
    // not on safety / epsilon / lambda_tilde / beta_rate.
    // W0, W1 are determined by queue state × M_MATRIX, independent of all
    // other parameters → the (W0,W1)→G cache built for any combo at a given
    // rho is fully reusable for every other combo at the same rho.
    // -----------------------------------------------------------------------
    for (int ri = 0; ri < N_r; ++ri) {
        double rho = cfg.rho_list[ri];

        RC_OUTSOURCE_THRESHOLD   = rho;
        RC_INWARD_REFLECTION     = cfg.inward_reflection;
        RC_WMP_LIMITING          = cfg.wmp_limiting;
        RC_WMP_EXTRA_RELATIVE    = cfg.wmp_extra_relative;
        RC_EPSILON_RELATIVE      = cfg.epsilon_relative;
        RC_PRINT_DETAILED_INFO   = cfg.print_detailed_info;

        // Reload the NN model whenever rho changes.
        double model_key   = std::round(rho             * 1000.0) / 1000.0;
        double current_key = std::round(current_model_rho * 1000.0) / 1000.0;
        if (model_key != current_key) {
            std::string model_path = make_model_path(rho, cfg.model_new_base_dir);
            printf("===========================================================\n");
            printf("[rho=%.4g] Loading NN model:\n  %s\n", rho, model_path.c_str());
            if (current_model_rho < 0) {
                nn_model_load(model_path);
            } else {
                nn_model_reload(model_path);
            }
            current_model_rho = rho;
            printf("Model loaded.\n");
            printf("===========================================================\n\n");
        }

        {
            for (int si = 0; si < N_s; ++si) {
                double safety = cfg.safety_list[si];
                for (int ei = 0; ei < N_e; ++ei) {
                    double epsilon = cfg.epsilon_list[ei];

                    RC_SAFETY_STOCK = safety;
                    RC_EPSILON_BAND = epsilon;

                    for (double lambda_tilde : cfg.lambda_tilde_list) {
                        for (double beta_rate_1 : cfg.beta_rate_1_list) {
                            // symmetric mode: beta_rate_2 = beta_rate_1; otherwise use list
                            const std::vector<double>& br2_list = cfg.beta_rate_symmetric
                                ? std::vector<double>{beta_rate_1}
                                : cfg.beta_rate_2_list;
                            for (double beta_rate_2 : br2_list) {

                                combo_idx++;
                                double beta_tilde_1 = beta_rate_1 * BETA_CONST * rho;
                                double beta_tilde_2 = beta_rate_2 * BETA_CONST * rho;

                                printf("[combo %d/%d] rho=%.4g  safety=%.4g  epsilon=%.4g  "
                                       "lambda_tilde=%.4g  beta_rate_1=%.4g  beta_rate_2=%.4g\n",
                                       combo_idx, total, rho, safety, epsilon,
                                       lambda_tilde, beta_rate_1, beta_rate_2);
                                printf("          → beta_tilde_1=%.4f  beta_tilde_2=%.4f\n",
                                       beta_tilde_1, beta_tilde_2);

                                RC_LAMBDA_TILDE = lambda_tilde;
                                RC_BETA_TILDE_1 = beta_tilde_1;
                                RC_BETA_TILDE_2 = beta_tilde_2;

                                clock_t t0 = clock();
                                SimResult res = run_simulation();
                                clock_t t1 = clock();
                                double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

                                printf("  → cost=%.4f ± %.4f  delay1=%.3f%%  delay2=%.3f%%  "
                                       "outsource1=%.2f%%  outsource2=%.2f%%  time=%.1fs\n",
                                       res.cost_mean, res.cost_std,
                                       res.delay_prob_mean[0], res.delay_prob_mean[1],
                                       res.outsource_frac_mean[0]*100.0,
                                       res.outsource_frac_mean[1]*100.0,
                                       elapsed);

                                SearchRecord rec;
                                rec.lambda_tilde          = lambda_tilde;
                                rec.beta_tilde_1          = beta_tilde_1;
                                rec.beta_tilde_2          = beta_tilde_2;
                                rec.rho                   = rho;
                                rec.cost_mean             = res.cost_mean;
                                rec.cost_std              = res.cost_std;
                                rec.delay1                = res.delay_prob_mean[0];
                                rec.delay2                = res.delay_prob_mean[1];
                                rec.delay1_std                  = res.delay_prob_std[0];
                                rec.delay2_std                  = res.delay_prob_std[1];
                                rec.pricing_cost_mean           = res.pricing_cost_mean;
                                rec.outsourcing_cost_mean       = res.outsourcing_cost_mean;
                                rec.outsourcing_band_cost_mean  = res.outsourcing_band_cost_mean;
                                rec.outsourcing_outer_cost_mean = res.outsourcing_outer_cost_mean;
                                rec.outsourcing_cost_1          = res.outsourcing_route_cost_mean[0];
                                rec.outsourcing_cost_2          = res.outsourcing_route_cost_mean[1];
                                all_records[ri][si * N_e + ei].push_back(rec);

                            } // beta_rate_2
                        } // beta_rate_1
                    } // lambda_tilde

                    // Save CSV for this (rho, safety, epsilon) block immediately
                    // Prefix encodes WMP scheduling variant: abs_/rel_/"" (see wmp_prefix).
                    std::string wmp_pfx = wmp_prefix(cfg.wmp_limiting, cfg.wmp_extra_relative);
                    std::string subfolder_rse = cfg.output_dir
                        + "/" + wmp_pfx + "rho" + fmt_param(rho)
                        + "_safety" + fmt_param(safety)
                        + "_epsilon" + fmt_param(epsilon);
                    mkdir_p(subfolder_rse);
                    std::string csv_path_rse = subfolder_rse + "/RS_final_results.csv";
                    printf("\n");
                    save_csv(csv_path_rse, all_records[ri][si * N_e + ei]);

                } // epsilon
            } // safety

        }
    } // rho

    // After all rho: find global optimal across ALL (rho, safety, epsilon)
    std::vector<SearchRecord> global_merged;
    for (int ri = 0; ri < N_r; ++ri)
        for (int si = 0; si < N_s; ++si)
            for (int ei = 0; ei < N_e; ++ei) {
                const auto& v = all_records[ri][si * N_e + ei];
                global_merged.insert(global_merged.end(), v.begin(), v.end());
            }

    // Prefix global optimal CSV by WMP variant so abs_ / rel_ runs don't overwrite each other.
    // ---- Top-level optimal_parameter.csv writing is disabled (uncomment to re-enable). ----
    // std::string global_optimal_path = cfg.output_dir + "/"
    //     + wmp_prefix(cfg.wmp_limiting, cfg.wmp_extra_relative) + "optimal_parameter.csv";
    printf("\n======================================================\n");
    printf("Global optimal across all rho, safety, epsilon:\n");
    printf("======================================================\n");
    // Pass an empty path so find_and_save_optimal still prints to stdout but does not write a file.
    // (The fopen/fwrite block inside that function is also commented out, so the path is unused.)
    find_and_save_optimal(global_merged, cfg.delay_constraint, /*optimal_path=*/"");

    clock_t t_wall_end = clock();
    double total_elapsed = (double)(t_wall_end - t_wall_start) / CLOCKS_PER_SEC;
    printf("\n=== Grid search complete ===\n");
    printf("Total wall-clock time: %.2f seconds\n", total_elapsed);

    nn_cache_stats();
    return 0;
}
