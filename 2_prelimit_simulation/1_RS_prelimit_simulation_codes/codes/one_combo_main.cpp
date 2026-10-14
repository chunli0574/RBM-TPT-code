/*
 * one_combo_main.cpp
 *
 * Single-combo simulation that records a delay-time histogram.
 *
 * Reads ONE parameter set from an INI-style txt file. The txt file holds:
 *   - Top-level common configuration (n_paths, hist_bucket, output_dir, NN model paths, ...)
 *   - One section per supported policy: [proposed], [A1]..[D2]
 *   - A top-level key  policy = X  selects which section's parameters to use.
 *
 * The selected section drives:
 *   - RC_PRICING_MODE / RC_OUTSOURCING_MODE / RC_SCHEDULING_MODE
 *     (via benchmark_kinds.h::set_policy_modes for benchmark sections; defaults
 *      stay at NN_PRICING / NN_OUTSOURCING / WMP_OBJ2 for [proposed]).
 *   - RC_PRICING_PARA[2][2]   from BCP linear pricing fields
 *   - RC_HL_PARA[2][3]        from High-Low pricing fields
 *   - RC_LAMBDA_TILDE / RC_BETA_TILDE_1 / RC_BETA_TILDE_2 (WMP scheduling)
 *   - RC_SAFETY_STOCK / RC_OUTSOURCE_THRESHOLD / RC_EPSILON_BAND
 *   - RC_INWARD_REFLECTION / RC_WMP_LIMITING / RC_WMP_EXTRA_RELATIVE
 *
 * Writes TWO CSV files into output_dir:
 *   {prefix}{policy_name}_summary.csv      1 row: parameters + cost mean/std + delay1/2 mean/std
 *   {prefix}{policy_name}_delay_hist.csv   long format: bucket_lower, count_route0, count_route1
 *
 * prefix encodes the WMP scheduling variant (only meaningful for WMP-using
 * sections: proposed, C1, C2, D1, D2):
 *   wmp_limiting=1                          → ""
 *   wmp_limiting=0, wmp_extra_relative=0    → "abs_"
 *   wmp_limiting=0, wmp_extra_relative=1    → "rel_"
 * For A/B sections (non-WMP) the prefix is always "".
 */

#include "params.h"
#include "run_config.h"
#include "simulation.h"
#include "nn_model.h"
#include "nn_model_paths.h"
#include "profiling.h"
#include "benchmark_kinds.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Filesystem helper: mkdir -p
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

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// ---------------------------------------------------------------------------
// Per-section parameter table + INI parser
// ---------------------------------------------------------------------------
struct SectionParams {
    std::map<std::string, std::string> values;

    bool has(const std::string& key) const { return values.count(key) > 0; }
    const std::string& raw(const std::string& key) const { return values.at(key); }
    double get_d(const std::string& key) const { return std::stod(values.at(key)); }
    int    get_i(const std::string& key) const { return std::stoi(values.at(key)); }
};

struct OneComboCommon {
    std::string policy_name;            // required
    int    n_paths             = N_PATHS;
    double hist_bucket         = 0.1;
    std::string output_dir     = "delay_distribution_results";
    int    print_detailed_info = 0;
    int    inward_reflection   = 0;     // proposed only
    int    epsilon_relative    = 0;
    int    beta_rate_symmetric = 1;
    std::string model_new_base_dir;      // proposed only (empty = NN_MODEL_DIR_DEFAULT)
};

struct ParsedInput {
    OneComboCommon common;
    std::map<std::string, SectionParams> sections;  // "proposed"/"A1".."D2" → params
};

static ParsedInput parse_input_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open input file: %s\n", path.c_str());
        exit(1);
    }
    ParsedInput out;
    std::string current_section;  // "" = top-level common
    std::string line;
    while (std::getline(f, line)) {
        // Strip comment
        size_t hashp = line.find('#');
        if (hashp != std::string::npos) line = line.substr(0, hashp);
        line = trim(line);
        if (line.empty()) continue;

        // Section header [name]
        if (line.front() == '[' && line.back() == ']') {
            current_section = trim(line.substr(1, line.size() - 2));
            // Materialize the section so a [name] with no fields still exists.
            (void)out.sections[current_section];
            continue;
        }

        // key = value
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (key.empty()) continue;

        if (current_section.empty()) {
            // Top-level common keys
            OneComboCommon& c = out.common;
            if      (key == "policy")             c.policy_name        = val;
            else if (key == "n_paths")            c.n_paths            = std::stoi(val);
            else if (key == "hist_bucket")        c.hist_bucket        = std::stod(val);
            else if (key == "output_dir")         c.output_dir         = val;
            else if (key == "print_detailed_info")c.print_detailed_info= std::stoi(val);
            else if (key == "inward_reflection")  c.inward_reflection  = std::stoi(val);
            else if (key == "epsilon_relative")   c.epsilon_relative   = std::stoi(val);
            else if (key == "beta_rate_symmetric")c.beta_rate_symmetric= std::stoi(val);
            else if (key == "model_new_base_dir") c.model_new_base_dir = val;
            else {
                fprintf(stderr, "WARNING: unknown top-level key '%s' (ignored)\n", key.c_str());
            }
        } else {
            out.sections[current_section].values[key] = val;
        }
    }

    if (out.common.policy_name.empty()) {
        fprintf(stderr, "ERROR: input file missing top-level 'policy = ...' selector.\n");
        exit(1);
    }
    if (!out.sections.count(out.common.policy_name)) {
        fprintf(stderr, "ERROR: input file has no [%s] section (policy = %s).\n",
                out.common.policy_name.c_str(), out.common.policy_name.c_str());
        exit(1);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Per-policy validation + RC_* dispatch
// ---------------------------------------------------------------------------
struct ResolvedPolicy {
    bool   is_proposed;
    BmkKind bkind;                  // valid only when !is_proposed
    bool   uses_wmp;
    int    wmp_limiting;
    int    wmp_extra_relative;

    double safety;
    double rho;
    double epsilon;                 // 0 if not provided (B/D)
    bool   has_epsilon;
};

// Whitelist of recognised section keys per policy (for early-fail validation).
static void require(const SectionParams& sp, const std::string& key,
                    const std::string& policy) {
    if (!sp.has(key)) {
        fprintf(stderr, "ERROR: section [%s] missing required key '%s'\n",
                policy.c_str(), key.c_str());
        exit(1);
    }
}

static ResolvedPolicy apply_input(const ParsedInput& in) {
    const OneComboCommon& c = in.common;
    const SectionParams& sp = in.sections.at(c.policy_name);
    ResolvedPolicy r;
    r.is_proposed = (c.policy_name == "proposed");
    r.uses_wmp    = false;

    // --- Common required ---
    require(sp, "safety", c.policy_name);
    require(sp, "rho",    c.policy_name);
    r.safety = sp.get_d("safety");
    r.rho    = sp.get_d("rho");

    // --- Top-level globals (always set) ---
    RC_INWARD_REFLECTION   = c.inward_reflection;
    RC_EPSILON_RELATIVE    = c.epsilon_relative;
    RC_PRINT_DETAILED_INFO = c.print_detailed_info;
    RC_RECORD_DELAY_HIST   = 1;            // sim_one_combo always records hist
    RC_N_PATHS             = c.n_paths;
    RC_SAFETY_STOCK        = r.safety;
    RC_OUTSOURCE_THRESHOLD = r.rho;

    // --- epsilon (A/C/proposed have it; B/D omit) ---
    bool needs_epsilon = false;
    bool needs_bcp     = false;
    bool needs_hl      = false;
    bool needs_wmp     = false;

    if (r.is_proposed) {
        needs_epsilon = true;
        needs_wmp     = true;
        // proposed uses NN_PRICING + NN_OUTSOURCING + WMP_OBJ2 (compile-time defaults
        // in params.h); proposed_policy_main does not touch RC_*_MODE, so we don't
        // either. Just leave RC_PRICING_MODE / RC_OUTSOURCING_MODE / RC_SCHEDULING_MODE
        // at their defaults. (Other binaries may have overwritten them at run-time;
        // explicitly reset them to the params.h defaults so a stale RC_* cannot
        // bleed in.)
        RC_PRICING_MODE     = (int)PRICING_MODE;
        RC_OUTSOURCING_MODE = (int)OUTSOURCING_MODE;
        RC_SCHEDULING_MODE  = (int)SCHEDULING_MODE;
    } else {
        // Benchmarks: dispatch via benchmark_kinds.h
        r.bkind = bmk_kind_from_name(c.policy_name);
        set_policy_modes(r.bkind);

        needs_epsilon = (r.bkind.outsource == BmkOutsource::SinglePushing);  // A/C
        needs_bcp     = (r.bkind.pricing   == BmkPricing::BCPlinear);
        needs_hl      = (r.bkind.pricing   == BmkPricing::HighLow);
        needs_wmp     = (r.bkind.sched     == BmkScheduling::WMP);
    }

    if (needs_epsilon) {
        require(sp, "epsilon", c.policy_name);
        r.epsilon       = sp.get_d("epsilon");
        r.has_epsilon   = true;
        RC_EPSILON_BAND = r.epsilon;
    } else {
        r.epsilon       = 0.0;
        r.has_epsilon   = false;
        RC_EPSILON_BAND = 0.0;  // ignored by ThresholdBased outsourcing
    }

    // --- BCP pricing (4 fields) ---
    if (needs_bcp) {
        require(sp, "phi_own_0",   c.policy_name);
        require(sp, "phi_other_0", c.policy_name);
        require(sp, "phi_own_1",   c.policy_name);
        require(sp, "phi_other_1", c.policy_name);
        double flat[4] = {
            sp.get_d("phi_own_0"),  sp.get_d("phi_other_0"),
            sp.get_d("phi_own_1"),  sp.get_d("phi_other_1")
        };
        set_bcp_pricing_para(flat);
    }

    // --- High-Low pricing (6 fields, with Q_loop fraction → z_thresh) ---
    if (needs_hl) {
        require(sp, "Q_loop_1", c.policy_name);
        require(sp, "zeta_L_1", c.policy_name);
        require(sp, "zeta_H_1", c.policy_name);
        require(sp, "Q_loop_2", c.policy_name);
        require(sp, "zeta_L_2", c.policy_name);
        require(sp, "zeta_H_2", c.policy_name);
        double Q1 = sp.get_d("Q_loop_1");
        double Q2 = sp.get_d("Q_loop_2");
        // Same conversion as benchmark_search_main.cpp:
        //   z_thresh = Q_loop * Q_BOUND_MAX * rho * INV_SQRT_N
        double z1 = Q1 * (double)Q_BOUND_MAX * r.rho * INV_SQRT_N;
        double z2 = Q2 * (double)Q_BOUND_MAX * r.rho * INV_SQRT_N;
        double flat[6] = {
            z1, sp.get_d("zeta_L_1"), sp.get_d("zeta_H_1"),
            z2, sp.get_d("zeta_L_2"), sp.get_d("zeta_H_2")
        };
        set_hl_pricing_para(flat);
    }

    // --- WMP scheduling (3 fields + 2 flags) for proposed / C* / D* ---
    if (needs_wmp) {
        r.uses_wmp = true;
        require(sp, "lambda_tilde", c.policy_name);
        require(sp, "beta_rate_1",  c.policy_name);
        double lam = sp.get_d("lambda_tilde");
        double br1 = sp.get_d("beta_rate_1");
        double br2;
        if (c.beta_rate_symmetric == 1) {
            br2 = br1;
        } else {
            require(sp, "beta_rate_2", c.policy_name);
            br2 = sp.get_d("beta_rate_2");
        }
        RC_LAMBDA_TILDE = lam;
        RC_BETA_TILDE_1 = br1 * BETA_CONST * r.rho;
        RC_BETA_TILDE_2 = br2 * BETA_CONST * r.rho;

        r.wmp_limiting       = sp.has("wmp_limiting")       ? sp.get_i("wmp_limiting")       : 0;
        r.wmp_extra_relative = sp.has("wmp_extra_relative") ? sp.get_i("wmp_extra_relative") : 0;
        RC_WMP_LIMITING       = r.wmp_limiting;
        RC_WMP_EXTRA_RELATIVE = r.wmp_extra_relative;
    } else {
        // Non-WMP: defaults — ignored by LRS scheduling, but reset for cleanliness.
        r.wmp_limiting       = 0;
        r.wmp_extra_relative = 0;
        RC_WMP_LIMITING       = 0;
        RC_WMP_EXTRA_RELATIVE = 0;
    }

    return r;
}

// ---------------------------------------------------------------------------
// File-name prefix encoding the WMP variant for output files.
// ---------------------------------------------------------------------------
static std::string wmp_prefix(const ResolvedPolicy& r) {
    if (!r.uses_wmp)            return "";
    if (r.wmp_limiting != 0)    return "";
    return (r.wmp_extra_relative == 0) ? "abs_" : "rel_";
}

// ---------------------------------------------------------------------------
// Echo selected parameters to stdout (debug aid).
// ---------------------------------------------------------------------------
static void print_input_summary(const ParsedInput& in, const ResolvedPolicy& r) {
    const OneComboCommon& c = in.common;
    const SectionParams& sp = in.sections.at(c.policy_name);
    printf("=================================================================\n");
    printf("sim_one_combo: policy = %s\n", c.policy_name.c_str());
    printf("=================================================================\n");
    printf("  n_paths             = %d\n", c.n_paths);
    printf("  hist_bucket         = %g\n", c.hist_bucket);
    printf("  output_dir          = %s\n", c.output_dir.c_str());
    printf("  print_detailed_info = %d\n", c.print_detailed_info);
    printf("  epsilon_relative    = %d\n", c.epsilon_relative);
    printf("  beta_rate_symmetric = %d\n", c.beta_rate_symmetric);
    printf("  inward_reflection   = %d  (used only by [proposed])\n", c.inward_reflection);
    if (r.is_proposed) {
        printf("  model_new_base_dir  = %s\n", c.model_new_base_dir.c_str());
    }
    printf("---\n");
    printf("[%s] section:\n", c.policy_name.c_str());
    printf("  safety  = %g\n", r.safety);
    printf("  rho     = %g\n", r.rho);
    if (r.has_epsilon) printf("  epsilon = %g\n", r.epsilon);
    if (sp.has("phi_own_0")) {
        printf("  BCP pricing: phi_own_0=%s phi_other_0=%s phi_own_1=%s phi_other_1=%s\n",
               sp.raw("phi_own_0").c_str(), sp.raw("phi_other_0").c_str(),
               sp.raw("phi_own_1").c_str(), sp.raw("phi_other_1").c_str());
    }
    if (sp.has("Q_loop_1")) {
        printf("  HighLow pricing: Q_loop_1=%s zeta_L_1=%s zeta_H_1=%s\n",
               sp.raw("Q_loop_1").c_str(), sp.raw("zeta_L_1").c_str(), sp.raw("zeta_H_1").c_str());
        printf("                   Q_loop_2=%s zeta_L_2=%s zeta_H_2=%s\n",
               sp.raw("Q_loop_2").c_str(), sp.raw("zeta_L_2").c_str(), sp.raw("zeta_H_2").c_str());
        printf("  → z_thresh_1=%g  z_thresh_2=%g  (Q_BOUND_MAX=%d, INV_SQRT_N=%g)\n",
               RC_HL_PARA[0][0], RC_HL_PARA[1][0], Q_BOUND_MAX, INV_SQRT_N);
    }
    if (r.uses_wmp) {
        printf("  WMP: lambda_tilde=%g  beta_rate_1=%s  beta_rate_2=%s\n",
               RC_LAMBDA_TILDE,
               sp.raw("beta_rate_1").c_str(),
               in.common.beta_rate_symmetric ? sp.raw("beta_rate_1").c_str()
                                             : sp.raw("beta_rate_2").c_str());
        printf("  → beta_tilde_1=%g  beta_tilde_2=%g\n",
               RC_BETA_TILDE_1, RC_BETA_TILDE_2);
        printf("  wmp_limiting=%d  wmp_extra_relative=%d  → prefix='%s'\n",
               r.wmp_limiting, r.wmp_extra_relative, wmp_prefix(r).c_str());
    }
    printf("=================================================================\n");
}

// ---------------------------------------------------------------------------
// CSV writers
// ---------------------------------------------------------------------------
static void write_summary_csv(const std::string& path,
                              const ParsedInput& in,
                              const ResolvedPolicy& r,
                              const SimResult& res,
                              double elapsed_sec) {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot write %s\n", path.c_str());
        return;
    }
    const SectionParams& sp = in.sections.at(in.common.policy_name);

    // Header: meta + parameters + results
    fprintf(fp,
        "policy,n_paths,safety,rho,epsilon,"
        "phi_own_0,phi_other_0,phi_own_1,phi_other_1,"
        "Q_loop_1,zeta_L_1,zeta_H_1,Q_loop_2,zeta_L_2,zeta_H_2,"
        "lambda_tilde,beta_rate_1,beta_rate_2,beta_tilde_1,beta_tilde_2,"
        "wmp_limiting,wmp_extra_relative,"
        "cost_mean,cost_std,delay1_mean,delay1_std,delay2_mean,delay2_std,"
        "elapsed_sec\n");

    auto field = [&](const std::string& k) -> std::string {
        return sp.has(k) ? sp.raw(k) : std::string("");
    };

    fprintf(fp, "%s,%d,%g,%g,%s,",
            in.common.policy_name.c_str(), in.common.n_paths,
            r.safety, r.rho,
            r.has_epsilon ? std::to_string(r.epsilon).c_str() : "");

    // BCP fields
    fprintf(fp, "%s,%s,%s,%s,",
            field("phi_own_0").c_str(), field("phi_other_0").c_str(),
            field("phi_own_1").c_str(), field("phi_other_1").c_str());

    // HighLow fields
    fprintf(fp, "%s,%s,%s,%s,%s,%s,",
            field("Q_loop_1").c_str(), field("zeta_L_1").c_str(), field("zeta_H_1").c_str(),
            field("Q_loop_2").c_str(), field("zeta_L_2").c_str(), field("zeta_H_2").c_str());

    // WMP fields
    if (r.uses_wmp) {
        std::string br1 = field("beta_rate_1");
        std::string br2 = (in.common.beta_rate_symmetric == 1) ? br1 : field("beta_rate_2");
        fprintf(fp, "%g,%s,%s,%g,%g,%d,%d,",
                RC_LAMBDA_TILDE, br1.c_str(), br2.c_str(),
                RC_BETA_TILDE_1, RC_BETA_TILDE_2,
                r.wmp_limiting, r.wmp_extra_relative);
    } else {
        fprintf(fp, ",,,,,,,");  // empty WMP columns for A/B
    }

    // Results
    fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
            res.cost_mean, res.cost_std,
            res.delay_prob_mean[0], res.delay_prob_std[0],
            res.delay_prob_mean[1], res.delay_prob_std[1],
            elapsed_sec);
    fclose(fp);
    printf("Saved summary  : %s\n", path.c_str());
}

static void write_histogram_csv(const std::string& path,
                                const DelayHistogram& h) {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot write %s\n", path.c_str());
        return;
    }
    fprintf(fp, "bucket_lower,count_route0,count_route1\n");

    // Outer-join the two route maps' key sets
    std::vector<int> keys;
    keys.reserve(h.hist[0].size() + h.hist[1].size());
    for (const auto& kv : h.hist[0]) keys.push_back(kv.first);
    for (const auto& kv : h.hist[1]) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    for (int k : keys) {
        uint64_t c0 = 0, c1 = 0;
        auto it0 = h.hist[0].find(k); if (it0 != h.hist[0].end()) c0 = it0->second;
        auto it1 = h.hist[1].find(k); if (it1 != h.hist[1].end()) c1 = it1->second;
        double bucket_lower = (double)k * h.bucket_width;
        fprintf(fp, "%.6g,%llu,%llu\n",
                bucket_lower,
                (unsigned long long)c0, (unsigned long long)c1);
    }
    fclose(fp);
    printf("Saved histogram: %s\n", path.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Usage:
    //   ./sim_one_combo                       (uses ./one_combo_input.txt, policy from file)
    //   ./sim_one_combo INPUT                 (custom input file, policy from file)
    //   ./sim_one_combo INPUT POLICY          (custom input file, policy overridden by CLI)
    //                                         (POLICY ∈ {proposed, A1..D2})
    std::string config_path = "one_combo_input.txt";
    if (argc >= 2) config_path = argv[1];

    printf("Reading input file: %s\n", config_path.c_str());
    ParsedInput parsed = parse_input_file(config_path);

    // Optional CLI override for the policy selector — lets one input file with
    // many [section] blocks be reused across multiple policies in a SLURM script.
    if (argc >= 3) {
        std::string cli_policy = argv[2];
        if (!parsed.sections.count(cli_policy)) {
            fprintf(stderr, "ERROR: CLI policy '%s' has no matching [%s] section in %s.\n",
                    cli_policy.c_str(), cli_policy.c_str(), config_path.c_str());
            return 1;
        }
        printf("CLI override: policy = %s (was %s)\n",
               cli_policy.c_str(), parsed.common.policy_name.c_str());
        parsed.common.policy_name = cli_policy;
    }

    // Apply parsed values to RC_* runtime variables (mutates RC_*).
    ResolvedPolicy r = apply_input(parsed);

    // Load NN model only when policy = proposed.
    if (r.is_proposed) {
        std::string model_path = make_model_path(r.rho, parsed.common.model_new_base_dir);
        printf("Loading NN model for rho=%.4g:\n  %s\n", r.rho, model_path.c_str());
        if (!nn_model_path_exists(model_path)) {
            fprintf(stderr, "ERROR: NN model not found: %s\n", model_path.c_str());
            return 1;
        }
        nn_model_load(model_path);
        printf("NN model loaded.\n");
    }

    print_input_summary(parsed, r);

    // ---- Run simulation with histogram ----
    DelayHistogram hist;
    hist.bucket_width = parsed.common.hist_bucket;

    clock_t t0 = clock();
    SimResult res = run_simulation(/*out_paths=*/nullptr, /*out_hist=*/&hist);
    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("\n");
    printf("=================================================================\n");
    printf("Result: cost=%.4f ± %.4f  delay1=%.3f%%±%.3f%%  delay2=%.3f%%±%.3f%%  time=%.2fs\n",
           res.cost_mean, res.cost_std,
           res.delay_prob_mean[0], res.delay_prob_std[0],
           res.delay_prob_mean[1], res.delay_prob_std[1],
           elapsed);
    printf("Histogram: route0 has %zu non-empty buckets, route1 has %zu\n",
           hist.hist[0].size(), hist.hist[1].size());
    printf("=================================================================\n\n");

    // ---- Write outputs ----
    mkdir_p(parsed.common.output_dir);
    std::string prefix = wmp_prefix(r);
    std::string base   = parsed.common.output_dir + "/" + prefix + parsed.common.policy_name;
    std::string sumcsv = base + "_summary.csv";
    std::string hstcsv = base + "_delay_hist.csv";

    write_summary_csv(sumcsv, parsed, r, res, elapsed);
    write_histogram_csv(hstcsv, hist);

    return 0;
}
