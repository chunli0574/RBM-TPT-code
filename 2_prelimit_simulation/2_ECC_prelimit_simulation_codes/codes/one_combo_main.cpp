/*
 * one_combo_main.cpp
 *
 * Single-combo simulation that records a delay-time histogram per route.
 * Reads ONE parameter set from an INI-style txt file:
 *   - Top-level common configuration (n_paths, hist_bucket, output_dir, NN model paths,...)
 *   - One section per supported policy: [proposed], [A1]..[D2]
 *   - A top-level key  policy = X  selects which section's parameters to use.
 *   - Optional 2nd CLI argument overrides the policy selector.
 *
 * The selected section drives:
 *   - RC_PRICING_MODE / RC_OUTSOURCING_MODE / RC_SCHEDULING_MODE
 *     (via benchmark_kinds::set_policy_modes for benchmark sections; defaults
 *      stay at NN_PRICING / NN_OUTSOURCING / WMP_OBJ2 for [proposed]).
 *   - RC_PRICING_PHI[N_ROUTES][2]   from BCP linear pricing fields (N_ROUTES*2)
 *   - RC_HL_PARA[N_ROUTES][3]       from High-Low pricing fields (N_ROUTES*3,
 *                                   Q_loop_r * QUEUE_LEN_BOUND[r] * rho * INV_SQRT_N)
 *   - RC_LAMBDA_TILDE / RC_BETA_TILDE_V / RC_BETA_TILDE_H (WMP scheduling)
 *   - RC_SAFETY_STOCK / RC_OUTSOURCE_THRESHOLD / RC_EPSILON_BAND
 *   - RC_INWARD_REFLECTION / RC_EPSILON_RELATIVE
 *   - RC_WMP_RELATIVE / RC_WMP_NO_QUEUE / RC_WMP_EXTRA_RELATIVE
 *
 * Writes TWO CSV files into output_dir:
 *   {prefix}{policy_name}_summary.csv      1 row: parameters + cost mean/std + delayN mean/std
 *   {prefix}{policy_name}_delay_hist.csv   long format: bucket_lower, count_route0..count_routeN-1
 *
 * prefix encodes the WMP scheduling variant (only meaningful for WMP-using
 * sections: proposed, C1, C2, D1, D2):
 *   wmp_extra_relative=0   → "abs_"
 *   wmp_extra_relative=1   → "rel_"
 * For A/B sections (non-WMP) the prefix is always "".
 *
 * N_ROUTES is compile-time (DIM_K_SELECT=3 → 4 routes; DIM_K_SELECT=10 → 11 routes).
 * The CC3 / CC10 input txt files differ in how many BCP / HighLow fields are
 * required per section.
 */

#include "params.h"
#include "run_config.h"
#include "simulation.h"
#include "nn_model.h"
#include "benchmark_kinds.h"
#include "model_path.h"

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
// Filesystem / string helpers
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

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// ---------------------------------------------------------------------------
// INI parser: top-level keys + per-section [name] key=value blocks
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
    int    wmp_relative        = 0;     // ECC-specific WMP flag
    int    wmp_no_queue        = 0;     // ECC-specific WMP flag
    int    beta_rate_symmetric = 0;     // 0 / 1 / 2
    std::string model_base_dir;         // proposed only
    std::string new_model_base_dir;     // proposed only
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
        size_t hashp = line.find('#');
        if (hashp != std::string::npos) line = line.substr(0, hashp);
        line = trim(line);
        if (line.empty()) continue;

        if (line.front() == '[' && line.back() == ']') {
            current_section = trim(line.substr(1, line.size() - 2));
            (void)out.sections[current_section];   // materialise empty section
            continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (key.empty()) continue;

        if (current_section.empty()) {
            OneComboCommon& c = out.common;
            if      (key == "policy")              c.policy_name        = val;
            else if (key == "n_paths")             c.n_paths            = std::stoi(val);
            else if (key == "hist_bucket")         c.hist_bucket        = std::stod(val);
            else if (key == "output_dir")          c.output_dir         = val;
            else if (key == "print_detailed_info") c.print_detailed_info= std::stoi(val);
            else if (key == "inward_reflection")   c.inward_reflection  = std::stoi(val);
            else if (key == "epsilon_relative")    c.epsilon_relative   = std::stoi(val);
            else if (key == "wmp_relative")        c.wmp_relative       = std::stoi(val);
            else if (key == "wmp_no_queue")        c.wmp_no_queue       = std::stoi(val);
            else if (key == "beta_rate_symmetric") c.beta_rate_symmetric= std::stoi(val);
            else if (key == "model_base_dir")      c.model_base_dir     = val;
            else if (key == "new_model_base_dir")  c.new_model_base_dir = val;
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
    int    wmp_extra_relative;

    double safety;
    double rho;
    double epsilon;                 // 0 if not provided (B/D)
    bool   has_epsilon;
};

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
    RC_WMP_RELATIVE        = c.wmp_relative;
    RC_WMP_NO_QUEUE        = c.wmp_no_queue;
    RC_PRINT_DETAILED_INFO = c.print_detailed_info;
    RC_RECORD_DELAY_HIST   = 1;            // sim_one_combo always records hist
    RC_N_PATHS             = c.n_paths;
    RC_SAFETY_STOCK        = r.safety;
    RC_OUTSOURCE_THRESHOLD = r.rho;

    bool needs_epsilon = false;
    bool needs_bcp     = false;
    bool needs_hl      = false;
    bool needs_wmp     = false;

    if (r.is_proposed) {
        needs_epsilon = true;
        needs_wmp     = true;
        // proposed uses NN_PRICING + NN_OUTSOURCING + WMP_OBJ2 (compile-time defaults
        // in params.h). proposed_policy_main does not touch RC_*_MODE; reset here
        // to the defaults so a stale RC_* (from a different binary in the same
        // process — unlikely but paranoia is cheap) cannot bleed in.
        RC_PRICING_MODE     = (int)PRICING_MODE;
        RC_OUTSOURCING_MODE = (int)OUTSOURCING_MODE;
        RC_SCHEDULING_MODE  = (int)SCHEDULING_MODE;
    } else {
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
        RC_EPSILON_BAND = 0.0;
    }

    // --- BCP pricing: N_ROUTES * 2 fields (phi_own_r, phi_other_r) ---
    if (needs_bcp) {
        std::vector<double> flat(N_ROUTES * 2);
        for (int rr = 0; rr < N_ROUTES; ++rr) {
            std::string k_own   = "phi_own_"   + std::to_string(rr);
            std::string k_other = "phi_other_" + std::to_string(rr);
            require(sp, k_own,   c.policy_name);
            require(sp, k_other, c.policy_name);
            flat[rr * 2 + 0] = sp.get_d(k_own);
            flat[rr * 2 + 1] = sp.get_d(k_other);
        }
        set_bcp_pricing_phi(flat.data());
    }

    // --- HighLow pricing: N_ROUTES * 3 fields (Q_loop_r, zeta_L_r, zeta_H_r) ---
    //   Q_loop_r is converted to z_thresh = Q_loop_r * QUEUE_LEN_BOUND[r] * rho * INV_SQRT_N
    if (needs_hl) {
        std::vector<double> flat(N_ROUTES * 3);
        for (int rr = 0; rr < N_ROUTES; ++rr) {
            std::string k_q  = "Q_loop_" + std::to_string(rr);
            std::string k_zL = "zeta_L_" + std::to_string(rr);
            std::string k_zH = "zeta_H_" + std::to_string(rr);
            require(sp, k_q,  c.policy_name);
            require(sp, k_zL, c.policy_name);
            require(sp, k_zH, c.policy_name);
            double Q  = sp.get_d(k_q);
            double zL = sp.get_d(k_zL);
            double zH = sp.get_d(k_zH);
            double z_thresh = Q * QUEUE_LEN_BOUND[rr] * r.rho * INV_SQRT_N;
            flat[rr * 3 + 0] = z_thresh;
            flat[rr * 3 + 1] = zL;
            flat[rr * 3 + 2] = zH;
        }
        set_hl_pricing_para(flat.data());
    }

    // --- WMP scheduling: lambda_tilde, beta_rate_1 [, beta_rate_2] + wmp_extra_relative ---
    if (needs_wmp) {
        r.uses_wmp = true;
        require(sp, "lambda_tilde", c.policy_name);
        require(sp, "beta_rate_1",  c.policy_name);
        double lam = sp.get_d("lambda_tilde");
        double br1 = sp.get_d("beta_rate_1");
        double br2;
        if (c.beta_rate_symmetric == 1) {
            br2 = br1;
        } else if (c.beta_rate_symmetric == 2) {
            // Same formula as benchmark_search / proposed_policy: br2 derived
            // from br1 via the V/H beta-constant ratio.
            br2 = 1.0 - (1.0 - br1) * BETA_CONST_V / BETA_CONST_H;
        } else {
            require(sp, "beta_rate_2", c.policy_name);
            br2 = sp.get_d("beta_rate_2");
        }
        RC_LAMBDA_TILDE = lam;
        RC_BETA_TILDE_V = br1 * BETA_CONST_V * r.rho;
        RC_BETA_TILDE_H = br2 * BETA_CONST_H * r.rho;

        r.wmp_extra_relative = sp.has("wmp_extra_relative")
                                 ? sp.get_i("wmp_extra_relative")
                                 : 0;
        RC_WMP_EXTRA_RELATIVE = r.wmp_extra_relative;
    } else {
        r.wmp_extra_relative   = 0;
        RC_WMP_EXTRA_RELATIVE  = 0;
    }

    return r;
}

// ---------------------------------------------------------------------------
// File-name prefix encoding the WMP variant.
// (ECC has no wmp_limiting concept; matches existing proposed_policy /
//  benchmark_search prefix logic — abs_/rel_ purely from wmp_extra_relative.)
// ---------------------------------------------------------------------------
static std::string wmp_prefix(const ResolvedPolicy& r) {
    if (!r.uses_wmp) return "";
    return (r.wmp_extra_relative == 0) ? "abs_" : "rel_";
}

// ---------------------------------------------------------------------------
// Echo selected parameters to stdout (debug aid).
// ---------------------------------------------------------------------------
static void print_input_summary(const ParsedInput& in, const ResolvedPolicy& r) {
    const OneComboCommon& c = in.common;
    const SectionParams& sp = in.sections.at(c.policy_name);
    printf("=================================================================\n");
    printf("sim_one_combo: policy = %s   (DIM_K=%d, N_ROUTES=%d)\n",
           c.policy_name.c_str(), DIM_K, N_ROUTES);
    printf("=================================================================\n");
    printf("  n_paths             = %d\n", c.n_paths);
    printf("  hist_bucket         = %g  (sim time units)\n", c.hist_bucket);
    printf("  output_dir          = %s\n", c.output_dir.c_str());
    printf("  print_detailed_info = %d\n", c.print_detailed_info);
    printf("  epsilon_relative    = %d\n", c.epsilon_relative);
    printf("  wmp_relative        = %d\n", c.wmp_relative);
    printf("  wmp_no_queue        = %d\n", c.wmp_no_queue);
    printf("  beta_rate_symmetric = %d\n", c.beta_rate_symmetric);
    printf("  inward_reflection   = %d  (used only by [proposed])\n", c.inward_reflection);
    if (r.is_proposed) {
        printf("  model_base_dir      = %s\n", c.model_base_dir.c_str());
        printf("  new_model_base_dir  = %s\n", c.new_model_base_dir.c_str());
    }
    printf("---\n");
    printf("[%s] section:\n", c.policy_name.c_str());
    printf("  safety  = %g\n", r.safety);
    printf("  rho     = %g\n", r.rho);
    if (r.has_epsilon) printf("  epsilon = %g\n", r.epsilon);
    if (sp.has("phi_own_0")) {
        printf("  BCP pricing (N_ROUTES*2 = %d fields):\n", N_ROUTES * 2);
        for (int rr = 0; rr < N_ROUTES; ++rr) {
            printf("    phi_own_%d=%s  phi_other_%d=%s\n",
                   rr, sp.raw("phi_own_"+std::to_string(rr)).c_str(),
                   rr, sp.raw("phi_other_"+std::to_string(rr)).c_str());
        }
    }
    if (sp.has("Q_loop_0")) {
        printf("  HighLow pricing (N_ROUTES*3 = %d fields, Q_BOUND=QUEUE_LEN_BOUND[r], INV_SQRT_N=%g):\n",
               N_ROUTES * 3, INV_SQRT_N);
        for (int rr = 0; rr < N_ROUTES; ++rr) {
            printf("    Q_loop_%d=%s  zeta_L_%d=%s  zeta_H_%d=%s  → z_thresh=%g\n",
                   rr, sp.raw("Q_loop_"+std::to_string(rr)).c_str(),
                   rr, sp.raw("zeta_L_"+std::to_string(rr)).c_str(),
                   rr, sp.raw("zeta_H_"+std::to_string(rr)).c_str(),
                   RC_HL_PARA[rr][0]);
        }
    }
    if (r.uses_wmp) {
        const char* br2_src = (in.common.beta_rate_symmetric == 1) ? "(=br1, symmetric=1)"
                            : (in.common.beta_rate_symmetric == 2) ? "(derived, symmetric=2)"
                            : "(from section)";
        printf("  WMP: lambda_tilde=%g  beta_rate_1=%s  beta_rate_2 %s\n",
               RC_LAMBDA_TILDE, sp.raw("beta_rate_1").c_str(), br2_src);
        printf("       → beta_tilde_V=%g  beta_tilde_H=%g\n", RC_BETA_TILDE_V, RC_BETA_TILDE_H);
        printf("  wmp_extra_relative=%d  → output prefix=\"%s\"\n",
               r.wmp_extra_relative, wmp_prefix(r).c_str());
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
    if (!fp) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return; }
    const SectionParams& sp = in.sections.at(in.common.policy_name);

    // ---- Header ----
    fprintf(fp, "policy,n_paths,safety,rho,epsilon");
    for (int rr = 0; rr < N_ROUTES; ++rr) fprintf(fp, ",phi_own_%d", rr);
    for (int rr = 0; rr < N_ROUTES; ++rr) fprintf(fp, ",phi_other_%d", rr);
    for (int rr = 0; rr < N_ROUTES; ++rr) fprintf(fp, ",Q_loop_%d", rr);
    for (int rr = 0; rr < N_ROUTES; ++rr) fprintf(fp, ",zeta_L_%d", rr);
    for (int rr = 0; rr < N_ROUTES; ++rr) fprintf(fp, ",zeta_H_%d", rr);
    fprintf(fp, ",lambda_tilde,beta_rate_1,beta_rate_2,beta_tilde_V,beta_tilde_H");
    fprintf(fp, ",wmp_extra_relative");
    fprintf(fp, ",cost_mean,cost_std");
    for (int rr = 0; rr < N_ROUTES; ++rr) fprintf(fp, ",delay%d_mean,delay%d_std", rr+1, rr+1);
    fprintf(fp, ",elapsed_sec\n");

    auto field = [&](const std::string& k) -> std::string {
        return sp.has(k) ? sp.raw(k) : std::string("");
    };

    // ---- Row ----
    fprintf(fp, "%s,%d,%g,%g,%s",
            in.common.policy_name.c_str(), in.common.n_paths,
            r.safety, r.rho,
            r.has_epsilon ? std::to_string(r.epsilon).c_str() : "");
    for (int rr = 0; rr < N_ROUTES; ++rr)
        fprintf(fp, ",%s", field("phi_own_"+std::to_string(rr)).c_str());
    for (int rr = 0; rr < N_ROUTES; ++rr)
        fprintf(fp, ",%s", field("phi_other_"+std::to_string(rr)).c_str());
    for (int rr = 0; rr < N_ROUTES; ++rr)
        fprintf(fp, ",%s", field("Q_loop_"+std::to_string(rr)).c_str());
    for (int rr = 0; rr < N_ROUTES; ++rr)
        fprintf(fp, ",%s", field("zeta_L_"+std::to_string(rr)).c_str());
    for (int rr = 0; rr < N_ROUTES; ++rr)
        fprintf(fp, ",%s", field("zeta_H_"+std::to_string(rr)).c_str());

    if (r.uses_wmp) {
        std::string br1 = field("beta_rate_1");
        std::string br2;
        if (in.common.beta_rate_symmetric == 1)      br2 = br1;
        else if (in.common.beta_rate_symmetric == 2) {
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%g",
                     1.0 - (1.0 - std::stod(br1)) * BETA_CONST_V / BETA_CONST_H);
            br2 = tmp;
        }
        else                                          br2 = field("beta_rate_2");
        fprintf(fp, ",%g,%s,%s,%g,%g,%d",
                RC_LAMBDA_TILDE, br1.c_str(), br2.c_str(),
                RC_BETA_TILDE_V, RC_BETA_TILDE_H, r.wmp_extra_relative);
    } else {
        fprintf(fp, ",,,,,,");  // 6 empty WMP columns (lam, br1, br2, btV, btH, extra_rel)
    }

    fprintf(fp, ",%.6f,%.6f", res.cost_mean, res.cost_std);
    for (int rr = 0; rr < N_ROUTES; ++rr)
        fprintf(fp, ",%.6f,%.6f", res.delay_prob_mean[rr], res.delay_prob_std[rr]);
    fprintf(fp, ",%.4f\n", elapsed_sec);
    fclose(fp);
    printf("Saved summary  : %s\n", path.c_str());
}

static void write_histogram_csv(const std::string& path,
                                const DelayHistogram& h) {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return; }

    // Header
    fprintf(fp, "bucket_lower");
    for (int rr = 0; rr < N_ROUTES; ++rr) fprintf(fp, ",count_route%d", rr);
    fprintf(fp, "\n");

    // Outer-join all per-route key sets to get the sorted union of bucket indices.
    std::vector<int> keys;
    size_t total = 0;
    for (int rr = 0; rr < N_ROUTES; ++rr) total += h.hist[rr].size();
    keys.reserve(total);
    for (int rr = 0; rr < N_ROUTES; ++rr)
        for (const auto& kv : h.hist[rr]) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    for (int k : keys) {
        double bucket_lower = (double)k * h.bucket_width;
        fprintf(fp, "%.6g", bucket_lower);
        for (int rr = 0; rr < N_ROUTES; ++rr) {
            uint64_t c = 0;
            auto it = h.hist[rr].find(k);
            if (it != h.hist[rr].end()) c = it->second;
            fprintf(fp, ",%llu", (unsigned long long)c);
        }
        fprintf(fp, "\n");
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
    std::string config_path = "one_combo_input.txt";
    if (argc >= 2) config_path = argv[1];

    printf("Reading input file: %s\n", config_path.c_str());
    ParsedInput parsed = parse_input_file(config_path);

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

    ResolvedPolicy r = apply_input(parsed);

    if (r.is_proposed) {
        if (parsed.common.model_base_dir.empty() && parsed.common.new_model_base_dir.empty()) {
            fprintf(stderr, "ERROR: [proposed] requires model_base_dir and/or new_model_base_dir.\n");
            return 1;
        }
        auto eta_theta     = build_eta_theta_map();
        auto eta_theta_b50 = build_eta_theta_b50_map();
        if (!rho_in_map(r.rho, eta_theta, eta_theta_b50)) {
            fprintf(stderr, "ERROR: rho=%.4g not in any eta_theta map (CC%d).\n",
                    r.rho, DIM_K);
            return 1;
        }
        std::string model_path = make_model_path(r.rho, eta_theta, eta_theta_b50,
                                                 parsed.common.model_base_dir,
                                                 parsed.common.new_model_base_dir);
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
    printf("Result: cost=%.4f ± %.4f", res.cost_mean, res.cost_std);
    for (int rr = 0; rr < N_ROUTES; ++rr)
        printf("  delay%d=%.3f%%±%.3f%%", rr+1, res.delay_prob_mean[rr], res.delay_prob_std[rr]);
    printf("  time=%.2fs\n", elapsed);
    printf("Histogram non-empty bucket counts:");
    for (int rr = 0; rr < N_ROUTES; ++rr)
        printf("  route%d=%zu", rr, hist.hist[rr].size());
    printf("\n");
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
