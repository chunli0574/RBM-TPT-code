/*
 * benchmark_kinds.h
 *
 * Shared benchmark-naming logic used by both:
 *   - benchmark_search_main.cpp  (sim_benchmark / sim_benchmark_omp)
 *   - one_combo_main.cpp         (sim_one_combo)
 *
 * Provides:
 *   - enum class BmkPricing/BmkOutsource/BmkScheduling
 *   - struct BmkKind
 *   - bmk_kind_from_name("A1".."D2") → BmkKind
 *   - set_policy_modes(kind)         → mutates RC_PRICING_MODE / RC_OUTSOURCING_MODE / RC_SCHEDULING_MODE
 *   - set_bcp_pricing_para(flat[4])  → fill RC_PRICING_PARA[2][2]
 *   - set_hl_pricing_para(flat[6])   → fill RC_HL_PARA[2][3]
 */

#pragma once

#include <string>

// ---------------------------------------------------------------------------
// Benchmark kind taxonomy
// ---------------------------------------------------------------------------
enum class BmkPricing    { BCPlinear, HighLow };
enum class BmkOutsource  { SinglePushing, ThresholdBased };
enum class BmkScheduling { LRS_DelayTime, LRS_QueueLen, WMP };

struct BmkKind {
    BmkPricing    pricing;
    BmkOutsource  outsource;
    BmkScheduling sched;
    bool          has_sched_search;  // Group C/D: also search scheduling params
};

// Parse "A1".."D2" → BmkKind. Aborts with stderr message on unknown name.
BmkKind bmk_kind_from_name(const std::string& name);

// Set RC_PRICING_MODE / RC_OUTSOURCING_MODE / RC_SCHEDULING_MODE from BmkKind.
void set_policy_modes(const BmkKind& kind);

// Set RC_PRICING_PARA for BCPlinear from a flattened 4-element array.
//   [phi_own_0, phi_other_0, phi_own_1, phi_other_1]
void set_bcp_pricing_para(const double flat[4]);

// Set RC_HL_PARA for High-Low from a flattened 6-element array.
//   [t0, zL0, zH0, t1, zL1, zH1]
void set_hl_pricing_para(const double flat[6]);
