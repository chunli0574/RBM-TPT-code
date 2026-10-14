/*
 * benchmark_kinds.h
 *
 * Shared benchmark-naming logic used by:
 *   - benchmark_search_main.cpp  (sim_benchmark / sim_benchmark_omp)
 *   - one_combo_main.cpp         (sim_one_combo / sim_one_combo_batch)
 *
 * Provides:
 *   - enum class BmkPricing/BmkOutsource/BmkScheduling
 *   - struct BmkKind
 *   - bmk_kind_from_name("A1".."D2") → BmkKind
 *   - set_policy_modes(kind)         → mutates RC_PRICING_MODE / RC_OUTSOURCING_MODE / RC_SCHEDULING_MODE
 *   - set_bcp_pricing_phi(flat)      → fill RC_PRICING_PHI[N_ROUTES][2]    (N_ROUTES*2 elements)
 *   - set_hl_pricing_para(flat)      → fill RC_HL_PARA[N_ROUTES][3]        (N_ROUTES*3 elements)
 *
 * N_ROUTES is compile-time (3 for CC3 / 11 for CC10), selected via DIM_K_SELECT
 * in params.h.
 */

#pragma once

#include "params.h"   // for N_ROUTES
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

// Set RC_PRICING_PHI for BCPlinear from a flattened (N_ROUTES * 2)-element array.
//   flat[r*2 + 0] = phi_r    (own-buffer coefficient for route r)
//   flat[r*2 + 1] = phi_{-r} (other-buffer coefficient for route r)
void set_bcp_pricing_phi(const double flat[N_ROUTES * 2]);

// Set RC_HL_PARA for High-Low from a flattened (N_ROUTES * 3)-element array.
//   flat[r*3 + 0] = z_thresh (Z-scale, already × INV_SQRT_N)
//   flat[r*3 + 1] = zeta_L
//   flat[r*3 + 2] = zeta_H
void set_hl_pricing_para(const double flat[N_ROUTES * 3]);
