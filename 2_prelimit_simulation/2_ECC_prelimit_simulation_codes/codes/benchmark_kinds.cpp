/*
 * benchmark_kinds.cpp
 *
 * Implementation of benchmark naming + RC_* mode dispatch.
 * Originally lived in benchmark_search_main.cpp; extracted so sim_one_combo
 * (and any future binary) can share the exact same logic without duplication.
 *
 * Pure refactor — no behavioural change vs the original definitions.
 */

#include "benchmark_kinds.h"
#include "params.h"        // for enum constants (BCP_LINEAR, HIGH_LOW, ...)
#include "run_config.h"    // for RC_PRICING_MODE / RC_OUTSOURCING_MODE / RC_SCHEDULING_MODE / RC_PRICING_PHI / RC_HL_PARA

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Derive benchmark kind from name string ("A1".."D2").
// ---------------------------------------------------------------------------
BmkKind bmk_kind_from_name(const std::string& name) {
    if      (name == "A1") return {BmkPricing::BCPlinear, BmkOutsource::SinglePushing,  BmkScheduling::LRS_DelayTime, false};
    else if (name == "A2") return {BmkPricing::BCPlinear, BmkOutsource::SinglePushing,  BmkScheduling::LRS_QueueLen,  false};
    else if (name == "A3") return {BmkPricing::HighLow,   BmkOutsource::SinglePushing,  BmkScheduling::LRS_DelayTime, false};
    else if (name == "A4") return {BmkPricing::HighLow,   BmkOutsource::SinglePushing,  BmkScheduling::LRS_QueueLen,  false};
    else if (name == "B1") return {BmkPricing::BCPlinear, BmkOutsource::ThresholdBased, BmkScheduling::LRS_DelayTime, false};
    else if (name == "B2") return {BmkPricing::BCPlinear, BmkOutsource::ThresholdBased, BmkScheduling::LRS_QueueLen,  false};
    else if (name == "B3") return {BmkPricing::HighLow,   BmkOutsource::ThresholdBased, BmkScheduling::LRS_DelayTime, false};
    else if (name == "B4") return {BmkPricing::HighLow,   BmkOutsource::ThresholdBased, BmkScheduling::LRS_QueueLen,  false};
    else if (name == "C1") return {BmkPricing::BCPlinear, BmkOutsource::SinglePushing,  BmkScheduling::WMP,           true};
    else if (name == "C2") return {BmkPricing::HighLow,   BmkOutsource::SinglePushing,  BmkScheduling::WMP,           true};
    else if (name == "D1") return {BmkPricing::BCPlinear, BmkOutsource::ThresholdBased, BmkScheduling::WMP,           true};
    else if (name == "D2") return {BmkPricing::HighLow,   BmkOutsource::ThresholdBased, BmkScheduling::WMP,           true};
    else {
        fprintf(stderr, "ERROR: unknown benchmark '%s'. Valid: A1-A4, B1-B4, C1-C2, D1-D2\n", name.c_str());
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Set RC_* policy modes from benchmark kind
// ---------------------------------------------------------------------------
void set_policy_modes(const BmkKind& kind) {
    RC_PRICING_MODE     = (kind.pricing == BmkPricing::BCPlinear) ? BCP_LINEAR : HIGH_LOW;
    RC_OUTSOURCING_MODE = (kind.outsource == BmkOutsource::SinglePushing) ? SINGLE_PUSHING : THRESHOLD_BASED;
    if (kind.sched == BmkScheduling::WMP)
        RC_SCHEDULING_MODE = WMP_OBJ2;
    else if (kind.sched == BmkScheduling::LRS_QueueLen)
        RC_SCHEDULING_MODE = LRS_QUEUE_LEN;
    else
        RC_SCHEDULING_MODE = LRS_DELAY_TIME;
}

// ---------------------------------------------------------------------------
// Set RC_PRICING_PHI from flat array (BCPlinear, N_ROUTES × 2).
// ---------------------------------------------------------------------------
void set_bcp_pricing_phi(const double flat[N_ROUTES * 2]) {
    for (int r = 0; r < N_ROUTES; ++r) {
        RC_PRICING_PHI[r][0] = flat[r * 2 + 0];
        RC_PRICING_PHI[r][1] = flat[r * 2 + 1];
    }
}

// ---------------------------------------------------------------------------
// Set RC_HL_PARA from flat array (High_low, N_ROUTES × 3).
// ---------------------------------------------------------------------------
void set_hl_pricing_para(const double flat[N_ROUTES * 3]) {
    for (int r = 0; r < N_ROUTES; ++r)
        for (int k = 0; k < 3; ++k)
            RC_HL_PARA[r][k] = flat[r * 3 + k];
}
