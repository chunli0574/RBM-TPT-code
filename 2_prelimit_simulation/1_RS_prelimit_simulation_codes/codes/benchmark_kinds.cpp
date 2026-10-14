/*
 * benchmark_kinds.cpp
 *
 * Implementation of benchmark naming + RC_* mode dispatch.
 * Originally lived in benchmark_search_main.cpp; extracted so sim_one_combo
 * can share the exact same logic without duplication.
 */

#include "benchmark_kinds.h"
#include "params.h"        // for enum constants (BCP_LINEAR, HIGH_LOW, ...)
#include "run_config.h"    // for RC_PRICING_MODE / RC_OUTSOURCING_MODE / RC_SCHEDULING_MODE / RC_PRICING_PARA / RC_HL_PARA

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
    // Pricing mode
    RC_PRICING_MODE = (kind.pricing == BmkPricing::BCPlinear) ? BCP_LINEAR : HIGH_LOW;
    // Outsourcing mode
    RC_OUTSOURCING_MODE = (kind.outsource == BmkOutsource::SinglePushing) ? SINGLE_PUSHING : THRESHOLD_BASED;
    // Scheduling mode
    if (kind.sched == BmkScheduling::WMP)
        RC_SCHEDULING_MODE = WMP_OBJ2;
    else if (kind.sched == BmkScheduling::LRS_QueueLen)
        RC_SCHEDULING_MODE = LRS_QUEUE_LEN;
    else
        RC_SCHEDULING_MODE = LRS_DELAY_TIME;
}

// ---------------------------------------------------------------------------
// Set RC_PRICING_PARA for BCPlinear from a flattened 4-element array.
// Compact layout (2 scalars per route): [phi_own_0, phi_other_0, phi_own_1, phi_other_1]
// ---------------------------------------------------------------------------
void set_bcp_pricing_para(const double flat[4]) {
    RC_PRICING_PARA[0][0] = flat[0];  // phi_own for route 0
    RC_PRICING_PARA[0][1] = flat[1];  // phi_other for route 0
    RC_PRICING_PARA[1][0] = flat[2];  // phi_own for route 1
    RC_PRICING_PARA[1][1] = flat[3];  // phi_other for route 1
}

// ---------------------------------------------------------------------------
// Set RC_HL_PARA for High-Low from a flattened 6-element array
//   flat = [t0, zL0, zH0, t1, zL1, zH1]
// ---------------------------------------------------------------------------
void set_hl_pricing_para(const double flat[6]) {
    for (int k = 0; k < 3; k++) {
        RC_HL_PARA[0][k] = flat[k];
        RC_HL_PARA[1][k] = flat[3 + k];
    }
}
