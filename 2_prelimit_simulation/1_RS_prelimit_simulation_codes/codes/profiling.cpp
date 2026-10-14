/*
 * profiling.cpp
 *
 * Counters + atexit-registered summary printer for the 3-level profiling.
 * All content is guarded by ENABLE_PROFILING so a build without -DENABLE_PROFILING=1
 * links to an empty object.
 */

#include "profiling.h"

#ifdef ENABLE_PROFILING

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Counter storage
// ---------------------------------------------------------------------------
std::atomic<uint64_t> prof_L1_counters[PROF_L1_COUNT];
std::atomic<uint64_t> prof_L2_counters[PROF_L2_COUNT];
std::atomic<uint64_t> prof_L3_counters[PROF_L3_COUNT];

// ---------------------------------------------------------------------------
// Section names (display strings)
// ---------------------------------------------------------------------------
static const char* const prof_L1_names[PROF_L1_COUNT] = {
    "S1_delay_time",
    "S2_compute_policy",
    "S3_next_event",
    "S4_accumulate_cost",
    "S5_state_and_queue",
    "path_init",
    "aggregate_mean_std"
};

static const char* const prof_L2_names[PROF_L2_COUNT] = {
    "P1_Z_W_prep",
    "P2_pricing",
    "P3_scheduling",
    "P4_outsourcing",
    "P5_hardstop_idling"
};

static const char* const prof_L3_names[PROF_L3_COUNT] = {
    "run_one_bmk (sim)",
    "save_bmk_csv",
    "find_min_cost",
    "total_search"
};

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------

// L1 / L2: cumulative table, denominator = sum of all entries.
static void print_cumulative_table(const char* title,
                                    const char* const* names,
                                    std::atomic<uint64_t>* counters,
                                    int count) {
    uint64_t sum = 0;
    for (int i = 0; i < count; ++i) sum += counters[i].load();
    uint64_t denom = (sum > 0) ? sum : 1;

    printf("========== PROFILING (%s, cumulative) ==========\n", title);
    printf("  %-28s %18s %8s\n", "section", "ns", "%sum");
    for (int i = 0; i < count; ++i) {
        uint64_t ns = counters[i].load();
        printf("  %-28s %18llu %7.2f%%\n",
               names[i],
               (unsigned long long)ns,
               100.0 * (double)ns / (double)denom);
    }
    printf("  %-28s %18llu %7.2f%%\n",
           "sum", (unsigned long long)sum, 100.00);
    printf("============================================================\n\n");
}

// L3: benchmark search outer. Denominator = total_search. Prints an implicit
// "other (setup/loop)" = total_search - (run_one_bmk + save_bmk_csv + find_min_cost).
static void print_search_outer_table() {
    const uint64_t total = prof_L3_counters[PROF_L3_TOTAL_SEARCH].load();
    uint64_t denom = (total > 0) ? total : 1;

    uint64_t known = 0;
    for (int i = 0; i < PROF_L3_COUNT; ++i) {
        if (i == PROF_L3_TOTAL_SEARCH) continue;
        known += prof_L3_counters[i].load();
    }
    uint64_t other = (total > known) ? (total - known) : 0;

    printf("========== PROFILING (benchmark search outer) ==========\n");
    printf("  %-28s %18s %8s\n", "section", "ns", "%total");
    // Non-total rows first (in enum order)
    for (int i = 0; i < PROF_L3_COUNT; ++i) {
        if (i == PROF_L3_TOTAL_SEARCH) continue;
        uint64_t ns = prof_L3_counters[i].load();
        printf("  %-28s %18llu %7.2f%%\n",
               prof_L3_names[i],
               (unsigned long long)ns,
               100.0 * (double)ns / (double)denom);
    }
    // Implicit "other (setup/loop)"
    printf("  %-28s %18llu %7.2f%%\n",
           "other (setup/loop)",
           (unsigned long long)other,
           100.0 * (double)other / (double)denom);
    // Finally total_search
    printf("  %-28s %18llu %7.2f%%\n",
           prof_L3_names[PROF_L3_TOTAL_SEARCH],
           (unsigned long long)total,
           100.00);
    printf("============================================================\n\n");
}

// ---------------------------------------------------------------------------
// Public: print all three tables
// ---------------------------------------------------------------------------
void prof_print_all() {
    print_cumulative_table("compute_policy",
                           prof_L2_names, prof_L2_counters, PROF_L2_COUNT);
    print_cumulative_table("serial run_one_path",
                           prof_L1_names, prof_L1_counters, PROF_L1_COUNT);
    print_search_outer_table();
}

// ---------------------------------------------------------------------------
// Auto-register atexit so every binary that links profiling.o prints on exit.
// ---------------------------------------------------------------------------
namespace {
struct ProfAtExitRegistrar {
    ProfAtExitRegistrar() { std::atexit(prof_print_all); }
};
static ProfAtExitRegistrar _prof_atexit_registrar;
}  // anonymous namespace

#endif  // ENABLE_PROFILING
