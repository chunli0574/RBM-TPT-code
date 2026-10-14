/*
 * profiling.h
 *
 * Optional 3-level TIC/TOC profiling, guarded by ENABLE_PROFILING.
 *
 * Build with `make PROFILING=1 <target>` to enable; omit PROFILING= for zero
 * overhead (all macros expand to no-ops).
 *
 * Groups:
 *   L1  — inside run_one_path (simulation.cpp): S1..S5 + path_init + aggregate_mean_std
 *   L2  — inside compute_policy (policy.cpp):   P1..P5
 *   L3  — benchmark search outer (benchmark_search_main.cpp)
 *
 * Output: atexit handler prints three cumulative tables (ns + %sum or %total).
 * Thread-safe under OpenMP: counters are std::atomic<uint64_t> in nanoseconds.
 */

#pragma once

#ifdef ENABLE_PROFILING

#include <atomic>
#include <chrono>
#include <cstdint>

// ---------------------------------------------------------------------------
// L1: run_one_path sections (simulation.cpp)
// ---------------------------------------------------------------------------
enum ProfL1Id {
    PROF_L1_S1_DELAY_TIME = 0,
    PROF_L1_S2_COMPUTE_POLICY,
    PROF_L1_S3_NEXT_EVENT,
    PROF_L1_S4_ACCUMULATE_COST,
    PROF_L1_S5_STATE_AND_QUEUE,
    PROF_L1_PATH_INIT,
    PROF_L1_AGGREGATE_MEAN_STD,
    PROF_L1_COUNT
};

// ---------------------------------------------------------------------------
// L2: compute_policy sections (policy.cpp)
// ---------------------------------------------------------------------------
enum ProfL2Id {
    PROF_L2_P1_Z_W_PREP = 0,
    PROF_L2_P2_PRICING,
    PROF_L2_P3_SCHEDULING,
    PROF_L2_P4_OUTSOURCING,
    PROF_L2_P5_HARDSTOP_IDLING,
    PROF_L2_COUNT
};

// ---------------------------------------------------------------------------
// L3: benchmark search outer (benchmark_search_main.cpp)
// TOTAL_SEARCH is special: it brackets main()'s body; "other (setup/loop)"
// is derived as TOTAL_SEARCH - (RUN_ONE_BMK + SAVE_BMK_CSV + FIND_MIN_COST).
// ---------------------------------------------------------------------------
enum ProfL3Id {
    PROF_L3_RUN_ONE_BMK = 0,
    PROF_L3_SAVE_BMK_CSV,
    PROF_L3_FIND_MIN_COST,
    PROF_L3_TOTAL_SEARCH,
    PROF_L3_COUNT
};

extern std::atomic<uint64_t> prof_L1_counters[PROF_L1_COUNT];
extern std::atomic<uint64_t> prof_L2_counters[PROF_L2_COUNT];
extern std::atomic<uint64_t> prof_L3_counters[PROF_L3_COUNT];

// RAII scope that measures elapsed nanoseconds on destruction and accumulates
// atomically into the given counter. Exactly one per {} block — use the
// PROF_L* macros below, which declare a unique-named instance.
struct ProfScope {
    std::atomic<uint64_t>* counter;
    std::chrono::steady_clock::time_point t0;
    explicit ProfScope(std::atomic<uint64_t>* c)
        : counter(c), t0(std::chrono::steady_clock::now()) {}
    ~ProfScope() {
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        counter->fetch_add(static_cast<uint64_t>(ns),
                           std::memory_order_relaxed);
    }
};

#define PROF_CAT_(a, b) a##b
#define PROF_CAT(a, b)  PROF_CAT_(a, b)

// One-liner macros: declare a local ProfScope with a unique name so multiple
// can coexist in the same outer scope (even though typically only one per {}).
#define PROF_L1(id) ProfScope PROF_CAT(_prof_l1_, __LINE__)(&prof_L1_counters[PROF_L1_##id])
#define PROF_L2(id) ProfScope PROF_CAT(_prof_l2_, __LINE__)(&prof_L2_counters[PROF_L2_##id])
#define PROF_L3(id) ProfScope PROF_CAT(_prof_l3_, __LINE__)(&prof_L3_counters[PROF_L3_##id])

// Print all three tables; automatically registered with std::atexit.
void prof_print_all();

#else  // !ENABLE_PROFILING

// Zero-overhead no-op stubs
#define PROF_L1(id) ((void)0)
#define PROF_L2(id) ((void)0)
#define PROF_L3(id) ((void)0)

inline void prof_print_all() {}

#endif  // ENABLE_PROFILING
