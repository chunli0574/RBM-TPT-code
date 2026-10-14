/*
 * main.cpp
 *
 * Entry point: prints parameters, runs N_PATHS simulation paths, prints results.
 *
 * Python equivalent: calling simulation() then print_simulation_results() (line 920).
 *
 * TO RUN:
 *   make run
 * TO REBUILD FROM SCRATCH:
 *   make clean && make run
 */

#include "params.h"
#include "simulation.h"
#include "nn_model.h"
#include <cstdio>
#include <cmath>
#include <ctime>

static void print_config() {
    printf("=== W-Network Simulation (C++ — %d paths) ===\n\n", N_PATHS);
    printf("--- System parameters ---\n");
    printf("  n (scale)          = %d\n",   N_SCALE);
    printf("  inv_sqrt_n         = %.6f\n", INV_SQRT_N);
    printf("  N_JUMPS per path   = %d\n",   N_JUMPS);
    printf("  N_PATHS            = %d\n",   N_PATHS);
    printf("  INTEREST_RATE      = %.6f\n", INTEREST_RATE);
    printf("  LAMBDA             = [%.6f, %.6f]\n", LAMBDA[0], LAMBDA[1]);
    printf("  MUS                = [%.2f, %.2f, %.2f, %.2f]\n",
           MUS[0], MUS[1], MUS[2], MUS[3]);
    printf("  B_VEC              = [%.6f, %.6f]\n", B_VEC[0], B_VEC[1]);
    printf("  INITIAL_STATE      = [%d, %d, %d, %d]\n",
           INITIAL_STATE[0], INITIAL_STATE[1], INITIAL_STATE[2], INITIAL_STATE[3]);
    printf("  DELAY_BOUND        = [%.4f, %.4f]\n", DELAY_BOUND[0], DELAY_BOUND[1]);
    printf("  BUFFER_SIZE        = %d\n",   BUFFER_SIZE);
    printf("  DELAY_RECORD_START = %d\n",   DELAY_PROB_RECORD_START);
    printf("\n--- Policy parameters (fixed for this run) ---\n");
    // Pricing
    if (PRICING_MODE == BCP_LINEAR) {
        printf("  [Pricing]   Mode              = BCP_LINEAR\n");
        printf("  [Pricing]   PRICING_PARA row0 = [phi_own=%.4f, phi_other=%.4f]\n",
               PRICING_PARA[0][0], PRICING_PARA[0][1]);
        printf("  [Pricing]   PRICING_PARA row1 = [phi_own=%.4f, phi_other=%.4f]\n",
               PRICING_PARA[1][0], PRICING_PARA[1][1]);
    } else if (PRICING_MODE == HIGH_LOW) {
        printf("  [Pricing]   Mode              = HIGH_LOW\n");
        printf("  [Pricing]   HL_PARA row0      = [thr_z=%.4f, low=%.4f, high=%.4f]\n",
               HL_PARA[0][0], HL_PARA[0][1], HL_PARA[0][2]);
        printf("  [Pricing]   HL_PARA row1      = [thr_z=%.4f, low=%.4f, high=%.4f]\n",
               HL_PARA[1][0], HL_PARA[1][1], HL_PARA[1][2]);
    } else {
        printf("  [Pricing]   Mode              = NN_PRICING  (hold_last_price)\n");
        printf("  [Pricing]   NN_MODEL_PATH     = %s\n", NN_MODEL_PATH.c_str());
        printf("  [Pricing]   NN_PRICING_MATRIX = [[%.4f, %.4f], [%.4f, %.4f]]\n",
               NN_PRICING_MATRIX[0][0], NN_PRICING_MATRIX[0][1],
               NN_PRICING_MATRIX[1][0], NN_PRICING_MATRIX[1][1]);
    }
    // Scheduling
    if (SCHEDULING_MODE == WMP_OBJ2) {
        printf("  [Sched]     Mode              = WMP_OBJ2\n");
        printf("  [Sched]     LAMBDA_TILDE      = %.6f\n", LAMBDA_TILDE);
        printf("  [Sched]     BETA_TILDE_1      = %.6f\n", BETA_TILDE_1);
        printf("  [Sched]     BETA_TILDE_2      = %.6f\n", BETA_TILDE_2);
    } else if (SCHEDULING_MODE == LRS_QUEUE_LEN) {
        printf("  [Sched]     Mode              = LRS_QUEUE_LEN  (no parameters)\n");
    } else if (SCHEDULING_MODE == LRS_DELAY_TIME) {
        printf("  [Sched]     Mode              = LRS_DELAY_TIME  (no parameters)\n");
    } else {
        printf("  [Sched]     Mode              = LRS_REMAIN_DELAY  (no parameters)\n");
    }
    // Outsourcing
    if (OUTSOURCING_MODE == THRESHOLD_BASED) {
        printf("  [Outsource] Mode              = THRESHOLD_BASED\n");
        printf("  [Outsource] rho (threshold)   = %.6f\n", OUTSOURCE_THRESHOLD);
    } else if (OUTSOURCING_MODE == SINGLE_PUSHING) {
        printf("  [Outsource] Mode              = SINGLE_PUSHING\n");
        printf("  [Outsource] rho (face bound)  = %.6f\n", OUTSOURCE_THRESHOLD);
        printf("  [Outsource] epsilon_band      = %.6f\n", EPSILON_BAND);
    } else {
        // NN_OUTSOURCING
        const char* variant_str =
            (NN_OUTSOURCING_VARIANT == NN_INWARD)
            ? "epsilon_band_inward_outsource_simultaneously"
            : "epsilon_band_hold_last_outsource_simultaneously_no_takeover";
        printf("  [Outsource] Mode              = NN_OUTSOURCING\n");
        printf("  [Outsource] variant           = %s\n", variant_str);
        printf("  [Outsource] rho (face bound)  = %.6f\n", OUTSOURCE_THRESHOLD);
        printf("  [Outsource] epsilon_band      = %.6f\n", EPSILON_BAND);
    }
    printf("\n");
}

// print_result: display aggregate statistics over all N_PATHS paths.
//
// Python equivalent: SimulationInfo.print_simulation_results() (lines 804–824)
//   print(f"Cost: {cost_mean:.4f}, Std: {cost_std:.4f}")
//   print(f"Delay probability of each route: {delay_prob} %")
//   print(f"Outsourced percentage of each route: {outsourcing_percentage} %")
static void print_result(const SimResult& res) {
    printf("\n=== Results (mean ± stderr over %d paths) ===\n", N_PATHS);

    // Discounted cost
    // Python: cost_mean, cost_std = calculate_mean_std(rec_cost, M)
    //         print(f"Cost: {cost_mean:.4f}, Std: {cost_std:.4f}")
    printf("  Discounted cost      : %.4f  ±  %.4f\n", res.cost_mean, res.cost_std);

    // Delay probability per route
    // Python: print(f"Delay probability of each route: {np.round(delay_prob, 2)} %")
    printf("  Delay prob (route 1) : %.3f  ±  %.3f %%\n",
           res.delay_prob_mean[0], res.delay_prob_std[0]);
    printf("  Delay prob (route 2) : %.3f  ±  %.3f %%\n",
           res.delay_prob_mean[1], res.delay_prob_std[1]);

    // Outsourcing fraction
    // Python: outsourcing_percentage = np.mean(t_outsourced / t) * 100
    printf("  Outsource (route 1)  : %.2f %%\n", res.outsource_frac_mean[0] * 100.0);
    printf("  Outsource (route 2)  : %.2f %%\n", res.outsource_frac_mean[1] * 100.0);

    printf("  Wall-clock time      : %.2f seconds\n", res.elapsed_seconds);
}

int main() {
    print_config();

    // Load NN model if NN_PRICING or NN_OUTSOURCING is active.
    // The model is loaded once and reused across all N_PATHS simulation paths.
    // Python: model = load_savedmodel_as_keras(savedmodel_dir) in functions_general.py
    if (PRICING_MODE == NN_PRICING || OUTSOURCING_MODE == NN_OUTSOURCING) {
        printf("Loading NN model from:\n  %s\n", NN_MODEL_PATH.c_str());
        nn_model_load(NN_MODEL_PATH);
        printf("NN model loaded.\n\n");
    }

    clock_t t_start = clock();
    SimResult result = run_simulation();
    clock_t t_end   = clock();

    result.elapsed_seconds = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    print_result(result);
    if (PRICING_MODE == NN_PRICING || OUTSOURCING_MODE == NN_OUTSOURCING) {
        nn_cache_stats();
    }
    return 0;
}
