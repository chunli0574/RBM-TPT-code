/*
 * main.cpp
 *
 * Entry point for CC3 (Extended Criss-Cross, dim_K=3) simulation.
 * Prints parameters, runs N_PATHS simulation paths, prints results.
 *
 * Python equivalent: calling simulation() then print_simulation_results()
 *   in CC3_python_code/codes/functions_general.py
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
    printf("=== CC3 Extended Criss-Cross Simulation (C++ — %d paths) ===\n\n", N_PATHS);
    printf("--- System parameters ---\n");
    printf("  dim_K (stations)   = %d\n",   DIM_K);
    printf("  N_BUFFERS          = %d\n",   N_BUFFERS);
    printf("  N_ROUTES           = %d\n",   N_ROUTES);
    printf("  n (scale)          = %d\n",   N_SCALE);
    printf("  inv_sqrt_n         = %.6f\n", INV_SQRT_N);
    printf("  N_JUMPS per path   = %d\n",   N_JUMPS);
    printf("  N_PATHS            = %d\n",   N_PATHS);
    printf("  INTEREST_RATE      = %.6f\n", INTEREST_RATE);
    printf("  LAMBDA             = ["); for (int i=0;i<N_ROUTES; ++i) printf("%.6f%s",LAMBDA[i],      i<N_ROUTES-1?", ":""); printf("]\n");
    printf("  MUS                = ["); for (int i=0;i<N_BUFFERS;++i) printf("%.2f%s",  MUS[i],        i<N_BUFFERS-1?", ":""); printf("]\n");
    printf("  B_VEC              = ["); for (int i=0;i<N_ROUTES; ++i) printf("%.6f%s",B_VEC[i],        i<N_ROUTES-1?", ":""); printf("]\n");
    printf("  INITIAL_STATE      = ["); for (int i=0;i<N_BUFFERS;++i) printf("%d%s",  INITIAL_STATE[i],i<N_BUFFERS-1?", ":""); printf("]\n");
    printf("  DELAY_BOUND        = ["); for (int i=0;i<N_ROUTES; ++i) printf("%.4f%s",DELAY_BOUND[i],  i<N_ROUTES-1?", ":""); printf("]\n");
    printf("  DIAG_FACE_CONST    = %.6f\n", DIAG_FACE_CONST);
    printf("  OFFDIAG_FACE_CONST = %.6f\n", OFFDIAG_FACE_CONST);
    printf("  BUFFER_SIZE        = %d\n",   BUFFER_SIZE);
    printf("  DELAY_RECORD_START = %d\n",   DELAY_PROB_RECORD_START);
    printf("\n--- Policy parameters (fixed for this run) ---\n");

    // Pricing
    if (PRICING_MODE == NN_PRICING) {
        printf("  [Pricing]   Mode              = NN_PRICING  (hold_last_price)\n");
        printf("  [Pricing]   NN_MODEL_PATH     = %s\n", NN_MODEL_PATH.c_str());
        for (int r = 0; r < N_ROUTES; ++r) {
            printf("  [Pricing]   NN_PRICING_MATRIX row%d = [", r);
            for (int k = 0; k < DIM_K; ++k)
                printf("%.4f%s", NN_PRICING_MATRIX[r][k], k < DIM_K-1 ? ", " : "");
            printf("]\n");
        }
    } else {
        printf("  [Pricing]   Mode              = BCP_LINEAR\n");
    }

    // Scheduling
    if (SCHEDULING_MODE == WMP_OBJ2) {
        printf("  [Sched]     Mode              = WMP_OBJ2\n");
        printf("  [Sched]     LAMBDA_TILDE      = %.6f\n", LAMBDA_TILDE);
        printf("  [Sched]     BETA_TILDE_V      = %.6f\n", BETA_TILDE_V);
        printf("  [Sched]     BETA_TILDE_H      = %.6f\n", BETA_TILDE_H);
    } else {
        printf("  [Sched]     Mode              = LRS variant\n");
    }

    // Outsourcing
    if (OUTSOURCING_MODE == NN_OUTSOURCING) {
        printf("  [Outsource] Mode              = NN_OUTSOURCING\n");
        printf("  [Outsource] variant           = hold_last_outsource_simultaneously_no_takeover\n");
        printf("  [Outsource] rho               = %.6f\n", OUTSOURCE_THRESHOLD);
        printf("  [Outsource] epsilon_band      = %.6f\n", EPSILON_BAND);
        printf("  [Outsource] safety_stock      = %.4f\n", SAFETY_STOCK);
    } else {
        printf("  [Outsource] Mode              = THRESHOLD\n");
        printf("  [Outsource] rho               = %.6f\n", OUTSOURCE_THRESHOLD);
    }
    printf("\n");
}

static void print_result(const SimResult& res) {
    printf("\n=== Results (mean ± stderr over %d paths) ===\n", N_PATHS);

    // Discounted cost
    printf("  Discounted cost        : %.4f  ±  %.4f\n", res.cost_mean, res.cost_std);

    // Delay probability per route
    for (int r = 0; r < N_ROUTES; ++r) {
        printf("  Delay prob (route %d)   : %.3f  ±  %.3f %%\n",
               r + 1, res.delay_prob_mean[r], res.delay_prob_std[r]);
    }

    // Outsourcing fraction
    for (int r = 0; r < N_ROUTES; ++r) {
        printf("  Outsource (route %d)    : %.2f %%\n",
               r + 1, res.outsource_frac_mean[r] * 100.0);
    }

    printf("  Wall-clock time        : %.2f seconds\n", res.elapsed_seconds);
}

int main() {
    print_config();

    // Load NN model if NN pricing or NN outsourcing is active.
    // Model is loaded once and reused across all N_PATHS paths.
    // Python: model = load_savedmodel_as_keras(savedmodel_dir)
    bool need_nn = (PRICING_MODE    == NN_PRICING ||
                    OUTSOURCING_MODE == NN_OUTSOURCING);
    if (need_nn) {
        printf("Loading NN model from:\n  %s\n", NN_MODEL_PATH.c_str());
        nn_model_load(NN_MODEL_PATH);
        printf("NN model loaded.\n\n");
    }

    clock_t t_start = clock();
    SimResult result = run_simulation();
    clock_t t_end   = clock();

    result.elapsed_seconds = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    print_result(result);

    if (need_nn) {
        nn_cache_stats();
    }
    return 0;
}
