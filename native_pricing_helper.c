#include <stddef.h>
#include <stdint.h>

int pricing_dominance_scan(
    const int32_t *idx_array,
    size_t count,
    const double *time_after_service,
    const double *load_kg,
    const double *nonlabor_reduced_cost,
    const uint64_t *visited_mask,
    uint8_t *alive,
    double label_time,
    double label_load,
    double label_nonlabor,
    uint64_t label_mask,
    int incumbent_can_dominate,
    int label_can_dominate,
    int32_t *survivor_indices,
    size_t *survivor_count
) {
    const double tol = 1e-9;
    size_t survivor_total = 0;

    for (size_t pos = 0; pos < count; ++pos) {
        int32_t idx = idx_array[pos];
        if (!alive[idx]) {
            continue;
        }

        double incumbent_time = time_after_service[idx];
        double incumbent_load = load_kg[idx];
        double incumbent_nonlabor = nonlabor_reduced_cost[idx];
        uint64_t incumbent_mask = visited_mask[idx];

        if (
            incumbent_can_dominate
            && incumbent_time <= label_time + tol
            && incumbent_load <= label_load + tol
            && incumbent_nonlabor <= label_nonlabor + tol
            && (incumbent_mask & ~label_mask) == 0
        ) {
            *survivor_count = 0;
            return 1;
        }

        if (
            label_can_dominate
            && label_time <= incumbent_time + tol
            && label_load <= incumbent_load + tol
            && label_nonlabor <= incumbent_nonlabor + tol
            && (label_mask & ~incumbent_mask) == 0
        ) {
            alive[idx] = 0;
            continue;
        }

        survivor_indices[survivor_total++] = idx;
    }

    *survivor_count = survivor_total;
    return 0;
}
