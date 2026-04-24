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

double pricing_optimistic_reward(
    const uint64_t *stop_bits,
    const double *dual_rewards,
    const double *demands,
    const double *services,
    const uint64_t *separated_masks,
    const int32_t *forced_pred_nodes,
    const uint64_t *forced_pred_bits,
    size_t positive_count,
    const int32_t *density_order,
    size_t density_count,
    const int32_t *service_order,
    size_t service_count,
    uint64_t visited_mask,
    uint64_t requirement_mask,
    int32_t current_node,
    double optional_capacity,
    double optional_service_time
) {
    double mandatory_reward = 0.0;
    double capacity_reward = 0.0;
    double service_reward = 0.0;

    for (size_t i = 0; i < positive_count; ++i) {
        uint64_t stop_bit = stop_bits[i];
        if ((visited_mask & stop_bit) != 0) {
            continue;
        }
        if ((separated_masks[i] & visited_mask) != 0) {
            continue;
        }
        if (
            forced_pred_bits[i] != 0
            && forced_pred_nodes[i] != current_node
            && (visited_mask & forced_pred_bits[i]) != 0
        ) {
            continue;
        }
        if ((requirement_mask & stop_bit) != 0) {
            mandatory_reward += dual_rewards[i];
        }
    }

    if (optional_capacity > 1e-9) {
        for (size_t order_pos = 0; order_pos < density_count; ++order_pos) {
            int32_t idx = density_order[order_pos];
            uint64_t stop_bit = stop_bits[idx];
            if ((requirement_mask & stop_bit) != 0 || (visited_mask & stop_bit) != 0) {
                continue;
            }
            if ((separated_masks[idx] & visited_mask) != 0) {
                continue;
            }
            if (
                forced_pred_bits[idx] != 0
                && forced_pred_nodes[idx] != current_node
                && (visited_mask & forced_pred_bits[idx]) != 0
            ) {
                continue;
            }

            if (demands[idx] <= optional_capacity + 1e-9) {
                capacity_reward += dual_rewards[idx];
                optional_capacity -= demands[idx];
            } else {
                capacity_reward += dual_rewards[idx] * (optional_capacity / demands[idx]);
                break;
            }
        }
    }

    if (optional_service_time > 1e-9) {
        for (size_t order_pos = 0; order_pos < service_count; ++order_pos) {
            int32_t idx = service_order[order_pos];
            uint64_t stop_bit = stop_bits[idx];
            if ((requirement_mask & stop_bit) != 0 || (visited_mask & stop_bit) != 0) {
                continue;
            }
            if ((separated_masks[idx] & visited_mask) != 0) {
                continue;
            }
            if (
                forced_pred_bits[idx] != 0
                && forced_pred_nodes[idx] != current_node
                && (visited_mask & forced_pred_bits[idx]) != 0
            ) {
                continue;
            }

            if (services[idx] <= optional_service_time + 1e-9) {
                service_reward += dual_rewards[idx];
                optional_service_time -= services[idx];
            } else {
                service_reward += dual_rewards[idx] * (optional_service_time / services[idx]);
                break;
            }
        }
    }

    return mandatory_reward + (capacity_reward < service_reward ? capacity_reward : service_reward);
}
