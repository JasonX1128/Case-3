from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, time
from pprint import pprint
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from vrp_instance_data import (
    BIG_M,
    COST_PARAMETERS,
    DEPOT_CLOSE,
    DEPOT_OPEN,
    DISTANCE_MATRIX,
    STOPS,
    TRAVEL_TIME_MATRIX,
    VEHICLES,
)


DEPOT_NODE = 0
EARLY_FLAG_MINUTES = 0.001


@dataclass(frozen=True)
class StopData:
    stop_id: int
    customer_name: str
    demand_kg: float
    earliest_min: float
    latest_min: float
    service_min: float


@dataclass(frozen=True)
class VehicleData:
    vehicle_id: int
    vehicle_type: str
    capacity_kg: float
    cost_per_mile: float
    fixed_daily_cost: float
    available_from_min: float
    available_until_min: float


@dataclass(frozen=True)
class Instance:
    stops: dict[int, StopData]
    vehicles: dict[int, VehicleData]
    distance_miles: dict[tuple[int, int], float]
    travel_minutes: dict[tuple[int, int], float]
    cost_params: dict[str, float]
    depot_open_min: float
    depot_close_min: float
    big_m: float


def excel_time_to_minutes(value: Any) -> float:
    if value is None:
        raise ValueError("Encountered an empty time value.")
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute + value.second / 60
    if isinstance(value, time):
        return value.hour * 60 + value.minute + value.second / 60
    if isinstance(value, (int, float)):
        if 0 <= value <= 2:
            return float(value) * 24 * 60
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                return parsed.hour * 60 + parsed.minute + parsed.second / 60
            except ValueError:
                pass
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Unsupported time format: {value!r}") from exc
    raise TypeError(f"Unsupported time value type: {type(value)!r}")


def minutes_to_clock(minutes: float) -> str:
    total_minutes = int(round(minutes))
    hours, mins = divmod(total_minutes, 60)
    return f"{hours:02d}:{mins:02d}"


def load_instance() -> Instance:
    stops: dict[int, StopData] = {}
    for row in STOPS:
        stop_id = int(row["stop_id"])
        stops[stop_id] = StopData(
            stop_id=stop_id,
            customer_name=str(row["customer_name"]),
            demand_kg=float(row["demand_kg"]),
            earliest_min=excel_time_to_minutes(row["earliest"]),
            latest_min=excel_time_to_minutes(row["latest"]),
            service_min=float(row["service_min"]),
        )

    vehicles: dict[int, VehicleData] = {}
    for row in VEHICLES:
        vehicle_id = int(row["vehicle_id"])
        vehicles[vehicle_id] = VehicleData(
            vehicle_id=vehicle_id,
            vehicle_type=str(row["vehicle_type"]),
            capacity_kg=float(row["capacity_kg"]),
            cost_per_mile=float(row["cost_per_mile"]),
            fixed_daily_cost=float(row["fixed_daily_cost"]),
            available_from_min=excel_time_to_minutes(row["available_from"]),
            available_until_min=excel_time_to_minutes(row["available_until"]),
        )

    depot_open_min = excel_time_to_minutes(DEPOT_OPEN)
    depot_close_min = excel_time_to_minutes(DEPOT_CLOSE)

    node_ids = [DEPOT_NODE] + sorted(stops)
    expected_size = len(node_ids)
    if len(DISTANCE_MATRIX) != expected_size or any(len(row) != expected_size for row in DISTANCE_MATRIX):
        raise ValueError(
            "Distance matrix dimensions do not match the number of nodes in the Python data."
        )
    if len(TRAVEL_TIME_MATRIX) != expected_size or any(
        len(row) != expected_size for row in TRAVEL_TIME_MATRIX
    ):
        raise ValueError(
            "Travel-time matrix dimensions do not match the number of nodes in the Python data."
        )

    distance_miles: dict[tuple[int, int], float] = {}
    travel_minutes: dict[tuple[int, int], float] = {}
    for row_idx, from_node in enumerate(node_ids):
        for col_idx, to_node in enumerate(node_ids):
            distance_miles[from_node, to_node] = float(DISTANCE_MATRIX[row_idx][col_idx])
            travel_minutes[from_node, to_node] = float(TRAVEL_TIME_MATRIX[row_idx][col_idx])

    return Instance(
        stops=stops,
        vehicles=vehicles,
        distance_miles=distance_miles,
        travel_minutes=travel_minutes,
        cost_params={key: float(value) for key, value in COST_PARAMETERS.items()},
        depot_open_min=depot_open_min,
        depot_close_min=depot_close_min,
        big_m=float(BIG_M),
    )


def create_model() -> tuple[gp.Model, dict[str, Any]]:
    return build_model(load_instance())


def show_data() -> None:
    print("Stops:")
    pprint(STOPS, width=140)
    print("\nVehicles:")
    pprint(VEHICLES, width=140)
    print("\nCost Parameters:")
    pprint(COST_PARAMETERS, width=140)
    print(f"\nDepot: open={DEPOT_OPEN}, close={DEPOT_CLOSE}, big_m={BIG_M}")


def build_model(instance: Instance) -> tuple[gp.Model, dict[str, Any]]:
    model = gp.Model("excel_vrp_translation")

    stop_ids = sorted(instance.stops)
    vehicle_ids = sorted(instance.vehicles)
    node_ids = [DEPOT_NODE] + stop_ids
    node_pairs = [(i, j) for i in node_ids for j in node_ids if i != j]
    arc_keys = [(v, i, j) for v in vehicle_ids for i, j in node_pairs]

    x = model.addVars(arc_keys, vtype=GRB.BINARY, name="x")
    use_vehicle = model.addVars(vehicle_ids, vtype=GRB.BINARY, name="use_vehicle")
    depart_min = model.addVars(vehicle_ids, lb=0.0, ub=instance.big_m, name="depart_min")
    return_min = model.addVars(vehicle_ids, lb=0.0, ub=instance.big_m, name="return_min")
    service_start_min = model.addVars(
        stop_ids, lb=0.0, ub=instance.big_m, name="service_start_min"
    )
    early_min = model.addVars(stop_ids, lb=0.0, name="early_min")
    late_min = model.addVars(stop_ids, lb=0.0, name="late_min")
    early_flag = model.addVars(stop_ids, vtype=GRB.BINARY, name="early_flag")
    idle_min = model.addVars(vehicle_ids, lb=0.0, name="idle_min")
    regular_min = model.addVars(vehicle_ids, lb=0.0, name="regular_min")
    overtime_min = model.addVars(vehicle_ids, lb=0.0, name="overtime_min")

    incoming = {
        (v, s): gp.quicksum(x[v, i, s] for i in node_ids if i != s)
        for v in vehicle_ids
        for s in stop_ids
    }
    outgoing = {
        (v, s): gp.quicksum(x[v, s, j] for j in node_ids if j != s)
        for v in vehicle_ids
        for s in stop_ids
    }
    starts = {v: gp.quicksum(x[v, DEPOT_NODE, j] for j in stop_ids) for v in vehicle_ids}
    returns = {v: gp.quicksum(x[v, i, DEPOT_NODE] for i in stop_ids) for v in vehicle_ids}

    drive_min_expr = {
        v: gp.quicksum(
            instance.travel_minutes[i, j] * x[v, i, j] for i, j in node_pairs
        )
        for v in vehicle_ids
    }
    service_min_expr = {
        v: gp.quicksum(instance.stops[s].service_min * incoming[v, s] for s in stop_ids)
        for v in vehicle_ids
    }
    active_min_expr = {v: return_min[v] - depart_min[v] for v in vehicle_ids}
    distance_cost_expr = {
        v: gp.quicksum(
            instance.distance_miles[i, j]
            * (
                instance.vehicles[v].cost_per_mile
                + instance.cost_params["Fuel_Cost_per_Liter"]
                * instance.cost_params["Avg_Fuel_Consumption_L_per_mile"]
            )
            * x[v, i, j]
            for i, j in node_pairs
        )
        for v in vehicle_ids
    }

    for s in stop_ids:
        model.addConstr(
            gp.quicksum(incoming[v, s] for v in vehicle_ids) == 1,
            name=f"visit_once[{s}]",
        )
        model.addConstr(
            service_start_min[s] + early_min[s] >= instance.stops[s].earliest_min,
            name=f"earliest_cover[{s}]",
        )
        model.addConstr(
            service_start_min[s] <= instance.stops[s].latest_min + late_min[s],
            name=f"latest_cover[{s}]",
        )
        model.addConstr(
            early_min[s] <= instance.big_m * early_flag[s],
            name=f"early_flag_upper[{s}]",
        )
        model.addConstr(
            early_min[s] >= EARLY_FLAG_MINUTES * early_flag[s],
            name=f"early_flag_lower[{s}]",
        )

    for v in vehicle_ids:
        vehicle = instance.vehicles[v]
        start_limit = max(vehicle.available_from_min, instance.depot_open_min)
        end_limit = min(vehicle.available_until_min, instance.depot_close_min)

        model.addConstr(starts[v] == use_vehicle[v], name=f"use_vehicle_start[{v}]")
        model.addConstr(returns[v] == use_vehicle[v], name=f"use_vehicle_return[{v}]")

        for s in stop_ids:
            model.addConstr(
                incoming[v, s] == outgoing[v, s],
                name=f"flow[{v},{s}]",
            )

        model.addConstr(
            gp.quicksum(instance.stops[s].demand_kg * incoming[v, s] for s in stop_ids)
            <= vehicle.capacity_kg,
            name=f"capacity[{v}]",
        )

        model.addConstr(
            depart_min[v] >= start_limit * use_vehicle[v],
            name=f"depart_lb[{v}]",
        )
        model.addConstr(
            depart_min[v] <= end_limit * use_vehicle[v],
            name=f"depart_ub[{v}]",
        )
        model.addConstr(
            return_min[v] >= start_limit * use_vehicle[v],
            name=f"return_lb[{v}]",
        )
        model.addConstr(
            return_min[v] <= end_limit * use_vehicle[v],
            name=f"return_ub[{v}]",
        )
        model.addConstr(
            return_min[v] >= depart_min[v],
            name=f"nonnegative_route_duration[{v}]",
        )

        model.addConstr(
            idle_min[v] >= active_min_expr[v] - drive_min_expr[v] - service_min_expr[v],
            name=f"idle_linearization[{v}]",
        )
        model.addConstr(
            regular_min[v] + overtime_min[v] == active_min_expr[v],
            name=f"labor_partition[{v}]",
        )
        model.addConstr(
            regular_min[v]
            <= 60.0 * instance.cost_params["Regular_Hours_Before_Overtime"],
            name=f"regular_hour_cap[{v}]",
        )

        for s in stop_ids:
            model.addConstr(
                service_start_min[s] - depart_min[v]
                >= instance.travel_minutes[DEPOT_NODE, s]
                - instance.big_m * (1 - x[v, DEPOT_NODE, s]),
                name=f"time_from_depot[{v},{s}]",
            )
            model.addConstr(
                return_min[v] - service_start_min[s]
                >= instance.stops[s].service_min
                + instance.travel_minutes[s, DEPOT_NODE]
                - instance.big_m * (1 - x[v, s, DEPOT_NODE]),
                name=f"time_to_depot[{v},{s}]",
            )

        for i in stop_ids:
            for j in stop_ids:
                if i == j:
                    continue
                model.addConstr(
                    service_start_min[j] - service_start_min[i]
                    >= instance.stops[i].service_min
                    + instance.travel_minutes[i, j]
                    - instance.big_m * (1 - x[v, i, j]),
                    name=f"time_between_stops[{v},{i},{j}]",
                )

    objective = gp.quicksum(
        distance_cost_expr[v]
        + instance.vehicles[v].fixed_daily_cost * use_vehicle[v]
        + instance.cost_params["Driver_Hourly_Wage"] / 60.0 * regular_min[v]
        + instance.cost_params["Overtime_Hourly_Rate"] / 60.0 * overtime_min[v]
        + instance.cost_params["Vehicle_Idle_Cost_per_Hour"] / 60.0 * idle_min[v]
        for v in vehicle_ids
    ) + gp.quicksum(
        instance.cost_params["Early_Delivery_Penalty"] * early_flag[s]
        + instance.cost_params["Late_Delivery_Penalty_per_Hour"] / 60.0 * late_min[s]
        for s in stop_ids
    )
    model.setObjective(objective, GRB.MINIMIZE)

    model._data = {
        "x": x,
        "use_vehicle": use_vehicle,
        "depart_min": depart_min,
        "return_min": return_min,
        "service_start_min": service_start_min,
        "early_min": early_min,
        "late_min": late_min,
        "early_flag": early_flag,
        "idle_min": idle_min,
        "regular_min": regular_min,
        "overtime_min": overtime_min,
        "incoming": incoming,
        "starts": starts,
        "returns": returns,
        "drive_min_expr": drive_min_expr,
        "service_min_expr": service_min_expr,
        "active_min_expr": active_min_expr,
        "distance_cost_expr": distance_cost_expr,
        "node_ids": node_ids,
        "stop_ids": stop_ids,
        "vehicle_ids": vehicle_ids,
    }
    return model, model._data


def extract_route(data: dict[str, Any], vehicle_id: int) -> list[int]:
    x = data["x"]
    stop_ids = data["stop_ids"]

    route = [DEPOT_NODE]
    current = DEPOT_NODE
    seen_stops: set[int] = set()

    while True:
        next_nodes = [
            j
            for j in stop_ids + [DEPOT_NODE]
            if j != current and (vehicle_id, current, j) in x and x[vehicle_id, current, j].X > 0.5
        ]
        if not next_nodes:
            break
        next_node = next_nodes[0]
        route.append(next_node)
        if next_node == DEPOT_NODE:
            break
        if next_node in seen_stops:
            break
        seen_stops.add(next_node)
        current = next_node

    return route


def solve_model(time_limit: float | None, mip_gap: float | None) -> None:
    instance = load_instance()
    model, data = build_model(instance)

    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    model.optimize()

    if model.SolCount == 0:
        print(f"Status: {model.Status}")
        print("No feasible solution found.")
        return

    objective_value = sum(
        data["distance_cost_expr"][v].getValue()
        + instance.vehicles[v].fixed_daily_cost * data["use_vehicle"][v].X
        + instance.cost_params["Driver_Hourly_Wage"] / 60.0 * data["regular_min"][v].X
        + instance.cost_params["Overtime_Hourly_Rate"] / 60.0 * data["overtime_min"][v].X
        + instance.cost_params["Vehicle_Idle_Cost_per_Hour"] / 60.0 * data["idle_min"][v].X
        for v in data["vehicle_ids"]
    ) + sum(
        instance.cost_params["Early_Delivery_Penalty"] * data["early_flag"][s].X
        + instance.cost_params["Late_Delivery_Penalty_per_Hour"] / 60.0 * data["late_min"][s].X
        for s in data["stop_ids"]
    )

    print(f"Status: {model.Status}")
    print(f"Objective value: {objective_value:.2f}")
    print()

    for v in data["vehicle_ids"]:
        if data["use_vehicle"][v].X < 0.5:
            continue

        vehicle = instance.vehicles[v]
        route = extract_route(data, v)
        load = sum(
            instance.stops[s].demand_kg
            for s in data["stop_ids"]
            if data["incoming"][v, s].getValue() > 0.5
        )
        drive_min = data["drive_min_expr"][v].getValue()
        service_min = data["service_min_expr"][v].getValue()
        active_min = data["active_min_expr"][v].getValue()
        idle_min = data["idle_min"][v].X
        distance_cost = data["distance_cost_expr"][v].getValue()

        print(f"Vehicle {v} ({vehicle.vehicle_type})")
        print(
            "  "
            f"Depart {minutes_to_clock(data['depart_min'][v].X)}, "
            f"Return {minutes_to_clock(data['return_min'][v].X)}, "
            f"Load {load:.0f} / {vehicle.capacity_kg:.0f} kg"
        )
        print(
            "  "
            f"Active {active_min:.1f} min, Drive {drive_min:.1f} min, "
            f"Service {service_min:.1f} min, Idle {idle_min:.1f} min"
        )
        print(f"  Distance-based cost: {distance_cost:.2f}")
        print(
            "  Route: "
            + " -> ".join("Depot" if node == DEPOT_NODE else f"Stop {node}" for node in route)
        )

        served_stops = [s for s in route if s != DEPOT_NODE]
        for s in served_stops:
            start = data["service_start_min"][s].X
            early = max(instance.stops[s].earliest_min - start, 0.0)
            late = max(start - instance.stops[s].latest_min, 0.0)
            print(
                "    "
                f"Stop {s}: start {minutes_to_clock(start)}, "
                f"early {early:.1f} min, late {late:.1f} min"
            )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the standalone Python VRP model."
    )
    parser.add_argument(
        "--show-data",
        action="store_true",
        help="Print the Python data structures that define the instance and exit.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional Gurobi time limit in seconds.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Optional relative MIP gap target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_data:
        show_data()
        return
    solve_model(args.time_limit, args.mip_gap)


if __name__ == "__main__":
    main()
