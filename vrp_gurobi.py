from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

import gurobipy as gp
from gurobipy import GRB
from openpyxl import load_workbook


DEPOT_NODE = 0
REDUCED_COST_TOL = 1e-6


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
class VehicleTypeData:
    type_id: str
    vehicle_ids: tuple[int, ...]
    vehicle_type: str
    capacity_kg: float
    cost_per_mile: float
    fixed_daily_cost: float
    available_from_min: float
    available_until_min: float
    count: int


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


@dataclass
class RouteColumn:
    type_id: str
    vehicle_type: str
    stop_sequence: tuple[int, ...]
    stop_set: frozenset[int]
    cost: float
    load_kg: float
    distance_mi: float
    active_min: float
    return_min: float
    service_start_min: dict[int, float]
    late_min: dict[int, float]


@dataclass(frozen=True)
class ColumnGenerationResult:
    vehicle_types: dict[str, VehicleTypeData]
    routes: list[RouteColumn]
    iterations: int
    lp_objective: float


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


def load_instance(workbook_path: Path) -> Instance:
    wb = load_workbook(workbook_path, data_only=True)

    ws_stops = wb["Delivery Stops"]
    ws_vehicles = wb["Vehicle Fleet"]
    ws_depot = wb["Depot"]
    ws_dist = wb["Distance Matrix (miles)"]
    ws_time = wb["Travel Time Matrix (min)"]
    ws_cost = wb["Cost Parameters"]
    ws_solver = wb["Solver Setup"]

    big_m = float(ws_solver["B9"].value)

    stops: dict[int, StopData] = {}
    for row in range(2, ws_stops.max_row + 1):
        stop_id = int(ws_stops[f"A{row}"].value)
        stops[stop_id] = StopData(
            stop_id=stop_id,
            customer_name=str(ws_stops[f"B{row}"].value),
            demand_kg=float(ws_stops[f"D{row}"].value),
            earliest_min=excel_time_to_minutes(ws_stops[f"E{row}"].value),
            latest_min=excel_time_to_minutes(ws_stops[f"F{row}"].value),
            service_min=float(ws_stops[f"G{row}"].value),
        )

    vehicles: dict[int, VehicleData] = {}
    for row in range(2, ws_vehicles.max_row + 1):
        vehicle_id = int(ws_vehicles[f"A{row}"].value)
        vehicles[vehicle_id] = VehicleData(
            vehicle_id=vehicle_id,
            vehicle_type=str(ws_vehicles[f"B{row}"].value),
            capacity_kg=float(ws_vehicles[f"C{row}"].value),
            cost_per_mile=float(ws_vehicles[f"D{row}"].value),
            fixed_daily_cost=float(ws_vehicles[f"E{row}"].value),
            available_from_min=excel_time_to_minutes(ws_vehicles[f"F{row}"].value),
            available_until_min=excel_time_to_minutes(ws_vehicles[f"G{row}"].value),
        )

    depot_open_min = excel_time_to_minutes(ws_depot["C2"].value)
    depot_close_min = excel_time_to_minutes(ws_depot["D2"].value)

    node_ids = [DEPOT_NODE] + sorted(stops)
    expected_size = len(node_ids)
    if ws_dist.max_row != expected_size + 1 or ws_dist.max_column != expected_size + 1:
        raise ValueError(
            "Distance matrix dimensions do not match the number of nodes in the workbook."
        )
    if ws_time.max_row != expected_size + 1 or ws_time.max_column != expected_size + 1:
        raise ValueError(
            "Travel-time matrix dimensions do not match the number of nodes in the workbook."
        )

    distance_miles: dict[tuple[int, int], float] = {}
    travel_minutes: dict[tuple[int, int], float] = {}
    for row_offset, from_node in enumerate(node_ids, start=2):
        for col_offset, to_node in enumerate(node_ids, start=2):
            distance_miles[from_node, to_node] = float(ws_dist.cell(row_offset, col_offset).value)
            travel_minutes[from_node, to_node] = float(ws_time.cell(row_offset, col_offset).value)

    cost_params: dict[str, float] = {}
    for row in range(2, ws_cost.max_row + 1):
        key = str(ws_cost[f"A{row}"].value)
        cost_params[key] = float(ws_cost[f"B{row}"].value)

    return Instance(
        stops=stops,
        vehicles=vehicles,
        distance_miles=distance_miles,
        travel_minutes=travel_minutes,
        cost_params=cost_params,
        depot_open_min=depot_open_min,
        depot_close_min=depot_close_min,
        big_m=big_m,
    )


def validate_route_based_assumptions(instance: Instance) -> None:
    early_penalty = instance.cost_params.get("Early_Delivery_Penalty", 0.0)
    if abs(early_penalty) > 1e-9:
        raise NotImplementedError(
            "This route-based rewrite currently assumes zero early-delivery penalty. "
            "That matches the workbook scenario, where Early_Delivery_Penalty = 0."
        )


def build_vehicle_types(instance: Instance) -> dict[str, VehicleTypeData]:
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for vehicle_id, vehicle in sorted(instance.vehicles.items()):
        key = (
            vehicle.vehicle_type,
            vehicle.capacity_kg,
            vehicle.cost_per_mile,
            vehicle.fixed_daily_cost,
            vehicle.available_from_min,
            vehicle.available_until_min,
        )
        grouped[key].append(vehicle_id)

    vehicle_types: dict[str, VehicleTypeData] = {}
    for idx, (key, vehicle_ids) in enumerate(
        sorted(grouped.items(), key=lambda item: min(item[1])), start=1
    ):
        vehicle_type, capacity_kg, cost_per_mile, fixed_daily_cost, available_from_min, available_until_min = key
        type_id = f"T{idx}"
        vehicle_types[type_id] = VehicleTypeData(
            type_id=type_id,
            vehicle_ids=tuple(vehicle_ids),
            vehicle_type=str(vehicle_type),
            capacity_kg=float(capacity_kg),
            cost_per_mile=float(cost_per_mile),
            fixed_daily_cost=float(fixed_daily_cost),
            available_from_min=float(available_from_min),
            available_until_min=float(available_until_min),
            count=len(vehicle_ids),
        )
    return vehicle_types


def vehicle_start_limit(instance: Instance, vehicle_type: VehicleTypeData) -> float:
    return max(vehicle_type.available_from_min, instance.depot_open_min)


def vehicle_end_limit(instance: Instance, vehicle_type: VehicleTypeData) -> float:
    return min(vehicle_type.available_until_min, instance.depot_close_min)


def regular_limit_minutes(instance: Instance) -> float:
    return 60.0 * instance.cost_params["Regular_Hours_Before_Overtime"]


def route_distance_unit_cost(instance: Instance, vehicle_type: VehicleTypeData) -> float:
    return vehicle_type.cost_per_mile + (
        instance.cost_params["Fuel_Cost_per_Liter"]
        * instance.cost_params["Avg_Fuel_Consumption_L_per_mile"]
    )


def evaluate_route_sequence(
    instance: Instance,
    vehicle_type: VehicleTypeData,
    stop_sequence: tuple[int, ...],
) -> RouteColumn | None:
    if not stop_sequence:
        return None

    start_limit = vehicle_start_limit(instance, vehicle_type)
    end_limit = vehicle_end_limit(instance, vehicle_type)
    load_kg = sum(instance.stops[s].demand_kg for s in stop_sequence)
    if load_kg > vehicle_type.capacity_kg + 1e-9:
        return None

    current_time = start_limit
    current_node = DEPOT_NODE
    distance_mi = 0.0
    service_start_min: dict[int, float] = {}
    late_min: dict[int, float] = {}

    for stop_id in stop_sequence:
        stop = instance.stops[stop_id]
        current_time += instance.travel_minutes[current_node, stop_id]
        service_start_min[stop_id] = current_time
        late_min[stop_id] = max(current_time - stop.latest_min, 0.0)
        distance_mi += instance.distance_miles[current_node, stop_id]
        current_time += stop.service_min
        current_node = stop_id

    distance_mi += instance.distance_miles[current_node, DEPOT_NODE]
    current_time += instance.travel_minutes[current_node, DEPOT_NODE]
    return_min = current_time
    if return_min > end_limit + 1e-9:
        return None

    active_min = return_min - start_limit
    regular_min = min(active_min, regular_limit_minutes(instance))
    overtime_min = max(active_min - regular_min, 0.0)

    cost = vehicle_type.fixed_daily_cost
    cost += distance_mi * route_distance_unit_cost(instance, vehicle_type)
    cost += regular_min / 60.0 * instance.cost_params["Driver_Hourly_Wage"]
    cost += overtime_min / 60.0 * instance.cost_params["Overtime_Hourly_Rate"]
    cost += sum(late_min.values()) / 60.0 * instance.cost_params["Late_Delivery_Penalty_per_Hour"]

    return RouteColumn(
        type_id=vehicle_type.type_id,
        vehicle_type=vehicle_type.vehicle_type,
        stop_sequence=stop_sequence,
        stop_set=frozenset(stop_sequence),
        cost=cost,
        load_kg=load_kg,
        distance_mi=distance_mi,
        active_min=active_min,
        return_min=return_min,
        service_start_min=service_start_min,
        late_min=late_min,
    )


def initial_routes(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
) -> list[RouteColumn]:
    routes: list[RouteColumn] = []
    for vehicle_type in vehicle_types.values():
        for stop_id in sorted(instance.stops):
            route = evaluate_route_sequence(instance, vehicle_type, (stop_id,))
            if route is not None:
                routes.append(route)

    route_keys = {(route.type_id, route.stop_sequence) for route in routes}
    for route in construct_seed_partition_routes(instance, vehicle_types):
        key = (route.type_id, route.stop_sequence)
        if key not in route_keys:
            route_keys.add(key)
            routes.append(route)
    return routes


def best_stop_insertion(
    instance: Instance,
    vehicle_type: VehicleTypeData,
    current_sequence: tuple[int, ...],
    stop_id: int,
) -> RouteColumn | None:
    current_route = (
        evaluate_route_sequence(instance, vehicle_type, current_sequence)
        if current_sequence
        else None
    )
    current_cost = current_route.cost if current_route is not None else 0.0

    best_route: RouteColumn | None = None
    best_delta = float("inf")
    for pos in range(len(current_sequence) + 1):
        candidate_sequence = current_sequence[:pos] + (stop_id,) + current_sequence[pos:]
        candidate_route = evaluate_route_sequence(instance, vehicle_type, candidate_sequence)
        if candidate_route is None:
            continue
        delta = candidate_route.cost - current_cost
        if delta < best_delta - 1e-9:
            best_delta = delta
            best_route = candidate_route
    return best_route


def construct_seed_partition_routes(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
) -> list[RouteColumn]:
    slot_types: list[VehicleTypeData] = []
    for vehicle_type in sorted(
        vehicle_types.values(),
        key=lambda vt: (-vt.capacity_kg, vt.fixed_daily_cost, vt.type_id),
    ):
        slot_types.extend([vehicle_type] * vehicle_type.count)

    if not slot_types:
        return []

    stop_ids = sorted(instance.stops)
    round_trip_minutes = {
        stop_id: (
            instance.travel_minutes[DEPOT_NODE, stop_id]
            + instance.travel_minutes[stop_id, DEPOT_NODE]
        )
        for stop_id in stop_ids
    }
    orderings = [
        sorted(
            stop_ids,
            key=lambda stop_id: (
                -instance.stops[stop_id].demand_kg,
                -round_trip_minutes[stop_id],
                instance.stops[stop_id].latest_min,
            ),
        ),
        sorted(
            stop_ids,
            key=lambda stop_id: (
                -round_trip_minutes[stop_id],
                -instance.stops[stop_id].demand_kg,
                instance.stops[stop_id].latest_min,
            ),
        ),
        sorted(
            stop_ids,
            key=lambda stop_id: (
                instance.stops[stop_id].latest_min,
                -instance.stops[stop_id].demand_kg,
                -round_trip_minutes[stop_id],
            ),
        ),
    ]

    for ordering in orderings:
        slot_routes: list[RouteColumn | None] = [None] * len(slot_types)
        feasible = True

        for stop_id in ordering:
            best_slot: int | None = None
            best_candidate: RouteColumn | None = None
            best_delta = float("inf")

            for slot_idx, vehicle_type in enumerate(slot_types):
                current_route = slot_routes[slot_idx]
                current_sequence = (
                    current_route.stop_sequence if current_route is not None else tuple()
                )
                candidate = best_stop_insertion(
                    instance=instance,
                    vehicle_type=vehicle_type,
                    current_sequence=current_sequence,
                    stop_id=stop_id,
                )
                if candidate is None:
                    continue

                current_cost = current_route.cost if current_route is not None else 0.0
                delta = candidate.cost - current_cost
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_slot = slot_idx
                    best_candidate = candidate

            if best_slot is None or best_candidate is None:
                feasible = False
                break

            slot_routes[best_slot] = best_candidate

        if feasible:
            return [route for route in slot_routes if route is not None]

    return []


def solve_pricing_subproblem(
    instance: Instance,
    vehicle_type: VehicleTypeData,
    cover_duals: dict[int, float],
    type_dual: float,
    pricing_time_limit: float | None,
) -> RouteColumn | None:
    stop_ids = sorted(instance.stops)
    start_limit = vehicle_start_limit(instance, vehicle_type)
    end_limit = vehicle_end_limit(instance, vehicle_type)
    route_horizon = max(end_limit - start_limit, 0.0)
    route_used_max_regular = regular_limit_minutes(instance)

    stop_lb: dict[int, float] = {}
    stop_ub: dict[int, float] = {}
    feasible_stop: dict[int, bool] = {}
    late_ub: dict[int, float] = {}

    for stop_id in stop_ids:
        stop = instance.stops[stop_id]
        lb = start_limit + instance.travel_minutes[DEPOT_NODE, stop_id]
        ub = end_limit - stop.service_min - instance.travel_minutes[stop_id, DEPOT_NODE]
        feasible = stop.demand_kg <= vehicle_type.capacity_kg and lb <= ub
        stop_lb[stop_id] = lb
        stop_ub[stop_id] = ub
        feasible_stop[stop_id] = feasible
        late_ub[stop_id] = max(ub - stop.latest_min, 0.0) if feasible else 0.0

    arc_keys: list[tuple[int, int]] = []
    if route_horizon <= 0:
        return None

    for stop_id in stop_ids:
        if feasible_stop[stop_id]:
            arc_keys.append((DEPOT_NODE, stop_id))
            arc_keys.append((stop_id, DEPOT_NODE))

    for i in stop_ids:
        if not feasible_stop[i]:
            continue
        for j in stop_ids:
            if i == j or not feasible_stop[j]:
                continue
            if (
                stop_lb[i]
                + instance.stops[i].service_min
                + instance.travel_minutes[i, j]
                <= stop_ub[j]
            ):
                arc_keys.append((i, j))

    if not arc_keys:
        return None

    pricing = gp.Model(f"pricing_{vehicle_type.type_id}")
    pricing.Params.OutputFlag = 0
    if pricing_time_limit is not None:
        pricing.Params.TimeLimit = pricing_time_limit

    x = pricing.addVars(arc_keys, vtype=GRB.BINARY, name="x")
    y = pricing.addVars(stop_ids, vtype=GRB.BINARY, name="y")
    service_start = pricing.addVars(stop_ids, lb=0.0, name="service_start")
    late_min = pricing.addVars(stop_ids, lb=0.0, name="late_min")
    route_used = pricing.addVar(vtype=GRB.BINARY, name="route_used")
    return_min = pricing.addVar(lb=0.0, ub=end_limit, name="return_min")
    active_min = pricing.addVar(lb=0.0, ub=route_horizon, name="active_min")
    regular_min = pricing.addVar(
        lb=0.0, ub=route_used_max_regular, name="regular_min"
    )
    overtime_min = pricing.addVar(lb=0.0, ub=route_horizon, name="overtime_min")

    outgoing = {
        node: [head for tail, head in arc_keys if tail == node]
        for node in [DEPOT_NODE] + stop_ids
    }
    incoming = {
        node: [tail for tail, head in arc_keys if head == node]
        for node in [DEPOT_NODE] + stop_ids
    }

    pricing.addConstr(
        gp.quicksum(x[DEPOT_NODE, j] for j in outgoing[DEPOT_NODE]) == route_used,
        name="route_start",
    )
    pricing.addConstr(
        gp.quicksum(x[i, DEPOT_NODE] for i in incoming[DEPOT_NODE]) == route_used,
        name="route_end",
    )

    for stop_id in stop_ids:
        if not feasible_stop[stop_id]:
            pricing.addConstr(y[stop_id] == 0, name=f"infeasible_stop[{stop_id}]")
            pricing.addConstr(service_start[stop_id] == 0, name=f"no_time[{stop_id}]")
            pricing.addConstr(late_min[stop_id] == 0, name=f"no_late[{stop_id}]")
            continue

        pricing.addConstr(
            gp.quicksum(x[i, stop_id] for i in incoming[stop_id]) == y[stop_id],
            name=f"in_degree[{stop_id}]",
        )
        pricing.addConstr(
            gp.quicksum(x[stop_id, j] for j in outgoing[stop_id]) == y[stop_id],
            name=f"out_degree[{stop_id}]",
        )
        pricing.addConstr(
            service_start[stop_id] >= stop_lb[stop_id] * y[stop_id],
            name=f"time_lb[{stop_id}]",
        )
        pricing.addConstr(
            service_start[stop_id] <= stop_ub[stop_id] * y[stop_id],
            name=f"time_ub[{stop_id}]",
        )
        pricing.addConstr(
            late_min[stop_id]
            >= service_start[stop_id]
            - instance.stops[stop_id].latest_min
            - stop_ub[stop_id] * (1 - y[stop_id]),
            name=f"late_lb[{stop_id}]",
        )
        pricing.addConstr(
            late_min[stop_id] <= late_ub[stop_id] * y[stop_id],
            name=f"late_ub[{stop_id}]",
        )

        if (DEPOT_NODE, stop_id) in x:
            big_m = start_limit + instance.travel_minutes[DEPOT_NODE, stop_id]
            pricing.addConstr(
                service_start[stop_id]
                >= start_limit
                + instance.travel_minutes[DEPOT_NODE, stop_id]
                - big_m * (1 - x[DEPOT_NODE, stop_id]),
                name=f"from_depot[{stop_id}]",
            )
        if (stop_id, DEPOT_NODE) in x:
            big_m = (
                stop_ub[stop_id]
                + instance.stops[stop_id].service_min
                + instance.travel_minutes[stop_id, DEPOT_NODE]
            )
            pricing.addConstr(
                return_min
                >= service_start[stop_id]
                + instance.stops[stop_id].service_min
                + instance.travel_minutes[stop_id, DEPOT_NODE]
                - big_m * (1 - x[stop_id, DEPOT_NODE]),
                name=f"to_depot[{stop_id}]",
            )

    for i, j in arc_keys:
        if i == DEPOT_NODE or j == DEPOT_NODE:
            continue
        big_m = stop_ub[i] + instance.stops[i].service_min + instance.travel_minutes[i, j]
        pricing.addConstr(
            service_start[j]
            >= service_start[i]
            + instance.stops[i].service_min
            + instance.travel_minutes[i, j]
            - big_m * (1 - x[i, j]),
            name=f"time_link[{i},{j}]",
        )

    pricing.addConstr(
        gp.quicksum(instance.stops[s].demand_kg * y[s] for s in stop_ids)
        <= vehicle_type.capacity_kg * route_used,
        name="capacity",
    )
    pricing.addConstr(
        return_min <= end_limit * route_used,
        name="return_upper",
    )
    pricing.addConstr(
        active_min == return_min - start_limit * route_used,
        name="active_time",
    )
    pricing.addConstr(
        regular_min + overtime_min == active_min,
        name="labor_split",
    )
    pricing.addConstr(
        regular_min <= route_used_max_regular * route_used,
        name="regular_cap",
    )

    distance_cost_expr = gp.quicksum(
        instance.distance_miles[i, j]
        * route_distance_unit_cost(instance, vehicle_type)
        * x[i, j]
        for i, j in arc_keys
    )
    labor_cost_expr = (
        regular_min / 60.0 * instance.cost_params["Driver_Hourly_Wage"]
        + overtime_min / 60.0 * instance.cost_params["Overtime_Hourly_Rate"]
    )
    late_cost_expr = (
        gp.quicksum(late_min[s] for s in stop_ids)
        / 60.0
        * instance.cost_params["Late_Delivery_Penalty_per_Hour"]
    )
    reduced_cost_objective = (
        vehicle_type.fixed_daily_cost * route_used
        + distance_cost_expr
        + labor_cost_expr
        + late_cost_expr
        - gp.quicksum(cover_duals[s] * y[s] for s in stop_ids)
        - type_dual * route_used
    )
    pricing.setObjective(reduced_cost_objective, GRB.MINIMIZE)
    pricing.optimize()

    if pricing.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(
            f"Pricing problem for {vehicle_type.type_id} failed with status {pricing.Status}."
        )

    if pricing.SolCount == 0:
        return None

    if route_used.X < 0.5 or pricing.ObjVal >= -REDUCED_COST_TOL:
        return None

    sequence: list[int] = []
    current = DEPOT_NODE
    while True:
        next_nodes = [j for j in outgoing[current] if x[current, j].X > 0.5]
        if not next_nodes:
            break
        next_node = next_nodes[0]
        if next_node == DEPOT_NODE:
            break
        sequence.append(next_node)
        current = next_node

    route = evaluate_route_sequence(instance, vehicle_type, tuple(sequence))
    if route is None:
        raise RuntimeError(
            f"Pricing returned an invalid route for {vehicle_type.type_id}: {sequence}"
        )
    return route


def build_master_model(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
    routes: list[RouteColumn],
    relax: bool,
    allow_artificial: bool = False,
) -> tuple[gp.Model, dict[str, Any]]:
    model = gp.Model("route_based_vrp")
    model.Params.OutputFlag = 0

    route_indices = list(range(len(routes)))
    var_type = GRB.CONTINUOUS if relax else GRB.BINARY
    route_vars = model.addVars(route_indices, lb=0.0, ub=1.0, vtype=var_type, name="route")
    artificial_vars = (
        model.addVars(
            sorted(instance.stops),
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS if relax else GRB.BINARY,
            name="artificial_cover",
        )
        if allow_artificial
        else {}
    )
    artificial_penalty = 1_000_000.0

    cover_constraints: dict[int, gp.Constr] = {}
    for stop_id in sorted(instance.stops):
        candidate_routes = [idx for idx, route in enumerate(routes) if stop_id in route.stop_set]
        if not candidate_routes and not allow_artificial:
            raise ValueError(f"No route column covers stop {stop_id}.")
        cover_expr = gp.quicksum(route_vars[idx] for idx in candidate_routes)
        if allow_artificial:
            cover_expr += artificial_vars[stop_id]
        cover_constraints[stop_id] = model.addConstr(
            cover_expr == 1,
            name=f"cover[{stop_id}]",
        )

    type_constraints: dict[str, gp.Constr] = {}
    for type_id, vehicle_type in vehicle_types.items():
        candidate_routes = [idx for idx, route in enumerate(routes) if route.type_id == type_id]
        type_constraints[type_id] = model.addConstr(
            gp.quicksum(route_vars[idx] for idx in candidate_routes) <= vehicle_type.count,
            name=f"type_limit[{type_id}]",
        )

    model.setObjective(
        gp.quicksum(routes[idx].cost * route_vars[idx] for idx in route_indices)
        + (
            gp.quicksum(
                artificial_penalty * artificial_vars[stop_id]
                for stop_id in sorted(instance.stops)
            )
            if allow_artificial
            else 0.0
        ),
        GRB.MINIMIZE,
    )

    model._data = {
        "route_vars": route_vars,
        "artificial_vars": artificial_vars,
        "artificial_penalty": artificial_penalty if allow_artificial else None,
        "routes": routes,
        "vehicle_types": vehicle_types,
        "cover_constraints": cover_constraints,
        "type_constraints": type_constraints,
        "formulation": "route_set_partitioning",
    }
    return model, model._data


def generate_route_columns(
    instance: Instance,
    max_cg_iterations: int,
    pricing_time_limit: float | None,
) -> ColumnGenerationResult:
    validate_route_based_assumptions(instance)
    vehicle_types = build_vehicle_types(instance)
    routes = initial_routes(instance, vehicle_types)
    route_keys = {(route.type_id, route.stop_sequence) for route in routes}
    last_lp_objective = float("nan")

    for iteration in range(1, max_cg_iterations + 1):
        master_lp, lp_data = build_master_model(
            instance,
            vehicle_types,
            routes,
            relax=True,
            allow_artificial=True,
        )
        master_lp.optimize()
        if master_lp.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Column-generation master LP did not solve to optimality (status {master_lp.Status})."
            )

        last_lp_objective = master_lp.ObjVal
        cover_duals = {
            stop_id: lp_data["cover_constraints"][stop_id].Pi for stop_id in sorted(instance.stops)
        }
        type_duals = {
            type_id: lp_data["type_constraints"][type_id].Pi for type_id in vehicle_types
        }

        new_routes: list[RouteColumn] = []
        for vehicle_type in vehicle_types.values():
            route = solve_pricing_subproblem(
                instance=instance,
                vehicle_type=vehicle_type,
                cover_duals=cover_duals,
                type_dual=type_duals[vehicle_type.type_id],
                pricing_time_limit=pricing_time_limit,
            )
            if route is None:
                continue
            key = (route.type_id, route.stop_sequence)
            if key not in route_keys:
                route_keys.add(key)
                new_routes.append(route)

        if not new_routes:
            artificial_usage = sum(
                lp_data["artificial_vars"][stop_id].X for stop_id in sorted(instance.stops)
            )
            if artificial_usage > 1e-6:
                raise RuntimeError(
                    "Column generation stopped with artificial stop coverage still in use. "
                    "The route pool is not rich enough yet to represent a full solution."
                )
            return ColumnGenerationResult(
                vehicle_types=vehicle_types,
                routes=routes,
                iterations=iteration,
                lp_objective=last_lp_objective,
            )

        routes.extend(new_routes)

    return ColumnGenerationResult(
        vehicle_types=vehicle_types,
        routes=routes,
        iterations=max_cg_iterations,
        lp_objective=last_lp_objective,
    )


def create_model(
    workbook_path: Path,
    max_cg_iterations: int = 50,
    pricing_time_limit: float | None = None,
) -> tuple[gp.Model, dict[str, Any]]:
    instance = load_instance(workbook_path)
    cg_result = generate_route_columns(
        instance=instance,
        max_cg_iterations=max_cg_iterations,
        pricing_time_limit=pricing_time_limit,
    )
    model, data = build_master_model(
        instance=instance,
        vehicle_types=cg_result.vehicle_types,
        routes=cg_result.routes,
        relax=False,
    )
    data.update(
        {
            "instance": instance,
            "cg_iterations": cg_result.iterations,
            "lp_objective": cg_result.lp_objective,
            "column_count": len(cg_result.routes),
        }
    )
    return model, data


def solve_model(
    workbook_path: Path,
    time_limit: float | None,
    mip_gap: float | None,
    max_cg_iterations: int,
    pricing_time_limit: float | None,
) -> None:
    model, data = create_model(
        workbook_path=workbook_path,
        max_cg_iterations=max_cg_iterations,
        pricing_time_limit=pricing_time_limit,
    )
    instance: Instance = data["instance"]
    route_vars = data["route_vars"]
    routes: list[RouteColumn] = data["routes"]
    vehicle_types: dict[str, VehicleTypeData] = data["vehicle_types"]

    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    model.Params.OutputFlag = 1
    model.optimize()

    if model.SolCount == 0:
        print(f"Status: {model.Status}")
        print("No feasible solution found.")
        return

    print(f"Status: {model.Status}")
    print("Formulation: Route-Based Set Partitioning")
    print(f"Columns generated: {data['column_count']}")
    print(f"Column generation iterations: {data['cg_iterations']}")
    print(f"Master LP objective: {data['lp_objective']:.2f}")
    print(f"Objective value: {model.ObjVal:.2f}")
    print()

    selected_routes = [routes[idx] for idx in range(len(routes)) if route_vars[idx].X > 0.5]
    usage_counter: dict[str, int] = defaultdict(int)

    for route in sorted(
        selected_routes,
        key=lambda route: (route.type_id, route.return_min, route.stop_sequence),
    ):
        vehicle_type = vehicle_types[route.type_id]
        usage_counter[route.type_id] += 1
        vehicle_position = usage_counter[route.type_id] - 1
        vehicle_id = vehicle_type.vehicle_ids[vehicle_position]

        print(f"Vehicle {vehicle_id} ({vehicle_type.vehicle_type})")
        print(
            "  "
            f"Depart {minutes_to_clock(vehicle_start_limit(instance, vehicle_type))}, "
            f"Return {minutes_to_clock(route.return_min)}, "
            f"Load {route.load_kg:.0f} / {vehicle_type.capacity_kg:.0f} kg"
        )
        print(
            "  "
            f"Active {route.active_min:.1f} min, Distance {route.distance_mi:.2f} mi, "
            f"Route cost {route.cost:.2f}"
        )
        print(
            "  Route: "
            + " -> ".join(
                ["Depot"] + [f"Stop {stop_id}" for stop_id in route.stop_sequence] + ["Depot"]
            )
        )
        for stop_id in route.stop_sequence:
            start = route.service_start_min[stop_id]
            early = max(instance.stops[stop_id].earliest_min - start, 0.0)
            late = route.late_min[stop_id]
            print(
                "    "
                f"Stop {stop_id}: start {minutes_to_clock(start)}, "
                f"early {early:.1f} min, late {late:.1f} min"
            )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the workbook-driven VRP with a route-based set-partitioning formulation "
            "generated through column generation."
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        required=True,
        help="Path to the Excel workbook.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional Gurobi time limit in seconds for the final master MIP.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Optional relative MIP gap target for the final master MIP.",
    )
    parser.add_argument(
        "--max-cg-iterations",
        type=int,
        default=50,
        help="Maximum number of column-generation iterations.",
    )
    parser.add_argument(
        "--pricing-time-limit",
        type=float,
        default=None,
        help="Optional time limit in seconds for each pricing MIP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solve_model(
        workbook_path=args.workbook,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        max_cg_iterations=args.max_cg_iterations,
        pricing_time_limit=args.pricing_time_limit,
    )


if __name__ == "__main__":
    main()
