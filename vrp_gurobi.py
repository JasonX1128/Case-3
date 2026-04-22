from __future__ import annotations

import argparse
import heapq
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Any

import gurobipy as gp
from gurobipy import GRB
from openpyxl import load_workbook


DEPOT_NODE = 0
REDUCED_COST_TOL = 1e-6
INTEGRALITY_TOL = 1e-6
HEURISTIC_MASTER_TIME_LIMIT = 5.0


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
    arc_set: frozenset[tuple[int, int]]
    cost: float
    load_kg: float
    distance_mi: float
    active_min: float
    return_min: float
    service_start_min: dict[int, float]
    late_min: dict[int, float]


@dataclass(frozen=True)
class BranchState:
    forbidden_arcs: frozenset[tuple[int, int]]
    forced_arcs: frozenset[tuple[int, int]]
    excluded_assignments: frozenset[tuple[int, str]]
    forced_assignments: frozenset[tuple[int, str]]


@dataclass
class BranchContext:
    forbidden_arcs: frozenset[tuple[int, int]]
    forced_arcs: tuple[tuple[int, int], ...]
    forced_successor: dict[int, int]
    forced_predecessor: dict[int, int]
    forced_type_for_stop: dict[int, str]
    excluded_types_by_stop: dict[int, set[str]]


@dataclass
class NodeRelaxationResult:
    lower_bound: float
    routes: list[RouteColumn]
    route_values: list[float]
    cg_iterations: int


@dataclass
class BranchAndPriceResult:
    status: str
    objective: float | None
    best_bound: float | None
    gap: float | None
    selected_routes: list[RouteColumn]
    vehicle_types: dict[str, VehicleTypeData]
    nodes_processed: int
    nodes_pruned: int
    global_columns: int
    root_lp_objective: float | None


def excel_time_to_minutes(value: Any) -> float:
    if value is None:
        raise ValueError("Encountered an empty time value.")
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute + value.second / 60
    if isinstance(value, dt_time):
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


def remaining_time(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(deadline - perf_counter(), 0.0)


def relative_gap(incumbent: float | None, best_bound: float | None) -> float | None:
    if incumbent is None or best_bound is None:
        return None
    if abs(incumbent) <= INTEGRALITY_TOL:
        return 0.0 if abs(incumbent - best_bound) <= INTEGRALITY_TOL else None
    return max(incumbent - best_bound, 0.0) / abs(incumbent)


def route_arc_set(stop_sequence: tuple[int, ...]) -> frozenset[tuple[int, int]]:
    if not stop_sequence:
        return frozenset()

    arcs: list[tuple[int, int]] = [(DEPOT_NODE, stop_sequence[0])]
    for idx in range(len(stop_sequence) - 1):
        arcs.append((stop_sequence[idx], stop_sequence[idx + 1]))
    arcs.append((stop_sequence[-1], DEPOT_NODE))
    return frozenset(arcs)


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
            "This branch-and-price model currently assumes zero early-delivery penalty. "
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
        (
            vehicle_type,
            capacity_kg,
            cost_per_mile,
            fixed_daily_cost,
            available_from_min,
            available_until_min,
        ) = key
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
        arc_set=route_arc_set(stop_sequence),
        cost=cost,
        load_kg=load_kg,
        distance_mi=distance_mi,
        active_min=active_min,
        return_min=return_min,
        service_start_min=service_start_min,
        late_min=late_min,
    )


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


def build_branch_context(
    branch_state: BranchState,
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
) -> BranchContext | None:
    forced_successor: dict[int, int] = {}
    forced_predecessor: dict[int, int] = {}
    forced_type_for_stop: dict[int, str] = {}
    excluded_types_by_stop: dict[int, set[str]] = defaultdict(set)

    for arc in branch_state.forced_arcs:
        if arc in branch_state.forbidden_arcs:
            return None
        i, j = arc
        if i == DEPOT_NODE and j == DEPOT_NODE:
            return None
        if i != DEPOT_NODE:
            existing = forced_successor.get(i)
            if existing is not None and existing != j:
                return None
            forced_successor[i] = j
        if j != DEPOT_NODE:
            existing = forced_predecessor.get(j)
            if existing is not None and existing != i:
                return None
            forced_predecessor[j] = i

    for start_node in forced_successor:
        seen: set[int] = set()
        current = start_node
        while current != DEPOT_NODE and current in forced_successor:
            if current in seen:
                return None
            seen.add(current)
            next_node = forced_successor[current]
            if next_node == DEPOT_NODE:
                break
            current = next_node

    for stop_id, type_id in branch_state.forced_assignments:
        if type_id not in vehicle_types:
            return None
        existing = forced_type_for_stop.get(stop_id)
        if existing is not None and existing != type_id:
            return None
        forced_type_for_stop[stop_id] = type_id

    for stop_id, type_id in branch_state.excluded_assignments:
        if type_id not in vehicle_types:
            return None
        excluded_types_by_stop[stop_id].add(type_id)

    all_type_ids = set(vehicle_types)
    for stop_id, type_id in forced_type_for_stop.items():
        if type_id in excluded_types_by_stop[stop_id]:
            return None
    for stop_id in instance.stops:
        if stop_id not in forced_type_for_stop and excluded_types_by_stop.get(stop_id) == all_type_ids:
            return None

    return BranchContext(
        forbidden_arcs=branch_state.forbidden_arcs,
        forced_arcs=tuple(sorted(branch_state.forced_arcs)),
        forced_successor=forced_successor,
        forced_predecessor=forced_predecessor,
        forced_type_for_stop=forced_type_for_stop,
        excluded_types_by_stop=excluded_types_by_stop,
    )


def route_is_compatible(route: RouteColumn, branch_ctx: BranchContext) -> bool:
    if route.arc_set & branch_ctx.forbidden_arcs:
        return False

    for stop_id, forced_type in branch_ctx.forced_type_for_stop.items():
        if stop_id in route.stop_set and route.type_id != forced_type:
            return False

    for stop_id, excluded_types in branch_ctx.excluded_types_by_stop.items():
        if route.type_id in excluded_types and stop_id in route.stop_set:
            return False

    for i, j in branch_ctx.forced_arcs:
        if i == DEPOT_NODE:
            if j in route.stop_set and (i, j) not in route.arc_set:
                return False
        elif j == DEPOT_NODE:
            if i in route.stop_set and (i, j) not in route.arc_set:
                return False
        else:
            if (i in route.stop_set or j in route.stop_set) and (i, j) not in route.arc_set:
                return False

    return True


def solve_pricing_subproblem(
    instance: Instance,
    vehicle_type: VehicleTypeData,
    cover_duals: dict[int, float],
    type_dual: float,
    branch_ctx: BranchContext,
    pricing_time_limit: float | None,
    deadline: float | None,
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
        assignment_allowed = True
        forced_type = branch_ctx.forced_type_for_stop.get(stop_id)
        if forced_type is not None and forced_type != vehicle_type.type_id:
            assignment_allowed = False
        if vehicle_type.type_id in branch_ctx.excluded_types_by_stop.get(stop_id, set()):
            assignment_allowed = False

        feasible = (
            assignment_allowed
            and stop.demand_kg <= vehicle_type.capacity_kg
            and lb <= ub
        )
        stop_lb[stop_id] = lb
        stop_ub[stop_id] = ub
        feasible_stop[stop_id] = feasible
        late_ub[stop_id] = max(ub - stop.latest_min, 0.0) if feasible else 0.0

    if route_horizon <= 0:
        return None

    arc_keys: list[tuple[int, int]] = []
    for stop_id in stop_ids:
        if feasible_stop[stop_id] and (DEPOT_NODE, stop_id) not in branch_ctx.forbidden_arcs:
            arc_keys.append((DEPOT_NODE, stop_id))
        if feasible_stop[stop_id] and (stop_id, DEPOT_NODE) not in branch_ctx.forbidden_arcs:
            arc_keys.append((stop_id, DEPOT_NODE))

    for i in stop_ids:
        if not feasible_stop[i]:
            continue
        for j in stop_ids:
            if i == j or not feasible_stop[j]:
                continue
            if (i, j) in branch_ctx.forbidden_arcs:
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
    deadline_limit = remaining_time(deadline)
    if deadline_limit is not None:
        if deadline_limit <= 0:
            raise TimeoutError("Overall time limit reached during pricing.")
    effective_time_limit = pricing_time_limit
    if deadline_limit is not None:
        effective_time_limit = (
            min(deadline_limit, pricing_time_limit)
            if pricing_time_limit is not None
            else deadline_limit
        )
    if effective_time_limit is not None:
        pricing.Params.TimeLimit = effective_time_limit

    x = pricing.addVars(arc_keys, vtype=GRB.BINARY, name="x")
    y = pricing.addVars(stop_ids, vtype=GRB.BINARY, name="y")
    service_start = pricing.addVars(stop_ids, lb=0.0, name="service_start")
    late_min = pricing.addVars(stop_ids, lb=0.0, name="late_min")
    route_used = pricing.addVar(vtype=GRB.BINARY, name="route_used")
    return_min = pricing.addVar(lb=0.0, ub=end_limit, name="return_min")
    active_min = pricing.addVar(lb=0.0, ub=route_horizon, name="active_min")
    regular_min = pricing.addVar(lb=0.0, ub=route_used_max_regular, name="regular_min")
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

    for i, j in branch_ctx.forced_arcs:
        if i == DEPOT_NODE:
            if feasible_stop[j]:
                if (i, j) in x:
                    pricing.addConstr(x[i, j] == y[j], name=f"force_arc[{i},{j}]")
                else:
                    pricing.addConstr(y[j] == 0, name=f"force_off[{i},{j}]")
        elif j == DEPOT_NODE:
            if feasible_stop[i]:
                if (i, j) in x:
                    pricing.addConstr(x[i, j] == y[i], name=f"force_arc[{i},{j}]")
                else:
                    pricing.addConstr(y[i] == 0, name=f"force_off[{i},{j}]")
        else:
            if feasible_stop[i] and feasible_stop[j] and (i, j) in x:
                pricing.addConstr(x[i, j] == y[i], name=f"force_succ[{i},{j}]")
                pricing.addConstr(x[i, j] == y[j], name=f"force_pred[{i},{j}]")
            else:
                if feasible_stop[i]:
                    pricing.addConstr(y[i] == 0, name=f"force_stop_off[{i}]")
                if feasible_stop[j]:
                    pricing.addConstr(y[j] == 0, name=f"force_stop_off[{j}]")

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

    if pricing.Status == GRB.TIME_LIMIT:
        if deadline_limit is not None and (
            pricing_time_limit is None or deadline_limit <= pricing_time_limit + 1e-9
        ):
            raise TimeoutError("Overall time limit reached during pricing.")
        raise RuntimeError(
            "A pricing subproblem hit the user pricing time limit before optimality. "
            "Exact branch-and-price requires optimal pricing solves."
        )
    if pricing.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Pricing problem for {vehicle_type.type_id} failed with status {pricing.Status}."
        )

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
    if not route_is_compatible(route, branch_ctx):
        raise RuntimeError(
            f"Pricing returned a branch-incompatible route for {vehicle_type.type_id}: {sequence}"
        )
    return route


def build_master_model(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
    routes: list[RouteColumn],
    relax: bool,
    allow_artificial: bool = False,
) -> tuple[gp.Model, dict[str, Any]]:
    model = gp.Model("route_based_master")
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
        "cover_constraints": cover_constraints,
        "type_constraints": type_constraints,
        "routes": routes,
        "vehicle_types": vehicle_types,
    }
    return model, model._data


def solve_node_relaxation(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
    route_pool: list[RouteColumn],
    route_key_set: set[tuple[str, tuple[int, ...]]],
    branch_state: BranchState,
    max_cg_iterations: int,
    pricing_time_limit: float | None,
    deadline: float | None,
) -> NodeRelaxationResult | None:
    branch_ctx = build_branch_context(branch_state, instance, vehicle_types)
    if branch_ctx is None:
        return None

    compatible_routes = [route for route in route_pool if route_is_compatible(route, branch_ctx)]
    last_lp_objective = float("nan")

    for iteration in range(1, max_cg_iterations + 1):
        master_lp, lp_data = build_master_model(
            instance=instance,
            vehicle_types=vehicle_types,
            routes=compatible_routes,
            relax=True,
            allow_artificial=True,
        )
        master_time_limit = remaining_time(deadline)
        if master_time_limit is not None:
            if master_time_limit <= 0:
                raise TimeoutError("Overall time limit reached during node LP.")
            master_lp.Params.TimeLimit = master_time_limit
        master_lp.optimize()
        if master_lp.Status == GRB.TIME_LIMIT:
            raise TimeoutError("Overall time limit reached during node LP.")
        if master_lp.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Column-generation node LP did not solve to optimality (status {master_lp.Status})."
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
                branch_ctx=branch_ctx,
                pricing_time_limit=pricing_time_limit,
                deadline=deadline,
            )
            if route is None:
                continue
            key = (route.type_id, route.stop_sequence)
            if key not in route_key_set:
                route_key_set.add(key)
                route_pool.append(route)
                compatible_routes.append(route)
                new_routes.append(route)

        if not new_routes:
            artificial_usage = sum(
                lp_data["artificial_vars"][stop_id].X for stop_id in sorted(instance.stops)
            )
            if artificial_usage > INTEGRALITY_TOL:
                return None
            route_values = [
                lp_data["route_vars"][idx].X for idx in range(len(compatible_routes))
            ]
            return NodeRelaxationResult(
                lower_bound=last_lp_objective,
                routes=compatible_routes,
                route_values=route_values,
                cg_iterations=iteration,
            )

    raise RuntimeError(
        "Maximum column-generation iterations reached before node convergence. "
        "Increase --max-cg-iterations for an exact solve."
    )


def solve_restricted_integer_master(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
    routes: list[RouteColumn],
    deadline: float | None,
    heuristic_time_limit: float | None,
) -> tuple[float, list[RouteColumn]] | None:
    if not routes:
        return None

    model, data = build_master_model(
        instance=instance,
        vehicle_types=vehicle_types,
        routes=routes,
        relax=False,
        allow_artificial=False,
    )

    effective_time_limit = heuristic_time_limit
    remaining = remaining_time(deadline)
    if remaining is not None:
        if remaining <= 0:
            raise TimeoutError("Overall time limit reached during restricted master heuristic.")
        effective_time_limit = (
            min(heuristic_time_limit, remaining)
            if heuristic_time_limit is not None
            else remaining
        )
    if effective_time_limit is not None:
        model.Params.TimeLimit = effective_time_limit

    model.optimize()
    if model.Status == GRB.TIME_LIMIT and model.SolCount == 0:
        return None
    if model.SolCount == 0:
        return None

    route_vars = data["route_vars"]
    selected_routes = [routes[idx] for idx in range(len(routes)) if route_vars[idx].X > 0.5]
    return model.ObjVal, selected_routes


def is_integral_solution(route_values: list[float]) -> bool:
    return all(abs(value - round(value)) <= INTEGRALITY_TOL for value in route_values)


def compute_arc_values(
    routes: list[RouteColumn],
    route_values: list[float],
) -> dict[tuple[int, int], float]:
    arc_values: dict[tuple[int, int], float] = defaultdict(float)
    for route, value in zip(routes, route_values):
        if value <= INTEGRALITY_TOL:
            continue
        for arc in route.arc_set:
            arc_values[arc] += value
    return arc_values


def compute_assignment_values(
    routes: list[RouteColumn],
    route_values: list[float],
) -> dict[tuple[int, str], float]:
    assignment_values: dict[tuple[int, str], float] = defaultdict(float)
    for route, value in zip(routes, route_values):
        if value <= INTEGRALITY_TOL:
            continue
        for stop_id in route.stop_sequence:
            assignment_values[stop_id, route.type_id] += value
    return assignment_values


def select_fractional_arc(
    routes: list[RouteColumn],
    route_values: list[float],
    branch_ctx: BranchContext,
) -> tuple[int, int] | None:
    fixed_arcs = set(branch_ctx.forced_arcs) | set(branch_ctx.forbidden_arcs)
    best_arc: tuple[int, int] | None = None
    best_score: tuple[int, float] | None = None

    for arc, value in compute_arc_values(routes, route_values).items():
        if arc in fixed_arcs:
            continue
        if not (INTEGRALITY_TOL < value < 1.0 - INTEGRALITY_TOL):
            continue
        score = (1 if DEPOT_NODE in arc else 0, abs(value - 0.5))
        if best_score is None or score < best_score:
            best_score = score
            best_arc = arc

    return best_arc


def select_fractional_assignment(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
    routes: list[RouteColumn],
    route_values: list[float],
    branch_ctx: BranchContext,
) -> tuple[int, str] | None:
    assignment_values = compute_assignment_values(routes, route_values)
    best_assignment: tuple[int, str] | None = None
    best_score: float | None = None

    for stop_id in sorted(instance.stops):
        if stop_id in branch_ctx.forced_type_for_stop:
            continue
        for type_id in sorted(vehicle_types):
            if type_id in branch_ctx.excluded_types_by_stop.get(stop_id, set()):
                continue
            value = assignment_values.get((stop_id, type_id), 0.0)
            if not (INTEGRALITY_TOL < value < 1.0 - INTEGRALITY_TOL):
                continue
            score = abs(value - 0.5)
            if best_score is None or score < best_score:
                best_score = score
                best_assignment = (stop_id, type_id)

    return best_assignment


def create_model(
    workbook_path: Path,
    max_cg_iterations: int = 200,
    pricing_time_limit: float | None = None,
) -> tuple[gp.Model, dict[str, Any]]:
    instance = load_instance(workbook_path)
    validate_route_based_assumptions(instance)
    vehicle_types = build_vehicle_types(instance)
    route_pool = initial_routes(instance, vehicle_types)
    route_key_set = {(route.type_id, route.stop_sequence) for route in route_pool}
    root_state = BranchState(
        forbidden_arcs=frozenset(),
        forced_arcs=frozenset(),
        excluded_assignments=frozenset(),
        forced_assignments=frozenset(),
    )
    root_result = solve_node_relaxation(
        instance=instance,
        vehicle_types=vehicle_types,
        route_pool=route_pool,
        route_key_set=route_key_set,
        branch_state=root_state,
        max_cg_iterations=max_cg_iterations,
        pricing_time_limit=pricing_time_limit,
        deadline=None,
    )
    if root_result is None:
        raise RuntimeError("The root branch-and-price node is infeasible.")

    model, data = build_master_model(
        instance=instance,
        vehicle_types=vehicle_types,
        routes=root_result.routes,
        relax=False,
        allow_artificial=False,
    )
    data.update(
        {
            "instance": instance,
            "cg_iterations": root_result.cg_iterations,
            "lp_objective": root_result.lower_bound,
            "column_count": len(root_result.routes),
            "note": (
                "This is the root restricted master over generated columns. "
                "Exact solves happen through solve_model(), which runs branch-and-price."
            ),
        }
    )
    return model, data


def branch_and_price(
    instance: Instance,
    time_limit: float | None,
    mip_gap: float | None,
    max_cg_iterations: int,
    pricing_time_limit: float | None,
    max_bp_nodes: int | None,
) -> BranchAndPriceResult:
    validate_route_based_assumptions(instance)
    vehicle_types = build_vehicle_types(instance)
    route_pool = initial_routes(instance, vehicle_types)
    route_key_set = {(route.type_id, route.stop_sequence) for route in route_pool}

    root_state = BranchState(
        forbidden_arcs=frozenset(),
        forced_arcs=frozenset(),
        excluded_assignments=frozenset(),
        forced_assignments=frozenset(),
    )

    deadline = perf_counter() + time_limit if time_limit is not None else None
    active_nodes: list[tuple[float, int, int, BranchState]] = []
    sequence_counter = count()
    heapq.heappush(active_nodes, (0.0, 0, next(sequence_counter), root_state))

    incumbent_obj: float | None = None
    incumbent_routes: list[RouteColumn] = []
    root_lp_objective: float | None = None
    nodes_processed = 0
    nodes_pruned = 0
    status = "UNKNOWN"

    try:
        initial_heuristic = solve_restricted_integer_master(
            instance=instance,
            vehicle_types=vehicle_types,
            routes=route_pool,
            deadline=deadline,
            heuristic_time_limit=HEURISTIC_MASTER_TIME_LIMIT,
        )
    except TimeoutError:
        initial_heuristic = None
    if initial_heuristic is not None:
        incumbent_obj, incumbent_routes = initial_heuristic

    while active_nodes:
        if max_bp_nodes is not None and nodes_processed >= max_bp_nodes:
            status = "NODE_LIMIT"
            break
        if deadline is not None and remaining_time(deadline) <= 0:
            status = "TIME_LIMIT"
            break

        lower_bound_hint, depth, _, branch_state = heapq.heappop(active_nodes)
        if incumbent_obj is not None and lower_bound_hint >= incumbent_obj - INTEGRALITY_TOL:
            nodes_pruned += 1
            continue

        try:
            node_result = solve_node_relaxation(
                instance=instance,
                vehicle_types=vehicle_types,
                route_pool=route_pool,
                route_key_set=route_key_set,
                branch_state=branch_state,
                max_cg_iterations=max_cg_iterations,
                pricing_time_limit=pricing_time_limit,
                deadline=deadline,
            )
        except TimeoutError:
            heapq.heappush(
                active_nodes,
                (lower_bound_hint, depth, next(sequence_counter), branch_state),
            )
            status = "TIME_LIMIT"
            break

        nodes_processed += 1
        if node_result is None:
            continue
        if root_lp_objective is None:
            root_lp_objective = node_result.lower_bound

        node_lb = node_result.lower_bound
        if incumbent_obj is not None and node_lb >= incumbent_obj - INTEGRALITY_TOL:
            nodes_pruned += 1
            continue

        heuristic_obj: float | None = None
        if nodes_processed == 1 or incumbent_obj is None:
            try:
                heuristic_solution = solve_restricted_integer_master(
                    instance=instance,
                    vehicle_types=vehicle_types,
                    routes=node_result.routes,
                    deadline=deadline,
                    heuristic_time_limit=HEURISTIC_MASTER_TIME_LIMIT,
                )
            except TimeoutError:
                heapq.heappush(
                    active_nodes,
                    (node_lb, depth, next(sequence_counter), branch_state),
                )
                status = "TIME_LIMIT"
                break
            if heuristic_solution is not None:
                heuristic_obj, heuristic_routes = heuristic_solution
                if incumbent_obj is None or heuristic_obj < incumbent_obj - INTEGRALITY_TOL:
                    incumbent_obj = heuristic_obj
                    incumbent_routes = heuristic_routes

        if is_integral_solution(node_result.route_values):
            selected_routes = [
                route
                for route, value in zip(node_result.routes, node_result.route_values)
                if value > 0.5
            ]
            if incumbent_obj is None or node_lb < incumbent_obj - INTEGRALITY_TOL:
                incumbent_obj = node_lb
                incumbent_routes = selected_routes
            continue

        if heuristic_obj is not None and abs(heuristic_obj - node_lb) <= 1e-5:
            continue

        branch_ctx = build_branch_context(branch_state, instance, vehicle_types)
        if branch_ctx is None:
            continue

        child_states: list[BranchState] = []
        branch_arc = select_fractional_arc(
            routes=node_result.routes,
            route_values=node_result.route_values,
            branch_ctx=branch_ctx,
        )
        if branch_arc is not None:
            child_states = [
                BranchState(
                    forbidden_arcs=frozenset(set(branch_state.forbidden_arcs) | {branch_arc}),
                    forced_arcs=branch_state.forced_arcs,
                    excluded_assignments=branch_state.excluded_assignments,
                    forced_assignments=branch_state.forced_assignments,
                ),
                BranchState(
                    forbidden_arcs=branch_state.forbidden_arcs,
                    forced_arcs=frozenset(set(branch_state.forced_arcs) | {branch_arc}),
                    excluded_assignments=branch_state.excluded_assignments,
                    forced_assignments=branch_state.forced_assignments,
                ),
            ]
        else:
            branch_assignment = select_fractional_assignment(
                instance=instance,
                vehicle_types=vehicle_types,
                routes=node_result.routes,
                route_values=node_result.route_values,
                branch_ctx=branch_ctx,
            )
            if branch_assignment is not None:
                stop_id, type_id = branch_assignment
                child_states = [
                    BranchState(
                        forbidden_arcs=branch_state.forbidden_arcs,
                        forced_arcs=branch_state.forced_arcs,
                        excluded_assignments=frozenset(
                            set(branch_state.excluded_assignments) | {(stop_id, type_id)}
                        ),
                        forced_assignments=branch_state.forced_assignments,
                    ),
                    BranchState(
                        forbidden_arcs=branch_state.forbidden_arcs,
                        forced_arcs=branch_state.forced_arcs,
                        excluded_assignments=branch_state.excluded_assignments,
                        forced_assignments=frozenset(
                            set(branch_state.forced_assignments) | {(stop_id, type_id)}
                        ),
                    ),
                ]

        if not child_states:
            raise RuntimeError(
                "Branch-and-price could not find a valid branching object on a fractional node."
            )

        for child_state in child_states:
            if build_branch_context(child_state, instance, vehicle_types) is None:
                continue
            heapq.heappush(
                active_nodes,
                (node_lb, depth + 1, next(sequence_counter), child_state),
            )

        best_bound = min(bound for bound, *_ in active_nodes) if active_nodes else node_lb
        if incumbent_obj is not None and mip_gap is not None:
            gap = relative_gap(incumbent_obj, best_bound)
            if gap is not None and gap <= mip_gap + 1e-12:
                status = "GAP_LIMIT"
                break

    if status == "UNKNOWN":
        if active_nodes:
            status = "TIME_LIMIT" if deadline is not None and remaining_time(deadline) <= 0 else "NODE_LIMIT"
        elif incumbent_obj is None:
            status = "INFEASIBLE"
        else:
            status = "OPTIMAL"

    if active_nodes:
        best_bound = min(bound for bound, *_ in active_nodes)
    elif incumbent_obj is not None:
        best_bound = incumbent_obj
    else:
        best_bound = root_lp_objective

    return BranchAndPriceResult(
        status=status,
        objective=incumbent_obj,
        best_bound=best_bound,
        gap=relative_gap(incumbent_obj, best_bound),
        selected_routes=incumbent_routes,
        vehicle_types=vehicle_types,
        nodes_processed=nodes_processed,
        nodes_pruned=nodes_pruned,
        global_columns=len(route_pool),
        root_lp_objective=root_lp_objective,
    )


def solve_model(
    workbook_path: Path,
    time_limit: float | None,
    mip_gap: float | None,
    max_cg_iterations: int,
    pricing_time_limit: float | None,
    max_bp_nodes: int | None,
) -> None:
    instance = load_instance(workbook_path)
    result = branch_and_price(
        instance=instance,
        time_limit=time_limit,
        mip_gap=mip_gap,
        max_cg_iterations=max_cg_iterations,
        pricing_time_limit=pricing_time_limit,
        max_bp_nodes=max_bp_nodes,
    )

    print(f"Status: {result.status}")
    print("Formulation: Branch-and-Price Set Partitioning")
    print(f"Columns generated: {result.global_columns}")
    print(f"Nodes processed: {result.nodes_processed}")
    print(f"Nodes pruned: {result.nodes_pruned}")
    if result.root_lp_objective is not None:
        print(f"Root LP objective: {result.root_lp_objective:.2f}")
    if result.best_bound is not None:
        print(f"Best bound: {result.best_bound:.2f}")
    if result.objective is not None:
        print(f"Objective value: {result.objective:.2f}")
    if result.gap is not None:
        print(f"Relative gap: {100.0 * result.gap:.2f}%")
    print()

    if not result.selected_routes:
        print("No feasible incumbent route set found.")
        return

    usage_counter: dict[str, int] = defaultdict(int)
    for route in sorted(
        result.selected_routes,
        key=lambda candidate: (candidate.type_id, candidate.return_min, candidate.stop_sequence),
    ):
        vehicle_type = result.vehicle_types[route.type_id]
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
            "Solve the workbook-driven VRP with an exact branch-and-price algorithm "
            "over a route-based set-partitioning master problem."
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
        help="Optional overall branch-and-price time limit in seconds.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Optional relative optimality gap target for branch-and-price.",
    )
    parser.add_argument(
        "--max-cg-iterations",
        type=int,
        default=200,
        help="Maximum column-generation iterations allowed at each tree node.",
    )
    parser.add_argument(
        "--pricing-time-limit",
        type=float,
        default=None,
        help=(
            "Optional per-pricing time limit in seconds. "
            "If a pricing MIP hits this limit, the exact solve aborts."
        ),
    )
    parser.add_argument(
        "--max-bp-nodes",
        type=int,
        default=None,
        help="Optional cap on processed branch-and-price nodes.",
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
        max_bp_nodes=args.max_bp_nodes,
    )


if __name__ == "__main__":
    main()
