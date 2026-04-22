from __future__ import annotations

import argparse
import csv
import heapq
import math
import multiprocessing as mp
import os
import signal
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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
ARTIFICIAL_COVER_PENALTY = 1_000_000.0
INITIAL_ACTIVE_COVER_CANDIDATES_PER_STOP = 3
INITIAL_ACTIVE_CHEAPEST_ROUTES_PER_TYPE = 12
INITIAL_ACTIVE_RECENT_ROUTES_PER_TYPE = 12

_PRICING_WORKER_INSTANCE: Instance | None = None
_PRICING_WORKER_BRANCH_CTX: BranchContext | None = None
_INTERRUPT_SIGNAL_NAME: str | None = None
_MISSING = object()


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
    same_route_pairs: frozenset[tuple[int, int]]
    different_route_pairs: frozenset[tuple[int, int]]


@dataclass
class BranchContext:
    forbidden_arcs: frozenset[tuple[int, int]]
    forced_arcs: tuple[tuple[int, int], ...]
    forced_successor: dict[int, int]
    forced_predecessor: dict[int, int]
    forced_type_for_stop: dict[int, str]
    excluded_types_by_stop: dict[int, set[str]]
    together_groups: tuple[frozenset[int], ...]
    together_group_by_stop: dict[int, frozenset[int]]
    together_stop_pairs: frozenset[tuple[int, int]]
    separated_stop_pairs: frozenset[tuple[int, int]]
    separated_stops_by_stop: dict[int, set[int]]


@dataclass
class NodeRelaxationResult:
    lower_bound: float
    routes: list[RouteColumn]
    route_values: list[float]
    cg_iterations: int


@dataclass
class NodeMasterLP:
    model: gp.Model
    routes: list[RouteColumn]
    route_vars: list[gp.Var]
    artificial_vars: dict[int, gp.Var]
    cover_constraints: dict[int, gp.Constr]
    type_constraints: dict[str, gp.Constr]


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


@dataclass(frozen=True)
class PricingLabel:
    node: int
    visited_mask: int
    load_kg: float
    time_after_service: float
    reduced_cost: float
    nonlabor_reduced_cost: float
    predecessor_idx: int | None


class ProgressTracker:
    def __init__(self, enabled: bool, interval_seconds: float) -> None:
        self.enabled = enabled
        self.interval_seconds = max(interval_seconds, 0.1)
        self.start_time = perf_counter()
        self.last_emit = self.start_time - self.interval_seconds

    def emit(self, message: str, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        if not force and now - self.last_emit < self.interval_seconds:
            return
        elapsed = now - self.start_time
        print(f"[{elapsed:7.1f}s] {message}", flush=True)
        self.last_emit = now


class ObjectiveTraceLogger:
    FIELDNAMES = (
        "timestamp",
        "elapsed_seconds",
        "event",
        "status",
        "upper_bound",
        "lower_bound",
        "incumbent_objective",
        "best_bound",
        "relative_gap",
        "nodes_processed",
        "nodes_pruned",
        "global_columns",
        "queue_size",
        "node_depth",
        "note",
    )

    def __init__(self, path: Path | None, workbook_path: Path | None) -> None:
        self.path = path
        self.workbook_path = workbook_path
        self.start_time = perf_counter()
        self.latest_incumbent: float | None = None
        self.latest_best_bound: float | None = None
        self.latest_gap: float | None = None
        self.latest_status: str | None = None
        self._file = None
        self._writer: csv.DictWriter[str] | None = None

        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._flush()

    def _flush(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())

    def record(
        self,
        *,
        event: str,
        status: str | None = None,
        incumbent_objective: float | None | object = _MISSING,
        best_bound: float | None | object = _MISSING,
        relative_gap: float | None | object = _MISSING,
        nodes_processed: int | None = None,
        nodes_pruned: int | None = None,
        global_columns: int | None = None,
        queue_size: int | None = None,
        node_depth: int | None = None,
        note: str = "",
    ) -> None:
        if incumbent_objective is not _MISSING:
            self.latest_incumbent = incumbent_objective
        if best_bound is not _MISSING:
            self.latest_best_bound = best_bound
        if relative_gap is not _MISSING:
            self.latest_gap = relative_gap
        if status is not None:
            self.latest_status = status

        if self._writer is None:
            return

        elapsed = perf_counter() - self.start_time
        self._writer.writerow(
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "elapsed_seconds": f"{elapsed:.6f}",
                "event": event,
                "status": self.latest_status or "",
                "upper_bound": (
                    "" if self.latest_incumbent is None else f"{self.latest_incumbent:.10f}"
                ),
                "lower_bound": (
                    "" if self.latest_best_bound is None else f"{self.latest_best_bound:.10f}"
                ),
                "incumbent_objective": (
                    "" if self.latest_incumbent is None else f"{self.latest_incumbent:.10f}"
                ),
                "best_bound": (
                    "" if self.latest_best_bound is None else f"{self.latest_best_bound:.10f}"
                ),
                "relative_gap": "" if self.latest_gap is None else f"{self.latest_gap:.10f}",
                "nodes_processed": "" if nodes_processed is None else nodes_processed,
                "nodes_pruned": "" if nodes_pruned is None else nodes_pruned,
                "global_columns": "" if global_columns is None else global_columns,
                "queue_size": "" if queue_size is None else queue_size,
                "node_depth": "" if node_depth is None else node_depth,
                "note": note,
            }
        )
        self._flush()

    def close(self) -> None:
        if self._file is None:
            return
        self._flush()
        self._file.close()
        self._file = None
        self._writer = None


def optimize_with_heartbeat(
    model: gp.Model,
    progress: ProgressTracker | None,
    label: str,
) -> None:
    if progress is None or not progress.enabled:
        model.optimize()
        return

    active_callback_points = {
        GRB.Callback.SIMPLEX,
        GRB.Callback.BARRIER,
        GRB.Callback.MIP,
        GRB.Callback.MIPNODE,
        GRB.Callback.MIPSOL,
    }
    last_runtime_report = [-progress.interval_seconds]

    def callback(cb_model: gp.Model, where: int) -> None:
        if where not in active_callback_points:
            return
        runtime = cb_model.cbGet(GRB.Callback.RUNTIME)
        if runtime < progress.interval_seconds:
            return
        if runtime - last_runtime_report[0] < progress.interval_seconds:
            return
        progress.emit(
            f"{label}: still solving ({runtime:.1f}s in current Gurobi solve)",
            force=True,
        )
        last_runtime_report[0] = runtime

    model.optimize(callback)


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


def parse_optional_int_limit(value: str) -> int | None:
    text = value.strip().lower()
    if text in {"none", "unlimited", "inf", "infinite"}:
        return None

    parsed = int(text)
    if parsed <= 0:
        return None
    return parsed


def optional_float_changed(previous: float | None, current: float | None, tol: float = 1e-9) -> bool:
    if previous is None or current is None:
        return previous != current
    return abs(previous - current) > tol


def default_objective_trace_path(workbook_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return workbook_path.with_name(f"{workbook_path.stem}_objective_trace_{timestamp}.csv")


def install_interrupt_handlers() -> dict[int, Any]:
    previous_handlers: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        global _INTERRUPT_SIGNAL_NAME
        _INTERRUPT_SIGNAL_NAME = signal.Signals(signum).name
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous_handlers


def restore_interrupt_handlers(previous_handlers: dict[int, Any]) -> None:
    for signum, previous_handler in previous_handlers.items():
        signal.signal(signum, previous_handler)


def parse_closed_unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("Value must be between 0 and 1, inclusive.")
    return parsed


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_worker_count(value: str) -> int | None:
    text = value.strip().lower()
    if text == "auto":
        return None
    parsed = int(text)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Worker count must be a positive integer or 'auto'.")
    return parsed


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


def stabilize_dual_vector(
    raw_duals: dict[Any, float],
    previous_duals: dict[Any, float] | None,
    alpha: float,
) -> dict[Any, float]:
    if previous_duals is None or alpha >= 1.0 - 1e-12:
        return dict(raw_duals)
    if alpha <= 1e-12:
        return dict(previous_duals)
    return {
        key: alpha * raw_duals[key] + (1.0 - alpha) * previous_duals[key]
        for key in raw_duals
    }


def dual_shift_ratio(
    raw_duals: dict[Any, float],
    previous_raw_duals: dict[Any, float] | None,
) -> float:
    if previous_raw_duals is None or not raw_duals:
        return 0.0
    average_shift = sum(
        abs(raw_duals[key] - previous_raw_duals[key]) for key in raw_duals
    ) / len(raw_duals)
    average_magnitude = sum(abs(previous_raw_duals[key]) for key in previous_raw_duals) / len(
        previous_raw_duals
    )
    return average_shift / max(average_magnitude, 1.0)


def adaptive_stabilization_alpha(
    base_alpha: float,
    raw_cover_duals: dict[int, float],
    previous_raw_cover_duals: dict[int, float] | None,
    raw_type_duals: dict[str, float],
    previous_raw_type_duals: dict[str, float] | None,
    mode: str,
) -> float:
    if mode == "fixed" or base_alpha >= 1.0 - 1e-12:
        return base_alpha

    cover_shift = dual_shift_ratio(raw_cover_duals, previous_raw_cover_duals)
    type_shift = dual_shift_ratio(raw_type_duals, previous_raw_type_duals)
    weighted_shift = (
        cover_shift * len(raw_cover_duals) + type_shift * len(raw_type_duals)
    ) / max(len(raw_cover_duals) + len(raw_type_duals), 1)

    spread = min(base_alpha, 1.0 - base_alpha, 0.25)
    min_alpha = max(0.05, base_alpha - spread)
    max_alpha = min(1.0, base_alpha + spread)
    if weighted_shift <= 0.05:
        return max_alpha
    if weighted_shift >= 0.35:
        return min_alpha

    position = (weighted_shift - 0.05) / 0.30
    return max_alpha - position * (max_alpha - min_alpha)


def route_arc_set(stop_sequence: tuple[int, ...]) -> frozenset[tuple[int, int]]:
    if not stop_sequence:
        return frozenset()

    arcs: list[tuple[int, int]] = [(DEPOT_NODE, stop_sequence[0])]
    for idx in range(len(stop_sequence) - 1):
        arcs.append((stop_sequence[idx], stop_sequence[idx + 1]))
    arcs.append((stop_sequence[-1], DEPOT_NODE))
    return frozenset(arcs)


def route_reduced_cost(
    route: RouteColumn,
    cover_duals: dict[int, float],
    type_dual: float,
) -> float:
    return route.cost - sum(cover_duals[stop_id] for stop_id in route.stop_sequence) - type_dual


def route_column_key(route: RouteColumn) -> tuple[str, tuple[int, ...]]:
    return route.type_id, route.stop_sequence


def select_initial_active_routes(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
    compatible_routes: list[RouteColumn],
) -> list[RouteColumn]:
    if not compatible_routes:
        return []

    active_keys: set[tuple[str, tuple[int, ...]]] = set()
    routes_by_type: dict[str, list[RouteColumn]] = {type_id: [] for type_id in vehicle_types}
    routes_by_stop: dict[int, list[RouteColumn]] = {stop_id: [] for stop_id in instance.stops}

    for route in compatible_routes:
        routes_by_type[route.type_id].append(route)
        for stop_id in route.stop_set:
            routes_by_stop[stop_id].append(route)

    for stop_id, covering_routes in routes_by_stop.items():
        for route in sorted(
            covering_routes,
            key=lambda candidate: (
                candidate.cost,
                len(candidate.stop_sequence),
                candidate.return_min,
                candidate.stop_sequence,
            ),
        )[:INITIAL_ACTIVE_COVER_CANDIDATES_PER_STOP]:
            active_keys.add(route_column_key(route))

    for type_id, typed_routes in routes_by_type.items():
        cheapest_routes = sorted(
            typed_routes,
            key=lambda candidate: (
                candidate.cost,
                len(candidate.stop_sequence),
                candidate.return_min,
                candidate.stop_sequence,
            ),
        )[:INITIAL_ACTIVE_CHEAPEST_ROUTES_PER_TYPE]
        for route in cheapest_routes:
            active_keys.add(route_column_key(route))

        for route in typed_routes[-INITIAL_ACTIVE_RECENT_ROUTES_PER_TYPE:]:
            active_keys.add(route_column_key(route))

    return [route for route in compatible_routes if route_column_key(route) in active_keys]


def select_inactive_reactivation_routes(
    inactive_routes_by_type: dict[str, dict[tuple[str, tuple[int, ...]], RouteColumn]],
    cover_duals: dict[int, float],
    type_duals: dict[str, float],
    max_per_type: int,
) -> list[RouteColumn]:
    selected_routes: list[RouteColumn] = []
    for type_id, inactive_routes in inactive_routes_by_type.items():
        improving_routes = [
            (
                route_reduced_cost(
                    route,
                    cover_duals=cover_duals,
                    type_dual=type_duals[type_id],
                ),
                route,
            )
            for route in inactive_routes.values()
        ]
        improving_routes = [
            (reduced_cost, route)
            for reduced_cost, route in improving_routes
            if reduced_cost < -REDUCED_COST_TOL
        ]
        improving_routes.sort(
            key=lambda item: (
                item[0],
                item[1].cost,
                len(item[1].stop_sequence),
                item[1].stop_sequence,
            )
        )
        selected_routes.extend(route for _, route in improving_routes[:max_per_type])
    return selected_routes


def init_pricing_worker(instance: Instance, branch_ctx: BranchContext) -> None:
    global _PRICING_WORKER_INSTANCE, _PRICING_WORKER_BRANCH_CTX
    _PRICING_WORKER_INSTANCE = instance
    _PRICING_WORKER_BRANCH_CTX = branch_ctx


def run_pricing_worker(
    vehicle_type: VehicleTypeData,
    cover_duals: dict[int, float],
    type_dual: float,
    max_routes: int,
    excluded_stop_sequences: set[tuple[int, ...]],
    pricing_time_limit: float | None,
    deadline: float | None,
) -> tuple[str, list[RouteColumn]]:
    if _PRICING_WORKER_INSTANCE is None or _PRICING_WORKER_BRANCH_CTX is None:
        raise RuntimeError("Pricing worker was not initialized before solving pricing tasks.")
    return (
        vehicle_type.type_id,
        solve_pricing_subproblem(
            instance=_PRICING_WORKER_INSTANCE,
            vehicle_type=vehicle_type,
            cover_duals=cover_duals,
            type_dual=type_dual,
            branch_ctx=_PRICING_WORKER_BRANCH_CTX,
            max_routes=max_routes,
            excluded_stop_sequences=excluded_stop_sequences,
            pricing_time_limit=pricing_time_limit,
            deadline=deadline,
            progress=None,
            progress_label=None,
        ),
    )


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


def route_distance_unit_cost(instance: Instance, vehicle_type: VehicleTypeData) -> float:
    return vehicle_type.cost_per_mile + (
        instance.cost_params["Fuel_Cost_per_Liter"]
        * instance.cost_params["Avg_Fuel_Consumption_L_per_mile"]
    )


def billed_operating_hours(active_min: float) -> int:
    if active_min <= 1e-9:
        return 0
    return math.ceil(max(active_min - 1e-9, 0.0) / 60.0)


def operating_labor_cost(instance: Instance, active_min: float) -> float:
    return billed_operating_hours(active_min) * instance.cost_params["Driver_Hourly_Wage"]


def incremental_labor_cost(
    instance: Instance,
    elapsed_before_min: float,
    delta_min: float,
) -> float:
    if delta_min <= 0:
        return 0.0

    elapsed_after_min = max(elapsed_before_min + delta_min, 0.0)
    return operating_labor_cost(instance, elapsed_after_min) - operating_labor_cost(
        instance,
        max(elapsed_before_min, 0.0),
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

    cost = vehicle_type.fixed_daily_cost
    cost += distance_mi * route_distance_unit_cost(instance, vehicle_type)
    cost += operating_labor_cost(instance, active_min)
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

    stop_ids = sorted(instance.stops)
    parent = {stop_id: stop_id for stop_id in stop_ids}

    def find(stop_id: int) -> int:
        while parent[stop_id] != stop_id:
            parent[stop_id] = parent[parent[stop_id]]
            stop_id = parent[stop_id]
        return stop_id

    def union(stop_a: int, stop_b: int) -> None:
        root_a = find(stop_a)
        root_b = find(stop_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for stop_a, stop_b in branch_state.same_route_pairs:
        if stop_a not in instance.stops or stop_b not in instance.stops or stop_a == stop_b:
            return None
        union(stop_a, stop_b)

    group_members: dict[int, set[int]] = defaultdict(set)
    for stop_id in stop_ids:
        group_members[find(stop_id)].add(stop_id)

    together_groups = tuple(
        sorted((frozenset(group) for group in group_members.values() if len(group) > 1), key=min)
    )
    together_group_by_stop = {
        stop_id: group for group in together_groups for stop_id in group
    }

    together_stop_pairs: set[tuple[int, int]] = set()
    for group in together_groups:
        ordered_group = sorted(group)
        for idx, stop_a in enumerate(ordered_group):
            for stop_b in ordered_group[idx + 1 :]:
                together_stop_pairs.add((stop_a, stop_b))

    separated_stop_pairs: set[tuple[int, int]] = set()
    separated_stops_by_stop: dict[int, set[int]] = defaultdict(set)
    for stop_a, stop_b in branch_state.different_route_pairs:
        if stop_a not in instance.stops or stop_b not in instance.stops or stop_a == stop_b:
            return None
        root_a = find(stop_a)
        root_b = find(stop_b)
        if root_a == root_b:
            return None
        for group_stop_a in group_members[root_a]:
            for group_stop_b in group_members[root_b]:
                pair = tuple(sorted((group_stop_a, group_stop_b)))
                separated_stop_pairs.add(pair)
                separated_stops_by_stop[group_stop_a].add(group_stop_b)
                separated_stops_by_stop[group_stop_b].add(group_stop_a)

    return BranchContext(
        forbidden_arcs=branch_state.forbidden_arcs,
        forced_arcs=tuple(sorted(branch_state.forced_arcs)),
        forced_successor=forced_successor,
        forced_predecessor=forced_predecessor,
        forced_type_for_stop=forced_type_for_stop,
        excluded_types_by_stop=excluded_types_by_stop,
        together_groups=together_groups,
        together_group_by_stop=together_group_by_stop,
        together_stop_pairs=frozenset(together_stop_pairs),
        separated_stop_pairs=frozenset(separated_stop_pairs),
        separated_stops_by_stop=separated_stops_by_stop,
    )


def route_is_compatible(route: RouteColumn, branch_ctx: BranchContext) -> bool:
    if route.arc_set & branch_ctx.forbidden_arcs:
        return False

    for group in branch_ctx.together_groups:
        if route.stop_set & group and not group.issubset(route.stop_set):
            return False

    for stop_id in route.stop_set:
        if branch_ctx.separated_stops_by_stop.get(stop_id, set()) & route.stop_set:
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
    max_routes: int,
    excluded_stop_sequences: set[tuple[int, ...]] | None,
    pricing_time_limit: float | None,
    deadline: float | None,
    progress: ProgressTracker | None = None,
    progress_label: str | None = None,
) -> list[RouteColumn]:
    if max_routes <= 0:
        return []
    stop_ids = sorted(instance.stops)
    start_limit = vehicle_start_limit(instance, vehicle_type)
    end_limit = vehicle_end_limit(instance, vehicle_type)
    route_horizon = max(end_limit - start_limit, 0.0)
    if route_horizon <= 0:
        return []

    pricing_start = perf_counter()
    distance_unit_cost = route_distance_unit_cost(instance, vehicle_type)
    last_forced_stops = {
        i for i, j in branch_ctx.forced_arcs if i != DEPOT_NODE and j == DEPOT_NODE
    }
    excluded_stop_sequences = excluded_stop_sequences or set()

    def check_time_limit() -> None:
        now = perf_counter()
        if deadline is not None and now >= deadline - 1e-9:
            raise TimeoutError("Overall time limit reached during pricing.")
        if pricing_time_limit is not None and now >= pricing_start + pricing_time_limit - 1e-9:
            raise RuntimeError(
                "A pricing subproblem hit the user pricing time limit before optimality. "
                "Exact branch-and-price requires complete label-setting pricing."
            )

    stop_to_pos = {stop_id: idx for idx, stop_id in enumerate(stop_ids)}
    stop_mask = {stop_id: 1 << stop_to_pos[stop_id] for stop_id in stop_ids}
    together_group_masks = {
        sum(stop_mask[group_stop] for group_stop in group if group_stop in stop_mask)
        for group in branch_ctx.together_groups
    }
    separated_mask_by_stop = {
        stop_id: sum(
            stop_mask[other_stop]
            for other_stop in branch_ctx.separated_stops_by_stop.get(stop_id, set())
            if other_stop in stop_mask
        )
        for stop_id in stop_ids
    }

    feasible_stop: dict[int, bool] = {}
    potential_successors: dict[int, list[int]] = {DEPOT_NODE: []}
    for stop_id in stop_ids:
        potential_successors[stop_id] = []

    for stop_id in stop_ids:
        stop = instance.stops[stop_id]
        finish_time = (
            start_limit
            + instance.travel_minutes[DEPOT_NODE, stop_id]
            + stop.service_min
        )
        assignment_allowed = True
        forced_type = branch_ctx.forced_type_for_stop.get(stop_id)
        if forced_type is not None and forced_type != vehicle_type.type_id:
            assignment_allowed = False
        if vehicle_type.type_id in branch_ctx.excluded_types_by_stop.get(stop_id, set()):
            assignment_allowed = False

        feasible = (
            assignment_allowed
            and stop.demand_kg <= vehicle_type.capacity_kg + 1e-9
            and finish_time + instance.travel_minutes[stop_id, DEPOT_NODE] <= end_limit + 1e-9
        )
        feasible_stop[stop_id] = feasible
        if feasible and (DEPOT_NODE, stop_id) not in branch_ctx.forbidden_arcs:
            potential_successors[DEPOT_NODE].append(stop_id)

    for i in stop_ids:
        if not feasible_stop[i]:
            continue
        earliest_depart_i = (
            start_limit
            + instance.travel_minutes[DEPOT_NODE, i]
            + instance.stops[i].service_min
        )
        for j in stop_ids:
            if i == j or not feasible_stop[j]:
                continue
            if (i, j) in branch_ctx.forbidden_arcs:
                continue
            earliest_finish_j = (
                earliest_depart_i
                + instance.travel_minutes[i, j]
                + instance.stops[j].service_min
            )
            if earliest_finish_j + instance.travel_minutes[j, DEPOT_NODE] <= end_limit + 1e-9:
                potential_successors[i].append(j)

    positive_cover_duals = {
        stop_id: max(cover_duals[stop_id], 0.0) for stop_id in stop_ids
    }
    positive_dual_stops = [
        stop_id for stop_id in stop_ids if positive_cover_duals[stop_id] > REDUCED_COST_TOL
    ]
    group_mask_list = sorted(together_group_masks)
    stop_bit_pairs = [(stop_id, stop_mask[stop_id]) for stop_id in stop_ids]
    mask_resource_cache: dict[int, tuple[float, float]] = {}
    partial_group_requirement_cache: dict[int, int] = {}
    forced_chain_cache: dict[int, tuple[int, float] | None] = {}
    mandatory_remainder_cache: dict[tuple[int, int], tuple[int, float, float] | None] = {}
    optimistic_reward_cache: dict[tuple[int, int], float] = {}

    def mask_resources(mask: int) -> tuple[float, float]:
        cached = mask_resource_cache.get(mask)
        if cached is not None:
            return cached
        demand = 0.0
        service = 0.0
        for stop_id, stop_bit in stop_bit_pairs:
            if mask & stop_bit:
                stop = instance.stops[stop_id]
                demand += stop.demand_kg
                service += stop.service_min
        mask_resource_cache[mask] = (demand, service)
        return mask_resource_cache[mask]

    def partial_group_requirement_mask(visited_mask: int) -> int:
        cached = partial_group_requirement_cache.get(visited_mask)
        if cached is not None:
            return cached
        requirement_mask = 0
        for group_mask in group_mask_list:
            visited_group_mask = visited_mask & group_mask
            if visited_group_mask not in {0, group_mask}:
                requirement_mask |= group_mask & ~visited_mask
        partial_group_requirement_cache[visited_mask] = requirement_mask
        return requirement_mask

    def forced_chain_requirement(node: int) -> tuple[int, float] | None:
        if node in forced_chain_cache:
            return forced_chain_cache[node]
        if node == DEPOT_NODE:
            forced_chain_cache[node] = (0, 0.0)
            return forced_chain_cache[node]

        requirement_mask = 0
        minimum_extra_time = 0.0
        current = node
        seen: set[int] = set()
        while current in branch_ctx.forced_successor:
            next_node = branch_ctx.forced_successor[current]
            minimum_extra_time += instance.travel_minutes[current, next_node]
            if next_node == DEPOT_NODE:
                forced_chain_cache[node] = (requirement_mask, minimum_extra_time)
                return forced_chain_cache[node]
            if next_node in seen or not feasible_stop.get(next_node, False):
                forced_chain_cache[node] = None
                return None
            seen.add(next_node)
            requirement_mask |= stop_mask[next_node]
            minimum_extra_time += instance.stops[next_node].service_min
            current = next_node

        forced_chain_cache[node] = (requirement_mask, minimum_extra_time)
        return forced_chain_cache[node]

    label_store: list[PricingLabel] = []
    label_alive: list[bool] = []
    labels_at_node: dict[int, list[int]] = defaultdict(list)
    frontier: list[tuple[float, float, int]] = []
    explored_labels = 0

    def dominates(lhs: PricingLabel, rhs: PricingLabel) -> bool:
        return (
            lhs.node == rhs.node
            and lhs.time_after_service <= rhs.time_after_service + 1e-9
            and lhs.load_kg <= rhs.load_kg + 1e-9
            and lhs.nonlabor_reduced_cost <= rhs.nonlabor_reduced_cost + 1e-9
            and (lhs.visited_mask & ~rhs.visited_mask) == 0
        )

    def mandatory_remainder_info(label: PricingLabel) -> tuple[int, float, float] | None:
        cache_key = (label.node, label.visited_mask)
        if cache_key in mandatory_remainder_cache:
            return mandatory_remainder_cache[cache_key]

        chain_requirement = forced_chain_requirement(label.node)
        if chain_requirement is None:
            mandatory_remainder_cache[cache_key] = None
            return None
        chain_mask, chain_time_lb = chain_requirement
        if chain_mask & label.visited_mask:
            mandatory_remainder_cache[cache_key] = None
            return None

        group_mask = partial_group_requirement_mask(label.visited_mask)
        if group_mask & ~chain_mask:
            chain_terminal_node = label.node
            while chain_terminal_node in branch_ctx.forced_successor:
                next_node = branch_ctx.forced_successor[chain_terminal_node]
                if next_node == DEPOT_NODE:
                    mandatory_remainder_cache[cache_key] = None
                    return None
                chain_terminal_node = next_node

        requirement_mask = chain_mask | group_mask
        if requirement_mask & label.visited_mask:
            mandatory_remainder_cache[cache_key] = None
            return None

        for stop_id, stop_bit in stop_bit_pairs:
            if not (requirement_mask & stop_bit):
                continue
            if separated_mask_by_stop[stop_id] & (label.visited_mask | (requirement_mask ^ stop_bit)):
                mandatory_remainder_cache[cache_key] = None
                return None

        requirement_demand, requirement_service = mask_resources(requirement_mask)
        chain_demand, chain_service = mask_resources(chain_mask)
        mandatory_remainder_cache[cache_key] = (
            requirement_mask,
            requirement_demand,
            (requirement_service - chain_service) + chain_time_lb,
        )
        return mandatory_remainder_cache[cache_key]

    def optimistic_remaining_reward(label: PricingLabel) -> float:
        cache_key = (label.node, label.visited_mask)
        if cache_key in optimistic_reward_cache:
            return optimistic_reward_cache[cache_key]

        if label.node in last_forced_stops:
            optimistic_reward_cache[cache_key] = 0.0
            return 0.0

        reward = 0.0
        for stop_id in positive_dual_stops:
            stop_bit = stop_mask[stop_id]
            if label.visited_mask & stop_bit:
                continue
            if separated_mask_by_stop[stop_id] & label.visited_mask:
                continue
            forced_pred = branch_ctx.forced_predecessor.get(stop_id)
            if (
                forced_pred is not None
                and forced_pred != label.node
                and forced_pred in stop_mask
                and label.visited_mask & stop_mask[forced_pred]
            ):
                continue
            reward += positive_cover_duals[stop_id]

        optimistic_reward_cache[cache_key] = reward
        return reward

    def optimistic_completion_lower_bound(label: PricingLabel) -> float:
        mandatory_info = mandatory_remainder_info(label)
        if mandatory_info is None:
            return float("inf")
        _, requirement_demand, requirement_time_lb = mandatory_info
        if label.load_kg + requirement_demand > vehicle_type.capacity_kg + 1e-9:
            return float("inf")
        if label.time_after_service + requirement_time_lb > end_limit + 1e-9:
            return float("inf")
        if not can_return_from(label) and not candidate_successors(label):
            return float("inf")
        return label.reduced_cost - optimistic_remaining_reward(label)

    def can_return_from(label: PricingLabel) -> bool:
        if label.node in branch_ctx.forced_successor:
            return False
        if (label.node, DEPOT_NODE) in branch_ctx.forbidden_arcs:
            return False
        for group_mask in together_group_masks:
            visited_group_mask = label.visited_mask & group_mask
            if visited_group_mask not in {0, group_mask}:
                return False
        return (
            label.time_after_service + instance.travel_minutes[label.node, DEPOT_NODE]
            <= end_limit + 1e-9
        )

    def complete_reduced_cost(label: PricingLabel) -> float:
        elapsed_before = label.time_after_service - start_limit
        return_delta = instance.travel_minutes[label.node, DEPOT_NODE]
        return (
            label.reduced_cost
            + instance.distance_miles[label.node, DEPOT_NODE] * distance_unit_cost
            + incremental_labor_cost(instance, elapsed_before, return_delta)
        )

    def reconstruct_route(label_idx: int) -> RouteColumn:
        sequence: list[int] = []
        cursor = label_idx
        while cursor is not None:
            current_label = label_store[cursor]
            sequence.append(current_label.node)
            cursor = current_label.predecessor_idx
        sequence.reverse()
        route = evaluate_route_sequence(instance, vehicle_type, tuple(sequence))
        if route is None:
            raise RuntimeError(
                f"Label-setting pricing reconstructed an invalid route for "
                f"{vehicle_type.type_id}: {sequence}"
            )
        if not route_is_compatible(route, branch_ctx):
            raise RuntimeError(
                f"Label-setting pricing reconstructed a branch-incompatible route for "
                f"{vehicle_type.type_id}: {sequence}"
            )
        return route

    def exact_route_reduced_cost(route: RouteColumn) -> float:
        return route_reduced_cost(route, cover_duals=cover_duals, type_dual=type_dual)

    def add_label(label: PricingLabel) -> int | None:
        optimistic_lb = optimistic_completion_lower_bound(label)
        if optimistic_lb >= -REDUCED_COST_TOL:
            return None

        node_labels = labels_at_node[label.node]
        surviving: list[int] = []
        for idx in node_labels:
            incumbent = label_store[idx]
            if dominates(incumbent, label):
                return None
            if dominates(label, incumbent):
                label_alive[idx] = False
                continue
            surviving.append(idx)

        label_idx = len(label_store)
        label_store.append(label)
        label_alive.append(True)
        surviving.append(label_idx)
        labels_at_node[label.node] = surviving
        heapq.heappush(frontier, (optimistic_lb, label.reduced_cost, label_idx))
        return label_idx

    def candidate_successors(label: PricingLabel) -> list[int]:
        forced_next = branch_ctx.forced_successor.get(label.node)
        if forced_next is not None:
            if (
                forced_next in stop_mask
                and label.visited_mask & stop_mask[forced_next] == 0
                and forced_next in potential_successors[label.node]
            ):
                return [forced_next]
            return []

        if label.node in last_forced_stops:
            return []

        candidates: list[int] = []
        for next_stop in potential_successors[label.node]:
            if label.visited_mask & stop_mask[next_stop]:
                continue
            if separated_mask_by_stop[next_stop] & label.visited_mask:
                continue
            forced_pred = branch_ctx.forced_predecessor.get(next_stop)
            if forced_pred is not None and forced_pred != label.node:
                continue
            candidates.append(next_stop)
        return candidates

    improving_routes: list[tuple[float, RouteColumn]] = []
    improving_stop_sequences: set[tuple[int, ...]] = set()

    def maybe_add_route(label_idx: int) -> bool:
        label = label_store[label_idx]
        if not can_return_from(label):
            return False
        if complete_reduced_cost(label) >= -REDUCED_COST_TOL:
            return False
        route = reconstruct_route(label_idx)
        if (
            route.stop_sequence in excluded_stop_sequences
            or route.stop_sequence in improving_stop_sequences
        ):
            return False
        exact_rc = exact_route_reduced_cost(route)
        if exact_rc >= -REDUCED_COST_TOL:
            return False
        improving_stop_sequences.add(route.stop_sequence)
        improving_routes.append((exact_rc, route))
        return len(improving_routes) >= max_routes

    if progress is not None:
        progress.emit(
            f"{progress_label or f'pricing {vehicle_type.type_id}'}: starting label-setting "
            f"pricing over {sum(feasible_stop.values())} feasible stops"
        )

    for start_stop in potential_successors[DEPOT_NODE]:
        forced_pred = branch_ctx.forced_predecessor.get(start_stop)
        if forced_pred is not None and forced_pred != DEPOT_NODE:
            continue

        stop = instance.stops[start_stop]
        arrival = start_limit + instance.travel_minutes[DEPOT_NODE, start_stop]
        time_after_service = arrival + stop.service_min
        elapsed = time_after_service - start_limit
        reduced_cost = (
            vehicle_type.fixed_daily_cost
            - type_dual
            + instance.distance_miles[DEPOT_NODE, start_stop] * distance_unit_cost
            + incremental_labor_cost(instance, 0.0, elapsed)
            + max(arrival - stop.latest_min, 0.0)
            / 60.0
            * instance.cost_params["Late_Delivery_Penalty_per_Hour"]
            - cover_duals[start_stop]
        )

        label_idx = add_label(
            PricingLabel(
                node=start_stop,
                visited_mask=stop_mask[start_stop],
                load_kg=stop.demand_kg,
                time_after_service=time_after_service,
                reduced_cost=reduced_cost,
                nonlabor_reduced_cost=reduced_cost - operating_labor_cost(instance, elapsed),
                predecessor_idx=None,
            )
        )
        if label_idx is not None and maybe_add_route(label_idx):
            return [route for _, route in sorted(improving_routes, key=lambda item: item[0])]

    while frontier:
        _, _, label_idx = heapq.heappop(frontier)
        if not label_alive[label_idx]:
            continue

        explored_labels += 1
        if explored_labels % 250 == 0:
            check_time_limit()
            if progress is not None:
                progress.emit(
                    f"{progress_label or f'pricing {vehicle_type.type_id}'}: explored "
                    f"{explored_labels} labels, frontier {len(frontier)}"
                )

        label = label_store[label_idx]
        for next_stop in candidate_successors(label):
            next_stop_data = instance.stops[next_stop]
            new_load = label.load_kg + next_stop_data.demand_kg
            if new_load > vehicle_type.capacity_kg + 1e-9:
                continue

            arrival = label.time_after_service + instance.travel_minutes[label.node, next_stop]
            time_after_service = arrival + next_stop_data.service_min
            if time_after_service + instance.travel_minutes[next_stop, DEPOT_NODE] > end_limit + 1e-9:
                continue

            elapsed_before = label.time_after_service - start_limit
            delta_minutes = (
                instance.travel_minutes[label.node, next_stop]
                + next_stop_data.service_min
            )
            new_reduced_cost = (
                label.reduced_cost
                + instance.distance_miles[label.node, next_stop] * distance_unit_cost
                + incremental_labor_cost(instance, elapsed_before, delta_minutes)
                + max(arrival - next_stop_data.latest_min, 0.0)
                / 60.0
                * instance.cost_params["Late_Delivery_Penalty_per_Hour"]
                - cover_duals[next_stop]
            )

            new_label_idx = add_label(
                PricingLabel(
                    node=next_stop,
                    visited_mask=label.visited_mask | stop_mask[next_stop],
                    load_kg=new_load,
                    time_after_service=time_after_service,
                    reduced_cost=new_reduced_cost,
                    nonlabor_reduced_cost=(
                        label.nonlabor_reduced_cost
                        + instance.distance_miles[label.node, next_stop] * distance_unit_cost
                        + max(arrival - next_stop_data.latest_min, 0.0)
                        / 60.0
                        * instance.cost_params["Late_Delivery_Penalty_per_Hour"]
                        - cover_duals[next_stop]
                    ),
                    predecessor_idx=label_idx,
                )
            )
            if new_label_idx is None:
                continue

            if maybe_add_route(new_label_idx):
                return [route for _, route in sorted(improving_routes, key=lambda item: item[0])]

    check_time_limit()
    return [route for _, route in sorted(improving_routes, key=lambda item: item[0])]


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
                ARTIFICIAL_COVER_PENALTY * artificial_vars[stop_id]
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


def build_incremental_node_master(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
    routes: list[RouteColumn],
) -> NodeMasterLP:
    model = gp.Model("route_based_node_master")
    model.Params.OutputFlag = 0
    model.ModelSense = GRB.MINIMIZE

    artificial_vars = model.addVars(
        sorted(instance.stops),
        lb=0.0,
        ub=1.0,
        obj=ARTIFICIAL_COVER_PENALTY,
        vtype=GRB.CONTINUOUS,
        name="artificial_cover",
    )

    cover_constraints: dict[int, gp.Constr] = {}
    for stop_id in sorted(instance.stops):
        cover_constraints[stop_id] = model.addConstr(
            artificial_vars[stop_id] == 1.0,
            name=f"cover[{stop_id}]",
        )

    type_constraints: dict[str, gp.Constr] = {}
    for type_id, vehicle_type in vehicle_types.items():
        type_constraints[type_id] = model.addConstr(
            gp.LinExpr() <= vehicle_type.count,
            name=f"type_limit[{type_id}]",
        )

    node_master = NodeMasterLP(
        model=model,
        routes=[],
        route_vars=[],
        artificial_vars={stop_id: artificial_vars[stop_id] for stop_id in sorted(instance.stops)},
        cover_constraints=cover_constraints,
        type_constraints=type_constraints,
    )
    for route in routes:
        add_route_column_to_master(node_master, route)
    model.update()
    return node_master


def add_route_column_to_master(node_master: NodeMasterLP, route: RouteColumn) -> gp.Var:
    column = gp.Column()
    for stop_id in route.stop_set:
        column.addTerms(1.0, node_master.cover_constraints[stop_id])
    column.addTerms(1.0, node_master.type_constraints[route.type_id])

    route_index = len(node_master.routes)
    route_var = node_master.model.addVar(
        lb=0.0,
        ub=1.0,
        obj=route.cost,
        vtype=GRB.CONTINUOUS,
        column=column,
        name=f"route[{route_index}]",
    )
    node_master.routes.append(route)
    node_master.route_vars.append(route_var)
    return route_var


def solve_node_relaxation(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
    route_pool: list[RouteColumn],
    route_key_set: set[tuple[str, tuple[int, ...]]],
    branch_state: BranchState,
    max_cg_iterations: int | None,
    pricing_time_limit: float | None,
    pricing_columns_per_type: int,
    pricing_workers: int | None,
    dual_stabilization_alpha: float,
    dual_stabilization_mode: str,
    deadline: float | None,
    progress: ProgressTracker | None = None,
    node_label: str = "root",
) -> NodeRelaxationResult | None:
    branch_ctx = build_branch_context(branch_state, instance, vehicle_types)
    if branch_ctx is None:
        return None

    compatible_routes = [route for route in route_pool if route_is_compatible(route, branch_ctx)]
    initial_active_routes = select_initial_active_routes(
        instance=instance,
        vehicle_types=vehicle_types,
        compatible_routes=compatible_routes,
    )
    if not initial_active_routes:
        initial_active_routes = compatible_routes[:]
    node_master = build_incremental_node_master(
        instance=instance,
        vehicle_types=vehicle_types,
        routes=initial_active_routes,
    )
    last_lp_objective = float("nan")
    if progress is not None:
        progress.emit(
            f"{node_label}: starting node relaxation with {len(initial_active_routes)} active "
            f"columns out of {len(compatible_routes)} compatible global columns",
            force=True,
        )

    route_sequences_by_type: dict[str, set[tuple[int, ...]]] = {
        type_id: set() for type_id in vehicle_types
    }
    for type_id, stop_sequence in route_key_set:
        if type_id in route_sequences_by_type:
            route_sequences_by_type[type_id].add(stop_sequence)

    inactive_routes_by_type: dict[str, dict[tuple[str, tuple[int, ...]], RouteColumn]] = {
        type_id: {} for type_id in vehicle_types
    }
    active_route_keys = {route_column_key(route) for route in initial_active_routes}
    for route in compatible_routes:
        key = route_column_key(route)
        if key not in active_route_keys:
            inactive_routes_by_type[route.type_id][key] = route

    effective_pricing_workers = (
        min(pricing_workers, len(vehicle_types))
        if pricing_workers is not None
        else min(os.cpu_count() or 1, len(vehicle_types))
    )
    pricing_executor: ProcessPoolExecutor | None = None
    if effective_pricing_workers > 1:
        pricing_executor = ProcessPoolExecutor(
            max_workers=effective_pricing_workers,
            mp_context=mp.get_context("spawn"),
            initializer=init_pricing_worker,
            initargs=(instance, branch_ctx),
        )
        if progress is not None:
            progress.emit(
                f"{node_label}: pricing vehicle types in parallel with "
                f"{effective_pricing_workers} worker processes",
                force=True,
            )

    def collect_candidate_routes(
        cover_duals: dict[int, float],
        type_duals: dict[str, float],
        *,
        raw_pricing: bool,
        iteration_number: int,
    ) -> dict[str, list[RouteColumn]]:
        if pricing_executor is None:
            routes_by_type: dict[str, list[RouteColumn]] = {}
            for vehicle_type in vehicle_types.values():
                if progress is not None:
                    prefix = "raw pricing" if raw_pricing else "pricing"
                    progress.emit(
                        f"{node_label}: CG iteration {iteration_number}, {prefix} "
                        f"{vehicle_type.type_id} at LP bound {last_lp_objective:.2f}"
                    )
                routes_by_type[vehicle_type.type_id] = solve_pricing_subproblem(
                    instance=instance,
                    vehicle_type=vehicle_type,
                    cover_duals=cover_duals,
                    type_dual=type_duals[vehicle_type.type_id],
                    branch_ctx=branch_ctx,
                    max_routes=pricing_columns_per_type,
                    excluded_stop_sequences=route_sequences_by_type[vehicle_type.type_id],
                    pricing_time_limit=pricing_time_limit,
                    deadline=deadline,
                    progress=progress,
                    progress_label=(
                        f"{node_label}: {'raw pricing' if raw_pricing else 'pricing'} "
                        f"{vehicle_type.type_id} in CG iteration {iteration_number}"
                    ),
                )
            return routes_by_type

        if progress is not None:
            progress.emit(
                f"{node_label}: CG iteration {iteration_number}, running "
                f"{'raw ' if raw_pricing else ''}pricing for {len(vehicle_types)} vehicle types "
                f"in parallel"
            )

        future_by_type_id = {
            vehicle_type.type_id: pricing_executor.submit(
                run_pricing_worker,
                vehicle_type,
                cover_duals,
                type_duals[vehicle_type.type_id],
                pricing_columns_per_type,
                set(route_sequences_by_type[vehicle_type.type_id]),
                pricing_time_limit,
                deadline,
            )
            for vehicle_type in vehicle_types.values()
        }
        routes_by_type: dict[str, list[RouteColumn]] = {}
        for vehicle_type in vehicle_types.values():
            type_id, routes = future_by_type_id[vehicle_type.type_id].result()
            routes_by_type[type_id] = routes
        return routes_by_type

    stabilized_cover_duals: dict[int, float] | None = None
    stabilized_type_duals: dict[str, float] | None = None
    previous_raw_cover_duals: dict[int, float] | None = None
    previous_raw_type_duals: dict[str, float] | None = None
    try:
        iteration = 0
        while max_cg_iterations is None or iteration < max_cg_iterations:
            iteration += 1
            if progress is not None:
                progress.emit(
                    f"{node_label}: CG iteration {iteration}, solving master LP over "
                    f"{len(node_master.routes)} columns"
                )
            master_lp = node_master.model
            master_time_limit = remaining_time(deadline)
            if master_time_limit is not None:
                if master_time_limit <= 0:
                    raise TimeoutError("Overall time limit reached during node LP.")
                master_lp.Params.TimeLimit = master_time_limit
            optimize_with_heartbeat(
                master_lp,
                progress=progress,
                label=f"{node_label}: master LP iteration {iteration}",
            )
            if master_lp.Status == GRB.TIME_LIMIT:
                raise TimeoutError("Overall time limit reached during node LP.")
            if master_lp.Status != GRB.OPTIMAL:
                raise RuntimeError(
                    f"Column-generation node LP did not solve to optimality (status {master_lp.Status})."
                )

            last_lp_objective = master_lp.ObjVal
            raw_cover_duals = {
                stop_id: node_master.cover_constraints[stop_id].Pi
                for stop_id in sorted(instance.stops)
            }
            raw_type_duals = {
                type_id: node_master.type_constraints[type_id].Pi for type_id in vehicle_types
            }
            current_alpha = adaptive_stabilization_alpha(
                base_alpha=dual_stabilization_alpha,
                raw_cover_duals=raw_cover_duals,
                previous_raw_cover_duals=previous_raw_cover_duals,
                raw_type_duals=raw_type_duals,
                previous_raw_type_duals=previous_raw_type_duals,
                mode=dual_stabilization_mode,
            )
            stabilized_cover_duals = stabilize_dual_vector(
                raw_duals=raw_cover_duals,
                previous_duals=stabilized_cover_duals,
                alpha=current_alpha,
            )
            stabilized_type_duals = stabilize_dual_vector(
                raw_duals=raw_type_duals,
                previous_duals=stabilized_type_duals,
                alpha=current_alpha,
            )
            previous_raw_cover_duals = raw_cover_duals
            previous_raw_type_duals = raw_type_duals

            new_routes: list[RouteColumn] = []
            reactivated_routes = select_inactive_reactivation_routes(
                inactive_routes_by_type=inactive_routes_by_type,
                cover_duals=raw_cover_duals,
                type_duals=raw_type_duals,
                max_per_type=pricing_columns_per_type,
            )
            for route in reactivated_routes:
                key = route_column_key(route)
                if key in active_route_keys:
                    continue
                inactive_routes_by_type[route.type_id].pop(key, None)
                active_route_keys.add(key)
                new_routes.append(route)
                add_route_column_to_master(node_master, route)
            if reactivated_routes and progress is not None:
                progress.emit(
                    f"{node_label}: CG iteration {iteration} reactivated {len(reactivated_routes)} "
                    "stored columns from global memory"
                )
            if reactivated_routes:
                if progress is not None:
                    progress.emit(
                        f"{node_label}: CG iteration {iteration} added {len(new_routes)} routes; "
                        f"global column pool is now {len(route_pool)}"
                    )
                node_master.model.update()
                continue

            found_raw_improving_route = False
            pricing_results = collect_candidate_routes(
                cover_duals=stabilized_cover_duals,
                type_duals=stabilized_type_duals,
                raw_pricing=False,
                iteration_number=iteration,
            )
            for vehicle_type in vehicle_types.values():
                candidate_routes = pricing_results[vehicle_type.type_id]
                for route in candidate_routes:
                    if (
                        route_reduced_cost(
                            route,
                            cover_duals=raw_cover_duals,
                            type_dual=raw_type_duals[vehicle_type.type_id],
                        )
                        >= -REDUCED_COST_TOL
                    ):
                        continue
                    key = (route.type_id, route.stop_sequence)
                    if key in route_key_set:
                        continue
                    route_key_set.add(key)
                    route_sequences_by_type[route.type_id].add(route.stop_sequence)
                    active_route_keys.add(key)
                    route_pool.append(route)
                    new_routes.append(route)
                    add_route_column_to_master(node_master, route)
                    found_raw_improving_route = True

            if not found_raw_improving_route and current_alpha < 1.0 - 1e-12:
                if progress is not None:
                    progress.emit(
                        f"{node_label}: CG iteration {iteration}, no raw-improving routes found "
                        "under stabilized pricing; certifying with raw dual pricing"
                    )
                raw_pricing_results = collect_candidate_routes(
                    cover_duals=raw_cover_duals,
                    type_duals=raw_type_duals,
                    raw_pricing=True,
                    iteration_number=iteration,
                )
                for vehicle_type in vehicle_types.values():
                    for route in raw_pricing_results[vehicle_type.type_id]:
                        key = (route.type_id, route.stop_sequence)
                        if key in route_key_set:
                            continue
                        route_key_set.add(key)
                        route_sequences_by_type[route.type_id].add(route.stop_sequence)
                        active_route_keys.add(key)
                        route_pool.append(route)
                        new_routes.append(route)
                        add_route_column_to_master(node_master, route)

            if not new_routes:
                artificial_usage = sum(
                    node_master.artificial_vars[stop_id].X
                    for stop_id in sorted(instance.stops)
                )
                if artificial_usage > INTEGRALITY_TOL:
                    return None
                route_values = [route_var.X for route_var in node_master.route_vars]
                return NodeRelaxationResult(
                    lower_bound=last_lp_objective,
                    routes=list(node_master.routes),
                    route_values=route_values,
                    cg_iterations=iteration,
                )

            if progress is not None:
                progress.emit(
                    f"{node_label}: CG iteration {iteration} added {len(new_routes)} routes; "
                    f"global column pool is now {len(route_pool)}"
                )
            node_master.model.update()

        raise RuntimeError(
            "Maximum column-generation iterations reached before node convergence. "
            "Increase --max-cg-iterations for an exact solve."
        )
    finally:
        if pricing_executor is not None:
            pricing_executor.shutdown()


def solve_restricted_integer_master(
    instance: Instance,
    vehicle_types: dict[str, VehicleTypeData],
    routes: list[RouteColumn],
    deadline: float | None,
    heuristic_time_limit: float | None,
    progress: ProgressTracker | None = None,
    progress_label: str = "restricted master heuristic",
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

    optimize_with_heartbeat(model, progress=progress, label=progress_label)
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


def compute_pair_together_values(
    routes: list[RouteColumn],
    route_values: list[float],
) -> dict[tuple[int, int], float]:
    pair_values: dict[tuple[int, int], float] = defaultdict(float)
    for route, value in zip(routes, route_values):
        if value <= INTEGRALITY_TOL or len(route.stop_sequence) < 2:
            continue
        ordered_stops = sorted(route.stop_set)
        for idx, stop_a in enumerate(ordered_stops):
            for stop_b in ordered_stops[idx + 1 :]:
                pair_values[stop_a, stop_b] += value
    return pair_values


def route_conflicts_with_forced_arc(route: RouteColumn, arc: tuple[int, int]) -> bool:
    i, j = arc
    if arc in route.arc_set:
        return False
    if i == DEPOT_NODE:
        return j in route.stop_set
    if j == DEPOT_NODE:
        return i in route.stop_set
    return i in route.stop_set or j in route.stop_set


def route_conflicts_with_forced_assignment(
    route: RouteColumn,
    stop_id: int,
    type_id: str,
) -> bool:
    return stop_id in route.stop_set and route.type_id != type_id


def route_conflicts_with_excluded_assignment(
    route: RouteColumn,
    stop_id: int,
    type_id: str,
) -> bool:
    return stop_id in route.stop_set and route.type_id == type_id


def select_ryan_foster_pair(
    instance: Instance,
    routes: list[RouteColumn],
    route_values: list[float],
    branch_ctx: BranchContext,
) -> tuple[int, int] | None:
    pair_values = compute_pair_together_values(routes, route_values)
    fixed_pairs = set(branch_ctx.together_stop_pairs) | set(branch_ctx.separated_stop_pairs)
    best_pair: tuple[int, int] | None = None
    best_score: tuple[float, float, int, int, float] | None = None

    stop_ids = sorted(instance.stops)
    for idx, stop_a in enumerate(stop_ids):
        for stop_b in stop_ids[idx + 1 :]:
            pair = (stop_a, stop_b)
            if pair in fixed_pairs:
                continue
            together_value = pair_values.get(pair, 0.0)
            if not (INTEGRALITY_TOL < together_value < 1.0 - INTEGRALITY_TOL):
                continue

            together_mass = 0.0
            separate_mass = 0.0
            together_columns = 0
            separate_columns = 0
            for route, route_value in zip(routes, route_values):
                if route_value <= INTEGRALITY_TOL:
                    continue
                has_a = stop_a in route.stop_set
                has_b = stop_b in route.stop_set
                if has_a and has_b:
                    together_mass += route_value
                    together_columns += 1
                elif has_a or has_b:
                    separate_mass += route_value
                    separate_columns += 1

            score = (
                min(together_mass, separate_mass),
                together_mass + separate_mass,
                min(together_columns, separate_columns),
                together_columns + separate_columns,
                -abs(together_value - 0.5),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_pair = pair

    return best_pair


def select_fractional_arc(
    routes: list[RouteColumn],
    route_values: list[float],
    branch_ctx: BranchContext,
) -> tuple[int, int] | None:
    fixed_arcs = set(branch_ctx.forced_arcs) | set(branch_ctx.forbidden_arcs)
    arc_values = compute_arc_values(routes, route_values)
    fractional_arcs = {
        arc: value
        for arc, value in arc_values.items()
        if arc not in fixed_arcs and INTEGRALITY_TOL < value < 1.0 - INTEGRALITY_TOL
    }
    fractional_outgoing_count: dict[int, int] = defaultdict(int)
    fractional_incoming_count: dict[int, int] = defaultdict(int)
    for i, j in fractional_arcs:
        fractional_outgoing_count[i] += 1
        fractional_incoming_count[j] += 1

    best_arc: tuple[int, int] | None = None
    best_score: tuple[float, float, int, int, int, float] | None = None

    for arc, value in fractional_arcs.items():
        forbid_mass = 0.0
        force_mass = 0.0
        forbid_columns = 0
        force_columns = 0
        for route, route_value in zip(routes, route_values):
            if route_value <= INTEGRALITY_TOL:
                continue
            if arc in route.arc_set:
                forbid_mass += route_value
                forbid_columns += 1
            if route_conflicts_with_forced_arc(route, arc):
                force_mass += route_value
                force_columns += 1

        i, j = arc
        ambiguity = fractional_outgoing_count[i] + fractional_incoming_count[j]
        score = (
            min(forbid_mass, force_mass),
            forbid_mass + force_mass,
            min(forbid_columns, force_columns),
            ambiguity,
            0 if DEPOT_NODE in arc else 1,
            -abs(value - 0.5),
        )
        if best_score is None or score > best_score:
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
    best_score: tuple[float, float, int, int, float] | None = None

    for stop_id in sorted(instance.stops):
        if stop_id in branch_ctx.forced_type_for_stop:
            continue
        for type_id in sorted(vehicle_types):
            if type_id in branch_ctx.excluded_types_by_stop.get(stop_id, set()):
                continue
            value = assignment_values.get((stop_id, type_id), 0.0)
            if not (INTEGRALITY_TOL < value < 1.0 - INTEGRALITY_TOL):
                continue
            exclude_mass = 0.0
            force_mass = 0.0
            exclude_columns = 0
            force_columns = 0
            for route, route_value in zip(routes, route_values):
                if route_value <= INTEGRALITY_TOL:
                    continue
                if route_conflicts_with_excluded_assignment(route, stop_id, type_id):
                    exclude_mass += route_value
                    exclude_columns += 1
                if route_conflicts_with_forced_assignment(route, stop_id, type_id):
                    force_mass += route_value
                    force_columns += 1
            score = (
                min(exclude_mass, force_mass),
                exclude_mass + force_mass,
                min(exclude_columns, force_columns),
                exclude_columns + force_columns,
                -abs(value - 0.5),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = (stop_id, type_id)

    return best_assignment


def create_model(
    workbook_path: Path,
    max_cg_iterations: int | None = 200,
    pricing_time_limit: float | None = None,
    pricing_columns_per_type: int = 3,
    pricing_workers: int | None = None,
    dual_stabilization_alpha: float = 0.5,
    dual_stabilization_mode: str = "adaptive",
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
        same_route_pairs=frozenset(),
        different_route_pairs=frozenset(),
    )
    root_result = solve_node_relaxation(
        instance=instance,
        vehicle_types=vehicle_types,
        route_pool=route_pool,
        route_key_set=route_key_set,
        branch_state=root_state,
        max_cg_iterations=max_cg_iterations,
        pricing_time_limit=pricing_time_limit,
        pricing_columns_per_type=pricing_columns_per_type,
        pricing_workers=pricing_workers,
        dual_stabilization_alpha=dual_stabilization_alpha,
        dual_stabilization_mode=dual_stabilization_mode,
        deadline=None,
        progress=None,
        node_label="root",
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
            "pricing_columns_per_type": pricing_columns_per_type,
            "pricing_workers": pricing_workers,
            "dual_stabilization_alpha": dual_stabilization_alpha,
            "dual_stabilization_mode": dual_stabilization_mode,
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
    max_cg_iterations: int | None,
    pricing_time_limit: float | None,
    pricing_columns_per_type: int,
    pricing_workers: int | None,
    dual_stabilization_alpha: float,
    dual_stabilization_mode: str,
    max_bp_nodes: int | None,
    heuristic_time_limit: float | None,
    progress: ProgressTracker | None = None,
    trace_logger: ObjectiveTraceLogger | None = None,
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
        same_route_pairs=frozenset(),
        different_route_pairs=frozenset(),
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

    def current_global_bound(active_node_hint: float | None = None) -> float | None:
        if root_lp_objective is None and nodes_processed == 0:
            return None
        candidate_bounds: list[float] = []
        if active_node_hint is not None:
            candidate_bounds.append(active_node_hint)
        if active_nodes:
            candidate_bounds.append(min(bound for bound, *_ in active_nodes))
        if candidate_bounds:
            return min(candidate_bounds)
        if incumbent_obj is not None:
            return incumbent_obj
        return root_lp_objective

    last_logged_incumbent: float | None = None
    last_logged_bound: float | None = None

    def maybe_record_trace(
        event: str,
        *,
        active_node_hint: float | None = None,
        force: bool = False,
        note: str = "",
        node_depth: int | None = None,
    ) -> None:
        nonlocal last_logged_bound, last_logged_incumbent
        if trace_logger is None:
            return
        best_bound = current_global_bound(active_node_hint)
        incumbent_changed = optional_float_changed(last_logged_incumbent, incumbent_obj)
        bound_changed = optional_float_changed(last_logged_bound, best_bound)
        if not force and not incumbent_changed and not bound_changed:
            return
        gap = relative_gap(incumbent_obj, best_bound)
        trace_logger.record(
            event=event,
            status="RUNNING" if status == "UNKNOWN" else status,
            incumbent_objective=incumbent_obj,
            best_bound=best_bound,
            relative_gap=gap,
            nodes_processed=nodes_processed,
            nodes_pruned=nodes_pruned,
            global_columns=len(route_pool),
            queue_size=len(active_nodes),
            node_depth=node_depth,
            note=note,
        )
        last_logged_incumbent = incumbent_obj
        last_logged_bound = best_bound

    if progress is not None:
        progress.emit(
            f"Starting branch-and-price with {len(route_pool)} initial columns",
            force=True,
        )
    maybe_record_trace(
        "run_start",
        force=True,
        note=f"Initial route pool has {len(route_pool)} columns",
    )

    try:
        initial_heuristic = solve_restricted_integer_master(
            instance=instance,
            vehicle_types=vehicle_types,
            routes=route_pool,
            deadline=deadline,
            heuristic_time_limit=heuristic_time_limit,
            progress=progress,
            progress_label="initial restricted-master heuristic",
        )
    except TimeoutError:
        initial_heuristic = None
    if initial_heuristic is not None:
        incumbent_obj, incumbent_routes = initial_heuristic
        if progress is not None:
            progress.emit(
                f"Initial restricted-master incumbent: {incumbent_obj:.2f}",
                force=True,
            )
        maybe_record_trace(
            "incumbent_update",
            force=True,
            note="Initial restricted-master incumbent",
        )

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
            maybe_record_trace(
                "bound_update",
                note="Pruned queued node by incumbent cutoff",
            )
            continue

        if progress is not None:
            incumbent_text = "none" if incumbent_obj is None else f"{incumbent_obj:.2f}"
            progress.emit(
                f"Exploring node depth {depth}, queue {len(active_nodes)}, "
                f"best hint {lower_bound_hint:.2f}, incumbent {incumbent_text}"
            )
        try:
            node_result = solve_node_relaxation(
                instance=instance,
                vehicle_types=vehicle_types,
                route_pool=route_pool,
                route_key_set=route_key_set,
                branch_state=branch_state,
                max_cg_iterations=max_cg_iterations,
                pricing_time_limit=pricing_time_limit,
                pricing_columns_per_type=pricing_columns_per_type,
                pricing_workers=pricing_workers,
                dual_stabilization_alpha=dual_stabilization_alpha,
                dual_stabilization_mode=dual_stabilization_mode,
                deadline=deadline,
                progress=progress,
                node_label=f"node depth {depth}",
            )
        except TimeoutError:
            heapq.heappush(
                active_nodes,
                (lower_bound_hint, depth, next(sequence_counter), branch_state),
            )
            if progress is not None:
                progress.emit("Time limit reached while processing the current node", force=True)
            status = "TIME_LIMIT"
            break

        nodes_processed += 1
        if node_result is None:
            maybe_record_trace(
                "bound_update",
                note="Pruned infeasible node",
                node_depth=depth,
            )
            continue
        if root_lp_objective is None:
            root_lp_objective = node_result.lower_bound
            if progress is not None:
                progress.emit(
                    f"Root relaxation closed at {root_lp_objective:.2f}",
                    force=True,
                )
            maybe_record_trace(
                "bound_update",
                active_node_hint=node_result.lower_bound,
                force=True,
                note="Root relaxation closed",
                node_depth=depth,
            )

        node_lb = node_result.lower_bound
        if incumbent_obj is not None and node_lb >= incumbent_obj - INTEGRALITY_TOL:
            nodes_pruned += 1
            maybe_record_trace(
                "bound_update",
                note="Pruned processed node by incumbent cutoff",
                node_depth=depth,
            )
            continue

        heuristic_obj: float | None = None
        if nodes_processed == 1 or incumbent_obj is None:
            try:
                heuristic_solution = solve_restricted_integer_master(
                    instance=instance,
                    vehicle_types=vehicle_types,
                    routes=node_result.routes,
                    deadline=deadline,
                    heuristic_time_limit=heuristic_time_limit,
                    progress=progress,
                    progress_label=f"restricted-master heuristic at node depth {depth}",
                )
            except TimeoutError:
                heapq.heappush(
                    active_nodes,
                    (node_lb, depth, next(sequence_counter), branch_state),
                )
                if progress is not None:
                    progress.emit(
                        "Time limit reached while solving the restricted-master heuristic",
                        force=True,
                    )
                status = "TIME_LIMIT"
                break
            if heuristic_solution is not None:
                heuristic_obj, heuristic_routes = heuristic_solution
                if incumbent_obj is None or heuristic_obj < incumbent_obj - INTEGRALITY_TOL:
                    incumbent_obj = heuristic_obj
                    incumbent_routes = heuristic_routes
                    if progress is not None:
                        progress.emit(
                            f"New incumbent from restricted master: {incumbent_obj:.2f}",
                            force=True,
                        )
                    maybe_record_trace(
                        "incumbent_update",
                        active_node_hint=node_lb,
                        force=True,
                        note="Improved incumbent from restricted master",
                        node_depth=depth,
                    )

        if is_integral_solution(node_result.route_values):
            selected_routes = [
                route
                for route, value in zip(node_result.routes, node_result.route_values)
                if value > 0.5
            ]
            if incumbent_obj is None or node_lb < incumbent_obj - INTEGRALITY_TOL:
                incumbent_obj = node_lb
                incumbent_routes = selected_routes
                if progress is not None:
                    progress.emit(
                        f"Found integral node solution: {incumbent_obj:.2f}",
                        force=True,
                    )
                maybe_record_trace(
                    "incumbent_update",
                    force=True,
                    note="Found integral node solution",
                    node_depth=depth,
                )
            continue

        if heuristic_obj is not None and abs(heuristic_obj - node_lb) <= 1e-5:
            continue

        branch_ctx = build_branch_context(branch_state, instance, vehicle_types)
        if branch_ctx is None:
            continue

        child_states: list[BranchState] = []
        branch_pair = select_ryan_foster_pair(
            instance=instance,
            routes=node_result.routes,
            route_values=node_result.route_values,
            branch_ctx=branch_ctx,
        )
        if branch_pair is not None:
            if progress is not None:
                progress.emit(
                    f"Branching on Ryan-Foster pair ({branch_pair[0]}, {branch_pair[1]}) "
                    f"at node depth {depth}"
                )
            child_states = [
                BranchState(
                    forbidden_arcs=branch_state.forbidden_arcs,
                    forced_arcs=branch_state.forced_arcs,
                    excluded_assignments=branch_state.excluded_assignments,
                    forced_assignments=branch_state.forced_assignments,
                    same_route_pairs=branch_state.same_route_pairs,
                    different_route_pairs=frozenset(
                        set(branch_state.different_route_pairs) | {branch_pair}
                    ),
                ),
                BranchState(
                    forbidden_arcs=branch_state.forbidden_arcs,
                    forced_arcs=branch_state.forced_arcs,
                    excluded_assignments=branch_state.excluded_assignments,
                    forced_assignments=branch_state.forced_assignments,
                    same_route_pairs=frozenset(set(branch_state.same_route_pairs) | {branch_pair}),
                    different_route_pairs=branch_state.different_route_pairs,
                ),
            ]
        else:
            branch_arc = select_fractional_arc(
            routes=node_result.routes,
            route_values=node_result.route_values,
            branch_ctx=branch_ctx,
            )
            if branch_arc is not None:
                if progress is not None:
                    progress.emit(
                        f"Branching on arc {branch_arc[0]} -> {branch_arc[1]} at node depth {depth}"
                    )
                child_states = [
                    BranchState(
                        forbidden_arcs=frozenset(set(branch_state.forbidden_arcs) | {branch_arc}),
                        forced_arcs=branch_state.forced_arcs,
                        excluded_assignments=branch_state.excluded_assignments,
                        forced_assignments=branch_state.forced_assignments,
                        same_route_pairs=branch_state.same_route_pairs,
                        different_route_pairs=branch_state.different_route_pairs,
                    ),
                    BranchState(
                        forbidden_arcs=branch_state.forbidden_arcs,
                        forced_arcs=frozenset(set(branch_state.forced_arcs) | {branch_arc}),
                        excluded_assignments=branch_state.excluded_assignments,
                        forced_assignments=branch_state.forced_assignments,
                        same_route_pairs=branch_state.same_route_pairs,
                        different_route_pairs=branch_state.different_route_pairs,
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
                    if progress is not None:
                        progress.emit(
                            f"Branching on stop {stop_id} type assignment {type_id} at node depth {depth}"
                        )
                    child_states = [
                        BranchState(
                            forbidden_arcs=branch_state.forbidden_arcs,
                            forced_arcs=branch_state.forced_arcs,
                            excluded_assignments=frozenset(
                                set(branch_state.excluded_assignments) | {(stop_id, type_id)}
                            ),
                            forced_assignments=branch_state.forced_assignments,
                            same_route_pairs=branch_state.same_route_pairs,
                            different_route_pairs=branch_state.different_route_pairs,
                        ),
                        BranchState(
                            forbidden_arcs=branch_state.forbidden_arcs,
                            forced_arcs=branch_state.forced_arcs,
                            excluded_assignments=branch_state.excluded_assignments,
                            forced_assignments=frozenset(
                                set(branch_state.forced_assignments) | {(stop_id, type_id)}
                            ),
                            same_route_pairs=branch_state.same_route_pairs,
                            different_route_pairs=branch_state.different_route_pairs,
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

        maybe_record_trace(
            "bound_update",
            note="Updated global bound after branching",
            node_depth=depth,
        )

        best_bound = min(bound for bound, *_ in active_nodes) if active_nodes else node_lb
        if incumbent_obj is not None and mip_gap is not None:
            gap = relative_gap(incumbent_obj, best_bound)
            if gap is not None and gap <= mip_gap + 1e-12:
                if progress is not None:
                    progress.emit(
                        f"Gap target reached with incumbent {incumbent_obj:.2f} and "
                        f"bound {best_bound:.2f}",
                        force=True,
                    )
                status = "GAP_LIMIT"
                maybe_record_trace(
                    "gap_limit_reached",
                    force=True,
                    note="Reached requested relative gap target",
                    node_depth=depth,
                )
                break

    if status == "UNKNOWN":
        if active_nodes:
            status = "TIME_LIMIT" if deadline is not None and remaining_time(deadline) <= 0 else "NODE_LIMIT"
        elif incumbent_obj is None:
            status = "INFEASIBLE"
        else:
            status = "OPTIMAL"

    if root_lp_objective is None and nodes_processed == 0:
        best_bound = None
    elif active_nodes:
        best_bound = min(bound for bound, *_ in active_nodes)
    elif incumbent_obj is not None:
        best_bound = incumbent_obj
    else:
        best_bound = root_lp_objective

    if trace_logger is not None:
        trace_logger.record(
            event="solve_end",
            status=status,
            incumbent_objective=incumbent_obj,
            best_bound=best_bound,
            relative_gap=relative_gap(incumbent_obj, best_bound),
            nodes_processed=nodes_processed,
            nodes_pruned=nodes_pruned,
            global_columns=len(route_pool),
            queue_size=len(active_nodes),
            note="Branch-and-price finished",
        )

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
    max_cg_iterations: int | None,
    pricing_time_limit: float | None,
    pricing_columns_per_type: int,
    pricing_workers: int | None,
    dual_stabilization_alpha: float,
    dual_stabilization_mode: str,
    max_bp_nodes: int | None,
    heuristic_time_limit: float | None = HEURISTIC_MASTER_TIME_LIMIT,
    progress: ProgressTracker | None = None,
    trace_logger: ObjectiveTraceLogger | None = None,
) -> BranchAndPriceResult:
    instance = load_instance(workbook_path)
    result = branch_and_price(
        instance=instance,
        time_limit=time_limit,
        mip_gap=mip_gap,
        max_cg_iterations=max_cg_iterations,
        pricing_time_limit=pricing_time_limit,
        pricing_columns_per_type=pricing_columns_per_type,
        pricing_workers=pricing_workers,
        dual_stabilization_alpha=dual_stabilization_alpha,
        dual_stabilization_mode=dual_stabilization_mode,
        max_bp_nodes=max_bp_nodes,
        heuristic_time_limit=heuristic_time_limit,
        progress=progress,
        trace_logger=trace_logger,
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
        return result

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

    return result


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
        type=parse_optional_int_limit,
        default=200,
        help=(
            "Maximum column-generation iterations allowed at each tree node. "
            "Use 0, 'none', or 'unlimited' for no cap."
        ),
    )
    parser.add_argument(
        "--pricing-time-limit",
        type=float,
        default=None,
        help=(
            "Optional per-pricing time limit in seconds. "
            "If a pricing solve hits this limit, the exact solve aborts."
        ),
    )
    parser.add_argument(
        "--pricing-columns-per-type",
        type=parse_positive_int,
        default=3,
        help=(
            "Maximum number of distinct improving routes to add per vehicle type in each "
            "column-generation iteration."
        ),
    )
    parser.add_argument(
        "--pricing-workers",
        type=parse_worker_count,
        default=None,
        help=(
            "Number of worker processes for pricing across vehicle types. "
            "Default: auto-select up to the number of vehicle types. Use 1 to disable "
            "parallel pricing. You can also pass 'auto'."
        ),
    )
    parser.add_argument(
        "--dual-stabilization-alpha",
        type=parse_closed_unit_interval,
        default=0.5,
        help=(
            "Base exponential smoothing weight for pricing duals. "
            "In adaptive mode this is the center value; use 1.0 for raw duals."
        ),
    )
    parser.add_argument(
        "--dual-stabilization-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help=(
            "Use a fixed stabilization alpha every iteration, or adapt it based on "
            "how much the raw duals move between CG iterations."
        ),
    )
    parser.add_argument(
        "--max-bp-nodes",
        type=int,
        default=None,
        help="Optional cap on processed branch-and-price nodes.",
    )
    parser.add_argument(
        "--unlimited",
        action="store_true",
        help=(
            "Run with no user-facing time, gap, node, pricing, or column-generation limits. "
            "This also removes the restricted-master heuristic time cap."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show throttled progress updates during branch-and-price.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Minimum seconds between non-forced progress updates.",
    )
    parser.add_argument(
        "--objective-trace",
        type=Path,
        default=None,
        help=(
            "CSV path for incumbent and bound snapshots over time. "
            "Default: auto-generate a timestamped file beside the workbook."
        ),
    )
    parser.add_argument(
        "--no-objective-trace",
        action="store_true",
        help="Disable writing the objective trace CSV.",
    )
    return parser.parse_args()


def main() -> None:
    global _INTERRUPT_SIGNAL_NAME
    _INTERRUPT_SIGNAL_NAME = None
    args = parse_args()
    heuristic_time_limit = HEURISTIC_MASTER_TIME_LIMIT
    if args.unlimited:
        args.time_limit = None
        args.mip_gap = None
        args.max_cg_iterations = None
        args.pricing_time_limit = None
        args.max_bp_nodes = None
        heuristic_time_limit = None
    progress = ProgressTracker(
        enabled=args.progress,
        interval_seconds=args.progress_interval,
    )
    objective_trace_path = (
        None
        if args.no_objective_trace
        else (args.objective_trace or default_objective_trace_path(args.workbook))
    )
    trace_logger = ObjectiveTraceLogger(objective_trace_path, args.workbook)
    previous_handlers = install_interrupt_handlers()

    try:
        if trace_logger.path is not None:
            print(f"Objective trace: {trace_logger.path}")
            trace_logger.record(
                event="run_start",
                status="RUNNING",
                note=f"Workbook: {args.workbook}",
            )
        solve_model(
            workbook_path=args.workbook,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            max_cg_iterations=args.max_cg_iterations,
            pricing_time_limit=args.pricing_time_limit,
            pricing_columns_per_type=args.pricing_columns_per_type,
            pricing_workers=args.pricing_workers,
            dual_stabilization_alpha=args.dual_stabilization_alpha,
            dual_stabilization_mode=args.dual_stabilization_mode,
            max_bp_nodes=args.max_bp_nodes,
            heuristic_time_limit=heuristic_time_limit,
            progress=progress,
            trace_logger=trace_logger,
        )
    except KeyboardInterrupt:
        interrupt_note = "Interrupted by user"
        if _INTERRUPT_SIGNAL_NAME is not None:
            interrupt_note = f"Interrupted by {_INTERRUPT_SIGNAL_NAME}"
        if trace_logger.path is not None:
            trace_logger.record(
                event="interrupted",
                status="INTERRUPTED",
                note=interrupt_note,
            )
            print(f"\n{interrupt_note}. Partial objective trace saved to {trace_logger.path}")
        raise SystemExit(130) from None
    finally:
        restore_interrupt_handlers(previous_handlers)
        trace_logger.close()


if __name__ == "__main__":
    main()
