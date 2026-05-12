"""
Ultra-strict test script for COMP1037 campus_delivery.py

Put this file in the same folder as:
    campus_delivery.py
    campus_map.json

Run:
    PYTHONDONTWRITEBYTECODE=1 python test_campus_delivery_ultra.py

This script is read-only for your coursework files.
Do NOT submit this test file. Submit only campus_delivery.py.
"""

import sys
sys.dont_write_bytecode = True

import os
import ast
import io
import math
import json
import heapq
import copy
import inspect
import traceback
import importlib.util
from contextlib import redirect_stdout, redirect_stderr
from itertools import permutations, combinations


MODULE_FILE = "campus_delivery.py"
MAP_FILE = "campus_map.json"
EPS = 1e-6

FULL_ALL_PAIRS_TEST = True
FULL_MULTI_STOP_SAMPLE = True

VERBOSE_PASS = False

failures = []
passes = 0


# ============================================================
# Basic framework
# ============================================================

def pass_msg(message):
    global passes
    passes += 1
    if VERBOSE_PASS:
        print(f"[PASS] {message}")


def fail_msg(message):
    failures.append(message)
    print(f"[FAIL] {message}")


def check(condition, message):
    if condition:
        pass_msg(message)
    else:
        fail_msg(message)


def check_close(actual, expected, message, eps=EPS):
    if math.isinf(actual) and math.isinf(expected):
        pass_msg(message)
        return

    try:
        ok = abs(actual - expected) <= eps
    except Exception:
        ok = False

    if ok:
        pass_msg(message)
    else:
        fail_msg(f"{message}: expected {expected}, got {actual}")


def load_json_map():
    with open(MAP_FILE, "r") as f:
        data = json.load(f)
    return data["campus_graph"], data["campus_coords"], data["campus_bridges"]


def load_student_module_safely():
    module_path = os.path.abspath(MODULE_FILE)
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Cannot find {MODULE_FILE}")

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    spec = importlib.util.spec_from_file_location("campus_delivery_under_test", module_path)
    module = importlib.util.module_from_spec(spec)

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        spec.loader.exec_module(module)

    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()

    check(stdout_text == "", "importing campus_delivery.py produces no stdout")
    check(stderr_text == "", "importing campus_delivery.py produces no stderr")

    return module


# ============================================================
# Static checks
# ============================================================

def test_static_imports_and_structure():
    print("\n" + "=" * 80)
    print("0. Static checks: imports, main guard, forbidden obvious writes")
    print("=" * 80)

    with open(MODULE_FILE, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    allowed_modules = {"json", "math", "itertools"}

    imported_modules = set()
    suspicious_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])

        if isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

        if isinstance(node, ast.Call):
            func_name = None

            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in {"remove", "unlink", "rename", "replace", "rmtree", "copyfile", "move"}:
                suspicious_calls.append(func_name)

            if func_name == "open":
                # Flag open(..., "w"), open(..., "a"), open(..., "x") outside harmless templates.
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                    if isinstance(mode, str) and any(ch in mode for ch in ["w", "a", "x", "+"]):
                        suspicious_calls.append(f"open mode {mode!r}")

    extra_imports = imported_modules - allowed_modules
    check(extra_imports == set(), f"only allowed imports used: found {sorted(imported_modules)}")
    check(suspicious_calls == [], f"no obvious file-modifying calls: found {suspicious_calls}")

    check("if __name__ == '__main__':" in source or 'if __name__ == "__main__":' in source,
          "testing block is protected by __main__ guard")


def test_required_functions(cd):
    print("\n" + "=" * 80)
    print("1. Function existence, signatures, defaults")
    print("=" * 80)

    expected = {
        "make_node": ["state", "parent", "action", "cost", "depth"],
        "reconstruct_path": ["node"],
        "expand": ["problem", "node", "get_actions_fn", "action_cost_fn"],
        "load_campus_map": ["filename"],
        "validate_path": ["problem", "path"],
        "is_cycle": ["node", "k"],

        "make_campus_problem": ["start", "goal", "campus_graph", "campus_coords", "campus_bridges", "time_of_day"],
        "is_goal": ["problem", "state"],
        "campus_get_actions": ["problem", "state"],
        "campus_action_cost": ["problem", "state", "action"],
        "heuristic_euclidean": ["state", "problem"],
        "heuristic_scaled": ["state", "problem", "scale"],
        "find_best_scale": ["problem"],
        "a_star_search": ["problem", "heuristic_fn"],
        "multi_stop_delivery": ["problem", "locations_to_visit", "heuristic_fn"],
    }

    for fn_name, params in expected.items():
        check(hasattr(cd, fn_name), f"{fn_name} exists")
        if hasattr(cd, fn_name):
            sig = inspect.signature(getattr(cd, fn_name))
            actual = list(sig.parameters.keys())
            check(actual == params, f"{fn_name} parameters are exactly {params}")

    check(inspect.signature(cd.make_campus_problem).parameters["time_of_day"].default == 9,
          "make_campus_problem time_of_day default is 9")
    check(inspect.signature(cd.heuristic_scaled).parameters["scale"].default == 1.0,
          "heuristic_scaled scale default is 1.0")
    check(inspect.signature(cd.load_campus_map).parameters["filename"].default == "campus_map.json",
          "load_campus_map default filename is campus_map.json")
    check(inspect.signature(cd.is_cycle).parameters["k"].default == 30,
          "is_cycle default k is 30")


# ============================================================
# Independent reference implementation
# ============================================================

def ref_edge_cost(graph, bridges, time_of_day, state, nxt):
    if state not in graph or nxt not in graph[state]:
        return math.inf

    cost = graph[state][nxt]
    if nxt in bridges:
        cost += bridges[nxt].get("congestion", {}).get(str(time_of_day), 0)
    return cost


def ref_path_cost(graph, bridges, time_of_day, path):
    if path is None:
        return math.inf
    if len(path) <= 1:
        return 0

    total = 0
    for i in range(len(path) - 1):
        step = ref_edge_cost(graph, bridges, time_of_day, path[i], path[i + 1])
        if math.isinf(step):
            return math.inf
        total += step
    return total


def ref_dijkstra(graph, bridges, start, goal, time_of_day):
    if start == goal:
        return 0, [start]

    pq = [(0, start)]
    dist = {start: 0}
    parent = {}

    while pq:
        cost, state = heapq.heappop(pq)

        if cost > dist.get(state, math.inf):
            continue

        if state == goal:
            break

        for nxt in graph.get(state, {}):
            step = ref_edge_cost(graph, bridges, time_of_day, state, nxt)
            new_cost = cost + step

            if new_cost < dist.get(nxt, math.inf):
                dist[nxt] = new_cost
                parent[nxt] = state
                heapq.heappush(pq, (new_cost, nxt))

    if goal not in dist:
        return math.inf, None

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()

    return dist[goal], path


def ref_euclidean(coords, state, goal):
    x1, y1 = coords[state]
    x2, y2 = coords[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def ref_best_scale(graph, coords, bridges):
    candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    test_routes = [
        ("GATE_1", "PMB"),
        ("GATE_1", "GATE_2"),
        ("Library", "Staff_Hotel"),
        ("Hub", "Health_Centre"),
        ("TRENT", "Sports_Centre"),
    ]

    nodes = list(coords.keys())
    true_costs = {}

    for _, goal in test_routes:
        for node in nodes:
            true_costs[(node, goal)], _ = ref_dijkstra(graph, bridges, node, goal, 6)

    for scale in reversed(candidates):
        ok = True
        for _, goal in test_routes:
            for node in nodes:
                estimate = scale * ref_euclidean(coords, node, goal)
                if estimate > true_costs[(node, goal)] + EPS:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return scale

    return 1.0


def ref_multi_stop(graph, bridges, start, stops, hour):
    if not stops:
        return [start], 0

    best_path = None
    best_cost = math.inf

    for order in permutations(stops):
        checkpoints = [start] + list(order) + [start]
        total = 0
        combined = []
        ok = True

        for i in range(len(checkpoints) - 1):
            seg_cost, seg_path = ref_dijkstra(graph, bridges, checkpoints[i], checkpoints[i + 1], hour)

            if seg_path is None:
                ok = False
                break

            total += seg_cost

            if combined:
                combined.extend(seg_path[1:])
            else:
                combined.extend(seg_path)

        if ok and total < best_cost:
            best_cost = total
            best_path = combined

    return best_path, best_cost


# ============================================================
# Behaviour tests
# ============================================================

def test_helper_functions(cd):
    print("\n" + "=" * 80)
    print("2. Helper function behaviour")
    print("=" * 80)

    root = cd.make_node("A")
    check(isinstance(root, dict), "make_node returns dict")
    check(root == {"state": "A", "parent": None, "action": None, "cost": 0, "depth": 0},
          "make_node default node is correct")

    child = cd.make_node("B", parent=root, action="B", cost=5, depth=1)
    grandchild = cd.make_node("C", parent=child, action="C", cost=8, depth=2)

    check(cd.reconstruct_path(grandchild) == ["A", "B", "C"],
          "reconstruct_path reconstructs parent chain correctly")

    cycle_node = cd.make_node("A", parent=grandchild, action="A", cost=10, depth=3)
    check(cd.is_cycle(cycle_node) is True, "is_cycle detects ancestor state")
    check(cd.is_cycle(grandchild) is False, "is_cycle returns False for non-cycle")

    toy_problem = {
        "graph": {"A": {"B": 2, "C": 5}, "B": {}, "C": {}}
    }

    def toy_actions(problem, state):
        return [(n, n) for n in problem["graph"].get(state, {})]

    def toy_cost(problem, state, action):
        return problem["graph"].get(state, {}).get(action, math.inf)

    expanded = cd.expand(toy_problem, root, toy_actions, toy_cost)
    states = sorted(n["state"] for n in expanded)
    costs = sorted(n["cost"] for n in expanded)
    check(states == ["B", "C"], "expand creates correct child states")
    check(costs == [2, 5], "expand creates correct child costs")


def test_problem_actions_costs_exhaustive(cd, graph, coords, bridges):
    print("\n" + "=" * 80)
    print("3. Problem formulation, all actions, all edge costs")
    print("=" * 80)

    original_snapshot = copy.deepcopy((graph, coords, bridges))

    p = cd.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=9)

    check(isinstance(p, dict), "make_campus_problem returns dict")
    check(set(["initial", "goal", "graph", "coords", "bridges", "time_of_day"]).issubset(p.keys()),
          "problem has all required keys")
    check(p["graph"] is graph, "problem stores graph object")
    check(p["coords"] is coords, "problem stores coords object")
    check(p["bridges"] is bridges, "problem stores bridges object")

    check(type(cd.is_goal(p, "PMB")) is bool, "is_goal returns bool")
    check(cd.is_goal(p, "PMB") is True, "is_goal True at goal")
    check(cd.is_goal(p, "GATE_1") is False, "is_goal False away from goal")

    for state in coords:
        actions = cd.campus_get_actions(p, state)

        check(isinstance(actions, list), f"actions for {state} is list")
        check(all(isinstance(a, tuple) and len(a) == 2 for a in actions),
              f"actions for {state} are 2-tuples")

        expected_actions = {(n, n) for n in graph.get(state, {})}
        actual_actions = set(actions)

        check(actual_actions == expected_actions,
              f"actions for {state} exactly match graph neighbors")

    # Every real directed edge at every hour.
    for hour in range(24):
        p_hour = cd.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=hour)

        for state in graph:
            for nxt in graph[state]:
                expected = ref_edge_cost(graph, bridges, hour, state, nxt)
                actual = cd.campus_action_cost(p_hour, state, nxt)
                check_close(actual, expected, f"edge cost {state}->{nxt} at hour {hour}")

    # A few invalid moves.
    invalid_cases = [
        ("GATE_1", "Library"),
        ("Hub", "GATE_1"),
        ("Sports_Centre", "GATE_1"),
        ("NoSuchPlace", "GATE_1"),
        ("GATE_1", "NoSuchPlace"),
    ]

    for state, action in invalid_cases:
        actual = cd.campus_action_cost(p, state, action)
        check(math.isinf(actual), f"invalid move {state}->{action} returns infinity")

    # Explicit entering bridge vs leaving bridge checks.
    explicit_cases = [
        ("PMB", "Bridge_South", 9, 245 + 500),
        ("YANG_Fujia", "Bridge_South", 14, 266 + 500),
        ("Bridge_South", "PMB", 9, 245),
        ("Bridge_South", "YANG_Fujia", 14, 266),
        ("Library", "Bridge_North", 12, 167 + 500),
        ("Bridge_North", "Library", 12, 167),
        ("PB", "Bridge_Middle", 18, 269 + 500),
        ("Bridge_Middle", "PB", 18, 269),
    ]

    for state, action, hour, expected in explicit_cases:
        p_case = cd.make_campus_problem(state, action, graph, coords, bridges, time_of_day=hour)
        actual = cd.campus_action_cost(p_case, state, action)
        check_close(actual, expected, f"explicit congestion case {state}->{action} at hour {hour}")

    check((graph, coords, bridges) == original_snapshot,
          "problem/action/cost functions do not mutate map data")


def test_heuristics_exhaustive(cd, graph, coords, bridges):
    print("\n" + "=" * 80)
    print("4. Heuristic exactness, non-negativity, admissibility")
    print("=" * 80)

    nodes = list(coords.keys())

    for goal in nodes:
        p = cd.make_campus_problem("GATE_1", goal, graph, coords, bridges, time_of_day=6)

        for state in nodes:
            actual = cd.heuristic_euclidean(state, p)
            expected = ref_euclidean(coords, state, goal)

            if state == goal:
                check_close(actual, 0.0, f"h({state}) is zero at goal {goal}")

            check(isinstance(actual, float), f"heuristic_euclidean returns float for {state}->{goal}")
            check(actual >= -EPS, f"heuristic_euclidean non-negative for {state}->{goal}")
            check_close(actual, expected, f"heuristic_euclidean exact for {state}->{goal}")

            true_cost, _ = ref_dijkstra(graph, bridges, state, goal, 6)
            check(actual <= true_cost + EPS, f"heuristic_euclidean admissible for {state}->{goal}")

    scales = [0.0, 0.5, 1.0, 2.5, 3.0, 6.0]
    for goal in ["PMB", "GATE_2", "Sports_Centre"]:
        p = cd.make_campus_problem("GATE_1", goal, graph, coords, bridges, time_of_day=9)
        for state in nodes:
            base = cd.heuristic_euclidean(state, p)
            for scale in scales:
                actual = cd.heuristic_scaled(state, p, scale=scale)
                check_close(actual, scale * base, f"heuristic_scaled exact for {state}->{goal}, scale={scale}")


def test_astar_exhaustive(cd, graph, coords, bridges):
    print("\n" + "=" * 80)
    print("5. A* exhaustive comparison against independent Dijkstra")
    print("=" * 80)

    nodes = list(coords.keys())
    hours = list(range(24)) if FULL_ALL_PAIRS_TEST else [6, 9, 10, 12, 14, 18, 19]

    total = 0
    mismatch = 0

    for hour in hours:
        for start in nodes:
            for goal in nodes:
                total += 1

                p = cd.make_campus_problem(start, goal, graph, coords, bridges, time_of_day=hour)

                try:
                    result = cd.a_star_search(p, cd.heuristic_euclidean)
                except Exception as e:
                    fail_msg(f"A* exception for {start}->{goal} hour={hour}: {e}")
                    mismatch += 1
                    continue

                expected_cost, _ = ref_dijkstra(graph, bridges, start, goal, hour)

                if math.isinf(expected_cost):
                    if result is not None:
                        fail_msg(f"A* should return None for unreachable {start}->{goal} hour={hour}")
                        mismatch += 1
                    continue

                if result is None:
                    fail_msg(f"A* returned None for reachable {start}->{goal} hour={hour}")
                    mismatch += 1
                    continue

                if not isinstance(result, dict):
                    fail_msg(f"A* result not dict for {start}->{goal} hour={hour}")
                    mismatch += 1
                    continue

                for key in ["state", "parent", "action", "cost", "depth"]:
                    if key not in result:
                        fail_msg(f"A* result missing key {key!r} for {start}->{goal} hour={hour}")
                        mismatch += 1

                path = cd.reconstruct_path(result)

                if path[0] != start:
                    fail_msg(f"A* path wrong start for {start}->{goal} hour={hour}: {path}")
                    mismatch += 1

                if path[-1] != goal:
                    fail_msg(f"A* path wrong goal for {start}->{goal} hour={hour}: {path}")
                    mismatch += 1

                if start == goal:
                    if path != [start] or result["cost"] != 0 or result["depth"] != 0:
                        fail_msg(f"A* start==goal should return path [start], cost 0, depth 0 for {start}")
                        mismatch += 1

                if len(path) >= 2 and not cd.validate_path(p, path):
                    fail_msg(f"A* invalid path for {start}->{goal} hour={hour}: {path}")
                    mismatch += 1

                recomputed_cost = ref_path_cost(graph, bridges, hour, path)

                if abs(result["cost"] - expected_cost) > EPS:
                    fail_msg(
                        f"A* cost mismatch for {start}->{goal} hour={hour}: "
                        f"expected {expected_cost}, got {result['cost']}, path={path}"
                    )
                    mismatch += 1

                if abs(result["cost"] - recomputed_cost) > EPS:
                    fail_msg(
                        f"A* result cost does not equal path cost for {start}->{goal} hour={hour}: "
                        f"node cost {result['cost']}, path cost {recomputed_cost}, path={path}"
                    )
                    mismatch += 1

                if result["depth"] != len(path) - 1:
                    fail_msg(
                        f"A* depth mismatch for {start}->{goal} hour={hour}: "
                        f"depth {result['depth']}, path length {len(path)}"
                    )
                    mismatch += 1

                # Check action chain consistency: each non-root node's action should match its state.
                cur = result
                while cur.get("parent") is not None:
                    if cur.get("action") != cur.get("state"):
                        fail_msg(f"A* action/state mismatch in node {cur}")
                        mismatch += 1
                        break
                    cur = cur["parent"]

    if mismatch == 0:
        pass_msg(f"A* matches Dijkstra for {total} cases")
        print(f"[PASS] A* matches Dijkstra for {total} cases")
    else:
        fail_msg(f"A* had {mismatch} mismatches out of {total} cases")


def test_astar_with_scaled_heuristics(cd, graph, coords, bridges):
    print("\n" + "=" * 80)
    print("6. A* with returned best scale on official goals")
    print("=" * 80)

    official_routes = [
        ("GATE_1", "PMB"),
        ("GATE_1", "GATE_2"),
        ("Library", "Staff_Hotel"),
        ("Hub", "Health_Centre"),
        ("TRENT", "Sports_Centre"),
    ]

    nodes = list(coords.keys())

    p0 = cd.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=6)
    best_scale = cd.find_best_scale(p0)

    def h_best(state, problem):
        return cd.heuristic_scaled(state, problem, scale=best_scale)

    for _, goal in official_routes:
        for start in nodes:
            p = cd.make_campus_problem(start, goal, graph, coords, bridges, time_of_day=6)
            result = cd.a_star_search(p, h_best)
            expected_cost, _ = ref_dijkstra(graph, bridges, start, goal, 6)

            check(result is not None, f"A* with best scale finds path {start}->{goal}")
            if result is not None:
                check_close(result["cost"], expected_cost,
                            f"A* with best scale remains optimal for {start}->{goal}")


def test_find_best_scale_strict(cd, graph, coords, bridges):
    print("\n" + "=" * 80)
    print("7. find_best_scale strict check")
    print("=" * 80)

    p = cd.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=6)
    actual = cd.find_best_scale(p)
    expected = ref_best_scale(graph, coords, bridges)

    candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    check(type(actual) is float, "find_best_scale returns float, not int/string")
    check(actual in candidates, "find_best_scale result is one of official candidates")
    check_close(actual, expected, "find_best_scale equals independent reference")

    # Verify actual scale is admissible for official goals and next larger candidate is not.
    official_routes = [
        ("GATE_1", "PMB"),
        ("GATE_1", "GATE_2"),
        ("Library", "Staff_Hotel"),
        ("Hub", "Health_Centre"),
        ("TRENT", "Sports_Centre"),
    ]
    nodes = list(coords.keys())

    def is_scale_admissible(scale):
        for _, goal in official_routes:
            for node in nodes:
                h = scale * ref_euclidean(coords, node, goal)
                true_cost, _ = ref_dijkstra(graph, bridges, node, goal, 6)
                if h > true_cost + EPS:
                    return False
        return True

    check(is_scale_admissible(actual), f"returned scale {actual} is admissible")

    index = candidates.index(actual)
    if index < len(candidates) - 1:
        larger = candidates[index + 1]
        check(not is_scale_admissible(larger), f"next larger scale {larger} is not admissible")


def test_multi_stop_strict(cd, graph, coords, bridges):
    print("\n" + "=" * 80)
    print("8. multi_stop_delivery strict tests")
    print("=" * 80)

    base_cases = [
        ("GATE_1", ["PMB", "Library", "DB"], 9),
        ("GATE_1", ["PMB", "Library", "DB"], 6),
        ("PMB", ["Sports_Centre", "Staff_Hotel", "GATE_2"], 12),
        ("Library", ["Health_Centre", "GATE_1", "Student_Canteen"], 18),
        ("Staff_Hotel", ["DB", "New_Audi", "Hub"], 6),
        ("GATE_2", ["Bridge_North", "Bridge_Middle", "Bridge_South"], 14),
        ("Hub", ["PB"], 9),
        ("Hub", ["PB", "Library"], 12),
        ("GATE_1", [], 9),
    ]

    extra_cases = []

    if FULL_MULTI_STOP_SAMPLE:
        starts = ["GATE_1", "PMB", "Library", "Staff_Hotel", "Sports_Centre"]
        stop_sets = [
            ["PMB", "Library", "DB"],
            ["GATE_2", "Student_Canteen", "Health_Centre"],
            ["Bridge_North", "Bridge_Middle", "Bridge_South"],
            ["TRENT", "NICC", "Sports_Centre"],
            ["IAMET", "Staff_Hotel", "GATE_4"],
        ]
        hours = [6, 9, 12, 14, 18, 19]

        for start in starts:
            for stops in stop_sets:
                if start not in stops:
                    for hour in hours:
                        extra_cases.append((start, stops, hour))

    all_cases = base_cases + extra_cases

    for start, stops, hour in all_cases:
        p = cd.make_campus_problem(start, start, graph, coords, bridges, time_of_day=hour)

        try:
            result = cd.multi_stop_delivery(p, stops, cd.heuristic_euclidean)
        except Exception as e:
            fail_msg(f"multi_stop exception for start={start}, stops={stops}, hour={hour}: {e}")
            continue

        check(isinstance(result, tuple) and len(result) == 2,
              f"multi_stop returns tuple(path, cost), start={start}, stops={stops}, hour={hour}")

        if not (isinstance(result, tuple) and len(result) == 2):
            continue

        path, cost = result
        expected_path, expected_cost = ref_multi_stop(graph, bridges, start, stops, hour)

        check(isinstance(path, list), f"multi_stop path is list, start={start}, stops={stops}, hour={hour}")
        check(isinstance(cost, (int, float)), f"multi_stop cost is numeric, start={start}, stops={stops}, hour={hour}")

        if path:
            check(path[0] == start, f"multi_stop path starts correctly, start={start}, stops={stops}, hour={hour}")
            check(path[-1] == start, f"multi_stop path returns to start, start={start}, stops={stops}, hour={hour}")

            for stop in stops:
                check(stop in path, f"multi_stop visits {stop}, start={start}, hour={hour}")

            if len(path) >= 2:
                check(cd.validate_path(p, path), f"multi_stop path is valid, start={start}, stops={stops}, hour={hour}")

            for i in range(len(path) - 1):
                check(path[i] != path[i + 1], f"multi_stop has no adjacent duplicated node at index {i}")

            path_cost = ref_path_cost(graph, bridges, hour, path)
            check_close(path_cost, cost, f"multi_stop returned cost equals path cost, start={start}, stops={stops}, hour={hour}")

        check_close(cost, expected_cost,
                    f"multi_stop optimal cost, start={start}, stops={stops}, hour={hour}")


def test_artificial_graphs(cd):
    print("\n" + "=" * 80)
    print("9. Artificial graph tests: unreachable, custom bridges, cycles")
    print("=" * 80)

    small_graph = {
        "A": {"B": 1, "X": 10},
        "B": {"C": 1},
        "C": {"A": 1},
        "X": {},
        "D": {}
    }
    small_coords = {
        "A": [0, 0],
        "B": [1, 0],
        "C": [2, 0],
        "X": [10, 0],
        "D": [100, 0]
    }
    small_bridges = {
        "B": {
            "description": "toy bridge",
            "congestion": {"9": 100}
        }
    }

    p = cd.make_campus_problem("A", "C", small_graph, small_coords, small_bridges, time_of_day=9)

    check_close(cd.campus_action_cost(p, "A", "B"), 101, "toy graph entering bridge B at 9 adds penalty")
    check_close(cd.campus_action_cost(p, "B", "C"), 1, "toy graph leaving bridge B does not add penalty")
    check(math.isinf(cd.campus_action_cost(p, "A", "D")), "toy graph invalid edge returns inf")

    result = cd.a_star_search(p, cd.heuristic_euclidean)
    check(result is not None, "toy graph A* finds A->C")
    if result:
        path = cd.reconstruct_path(result)
        check(path == ["A", "B", "C"], f"toy graph A* path is A-B-C, got {path}")
        check_close(result["cost"], 102, "toy graph A* cost includes bridge congestion")

    unreachable = cd.make_campus_problem("A", "D", small_graph, small_coords, small_bridges, time_of_day=9)
    result2 = cd.a_star_search(unreachable, cd.heuristic_euclidean)
    check(result2 is None, "A* returns None for unreachable artificial goal")


def test_no_mutation(cd, graph, coords, bridges):
    print("\n" + "=" * 80)
    print("10. No mutation check")
    print("=" * 80)

    before = copy.deepcopy((graph, coords, bridges))

    nodes = list(coords.keys())

    for hour in [6, 9, 12, 14, 18]:
        for start in nodes[:5]:
            for goal in nodes[-5:]:
                p = cd.make_campus_problem(start, goal, graph, coords, bridges, time_of_day=hour)
                cd.campus_get_actions(p, start)
                cd.campus_action_cost(p, start, goal)
                cd.heuristic_euclidean(start, p)
                cd.a_star_search(p, cd.heuristic_euclidean)

    p_multi = cd.make_campus_problem("GATE_1", "GATE_1", graph, coords, bridges, time_of_day=9)
    cd.multi_stop_delivery(p_multi, ["PMB", "Library", "DB"], cd.heuristic_euclidean)
    cd.find_best_scale(p_multi)

    after = (graph, coords, bridges)
    check(after == before, "student functions do not mutate graph/coords/bridges")


# ============================================================
# Runner
# ============================================================

def run_all():
    print("=" * 80)
    print("ULTRA-STRICT TESTS FOR campus_delivery.py")
    print("=" * 80)

    if not os.path.exists(MODULE_FILE):
        print(f"[FATAL] Missing {MODULE_FILE}")
        return

    if not os.path.exists(MAP_FILE):
        print(f"[FATAL] Missing {MAP_FILE}")
        return

    try:
        test_static_imports_and_structure()
        cd = load_student_module_safely()
        graph, coords, bridges = load_json_map()

        print(f"\nLoaded {MODULE_FILE}")
        print(f"Loaded map: {len(coords)} nodes, {sum(len(v) for v in graph.values())} directed edges")

        test_required_functions(cd)
        test_helper_functions(cd)
        test_problem_actions_costs_exhaustive(cd, graph, coords, bridges)
        test_heuristics_exhaustive(cd, graph, coords, bridges)
        test_astar_exhaustive(cd, graph, coords, bridges)
        test_astar_with_scaled_heuristics(cd, graph, coords, bridges)
        test_find_best_scale_strict(cd, graph, coords, bridges)
        test_multi_stop_strict(cd, graph, coords, bridges)
        test_artificial_graphs(cd)
        test_no_mutation(cd, graph, coords, bridges)

    except Exception:
        fail_msg("Unexpected fatal error during test run")
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"PASS CHECKS: {passes}")
    print(f"FAILURES: {len(failures)}")

    if failures:
        print("\nFailure list:")
        for i, item in enumerate(failures, 1):
            print(f"{i}. {item}")
    else:
        print("ALL ULTRA-STRICT TESTS PASSED")


if __name__ == "__main__":
    run_all()