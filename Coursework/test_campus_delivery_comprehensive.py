"""
Comprehensive test script for COMP1037 campus_delivery.py

Put this file in the same folder as:
    campus_delivery.py
    campus_map.json

Run:
    python test_campus_delivery_comprehensive.py

Do NOT submit this test file. Submit only campus_delivery.py.
"""

import os
import math
import json
import heapq
import inspect
import traceback
import importlib.util
from itertools import permutations


MODULE_FILE = "campus_delivery.py"
MAP_FILE = "campus_map.json"
EPS = 1e-6

# Set this to False if you want a quicker sampled A* test.
# True tests all start-goal pairs at all 24 hours.
FULL_ALL_PAIRS_TEST = True


# ============================================================
# Basic test framework
# ============================================================

failures = []


def record_failure(message):
    failures.append(message)
    print(f"[FAIL] {message}")


def record_pass(message):
    print(f"[PASS] {message}")


def check(condition, message):
    if condition:
        record_pass(message)
    else:
        record_failure(message)


def check_close(actual, expected, message, eps=EPS):
    if math.isinf(actual) and math.isinf(expected):
        record_pass(message)
        return
    if abs(actual - expected) <= eps:
        record_pass(message)
    else:
        record_failure(f"{message}: expected {expected}, got {actual}")


def load_student_module():
    module_path = os.path.abspath(MODULE_FILE)
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Cannot find {MODULE_FILE} in current folder.")

    spec = importlib.util.spec_from_file_location("campus_delivery_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_map_data():
    if not os.path.exists(MAP_FILE):
        raise FileNotFoundError(f"Cannot find {MAP_FILE} in current folder.")

    with open(MAP_FILE, "r") as f:
        data = json.load(f)

    return data["campus_graph"], data["campus_coords"], data["campus_bridges"]


# ============================================================
# Independent reference implementation
# This does NOT call student's campus_action_cost or A*.
# ============================================================

def reference_edge_cost(graph, bridges, time_of_day, state, next_state):
    if state not in graph or next_state not in graph[state]:
        return math.inf

    base = graph[state][next_state]
    penalty = 0

    # Congestion is added only when ENTERING a bridge node.
    if next_state in bridges:
        congestion_table = bridges[next_state].get("congestion", {})
        penalty = congestion_table.get(str(time_of_day), 0)

    return base + penalty


def reference_path_cost(graph, bridges, time_of_day, path):
    if path is None:
        return math.inf

    if len(path) <= 1:
        return 0

    total = 0
    for i in range(len(path) - 1):
        step_cost = reference_edge_cost(graph, bridges, time_of_day, path[i], path[i + 1])
        if math.isinf(step_cost):
            return math.inf
        total += step_cost
    return total


def reference_dijkstra(graph, bridges, start, goal, time_of_day):
    """
    Independent Dijkstra implementation used as the gold-standard reference.
    Returns:
        (best_cost, best_path)
    """
    if start == goal:
        return 0, [start]

    frontier = [(0, start)]
    dist = {start: 0}
    parent = {}

    while frontier:
        current_cost, state = heapq.heappop(frontier)

        if current_cost > dist.get(state, math.inf):
            continue

        if state == goal:
            break

        for next_state in graph.get(state, {}):
            step_cost = reference_edge_cost(graph, bridges, time_of_day, state, next_state)
            if math.isinf(step_cost):
                continue

            new_cost = current_cost + step_cost

            if new_cost < dist.get(next_state, math.inf):
                dist[next_state] = new_cost
                parent[next_state] = state
                heapq.heappush(frontier, (new_cost, next_state))

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


def reference_euclidean(coords, state, goal):
    x1, y1 = coords[state]
    x2, y2 = coords[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def reference_best_scale(graph, coords, bridges):
    candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    test_routes = [
        ("GATE_1", "PMB"),
        ("GATE_1", "GATE_2"),
        ("Library", "Staff_Hotel"),
        ("Hub", "Health_Centre"),
        ("TRENT", "Sports_Centre"),
    ]

    all_nodes = list(coords.keys())

    true_costs = {}
    for _, goal in test_routes:
        for node in all_nodes:
            cost, _ = reference_dijkstra(graph, bridges, node, goal, time_of_day=6)
            true_costs[(node, goal)] = cost

    for scale in reversed(candidates):
        ok = True

        for _, goal in test_routes:
            for node in all_nodes:
                estimate = scale * reference_euclidean(coords, node, goal)
                true_cost = true_costs[(node, goal)]

                if estimate > true_cost + EPS:
                    ok = False
                    break

            if not ok:
                break

        if ok:
            return scale

    return 1.0


def reference_multi_stop(graph, bridges, start, locations_to_visit, time_of_day):
    if not locations_to_visit:
        return [start], 0

    best_path = None
    best_cost = math.inf

    for order in permutations(locations_to_visit):
        checkpoints = [start] + list(order) + [start]
        total = 0
        combined_path = []
        valid = True

        for i in range(len(checkpoints) - 1):
            segment_start = checkpoints[i]
            segment_goal = checkpoints[i + 1]

            segment_cost, segment_path = reference_dijkstra(
                graph, bridges, segment_start, segment_goal, time_of_day
            )

            if segment_path is None:
                valid = False
                break

            total += segment_cost

            if combined_path:
                combined_path.extend(segment_path[1:])
            else:
                combined_path.extend(segment_path)

        if valid and total < best_cost:
            best_cost = total
            best_path = combined_path

    return best_path, best_cost


# ============================================================
# Test groups
# ============================================================

def test_required_functions(cd):
    print("\n" + "=" * 70)
    print("1. Function existence and signatures")
    print("=" * 70)

    expected_signatures = {
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

    for fn_name, expected_params in expected_signatures.items():
        check(hasattr(cd, fn_name), f"{fn_name} exists")

        if hasattr(cd, fn_name):
            sig = inspect.signature(getattr(cd, fn_name))
            actual_params = list(sig.parameters.keys())
            check(
                actual_params == expected_params,
                f"{fn_name} parameter names/order are correct"
            )

    if hasattr(cd, "make_campus_problem"):
        sig = inspect.signature(cd.make_campus_problem)
        default_value = sig.parameters["time_of_day"].default
        check(default_value == 9, "make_campus_problem time_of_day default is 9")

    if hasattr(cd, "heuristic_scaled"):
        sig = inspect.signature(cd.heuristic_scaled)
        default_value = sig.parameters["scale"].default
        check(default_value == 1.0, "heuristic_scaled scale default is 1.0")


def test_problem_formulation_and_costs(cd, graph, coords, bridges):
    print("\n" + "=" * 70)
    print("2. Problem formulation, actions, and bridge congestion")
    print("=" * 70)

    p9 = cd.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=9)

    check(isinstance(p9, dict), "make_campus_problem returns dict")

    required_keys = {"initial", "goal", "graph", "coords", "bridges", "time_of_day"}
    check(required_keys.issubset(set(p9.keys())), "problem dict contains required keys")

    check(p9["initial"] == "GATE_1", "problem['initial'] is correct")
    check(p9["goal"] == "PMB", "problem['goal'] is correct")
    check(p9["time_of_day"] == 9, "problem['time_of_day'] is correct")

    check(type(cd.is_goal(p9, "PMB")) is bool, "is_goal returns bool")
    check(cd.is_goal(p9, "PMB") is True, "is_goal detects goal")
    check(cd.is_goal(p9, "GATE_1") is False, "is_goal rejects non-goal")

    actions = cd.campus_get_actions(p9, "GATE_1")
    check(isinstance(actions, list), "campus_get_actions returns list")
    check(all(isinstance(a, tuple) and len(a) == 2 for a in actions), "actions are tuples of length 2")
    check(("IAMET", "IAMET") in actions, "GATE_1 has action to IAMET")
    check(("IEB", "IEB") in actions, "GATE_1 has action to IEB")

    check_close(cd.campus_action_cost(p9, "GATE_1", "IAMET"), 431, "normal edge cost GATE_1 -> IAMET")
    check(math.isinf(cd.campus_action_cost(p9, "GATE_1", "Library")), "invalid edge returns math.inf")

    bridge_cases = [
        ("PMB", "Bridge_South", 9, 245 + 500),
        ("PMB", "Bridge_South", 10, 245 + 400),
        ("PMB", "Bridge_South", 6, 245),
        ("Bridge_South", "PMB", 9, 245),          # leaving bridge: no penalty
        ("Library", "Bridge_North", 12, 167 + 500),
        ("Bridge_North", "Library", 12, 167),    # leaving bridge: no penalty
        ("PB", "Bridge_Middle", 18, 269 + 500),
        ("Bridge_Middle", "PB", 18, 269),        # leaving bridge: no penalty
        ("PB", "Bridge_Middle", 19, 269 + 400),
    ]

    for state, action, hour, expected in bridge_cases:
        p = cd.make_campus_problem(state, action, graph, coords, bridges, time_of_day=hour)
        actual = cd.campus_action_cost(p, state, action)
        check_close(actual, expected, f"cost {state} -> {action} at hour {hour}")


def test_heuristics(cd, graph, coords, bridges):
    print("\n" + "=" * 70)
    print("3. Heuristic checks")
    print("=" * 70)

    nodes = list(coords.keys())

    for goal in ["PMB", "GATE_2", "Staff_Hotel", "Health_Centre", "Sports_Centre"]:
        p = cd.make_campus_problem("GATE_1", goal, graph, coords, bridges, time_of_day=6)

        h_goal = cd.heuristic_euclidean(goal, p)
        check_close(h_goal, 0.0, f"heuristic_euclidean({goal}) is 0 at goal")

        for state in nodes:
            h = cd.heuristic_euclidean(state, p)
            check(h >= -EPS, f"heuristic_euclidean is non-negative for {state} -> {goal}")

    p = cd.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=6)
    h1 = cd.heuristic_euclidean("GATE_1", p)
    h3 = cd.heuristic_scaled("GATE_1", p, scale=3.0)
    check_close(h3, 3.0 * h1, "heuristic_scaled equals scale * heuristic_euclidean")

    # Admissibility of Euclidean heuristic at off-peak time.
    # If it passes at off-peak, it also passes with congestion because congestion only adds cost.
    for goal in nodes:
        p = cd.make_campus_problem("GATE_1", goal, graph, coords, bridges, time_of_day=6)
        for state in nodes:
            h = cd.heuristic_euclidean(state, p)
            true_cost, _ = reference_dijkstra(graph, bridges, state, goal, time_of_day=6)
            check(
                h <= true_cost + EPS,
                f"Euclidean admissible for {state} -> {goal}"
            )


def test_a_star_against_dijkstra(cd, graph, coords, bridges):
    print("\n" + "=" * 70)
    print("4. A* compared against independent Dijkstra")
    print("=" * 70)

    nodes = list(coords.keys())

    if FULL_ALL_PAIRS_TEST:
        hours = list(range(24))
        print("Running full all-pairs test: 30 x 30 x 24 cases...")
    else:
        hours = [6, 9, 10, 12, 14, 18, 19]
        print("Running sampled all-pairs test...")

    total_cases = 0
    mismatches = 0

    for hour in hours:
        for start in nodes:
            for goal in nodes:
                total_cases += 1

                p = cd.make_campus_problem(start, goal, graph, coords, bridges, time_of_day=hour)

                try:
                    result = cd.a_star_search(p, cd.heuristic_euclidean)
                except Exception as e:
                    record_failure(f"A* raised exception for {start} -> {goal} at hour {hour}: {e}")
                    mismatches += 1
                    continue

                expected_cost, _ = reference_dijkstra(graph, bridges, start, goal, time_of_day=hour)

                if math.isinf(expected_cost):
                    if result is not None:
                        record_failure(f"A* should return None for unreachable {start} -> {goal} at hour {hour}")
                        mismatches += 1
                    continue

                if result is None:
                    record_failure(f"A* returned None for reachable {start} -> {goal} at hour {hour}")
                    mismatches += 1
                    continue

                if not isinstance(result, dict):
                    record_failure(f"A* result is not dict for {start} -> {goal} at hour {hour}")
                    mismatches += 1
                    continue

                for key in ["state", "parent", "action", "cost", "depth"]:
                    if key not in result:
                        record_failure(f"A* result missing key '{key}' for {start} -> {goal} at hour {hour}")
                        mismatches += 1
                        break

                path = cd.reconstruct_path(result)

                if path[0] != start:
                    record_failure(f"A* path does not start correctly for {start} -> {goal} at hour {hour}: {path}")
                    mismatches += 1

                if path[-1] != goal:
                    record_failure(f"A* path does not end correctly for {start} -> {goal} at hour {hour}: {path}")
                    mismatches += 1

                if len(path) >= 2 and not cd.validate_path(p, path):
                    record_failure(f"A* path is invalid for {start} -> {goal} at hour {hour}: {path}")
                    mismatches += 1

                actual_cost = result["cost"]
                recomputed_cost = reference_path_cost(graph, bridges, hour, path)

                if abs(actual_cost - expected_cost) > EPS:
                    record_failure(
                        f"A* cost mismatch for {start} -> {goal} at hour {hour}: "
                        f"expected {expected_cost}, got {actual_cost}, path={path}"
                    )
                    mismatches += 1

                if abs(recomputed_cost - actual_cost) > EPS:
                    record_failure(
                        f"A* path cost does not match node cost for {start} -> {goal} at hour {hour}: "
                        f"path_cost {recomputed_cost}, node_cost {actual_cost}, path={path}"
                    )
                    mismatches += 1

    if mismatches == 0:
        record_pass(f"A* matches independent Dijkstra for {total_cases} cases")
    else:
        record_failure(f"A* had {mismatches} mismatches out of {total_cases} cases")


def test_find_best_scale(cd, graph, coords, bridges):
    print("\n" + "=" * 70)
    print("5. find_best_scale")
    print("=" * 70)

    p = cd.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=6)

    try:
        actual = cd.find_best_scale(p)
    except Exception as e:
        record_failure(f"find_best_scale raised exception: {e}")
        return

    expected = reference_best_scale(graph, coords, bridges)

    candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    check(isinstance(actual, float), "find_best_scale returns float")
    check(actual in candidates, "find_best_scale returns one of candidate values")
    check_close(actual, expected, "find_best_scale matches independent reference")


def test_multi_stop(cd, graph, coords, bridges):
    print("\n" + "=" * 70)
    print("6. multi_stop_delivery")
    print("=" * 70)

    test_cases = [
        ("GATE_1", ["PMB", "Library", "DB"], 9),
        ("GATE_1", ["PMB", "Library", "DB"], 6),
        ("PMB", ["Sports_Centre", "Staff_Hotel", "GATE_2"], 12),
        ("Library", ["Health_Centre", "GATE_1", "Student_Canteen"], 18),
        ("Staff_Hotel", ["DB", "New_Audi", "Hub"], 6),
        ("GATE_2", ["Bridge_North", "Bridge_Middle", "Bridge_South"], 14),
    ]

    for start, stops, hour in test_cases:
        p = cd.make_campus_problem(start, start, graph, coords, bridges, time_of_day=hour)

        try:
            result = cd.multi_stop_delivery(p, stops, cd.heuristic_euclidean)
        except Exception as e:
            record_failure(f"multi_stop_delivery raised exception for {start}, {stops}, hour {hour}: {e}")
            continue

        check(isinstance(result, tuple) and len(result) == 2, f"multi_stop returns tuple(path, cost) for {start}, hour {hour}")

        if not (isinstance(result, tuple) and len(result) == 2):
            continue

        actual_path, actual_cost = result
        expected_path, expected_cost = reference_multi_stop(graph, bridges, start, stops, hour)

        check(isinstance(actual_path, list), f"multi_stop path is list for {start}, hour {hour}")
        check(isinstance(actual_cost, (int, float)), f"multi_stop cost is numeric for {start}, hour {hour}")

        if actual_path:
            check(actual_path[0] == start, f"multi_stop path starts at {start}, hour {hour}")
            check(actual_path[-1] == start, f"multi_stop path returns to {start}, hour {hour}")

            for stop in stops:
                check(stop in actual_path, f"multi_stop path visits {stop}, hour {hour}")

            if len(actual_path) >= 2:
                check(cd.validate_path(p, actual_path), f"multi_stop returned path is valid for {start}, hour {hour}")

            recomputed_cost = reference_path_cost(graph, bridges, hour, actual_path)
            check_close(recomputed_cost, actual_cost, f"multi_stop path cost matches returned cost for {start}, hour {hour}")

        check_close(actual_cost, expected_cost, f"multi_stop cost is optimal for {start}, stops={stops}, hour {hour}")

    # Empty stops edge case. This is not the main coursework case, but it is good robustness.
    p_empty = cd.make_campus_problem("GATE_1", "GATE_1", graph, coords, bridges, time_of_day=9)
    empty_path, empty_cost = cd.multi_stop_delivery(p_empty, [], cd.heuristic_euclidean)
    check(empty_path == ["GATE_1"], "multi_stop empty list returns start-only path")
    check_close(empty_cost, 0, "multi_stop empty list returns zero cost")


def test_unreachable_artificial_case(cd):
    print("\n" + "=" * 70)
    print("7. Artificial unreachable case")
    print("=" * 70)

    small_graph = {
        "A": {"B": 1},
        "B": {},
        "C": {}
    }
    small_coords = {
        "A": [0, 0],
        "B": [1, 0],
        "C": [10, 0]
    }
    small_bridges = {}

    p = cd.make_campus_problem("A", "C", small_graph, small_coords, small_bridges, time_of_day=9)

    try:
        result = cd.a_star_search(p, cd.heuristic_euclidean)
        check(result is None, "A* returns None when no path exists")
    except Exception as e:
        record_failure(f"A* raised exception on unreachable artificial case: {e}")


def run_all_tests():
    print("=" * 70)
    print("COMPREHENSIVE TESTS FOR campus_delivery.py")
    print("=" * 70)

    try:
        cd = load_student_module()
        graph, coords, bridges = load_map_data()
    except Exception:
        print("[FATAL] Could not load campus_delivery.py or campus_map.json")
        traceback.print_exc()
        return

    print(f"Loaded {MODULE_FILE}")
    print(f"Loaded map with {len(coords)} nodes and {sum(len(v) for v in graph.values())} directed edges")

    test_required_functions(cd)
    test_problem_formulation_and_costs(cd, graph, coords, bridges)
    test_heuristics(cd, graph, coords, bridges)
    test_a_star_against_dijkstra(cd, graph, coords, bridges)
    test_find_best_scale(cd, graph, coords, bridges)
    test_multi_stop(cd, graph, coords, bridges)
    test_unreachable_artificial_case(cd)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if failures:
        print(f"TOTAL FAILURES: {len(failures)}")
        print("\nFailure list:")
        for i, failure in enumerate(failures, 1):
            print(f"{i}. {failure}")
    else:
        print("ALL COMPREHENSIVE TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()