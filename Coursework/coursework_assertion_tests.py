import ast
import inspect
import itertools
import json
import math
import random
import heapq
import sys

import campus_delivery as stu


REPORT = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        REPORT.append((name, "PASS", detail))
    else:
        REPORT.append((name, "FAIL", detail))
        FAIL.append((name, detail))


def dijkstra(problem, start, goal):
    """Independent shortest-path baseline using student's action-cost semantics."""
    pq = [(0, start)]
    dist = {start: 0}
    parent = {start: None}
    while pq:
        g, u = heapq.heappop(pq)
        if g != dist[u]:
            continue
        if u == goal:
            break
        for v in problem["graph"].get(u, {}):
            w = stu.campus_action_cost(problem, u, v)
            ng = g + w
            if ng < dist.get(v, float("inf")):
                dist[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng, v))
    if goal not in dist:
        return None, float("inf")
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, dist[goal]


def dijkstra_cost_only(problem, start, goal):
    pq = [(0, start)]
    dist = {start: 0}
    while pq:
        g, u = heapq.heappop(pq)
        if g != dist[u]:
            continue
        if u == goal:
            return g
        for v in problem["graph"].get(u, {}):
            w = stu.campus_action_cost(problem, u, v)
            ng = g + w
            if ng < dist.get(v, float("inf")):
                dist[v] = ng
                heapq.heappush(pq, (ng, v))
    return float("inf")


def run():
    with open("campus_map.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    graph = data["campus_graph"]
    coords = data["campus_coords"]
    bridges = data["campus_bridges"]
    all_nodes = list(coords.keys())

    # 1) Helper unchanged checks against old_delivery.py
    old_src = open("old_delivery.py", "r", encoding="utf-8").read()
    new_src = open("campus_delivery.py", "r", encoding="utf-8").read()
    old_ast = ast.parse(old_src)
    new_ast = ast.parse(new_src)
    old_funcs = {n.name: n for n in old_ast.body if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
    helper_names = [
        "make_node",
        "reconstruct_path",
        "expand",
        "load_campus_map",
        "validate_path",
        "is_cycle",
    ]
    for h in helper_names:
        osrc = ast.get_source_segment(old_src, old_funcs[h])
        nsrc = ast.get_source_segment(new_src, new_funcs[h])
        check(f"helper_unchanged_{h}", osrc == nsrc, "provided helper must be unchanged")

    # 2) Signature checks
    expected_sigs = {
        "make_campus_problem": [
            "start",
            "goal",
            "campus_graph",
            "campus_coords",
            "campus_bridges",
            "time_of_day",
        ],
        "is_goal": ["problem", "state"],
        "campus_get_actions": ["problem", "state"],
        "campus_action_cost": ["problem", "state", "action"],
        "heuristic_euclidean": ["state", "problem"],
        "heuristic_scaled": ["state", "problem", "scale"],
        "find_best_scale": ["problem"],
        "a_star_search": ["problem", "heuristic_fn"],
        "multi_stop_delivery": ["problem", "locations_to_visit", "heuristic_fn"],
    }
    for fn, args in expected_sigs.items():
        sig = inspect.signature(getattr(stu, fn))
        check(f"signature_{fn}", list(sig.parameters.keys()) == args, str(sig))

    # 3) make_campus_problem
    p = stu.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=9)
    check("make_problem_type", isinstance(p, dict), type(p).__name__)
    check(
        "make_problem_keys",
        set(["initial", "goal", "graph", "coords", "bridges", "time_of_day"]).issubset(p.keys()),
        str(sorted(p.keys())),
    )
    check(
        "make_problem_values",
        p["initial"] == "GATE_1" and p["goal"] == "PMB" and p["time_of_day"] == 9,
        str((p.get("initial"), p.get("goal"), p.get("time_of_day"))),
    )

    # 4) is_goal
    res_goal = stu.is_goal(p, "PMB")
    res_not = stu.is_goal(p, "GATE_1")
    check("is_goal_type", isinstance(res_goal, bool) and isinstance(res_not, bool), f"{type(res_goal).__name__},{type(res_not).__name__}")
    check("is_goal_logic", res_goal is True and res_not is False, f"{res_goal},{res_not}")

    # 5) campus_get_actions
    for s in graph:
        acts = stu.campus_get_actions(p, s)
        check(f"actions_type_{s}", isinstance(acts, list), type(acts).__name__)
        check(f"actions_tuple_{s}", all(isinstance(x, tuple) and len(x) == 2 for x in acts), str(acts[:3]))
        expected = [(n, n) for n in graph[s].keys()]
        check(f"actions_exact_{s}", acts == expected, f"expected {len(expected)} got {len(acts)}")

    # 6) campus_action_cost
    check("action_cost_invalid_inf", math.isinf(stu.campus_action_cost(p, "GATE_1", "Library")), str(stu.campus_action_cost(p, "GATE_1", "Library")))
    p6 = stu.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=6)
    ok = True
    for u in graph:
        for v, base in graph[u].items():
            got = stu.campus_action_cost(p6, u, v)
            pen = bridges[v]["congestion"].get("6", 0) if v in bridges else 0
            if abs(got - (base + pen)) > 1e-9:
                ok = False
                break
        if not ok:
            break
    check("action_cost_all_edges_offpeak", ok, "")

    ok = True
    for hour in range(24):
        pr = stu.make_campus_problem("X", "Y", graph, coords, bridges, time_of_day=hour)
        hs = str(hour)
        for bname, bdata in bridges.items():
            preds = [u for u in graph if bname in graph[u]]
            if not preds:
                continue
            u = preds[0]
            base = graph[u][bname]
            exp = base + bdata["congestion"].get(hs, 0)
            got = stu.campus_action_cost(pr, u, bname)
            if abs(got - exp) > 1e-9:
                ok = False
                break
        if not ok:
            break
    check("action_cost_bridge_hours_0_23", ok, "")

    # 7) heuristic_euclidean
    ok_type = True
    ok_nonneg = True
    ok_goal0 = True
    for g in all_nodes:
        pr = stu.make_campus_problem("dummy", g, graph, coords, bridges, time_of_day=6)
        for s in all_nodes:
            h = stu.heuristic_euclidean(s, pr)
            if not isinstance(h, float):
                ok_type = False
            if h < -1e-12:
                ok_nonneg = False
        if abs(stu.heuristic_euclidean(g, pr) - 0.0) > 1e-12:
            ok_goal0 = False
    check("heuristic_type_float", ok_type, "")
    check("heuristic_nonnegative", ok_nonneg, "")
    check("heuristic_goal_zero", ok_goal0, "")

    ok_adm = True
    for g in all_nodes:
        pr = stu.make_campus_problem("dummy", g, graph, coords, bridges, time_of_day=6)
        for s in all_nodes:
            _, true_cost = dijkstra(pr, s, g)
            h = stu.heuristic_euclidean(s, pr)
            if h - true_cost > 1e-9:
                ok_adm = False
                break
        if not ok_adm:
            break
    check("heuristic_admissible_all_pairs_offpeak", ok_adm, "")

    # 8) heuristic_scaled
    hs = stu.heuristic_scaled("GATE_1", p, 3.0)
    he = stu.heuristic_euclidean("GATE_1", p)
    check("heuristic_scaled_type", isinstance(hs, float), type(hs).__name__)
    check("heuristic_scaled_value", abs(hs - 3.0 * he) < 1e-9, f"{hs} vs {3.0 * he}")

    # 9) find_best_scale
    candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    test_goals = ["PMB", "GATE_2", "Staff_Hotel", "Health_Centre", "Sports_Centre"]
    true_cost = {}
    for g in test_goals:
        pr = stu.make_campus_problem("dummy", g, graph, coords, bridges, time_of_day=6)
        for s in all_nodes:
            _, c = dijkstra(pr, s, g)
            true_cost[(s, g)] = c
    expected_best = 1.0
    for scale in reversed(candidates):
        good = True
        for g in test_goals:
            pr = stu.make_campus_problem("dummy", g, graph, coords, bridges, time_of_day=6)
            for s in all_nodes:
                h = stu.heuristic_scaled(s, pr, scale)
                if h - true_cost[(s, g)] > 1e-9:
                    good = False
                    break
            if not good:
                break
        if good:
            expected_best = scale
            break
    best = stu.find_best_scale(stu.make_campus_problem("GATE_1", "PMB", graph, coords, bridges, time_of_day=9))
    check("find_best_scale_type_float", isinstance(best, float), type(best).__name__)
    check("find_best_scale_in_candidates", best in candidates, str(best))
    check("find_best_scale_correct", abs(best - expected_best) < 1e-12, f"got {best} expected {expected_best}")

    # 10) a_star_search
    hours = [6, 9, 12, 18]
    ok_opt = True
    for hr in hours:
        for s in all_nodes:
            for g in all_nodes:
                pr = stu.make_campus_problem(s, g, graph, coords, bridges, time_of_day=hr)
                r = stu.a_star_search(pr, stu.heuristic_euclidean)
                if r is None:
                    ok_opt = False
                    break
                path = stu.reconstruct_path(r)
                if not isinstance(r, dict):
                    ok_opt = False
                    break
                if r.get("state") != g:
                    ok_opt = False
                    break
                if not stu.validate_path(pr, path) and s != g:
                    ok_opt = False
                    break
                _, bestc = dijkstra(pr, s, g)
                if abs(r["cost"] - bestc) > 1e-9:
                    ok_opt = False
                    break
            if not ok_opt:
                break
        if not ok_opt:
            break
    check("a_star_optimal_all_pairs_hours_6_9_12_18", ok_opt, "")

    pr = stu.make_campus_problem("PMB", "PMB", graph, coords, bridges, time_of_day=9)
    r = stu.a_star_search(pr, stu.heuristic_euclidean)
    check("a_star_start_equals_goal", isinstance(r, dict) and r["state"] == "PMB" and r["cost"] == 0, str(r))

    small_graph = {"A": {"B": 1}, "B": {}, "C": {}}
    small_coords = {"A": [0, 0], "B": [1, 0], "C": [2, 0]}
    small_bridges = {}
    pr = stu.make_campus_problem("A", "C", small_graph, small_coords, small_bridges, time_of_day=9)
    r = stu.a_star_search(pr, stu.heuristic_euclidean)
    check("a_star_no_path_none", r is None, str(r))

    # 11) multi_stop_delivery
    pc = stu.make_campus_problem("GATE_1", "GATE_1", graph, coords, bridges, time_of_day=9)
    path, cost = stu.multi_stop_delivery(pc, ["PMB", "Library", "DB"], stu.heuristic_euclidean)
    check("multi_return_tuple", isinstance((path, cost), tuple), "")
    check("multi_path_is_list", isinstance(path, list), type(path).__name__)
    check("multi_cost_is_number", isinstance(cost, (int, float)), type(cost).__name__)
    check("multi_start_end", len(path) > 0 and path[0] == "GATE_1" and path[-1] == "GATE_1", str(path[:2] + path[-2:] if len(path) >= 2 else path))
    check("multi_contains_all_targets", all(x in path for x in ["PMB", "Library", "DB"]), "")
    check("multi_path_valid", stu.validate_path(pc, path), "")

    st = "GATE_1"
    locs = ["PMB", "Library", "DB"]
    hr = 9
    best_cost = float("inf")
    for perm in itertools.permutations(locs):
        order = [st, *perm, st]
        total = 0
        feasible = True
        for i in range(4):
            pr = stu.make_campus_problem(order[i], order[i + 1], graph, coords, bridges, time_of_day=hr)
            _, c = dijkstra(pr, order[i], order[i + 1])
            if math.isinf(c):
                feasible = False
                break
            total += c
        if feasible and total < best_cost:
            best_cost = total
    check("multi_sample_optimal_cost", abs(cost - best_cost) < 1e-9, f"got {cost}, expected {best_cost}")

    random.seed(1037)
    ok_multi = True
    for _ in range(200):
        st = random.choice(all_nodes)
        locs = random.sample([n for n in all_nodes if n != st], 3)
        hr = random.choice([6, 9, 12, 18])
        pr = stu.make_campus_problem(st, st, graph, coords, bridges, time_of_day=hr)
        pth, cst = stu.multi_stop_delivery(pr, locs, stu.heuristic_euclidean)
        if not isinstance(pth, list) or not isinstance(cst, (int, float)):
            ok_multi = False
            break
        if len(pth) == 0 or pth[0] != st or pth[-1] != st:
            ok_multi = False
            break
        if not all(x in pth for x in locs):
            ok_multi = False
            break
        if not stu.validate_path(pr, pth):
            ok_multi = False
            break
        brute = float("inf")
        for perm in itertools.permutations(locs):
            order = [st, *perm, st]
            total = 0
            for i in range(4):
                pr2 = stu.make_campus_problem(order[i], order[i + 1], graph, coords, bridges, time_of_day=hr)
                _, cc = dijkstra(pr2, order[i], order[i + 1])
                total += cc
            brute = min(brute, total)
        if abs(cst - brute) > 1e-9:
            ok_multi = False
            break
    check("multi_random_200_optimal_and_valid", ok_multi, "")

    # 12) map-level consistency
    check("map_nodes_30", len(graph) == 30, str(len(graph)))
    check("map_edges_92", sum(len(v) for v in graph.values()) == 92, str(sum(len(v) for v in graph.values())))
    check("map_bridges_3", len(bridges) == 3, str(len(bridges)))

    # 13) imports should stay standard-library-only pattern from template
    mod = ast.parse(new_src)
    imports = []
    for n in mod.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                imports.append(a.name)
        elif isinstance(n, ast.ImportFrom):
            imports.append(f"{n.module}:{','.join(a.name for a in n.names)}")
    check("imports_only_expected", set(imports) == {"json", "math", "itertools:permutations"}, str(imports))

    # 14) strict A* exhaustive check: all pairs x all 24 hours
    ok_all_hours = True
    for hr in range(24):
        for s in all_nodes:
            for g in all_nodes:
                pr = stu.make_campus_problem(s, g, graph, coords, bridges, time_of_day=hr)
                r = stu.a_star_search(pr, stu.heuristic_euclidean)
                if r is None:
                    ok_all_hours = False
                    break
                best = dijkstra_cost_only(pr, s, g)
                if abs(r["cost"] - best) > 1e-9:
                    ok_all_hours = False
                    break
            if not ok_all_hours:
                break
        if not ok_all_hours:
            break
    check("a_star_all_pairs_all_hours", ok_all_hours, "")

    # 15) disconnected multi-stop edge case: keep return format (list, number)
    tiny_graph = {"A": {"B": 1}, "B": {"A": 1}, "C": {}, "D": {}}
    tiny_coords = {"A": [0, 0], "B": [1, 0], "C": [2, 0], "D": [3, 0]}
    tiny_bridges = {}
    tiny_p = stu.make_campus_problem("A", "A", tiny_graph, tiny_coords, tiny_bridges, time_of_day=9)
    t_path, t_cost = stu.multi_stop_delivery(tiny_p, ["B", "C", "D"], stu.heuristic_euclidean)
    check("multi_disconnected_path_list", isinstance(t_path, list), type(t_path).__name__)
    check("multi_disconnected_cost_number", isinstance(t_cost, (int, float)), type(t_cost).__name__)
    check("multi_disconnected_expected", t_path == [] and math.isinf(t_cost), f"path={t_path}, cost={t_cost}")

    total = len(REPORT)
    passed = sum(1 for _, s, _ in REPORT if s == "PASS")
    failed = len(FAIL)
    print(f"TOTAL_CHECKS {total}")
    print(f"PASS_COUNT {passed}")
    print(f"TOTAL_FAIL {failed}")

    if failed:
        print("FAILED_ITEMS:")
        for name, detail in FAIL:
            print(f"- {name}: {detail}")
        return 1

    key_items = [
        "find_best_scale_correct",
        "a_star_optimal_all_pairs_hours_6_9_12_18",
        "a_star_all_pairs_all_hours",
        "multi_random_200_optimal_and_valid",
        "helper_unchanged_make_node",
        "helper_unchanged_reconstruct_path",
        "helper_unchanged_expand",
        "helper_unchanged_load_campus_map",
        "helper_unchanged_validate_path",
        "helper_unchanged_is_cycle",
    ]
    print("KEY_RESULTS:")
    for k in key_items:
        rec = [r for r in REPORT if r[0] == k][0]
        print(f"- {rec[0]}: {rec[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
