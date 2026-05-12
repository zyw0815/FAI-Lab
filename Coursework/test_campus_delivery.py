"""
Edge case tests for campus_delivery.py — run before submitting.
Usage: python test_campus_delivery.py
"""

from campus_delivery import *
import ast
import inspect
import itertools
import heapq
import random

GREEN = '\033[92m'
RED   = '\033[91m'
CYAN  = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD  = '\033[1m'

# ---------------------------------------------------------------------------
results = []  # (section, test_name, pass/fail, detail)

def test(section, name, condition, detail=''):
    ok = bool(condition)
    results.append((section, name, ok, detail))
    mark = f'{GREEN}PASS{RESET}' if ok else f'{RED}FAIL{RESET}'
    print(f'  {mark}  {name}')
    if not ok and detail:
        print(f'         {RED}→ {detail}{RESET}')

# ---------------------------------------------------------------------------
# Reference Dijkstra (ground truth for optimality checks)
# ---------------------------------------------------------------------------

def dijkstra_path(problem, start, goal):
    """Reference shortest-path using student's campus_action_cost."""
    pq = [(0, start)]
    dist = {start: 0}
    parent = {start: None}
    while pq:
        g, u = heapq.heappop(pq)
        if g != dist[u]:
            continue
        if u == goal:
            break
        for v in problem['graph'].get(u, {}):
            w = campus_action_cost(problem, u, v)
            ng = g + w
            if ng < dist.get(v, float('inf')):
                dist[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng, v))
    if goal not in dist:
        return None, float('inf')
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, dist[goal]


def dijkstra_cost(problem, start, goal):
    pq = [(0, start)]
    dist = {start: 0}
    while pq:
        g, u = heapq.heappop(pq)
        if g != dist[u]:
            continue
        if u == goal:
            return g
        for v in problem['graph'].get(u, {}):
            w = campus_action_cost(problem, u, v)
            ng = g + w
            if ng < dist.get(v, float('inf')):
                dist[v] = ng
                heapq.heappush(pq, (ng, v))
    return float('inf')


def run_tests():
    data = load_campus_map()
    g = data['campus_graph']
    co = data['campus_coords']
    b = data['campus_bridges']
    all_nodes = list(co.keys())

    # ==================== Part A: Problem Formulation (10pts) ====================
    print(f'\n{BOLD}{CYAN}Part A — Problem Formulation (10pts){RESET}')

    p = make_campus_problem('GATE_1', 'PMB', g, co, b, time_of_day=9)
    test('A', 'returns dict with all required keys',
         all(k in p for k in ['initial', 'goal', 'graph', 'coords', 'bridges', 'time_of_day']))

    test('A', f"p['initial'] == 'GATE_1'", p['initial'] == 'GATE_1')
    test('A', f"p['goal'] == 'PMB'", p['goal'] == 'PMB')
    test('A', 'p[\'time_of_day\'] stores boundary value 0',
         make_campus_problem('A','B',g,co,b,time_of_day=0)['time_of_day'] == 0)
    test('A', 'p[\'time_of_day\'] stores boundary value 23',
         make_campus_problem('A','B',g,co,b,time_of_day=23)['time_of_day'] == 23)
    test('A', 'start==goal creates valid problem',
         make_campus_problem('PMB','PMB',g,co,b,12)['initial'] ==
         make_campus_problem('PMB','PMB',g,co,b,12)['goal'])

    test('A', 'is_goal returns True for goal', is_goal(p, 'PMB') == True)
    test('A', 'is_goal returns False for non-goal', is_goal(p, 'GATE_1') == False)

    acts = campus_get_actions(p, 'GATE_1')
    test('A', 'campus_get_actions returns list', isinstance(acts, list))
    test('A', 'each action is a tuple (not list)',
         all(isinstance(a, tuple) for a in acts),
         f'got: {[type(a).__name__ for a in acts]}')

    # ==================== Part B: Bridge Congestion (10pts) ====================
    print(f'\n{BOLD}{CYAN}Part B — Bridge Congestion (10pts){RESET}')

    bridge_cases = [
        ('Bridge_South off-peak (6am)',   'PMB', 'Bridge_South',   6, 245),
        ('Bridge_South peak +500 (9am)',  'PMB', 'Bridge_South',   9, 745),
        ('Bridge_South peak +400 (10am)', 'PMB', 'Bridge_South',  10, 645),
        ('Bridge_North peak +500 (12pm)', 'TRENT', 'Bridge_North', 12, 881),
        ('Bridge_Middle peak +400 (7pm)', 'PB', 'Bridge_Middle',   19, 669),
    ]
    for desc, src, dst, hour, expected in bridge_cases:
        pb = make_campus_problem(src, dst, g, co, b, time_of_day=hour)
        got = campus_action_cost(pb, src, dst)
        test('B', desc, got == expected, f'expected {expected}, got {got}')

    pb = make_campus_problem('Bridge_South', 'PMB', g, co, b, time_of_day=9)
    got = campus_action_cost(pb, 'Bridge_South', 'PMB')
    test('B', 'leaving bridge → no penalty (Bridge_South→PMB at 9am)',
         got == 245, f'expected 245, got {got}')

    test('B', 'normal edge (GATE_1→IAMET, no bridge)',
         campus_action_cost(make_campus_problem('G','I',g,co,b,9), 'GATE_1','IAMET') == 431,
         f'got {campus_action_cost(make_campus_problem("G","I",g,co,b,9), "GATE_1","IAMET")}')

    pb = make_campus_problem('GATE_1', 'Library', g, co, b, 9)
    got = campus_action_cost(pb, 'GATE_1', 'Library')
    test('B', 'invalid edge returns math.inf', got == float('inf'),
         f'expected inf, got {got}')
    test('B', 'invalid state returns math.inf',
         campus_action_cost(make_campus_problem('G','I',g,co,b,9),
                            'NonExistent', 'PMB') == float('inf'))

    # ==================== Part C: Heuristic (10pts) ====================
    print(f'\n{BOLD}{CYAN}Part C — Heuristic (10pts){RESET}')

    goals = ['PMB', 'GATE_2', 'Staff_Hotel', 'Health_Centre', 'Sports_Centre']
    for goal in goals:
        pg = make_campus_problem(goal, goal, g, co, b, 6)
        hval = heuristic_euclidean(goal, pg)
        test('C', f'h({goal}) == 0', hval == 0.0,
             f'got {hval}' if hval != 0.0 else '')

    p = make_campus_problem('GATE_1', 'PMB', g, co, b, 6)
    hval = heuristic_euclidean('GATE_1', p)
    test('C', 'h(GATE_1) >= 0', hval >= 0)
    test('C', 'h returns float', isinstance(hval, float),
         f'got type {type(hval).__name__}')

    routes = [('GATE_1','PMB'), ('GATE_1','GATE_2'), ('Library','Staff_Hotel'),
              ('Hub','Health_Centre'), ('TRENT','Sports_Centre')]
    for start, goal in routes:
        pg = make_campus_problem(start, goal, g, co, b, 6)
        hval = heuristic_euclidean(start, pg)
        r = a_star_search(pg, heuristic_euclidean)
        test('C', f'admissible: h({start}→{goal}) ≤ true_cost',
             hval <= r['cost'],
             f'h={hval:.0f}, true_cost={r["cost"]}' if hval > r['cost'] else '')

    # ==================== Part D: A* Search (50pts) ====================
    print(f'\n{BOLD}{CYAN}Part D — A* Search (50pts){RESET}')

    r = a_star_search(make_campus_problem('PMB', 'PMB', g, co, b, 9), heuristic_euclidean)
    test('D', 'start==goal returns cost=0',
         r is not None and r['cost'] == 0 and reconstruct_path(r) == ['PMB'],
         f'cost={r["cost"] if r else None}, path={reconstruct_path(r) if r else None}')

    r = a_star_search(make_campus_problem('GATE_1', 'PMB', g, co, b, 9), heuristic_euclidean)
    test('D', 'GATE_1→PMB cost=615', r is not None and r['cost'] == 615,
         f'cost={r["cost"] if r else None}')

    r6 = a_star_search(make_campus_problem('PMB', 'Sports_Centre', g, co, b, 6),
                       heuristic_euclidean)
    path6 = reconstruct_path(r6) if r6 else []
    test('D', '6am PMB→Sports: uses Bridge_South (no congestion)',
         'Bridge_South' in path6, f'path={path6}')

    r9 = a_star_search(make_campus_problem('PMB', 'Sports_Centre', g, co, b, 9),
                       heuristic_euclidean)
    path9 = reconstruct_path(r9) if r9 else []
    test('D', '9am PMB→Sports: avoids Bridge_South (congested)',
         'Bridge_South' not in path9, f'path={path9}')

    test('D', '6am cost < 9am cost (congestion penalty works)',
         r6 and r9 and r6['cost'] < r9['cost'],
         f'6am={r6["cost"] if r6 else "?"}, 9am={r9["cost"] if r9 else "?"}')

    r_ucs = a_star_search(make_campus_problem('GATE_1', 'PMB', g, co, b, 9),
                          lambda s, p: 0)
    test('D', 'zero heuristic (UCS mode) still finds optimal',
         r_ucs is not None and r_ucs['cost'] == 615,
         f'cost={r_ucs["cost"] if r_ucs else None}')

    r = a_star_search(make_campus_problem('Bridge_Middle', 'PB', g, co, b, 12),
                      heuristic_euclidean)
    test('D', 'bridge→non-bridge: no penalty for leaving',
         r is not None and r['cost'] == 269,
         f'cost={r["cost"] if r else None}')

    r = a_star_search(make_campus_problem('GATE_2', 'IEB', g, co, b, 6), heuristic_euclidean)
    test('D', 'east→west: must cross at least one bridge',
         r is not None and any(n.startswith('Bridge') for n in reconstruct_path(r)),
         f'path={reconstruct_path(r) if r else None}')

    r = a_star_search(make_campus_problem('GATE_4', 'Academician_Park', g, co, b, 15),
                      heuristic_euclidean)
    p_long = make_campus_problem('GATE_4', 'Academician_Park', g, co, b, 15)
    test('D', 'long diagonal route: valid path',
         r is not None and validate_path(p_long, reconstruct_path(r)),
         f'cost={r["cost"] if r else "None"}')

    # At 13pm, Bridge_North/Middle have +400, Bridge_South is clear
    r13 = a_star_search(make_campus_problem('Sports_Centre', 'DB', g, co, b, 13),
                        heuristic_euclidean)
    path13 = reconstruct_path(r13) if r13 else []
    test('D', '13pm Sport→DB: bypasses mealtime bridges via Bridge_South',
         r13 is not None and 'Bridge_North' not in path13 and 'Bridge_Middle' not in path13,
         f'path={path13}')

    # ==================== Challenge A: Best Scale (10pts) ====================
    print(f'\n{BOLD}{CYAN}Challenge A — Best Scale Factor (10pts){RESET}')

    exp = make_campus_problem('GATE_1', 'PMB', g, co, b, time_of_day=6)
    best = find_best_scale(exp)
    candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    test('CA', 'returns float', isinstance(best, float),
         f'got type {type(best).__name__}')
    test('CA', f'value in candidate list (got {best})',
         best in candidates, f'best={best}')

    exp9 = make_campus_problem('GATE_1', 'PMB', g, co, b, time_of_day=9)
    best9 = find_best_scale(exp9)
    test('CA', 'uses time_of_day=6 regardless of input',
         best9 == best, f'time_of_day=9 gave {best9}, 6 gave {best}')

    r25 = a_star_search(exp, lambda s, p: heuristic_scaled(s, p, best))
    test('CA', f'A* with best scale ({best}) still optimal (cost=615)',
         r25 is not None and r25['cost'] == 615,
         f'cost={r25["cost"] if r25 else None}')

    r10 = a_star_search(exp, lambda s, p: heuristic_scaled(s, p, 1.0))
    test('CA', 'scale=1.0 is optimal (sanity check)',
         r10 is not None and r10['cost'] == 615)

    # --- find_best_scale: independently verify via Dijkstra ---
    test_goals = ['PMB', 'GATE_2', 'Staff_Hotel', 'Health_Centre', 'Sports_Centre']
    true_cost = {}
    for goal in test_goals:
        pr = make_campus_problem('dummy', goal, g, co, b, time_of_day=6)
        for s in all_nodes:
            _, c = dijkstra_path(pr, s, goal)
            true_cost[(s, goal)] = c
    expected_best = 1.0
    for scale in reversed(candidates):
        good = True
        for goal in test_goals:
            pr = make_campus_problem('dummy', goal, g, co, b, time_of_day=6)
            for s in all_nodes:
                h = heuristic_scaled(s, pr, scale)
                if h - true_cost[(s, goal)] > 1e-9:
                    good = False
                    break
            if not good:
                break
        if good:
            expected_best = scale
            break
    test('CA', 'find_best_scale correct (Dijkstra-verified)',
         abs(best - expected_best) < 1e-12,
         f'got {best}, expected {expected_best}')

    # ==================== Challenge B: Multi-Stop (10pts) ====================
    print(f'\n{BOLD}{CYAN}Challenge B — Multi-Stop Delivery (10pts){RESET}')

    pc = make_campus_problem('GATE_1', 'GATE_1', g, co, b, time_of_day=9)
    path, cost = multi_stop_delivery(pc, ['PMB', 'Library', 'DB'], heuristic_euclidean)

    test('CB', 'returns tuple (not list)', isinstance((path, cost), tuple))
    test('CB', 'path is list', isinstance(path, list))
    test('CB', 'cost is number', isinstance(cost, (int, float)))
    test('CB', 'starts at GATE_1', path[0] == 'GATE_1' if path else False,
         f'path[0]={path[0] if path else None}')
    test('CB', 'ends at GATE_1', path[-1] == 'GATE_1' if path else False,
         f'path[-1]={path[-1] if path else None}')
    for loc in ['PMB', 'Library', 'DB']:
        test('CB', f'visits {loc}', loc in path if path else False)
    test('CB', 'no duplicate consecutive nodes',
         all(path[i] != path[i+1] for i in range(len(path)-1)) if path else False)

    # Alternate route
    pc2 = make_campus_problem('TRENT', 'TRENT', g, co, b, time_of_day=15)
    path2, _ = multi_stop_delivery(pc2,
        ['Sports_Centre', 'GATE_4', 'Staff_Hotel'], heuristic_euclidean)
    test('CB', 'alternate start/locations: TRENT → ... → TRENT',
         path2 is not None and path2[0] == 'TRENT' and path2[-1] == 'TRENT',
         f'path[0]={path2[0] if path2 else None}, path[-1]={path2[-1] if path2 else None}')
    for loc in ['Sports_Centre', 'GATE_4', 'Staff_Hotel']:
        test('CB', f'alternate visits {loc}',
             loc in path2 if path2 else False)

    # Congestion impact
    pc6 = make_campus_problem('GATE_1', 'GATE_1', g, co, b, time_of_day=6)
    _, c6 = multi_stop_delivery(pc6, ['PMB', 'Library', 'DB'], heuristic_euclidean)
    pc9 = make_campus_problem('GATE_1', 'GATE_1', g, co, b, time_of_day=9)
    _, c9 = multi_stop_delivery(pc9, ['PMB', 'Library', 'DB'], heuristic_euclidean)
    test('CB', 'congestion changes multi-stop cost',
         c6 != c9, f'6am={c6}, 9am={c9}')
    test('CB', '6am cost ≤ 9am cost (congestion adds, not removes)',
         c6 <= c9, f'6am={c6}, 9am={c9}')

    # --- Disconnected / missing location edge cases ---
    print(f'\n  {YELLOW}disconnected location edge cases:{RESET}')

    # Build modified graph with an isolated node
    g_iso = {k: dict(v) for k, v in g.items()}
    g_iso['Isolated_Node'] = {}  # exists in graph, but no edges
    co_iso = dict(co)
    co_iso['Isolated_Node'] = [300, 300]  # arbitrary coords

    # CB-disco1: isolated node (in graph, in coords, no edges)
    pc_iso = make_campus_problem('GATE_1', 'GATE_1', g_iso, co_iso, b, time_of_day=9)
    path_iso, cost_iso = multi_stop_delivery(pc_iso, ['PMB', 'Isolated_Node', 'DB'], heuristic_euclidean)
    test('CB', 'isolated node in graph: returns empty list (not crash)',
         path_iso == [] and cost_iso == float('inf'),
         f'path={path_iso}, cost={cost_iso}')
    test('CB', 'isolated node: path is list (isinstance check)',
         isinstance(path_iso, list),
         f'got type {type(path_iso).__name__}')

    # CB-disco2: node missing from graph entirely
    pc_miss = make_campus_problem('GATE_1', 'GATE_1', g, co, b, time_of_day=9)
    path_miss, cost_miss = multi_stop_delivery(pc_miss, ['PMB', 'Fake_Location', 'DB'], heuristic_euclidean)
    test('CB', 'node not in graph: no crash, returns empty list',
         path_miss == [] and cost_miss == float('inf'),
         f'path={path_miss}, cost={cost_miss}')
    test('CB', 'missing node: path is list (isinstance check)',
         isinstance(path_miss, list),
         f'got type {type(path_miss).__name__}')

    # CB-disco3: all 3 locations unreachable → still no crash
    pc_all = make_campus_problem('GATE_1', 'GATE_1', g_iso, co_iso, b, time_of_day=9)
    path_all, cost_all = multi_stop_delivery(pc_all, ['Isolated_Node', 'Fake_A', 'Fake_B'], heuristic_euclidean)
    test('CB', 'all 3 locations unreachable: returns empty list + inf',
         path_all == [] and cost_all == float('inf'),
         f'path={path_all}, cost={cost_all}')

    # CB-disco4: campus_get_actions handles missing state gracefully
    p_test = make_campus_problem('GATE_1', 'PMB', g, co, b, 9)
    acts = campus_get_actions(p_test, 'NonExistent')
    test('CB', 'campus_get_actions on missing state: returns []',
         acts == [], f'got {acts}')

    # CB-disco5: tiny disconnected graph
    tiny_graph = {'A': {'B': 1}, 'B': {'A': 1}, 'C': {}, 'D': {}}
    tiny_coords = {'A': [0, 0], 'B': [1, 0], 'C': [2, 0], 'D': [3, 0]}
    tiny_bridges = {}
    tiny_p = make_campus_problem('A', 'A', tiny_graph, tiny_coords, tiny_bridges, time_of_day=9)
    t_path, t_cost = multi_stop_delivery(tiny_p, ['B', 'C', 'D'], heuristic_euclidean)
    test('CB', 'tiny graph: disconnected path is list', isinstance(t_path, list),
         f'got type {type(t_path).__name__}')
    test('CB', 'tiny graph: disconnected cost is number', isinstance(t_cost, (int, float)),
         f'got type {type(t_cost).__name__}')
    test('CB', 'tiny graph: returns empty list + inf', t_path == [] and math.isinf(t_cost),
         f'path={t_path}, cost={t_cost}')

    # --- Multi-stop optimality (Dijkstra brute-force) ---
    test('CB', 'multi_stop path is valid', validate_path(pc, path), 'path validation failed')
    brute = float('inf')
    for perm in itertools.permutations(['PMB', 'Library', 'DB']):
        order = ['GATE_1'] + list(perm) + ['GATE_1']
        total = 0
        feasible = True
        for i in range(4):
            pr = make_campus_problem(order[i], order[i + 1], g, co, b, time_of_day=9)
            _, c = dijkstra_path(pr, order[i], order[i + 1])
            if math.isinf(c):
                feasible = False
                break
            total += c
        if feasible and total < brute:
            brute = total
    test('CB', 'multi_stop cost is optimal (Dijkstra brute-force)',
         abs(cost - brute) < 1e-9,
         f'got {cost}, brute-force optimal {brute}')

    # ===================================================================
    # EXTRA VERIFICATION from coursework_assertion_tests.py
    # ===================================================================

    # ==================== Part E: Helper Integrity ====================
    print(f'\n{BOLD}{CYAN}Part E — Helper Integrity{RESET}')

    helper_names = [
        'make_node', 'reconstruct_path', 'expand',
        'load_campus_map', 'validate_path', 'is_cycle',
    ]
    old_src = open('old_delivery.py', 'r', encoding='utf-8').read()
    new_src = open('campus_delivery.py', 'r', encoding='utf-8').read()
    old_ast = ast.parse(old_src)
    new_ast = ast.parse(new_src)
    old_funcs = {n.name: n for n in old_ast.body if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
    for h in helper_names:
        o_src = ast.get_source_segment(old_src, old_funcs[h])
        n_src = ast.get_source_segment(new_src, new_funcs[h])
        test('E', f'helper_unchanged_{h}', o_src == n_src,
             f'{h} differs from old_delivery.py' if o_src != n_src else '')

    # ==================== Part F: Function Signatures ====================
    print(f'\n{BOLD}{CYAN}Part F — Function Signatures{RESET}')

    expected_sigs = {
        'make_campus_problem': ['start', 'goal', 'campus_graph', 'campus_coords',
                                'campus_bridges', 'time_of_day'],
        'is_goal': ['problem', 'state'],
        'campus_get_actions': ['problem', 'state'],
        'campus_action_cost': ['problem', 'state', 'action'],
        'heuristic_euclidean': ['state', 'problem'],
        'heuristic_scaled': ['state', 'problem', 'scale'],
        'find_best_scale': ['problem'],
        'a_star_search': ['problem', 'heuristic_fn'],
        'multi_stop_delivery': ['problem', 'locations_to_visit', 'heuristic_fn'],
    }
    for fn, args in expected_sigs.items():
        func = globals()[fn]
        actual = list(inspect.signature(func).parameters.keys())
        test('F', f'signature_{fn}', actual == args,
             f'expected {args}, got {actual}')

    # ==================== Part G: Exhaustive Action Cost ====================
    print(f'\n{BOLD}{CYAN}Part G — Exhaustive Action Cost{RESET}')

    p6 = make_campus_problem('GATE_1', 'PMB', g, co, b, time_of_day=6)
    ok_offpeak = True
    for u in g:
        for v, base in g[u].items():
            got = campus_action_cost(p6, u, v)
            pen = b[v]['congestion'].get('6', 0) if v in b else 0
            if abs(got - (base + pen)) > 1e-9:
                ok_offpeak = False
                break
        if not ok_offpeak:
            break
    test('G', 'all edges correct at off-peak (6am)', ok_offpeak, '')

    ok_bridge_hours = True
    for hour in range(24):
        pr = make_campus_problem('X', 'Y', g, co, b, time_of_day=hour)
        hs = str(hour)
        for bname, bdata in b.items():
            preds = [u for u in g if bname in g[u]]
            if not preds:
                continue
            u = preds[0]
            base = g[u][bname]
            exp = base + bdata['congestion'].get(hs, 0)
            got = campus_action_cost(pr, u, bname)
            if abs(got - exp) > 1e-9:
                ok_bridge_hours = False
                break
        if not ok_bridge_hours:
            break
    test('G', 'bridge congestion correct for all 24 hours', ok_bridge_hours, '')

    # campus_get_actions exact match for every node
    ok_actions = True
    for s in g:
        acts = campus_get_actions(p, s)
        expected_acts = [(n, n) for n in g[s].keys()]
        if acts != expected_acts:
            ok_actions = False
            break
    test('G', 'campus_get_actions exact match for all nodes', ok_actions, '')

    # ==================== Part H: Exhaustive Heuristic ====================
    print(f'\n{BOLD}{CYAN}Part H — Exhaustive Heuristic{RESET}')

    ok_type = True
    ok_nonneg = True
    ok_goal0 = True
    for goal in all_nodes:
        pr = make_campus_problem('dummy', goal, g, co, b, time_of_day=6)
        for s in all_nodes:
            h = heuristic_euclidean(s, pr)
            if not isinstance(h, float):
                ok_type = False
            if h < -1e-12:
                ok_nonneg = False
        if abs(heuristic_euclidean(goal, pr) - 0.0) > 1e-12:
            ok_goal0 = False
    test('H', 'heuristic always returns float', ok_type, '')
    test('H', 'heuristic always non-negative', ok_nonneg, '')
    test('H', 'heuristic zero for every goal node', ok_goal0, '')

    # Admissible for ALL pairs (off-peak: time_of_day=6)
    ok_adm = True
    for goal in all_nodes:
        pr = make_campus_problem('dummy', goal, g, co, b, time_of_day=6)
        for s in all_nodes:
            _, true_c = dijkstra_path(pr, s, goal)
            h = heuristic_euclidean(s, pr)
            if h - true_c > 1e-9:
                ok_adm = False
                break
        if not ok_adm:
            break
    test('H', 'heuristic admissible for ALL pairs (off-peak)', ok_adm, '')

    # heuristic_scaled type and value
    hs = heuristic_scaled('GATE_1', p, 3.0)
    he_val = heuristic_euclidean('GATE_1', p)
    test('H', 'heuristic_scaled returns float', isinstance(hs, float),
         f'got type {type(hs).__name__}')
    test('H', 'heuristic_scaled = scale * Euclidean',
         abs(hs - 3.0 * he_val) < 1e-9,
         f'{hs} vs {3.0 * he_val}')

    # ==================== Part I: Exhaustive A* Search ====================
    print(f'\n{BOLD}{CYAN}Part I — A* Search Exhaustive{RESET}')

    # A* start == goal returns dict with correct state and cost=0
    r_self = a_star_search(make_campus_problem('PMB', 'PMB', g, co, b, 9),
                           heuristic_euclidean)
    test('I', 'A* start==goal returns dict with state==goal and cost==0',
         isinstance(r_self, dict) and r_self.get('state') == 'PMB' and r_self.get('cost') == 0,
         str(r_self))

    # A* returns None when no path exists
    small_graph = {'A': {'B': 1}, 'B': {}, 'C': {}}
    small_coords = {'A': [0, 0], 'B': [1, 0], 'C': [2, 0]}
    small_bridges = {}
    pr_small = make_campus_problem('A', 'C', small_graph, small_coords, small_bridges, time_of_day=9)
    r_none = a_star_search(pr_small, heuristic_euclidean)
    test('I', 'A* returns None when no path exists', r_none is None, str(r_none))

    # Optimal for all pairs at key hours (6, 9, 12, 18)
    print(f'  {YELLOW}all-pairs optimality at hours [6, 9, 12, 18] (may take a moment)...{RESET}')
    hours_key = [6, 9, 12, 18]
    ok_opt = True
    for hr in hours_key:
        for s in all_nodes:
            for g_node in all_nodes:
                pr = make_campus_problem(s, g_node, g, co, b, time_of_day=hr)
                r = a_star_search(pr, heuristic_euclidean)
                if r is None:
                    ok_opt = False
                    break
                path = reconstruct_path(r)
                if not isinstance(r, dict) or r.get('state') != g_node:
                    ok_opt = False
                    break
                if s != g_node and not validate_path(pr, path):
                    ok_opt = False
                    break
                _, bestc = dijkstra_path(pr, s, g_node)
                if abs(r['cost'] - bestc) > 1e-9:
                    ok_opt = False
                    break
            if not ok_opt:
                break
        if not ok_opt:
            break
    test('I', 'A* optimal for all pairs at hours [6,9,12,18]', ok_opt, '')

    # Optimal for all pairs at ALL 24 hours
    print(f'  {YELLOW}all-pairs optimality at all 24 hours (this takes longer)...{RESET}')
    ok_all_hours = True
    for hr in range(24):
        for s in all_nodes:
            for g_node in all_nodes:
                pr = make_campus_problem(s, g_node, g, co, b, time_of_day=hr)
                r = a_star_search(pr, heuristic_euclidean)
                if r is None:
                    ok_all_hours = False
                    break
                bestc = dijkstra_cost(pr, s, g_node)
                if abs(r['cost'] - bestc) > 1e-9:
                    ok_all_hours = False
                    break
            if not ok_all_hours:
                break
        if not ok_all_hours:
            break
    test('I', 'A* optimal for all pairs at ALL 24 hours', ok_all_hours, '')

    # ==================== Part J: Multi-Stop Random 200 ====================
    print(f'\n{BOLD}{CYAN}Part J — Multi-Stop Random 200{RESET}')
    print(f'  {YELLOW}random multi-stop validation (200 runs)...{RESET}')

    random.seed(1037)
    ok_multi = True
    for _ in range(200):
        st = random.choice(all_nodes)
        locs = random.sample([n for n in all_nodes if n != st], 3)
        hr = random.choice([6, 9, 12, 18])
        pr = make_campus_problem(st, st, g, co, b, time_of_day=hr)
        pth, cst = multi_stop_delivery(pr, locs, heuristic_euclidean)
        if not isinstance(pth, list) or not isinstance(cst, (int, float)):
            ok_multi = False
            break
        if len(pth) == 0 or pth[0] != st or pth[-1] != st:
            ok_multi = False
            break
        if not all(x in pth for x in locs):
            ok_multi = False
            break
        if not validate_path(pr, pth):
            ok_multi = False
            break
        brute = float('inf')
        for perm in itertools.permutations(locs):
            order = [st] + list(perm) + [st]
            total = 0
            for i in range(4):
                pr2 = make_campus_problem(order[i], order[i + 1], g, co, b, time_of_day=hr)
                _, cc = dijkstra_path(pr2, order[i], order[i + 1])
                total += cc
            brute = min(brute, total)
        if abs(cst - brute) > 1e-9:
            ok_multi = False
            break
    test('J', 'multi_stop random 200: all valid', ok_multi, '')

    # ==================== Part K: Map & Imports Consistency ====================
    print(f'\n{BOLD}{CYAN}Part K — Map & Imports Consistency{RESET}')

    test('K', 'map has 30 nodes', len(g) == 30, str(len(g)))
    test('K', 'map has 92 edges', sum(len(v) for v in g.values()) == 92,
         str(sum(len(v) for v in g.values())))
    test('K', 'map has 3 bridges', len(b) == 3, str(len(b)))

    # Imports: only json, math, itertools:permutations
    new_src = open('campus_delivery.py', 'r', encoding='utf-8').read()
    new_ast = ast.parse(new_src)
    imports = []
    for n in new_ast.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                imports.append(a.name)
        elif isinstance(n, ast.ImportFrom):
            imports.append(f'{n.module}:{",".join(a.name for a in n.names)}')
    expected_imports = {'json', 'math', 'itertools:permutations'}
    test('K', 'imports match template (json, math, itertools:permutations)',
         set(imports) == expected_imports,
         f'got {set(imports)}, expected {expected_imports}')

    # is_goal return type is bool
    res_goal = is_goal(p, 'PMB')
    res_not = is_goal(p, 'GATE_1')
    test('K', 'is_goal returns bool (True)', isinstance(res_goal, bool) and res_goal is True,
         f'{type(res_goal).__name__}={res_goal}')
    test('K', 'is_goal returns bool (False)', isinstance(res_not, bool) and res_not is False,
         f'{type(res_not).__name__}={res_not}')

    return best, cost, c6


# ===================== Run & Summary =====================

print(f'{BOLD}{"─" * 60}{RESET}')
print(f'{BOLD}  Campus Delivery — Pre-Submission Test Suite{RESET}')
print(f'{BOLD}{"─" * 60}{RESET}')

best, multi_9am, multi_6am = run_tests()

# Summary
total = len(results)
passed = sum(1 for _, _, ok, _ in results if ok)
failed = total - passed

print(f'\n{BOLD}{"─" * 60}{RESET}')

sections_order = ['A', 'B', 'C', 'D', 'CA', 'CB', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
section_names = {
    'A': 'Part A: Problem Formulation',
    'B': 'Part B: Bridge Congestion',
    'C': 'Part C: Heuristic',
    'D': 'Part D: A* Search',
    'CA': 'Challenge A: Best Scale',
    'CB': 'Challenge B: Multi-Stop',
    'E': 'Part E: Helper Integrity',
    'F': 'Part F: Function Signatures',
    'G': 'Part G: Exhaustive Action Cost',
    'H': 'Part H: Exhaustive Heuristic',
    'I': 'Part I: A* Search Exhaustive',
    'J': 'Part J: Multi-Stop Random 200',
    'K': 'Part K: Map & Imports',
}
section_pts = {'A': 10, 'B': 10, 'C': 10, 'D': 50, 'CA': 10, 'CB': 10,
               'E': '—', 'F': '—', 'G': '—', 'H': '—', 'I': '—', 'J': '—', 'K': '—'}

for sec in sections_order:
    sec_tests = [r for r in results if r[0] == sec]
    sec_passed = sum(1 for _, _, ok, _ in sec_tests if ok)
    sec_total = len(sec_tests)
    status = f'{GREEN}✓{RESET}' if sec_passed == sec_total else f'{RED}✗{RESET}'
    bar = f'{GREEN}{"█" * sec_passed}{RED}{"█" * (sec_total - sec_passed)}{RESET}'
    pts = section_pts[sec]
    print(f'  {status} {section_names[sec]:35s}  {sec_passed}/{sec_total}  {bar}  [{pts}pts]')

print(f'{BOLD}{"─" * 60}{RESET}')
if failed == 0:
    print(f'\n  {GREEN}{BOLD}ALL {total} TESTS PASSED{RESET}')
else:
    print(f'\n  {RED}{BOLD}{failed}/{total} TESTS FAILED{RESET}')
    print(f'\n  {RED}Failed:{RESET}')
    for sec, name, ok, detail in results:
        if not ok:
            print(f'    {RED}✗{RESET} [{sec}] {name}')
            if detail:
                print(f'       {RED}{detail}{RESET}')

print(f'\n  Key results:')
print(f'    find_best_scale = {best}')
print(f'    multi_stop cost = {multi_9am} (9am) / {multi_6am} (6am)')
print()
