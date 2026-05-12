"""
Campus Delivery Agent -- FAI Coursework

Implement A* search to find optimal paths on a campus map with bridge congestion.

Student: [Your name]
Date: [Submission date]
"""

import json
import math
from itertools import permutations

# =====================================================================
# PART 1: HELPER FUNCTIONS (Provided -- Similar to Lab 3/4)
# =====================================================================

def make_node(state, parent=None, action=None, cost=0, depth=0):
    """Create a node dict for search tree."""
    return {
        'state': state,
        'parent': parent,
        'action': action,
        'cost': cost,
        'depth': depth
    }


def reconstruct_path(node):
    """Reconstruct path from goal node back to start."""
    path = []
    while node:
        path.append(node['state'])
        node = node['parent']
    return list(reversed(path))


def expand(problem, node, get_actions_fn, action_cost_fn):
    """Generate child nodes by applying possible actions."""
    children = []
    for action_name, next_state in get_actions_fn(problem, node['state']):
        cost_increment = action_cost_fn(problem, node['state'], action_name)
        if math.isinf(cost_increment):
            continue
        child = make_node(
            state=next_state, parent=node, action=action_name,
            cost=node['cost'] + cost_increment, depth=node['depth'] + 1
        )
        children.append(child)
    return children


def load_campus_map(filename='campus_map.json'):
    """Load campus map data from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def validate_path(problem, path):
    """Validate that a path follows valid edges. used by the testing case!!"""
    if not path or len(path) < 2:
        return False
    graph = problem['graph']
    for i in range(len(path) - 1):
        if path[i] not in graph or path[i + 1] not in graph[path[i]]:
            return False
    return True


def is_cycle(node, k=30):
    """Check if node's state already appears in its ancestor chain.
    Returns True if a cycle of length <= k is found."""
    ancestor = node['parent']
    depth = 0
    while ancestor is not None and depth < k:
        if ancestor['state'] == node['state']:
            return True
        ancestor = ancestor['parent']
        depth += 1
    return False


# =====================================================================
# PART 2: PROBLEM FORMULATION (Student Implementation)
# =====================================================================

def make_campus_problem(start, goal, campus_graph, campus_coords, campus_bridges, time_of_day=9):
    """
    Create a problem dict that stores all information needed for search.

    Think about: what data does A* need access to during search?
    (Hint: review how make_romania_problem() works in Lab 3)

    Example:
        >>> p = make_campus_problem('GATE_1', 'PMB', graph, coords, bridges, 9)
        >>> p['initial']
        'GATE_1'
        >>> p['goal']
        'PMB'
        >>> p['time_of_day']
        9

    Returns:
        dict: must include 'initial', 'goal', and all data needed by
              other functions (graph, coordinates, bridges, time_of_day)
    """
    # TODO: return a dict containing all the search problem data
    return {}


def is_goal(problem, state):
    """
    Return True if state matches the goal in the problem.

    Example:
        >>> is_goal(problem, 'PMB')   # if goal is PMB
        True
        >>> is_goal(problem, 'GATE_1')
        False
    """
    # TODO: compare state with the goal stored in problem
    return False


def campus_get_actions(problem, state):
    """
    Return available moves from current state as a list of tuples.
    Each tuple is (action_name, next_state). On this campus, both values
    are the neighbor's name.

    Think about: where is the neighbor information stored in the problem?

    Example:
        >>> campus_get_actions(problem, 'GATE_1')
        [('IAMET', 'IAMET'), ('IEB', 'IEB')]
        >>> campus_get_actions(problem, 'Hub')
        [('PB', 'PB')]

    Returns:
        list of tuples, e.g., [('IAMET', 'IAMET'), ('IEB', 'IEB')]
    """
    # TODO: look up neighbors in the graph and return as list of tuples
    return []


def campus_action_cost(problem, state, action):
    """
    Return the cost of moving from state to action (the next location).

    The base cost comes from the graph. But there is an additional factor:
    bridges may have congestion at certain hours, adding extra cost.

    Think about:
    - Where is the base distance stored?
    - How do you check if the destination is a bridge?
    - Where is the congestion schedule, and how do you look up the current hour?
    - What should you return if the edge doesn't exist?

    Example:
        >>> campus_action_cost(problem, 'GATE_1', 'IAMET')  # normal edge
        431
        >>> campus_action_cost(problem, 'GATE_1', 'Library')  # not directly connected  
        inf
        >>> campus_action_cost(problem_9am, 'PMB', 'Bridge_South')  # congested
        745     # base 245 + congestion penalty
        >>> campus_action_cost(problem_6am, 'PMB', 'Bridge_South')  # not congested
        245     # base only, no penalty at this hour

    Returns:
        float: total cost (base + any congestion), or math.inf if invalid
    """
    # TODO: get base cost from graph, check for bridge congestion, return total
    return math.inf


# =====================================================================
# PART 3: HEURISTIC FUNCTIONS (Student Implementation)
# =====================================================================

def heuristic_euclidean(state, problem):
    """
    Estimate the remaining cost from state to goal using straight-line distance.

    Think about:
    - What is the formula for Euclidean distance between two (x,y) points?
    - What should this return when state IS the goal?
    - Why is straight-line distance guaranteed to never overestimate?

    Example:
        >>> heuristic_euclidean('PMB', problem)   # PMB is the goal
        0.0
        >>> heuristic_euclidean('GATE_1', problem)  # far from goal
        151.0

    Returns:
        float: estimated cost to goal (must be >= 0)
    """
    # TODO: compute straight-line distance from state to goal using coordinates
    return 0.0


def heuristic_scaled(state, problem, scale=1.0):
    """
    Multiply the Euclidean heuristic by a scale factor. It is set as 1.0 by default.

    A larger scale means a more aggressive estimate. At some point,
    the estimate becomes too aggressive and overestimates the true cost.
    When that happens, A* may no longer find the optimal path.

    Example:
        >>> heuristic_scaled('GATE_1', problem, scale=1.0)
        151.03
        >>> heuristic_scaled('GATE_1', problem, scale=3.0)
        453.09

    Returns:
        float: scale * Euclidean distance to goal
    """
    # TODO: call heuristic_euclidean and multiply by scale
    return 0.0


def find_best_scale(problem):
    """
    Find the LARGEST scale factor that keeps heuristic_scaled admissible.

    Candidates: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    A heuristic is admissible if h(n) <= true_cost(n -> goal) for ALL nodes n.

    Think about:
    - How can you compute the true optimal cost from any node to a goal?
      (Hint: you already have A* with an admissible heuristic at scale=1.0)
    - Why do you need to check ALL nodes, not just the start?
    - How do you test each candidate from largest to smallest?

    Test goals (use the goals from these routes):
        [('GATE_1','PMB'), ('GATE_1','GATE_2'), ('Library','Staff_Hotel'),
         ('Hub','Health_Centre'), ('TRENT','Sports_Centre')]

    Important: these routes only define which GOALS to check against.
    For each goal, you must verify admissibility across ALL nodes in the
    graph (not just the start of each route). A scale is only admissible
    if h(n) <= true_cost for every node n and every goal above.

    Returns:
        float: the largest admissible scale from the candidates list
    """
    candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    test_routes = [
        ('GATE_1', 'PMB'), ('GATE_1', 'GATE_2'), ('Library', 'Staff_Hotel'),
        ('Hub', 'Health_Centre'), ('TRENT', 'Sports_Centre'),
    ]
    all_nodes = list(problem['coords'].keys())

    # TODO: for each goal, compute optimal cost from every node using A*(scale=1.0)
    #       then test each candidate from largest to smallest
    #       return the largest where h(n) <= optimal_cost for ALL (node, goal) pairs
    return 1.0 #return the largest admissible scale factor


# =====================================================================
# PART 4: A* SEARCH ALGORITHM (Student Implementation)
# =====================================================================

def a_star_search(problem, heuristic_fn):
    """
    Find the optimal path using A* search.

    f(n) = g(n) + h(n)  where g = cost so far, h = heuristic estimate

    Hint: This follows Lab 4's general_search + enqueue_by_cost_gh pattern.
    Write the search loop (pop, goal test, expand) and the enqueue logic
    (append children, sort, clear, extend) in one function.
    The differences from Lab 4:
    - Pass campus_get_actions and campus_action_cost to expand()
    - Use heuristic_fn (a parameter) instead of a hardcoded heuristic

    Example:
        >>> result = a_star_search(problem, heuristic_euclidean)
        >>> result['state']
        'PMB'
        >>> result['cost']
        615
        >>> reconstruct_path(result)
        ['GATE_1', 'IAMET', 'PMB']

    Returns:
        dict: the goal node (use reconstruct_path() to get the path)
        None: if no path exists
    """
    # TODO: implement A* (refer to Lab 4's general_search + enqueue pattern)
    return None


# =====================================================================
# CHALLENGE B: MULTI-STOP DELIVERY (10 points)
# =====================================================================

def multi_stop_delivery(problem, locations_to_visit, heuristic_fn):
    """
    Visit 3 locations and return to start with minimum total cost.

    Think about:
    - How many different orderings of 3 locations exist?
    - For each ordering, how many A* searches do you need?
    - How do you join path segments without duplicating nodes at junctions?
    - Congestion still applies: pass time_of_day from the original problem
      to each sub-problem so bridge penalties are correctly included.

    Example:
        >>> path, cost = multi_stop_delivery(problem, ['PMB', 'Library', 'DB'], h)
        >>> path[0]
        'GATE_1'     # starts at initial
        >>> path[-1]
        'GATE_1'     # returns to initial
        >>> 'PMB' in path and 'Library' in path and 'DB' in path
        True          # visits all 3 locations

    Returns:
        tuple: (full_path_as_list, total_cost_as_number)
    """
    # TODO: try all permutations, solve A* for each segment, track the best
    return (None, float('inf'))


# =====================================================================
# TESTING (Run: python campus_delivery.py)
# =====================================================================

if __name__ == '__main__':
    campus_data = load_campus_map()
    graph = campus_data['campus_graph']
    coords = campus_data['campus_coords']
    bridges = campus_data['campus_bridges']

    print("=" * 70)
    print("CAMPUS DELIVERY AGENT -- LOCAL TESTS")
    print("=" * 70)

    # 1. Problem formulation
    print("\n1. Problem Formulation")
    p = make_campus_problem('GATE_1', 'PMB', graph, coords, bridges, time_of_day=9)
    print(f"   Initial={p.get('initial')}, Goal={p.get('goal')}, Time={p.get('time_of_day')}")

    # 2. Actions
    print("\n2. Actions from GATE_1")
    for a in campus_get_actions(p, 'GATE_1'):
        print(f"   GATE_1 -> {a[1]}: cost={campus_action_cost(p, 'GATE_1', a[1])}")

    # 3. Heuristic
    print("\n3. Heuristic")
    print(f"   h(GATE_1)={heuristic_euclidean('GATE_1', p):.1f}, h(PMB)={heuristic_euclidean('PMB', p):.1f}")

    # 4. A* Search
    print("\n4. A* Search: GATE_1 -> PMB")
    r = a_star_search(p, heuristic_euclidean)
    if r:
        print(f"   Path: {' -> '.join(reconstruct_path(r))}, Cost: {r['cost']:.0f}")
    else:
        print("   No path found (implement a_star_search first)")

    # 5. Congestion effect (time_of_day uses 24-hour format, matching campus_map.json)
    print("\n5. Congestion: PMB -> Sports_Centre")
    for hour in [6, 9]:  # 6 = 6am off-peak, 9 = 9am class hour (Bridge_South congested)
        pr = make_campus_problem('PMB', 'Sports_Centre', graph, coords, bridges, time_of_day=hour)
        r = a_star_search(pr, heuristic_euclidean)
        if r:
            br = [n for n in reconstruct_path(r) if n.startswith('Bridge')]
            print(f"   time_of_day={hour} ({hour:02d}:00): cost={r['cost']:.0f} via {br[0] if br else '?'}")

    # 6. Challenge A: Scaled Heuristic
    print("\n6. Challenge A: heuristic_scaled and find_best_scale")
    exp = make_campus_problem('GATE_1', 'PMB', graph, coords, bridges, time_of_day=6)
    print(f"   heuristic_scaled('GATE_1', problem, scale=1.0) = {heuristic_scaled('GATE_1', exp, 1.0):.1f}")
    print(f"   heuristic_scaled('GATE_1', problem, scale=3.0) = {heuristic_scaled('GATE_1', exp, 3.0):.1f}")
    # To use heuristic_scaled with a_star_search, wrap it in a function:
    def h_scaled(state, problem):
        return heuristic_scaled(state, problem, scale=6.0)
    r = a_star_search(exp, h_scaled)
    if r:
        print(f"   A* with scale=6.0: cost={r['cost']:.0f}")
    best = find_best_scale(exp)
    print(f"   find_best_scale() = {best}")

    # 7. Multi-stop (Challenge B)
    print("\n7. Multi-Stop (Challenge)")
    pc = make_campus_problem('GATE_1', 'GATE_1', graph, coords, bridges, time_of_day=9)
    path, cost = multi_stop_delivery(pc, ['PMB', 'Library', 'DB'], heuristic_euclidean)
    if path:
        print(f"   Cost: {cost:.0f}, Nodes: {len(path)}")
    else:
        print("   Not implemented yet")

    print("\n" + "=" * 70)
    print("Submit campus_delivery.py via Moodle for grading.")
    print("=" * 70)
