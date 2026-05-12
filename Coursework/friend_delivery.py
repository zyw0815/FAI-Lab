"""
Campus Delivery Agent -- FAI Coursework

Implement A* search to find optimal paths on a campus map with bridge congestion.

Student: Yuyang ZHOU
Date: 2026-5-11
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
    #Accept the given parameters, and return a dictionary including following keys.
    return {'initial':start,'goal':goal,'graph':campus_graph,'coords':campus_coords,'bridges':campus_bridges,'time_of_day':time_of_day}


def is_goal(problem, state):
    """
    Return True if state matches the goal in the problem.

    Example:
        >>> is_goal(problem, 'PMB')   # if goal is PMB
        True
        >>> is_goal(problem, 'GATE_1')
        False
    """
    return state == problem['goal']    #if current state reaches goal, return True, else return False


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
    actions=[]
    for neighbor in problem['graph'].get(state,{}):  #Use "get" method to get the state's available neighbors, if not exist, return empty dict
        actions.append((neighbor,neighbor))        #Tuple format
    return actions


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
    no_congested_cost = problem['graph'].get(state, {}).get(action, float('inf'))   #Get the cost of destination from the state
    
    if no_congested_cost== math.inf:    #The action is not available
        return math.inf
    
    if action in problem['bridges']:    #If the destination is a bridge, check if it's on the congestion time
        congestion_time=problem['bridges'][action].get('congestion',{})
        current_hour = str(problem['time_of_day'])
        total_cost = no_congested_cost+congestion_time.get(current_hour, 0)
        return total_cost
    
    return no_congested_cost


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
    if is_goal(problem,state):  
        return 0.0
    else:
        current_x,current_y=tuple(problem['coords'][state])     #Use x and y coordinates to calculate the euclidean distance
        goal_x,goal_y=tuple(problem['coords'][problem['goal']])
        distance=math.sqrt((goal_x-current_x)**2+(goal_y-current_y)**2)
        return float(distance)

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
    distance=heuristic_euclidean(state,problem)
    return float(scale*distance)


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
    
    test_goals = [route[1] for route in test_routes]  #Get test goals from given routes
    test_real_cost={}  #Store the real cost for all nodes to the goal
    
    for goal in test_goals:             #Test each goal in five goals
        test_problem=problem.copy()      #A duplication of the problem and set the goal and day to needed
        test_problem['goal']=goal
        test_problem['time_of_day']=6
        for node in all_nodes:
            test_problem['initial']=node    #Change the start point to each node
            real=a_star_search(test_problem,heuristic_euclidean)    #The real optimal path calculated by A*
            if real is None:  
                test_real_cost[(node,goal)]=float('inf')   #If the route is not reachable, the cost is infinity (for robustness)
            else:
                test_real_cost[(node,goal)]= float(real['cost'])  #Use (node,goal) for each node to each goal's cost
    
    for scale in reversed(candidates):       #From the largest to least, if the valid largest was found, break and return the value
        is_admissible = True
        for goal in test_goals:            
            test_problem=problem.copy()      
            test_problem['goal']=goal

            for node in all_nodes:            
                h_scaled_value=heuristic_scaled(node,test_problem,scale)    #Scaled heuristic estimate
                
                real_cost =test_real_cost.get((node, goal), float('inf')) #Real cost from the dictionary
                if h_scaled_value > real_cost + 1e-9:  #Costs compared with tolerance
                    is_admissible = False      #If the heuristic estimate is greater than the realcost, not admissible
                    break 
            if not is_admissible:
                break   
        if is_admissible: #If the best admissible heuristic is found, return the value
            return scale
    return 1.0 #Nothing is found, return 1.0
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
    frontier = [make_node(problem['initial'])]
    while True:
        if not frontier:
            return None                 # no solution found, return None
        
        node = frontier.pop(0)          # take the first node off the frontier
        
        if is_goal(problem, node['state']):
            return node                 # reach the goal
        
        children = expand(problem, node, campus_get_actions, campus_action_cost)  #Expand children nodes
        
        #check each child node and sort with the f-cost
        for child in children:
            if not is_cycle(child):
                frontier.append(child)
        frontier.sort(key=lambda node: node['cost']+heuristic_fn(node['state'], problem))    #Update the frontier


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
    start=problem['initial']
    full_path=[]
    total_cost=float('inf')
    for p in permutations(locations_to_visit):
        order=[start,p[0],p[1],p[2],start]
        #Define a zero initial cost and empty path for each order(total 6)
        curr_order_cost=0
        curr_order_path=[]
        for i in range(4):  #start->p[0], p[0]->p[1], p[1]->p[2], p[2]->start
            new_problem=problem.copy()
            new_problem['initial']=order[i]
            new_problem['goal']=order[i+1]
            
            subpath=a_star_search(new_problem,heuristic_fn)
            if subpath is None:
                curr_order_cost = float('inf')
                break
            curr_order_cost+=subpath['cost']
            path=reconstruct_path(subpath)
            #Handle the repeat elements in the path list, ensure the path is clean
            if i==0:
                curr_order_path.extend(path)
            else:
                curr_order_path.extend(path[1:])
        #Find the ordering that minimizes total travel cost
        if(curr_order_cost<total_cost):
            total_cost=curr_order_cost
            full_path=curr_order_path
    return (full_path,total_cost)


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
