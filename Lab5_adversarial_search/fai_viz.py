"""
fai_viz.py  —  FAI Course Visualization Library
================================================
Importable visualization toolkit for the Fundamentals of Artificial
Intelligence course notebooks (Demos + Labs).

Usage in Demo notebooks (02-Demo_Code/inclass_demo_code/):
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), '../..'))
    import fai_viz

Usage in Lab notebooks (03-Lab/):
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), '..'))
    import fai_viz

Sections
--------
1.  Constants          – Romania map data (city positions, roads, SLD heuristic)
2.  Reference Figures  – show_*() for problem introductions (self-contained)
3.  Grid              – plot_grid_path()
4.  Water Jug         – plot_jug_solution()
5.  BFS / DFS         – plot_bfs_dfs_exploration()
6.  Romania Maps      – draw_romania_ax(), plot_romania_algorithms(),
                        plot_algorithm_bar_chart()
7.  A* / f=g+h        – compute_fgh(), plot_astar_path_map(),
                        plot_fgh_decomposition()
8.  8-Puzzle          – plot_puzzle_heuristic_comparison()
9.  Tic-Tac-Toe       – draw_ttt_board(), draw_minimax_heatmap(),
                        plot_minimax_vs_alphabeta()
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS — Romania Road Map (AIMA Figure 3.2)
# ═══════════════════════════════════════════════════════════════════════

ROMANIA_CITY_POS = {
    "Oradea":    (1.3, 8.2), "Zerind":    (0.9, 7.2),
    "Arad":      (0.7, 6.1), "Timisoara": (1.2, 4.8),
    "Lugoj":     (2.0, 4.1), "Mehadia":   (2.1, 3.3),
    "Dobreta":   (2.2, 2.4), "Craiova":   (3.8, 1.8),
    "Sibiu":     (3.8, 6.0), "Rimnicu":   (4.4, 5.2),
    "Fagaras":   (5.2, 6.6), "Pitesti":   (5.5, 4.2),
    "Bucharest": (6.5, 3.5), "Giurgiu":   (6.2, 2.2),
    "Urziceni":  (7.5, 3.8), "Hirsova":   (8.3, 4.7),
    "Eforie":    (8.4, 3.3), "Vaslui":    (8.0, 6.5),
    "Iasi":      (7.7, 7.5), "Neamt":     (6.7, 7.9),
}

ROMANIA_ROADS = {
    "Arad":      {"Zerind": 75,  "Sibiu": 140,  "Timisoara": 118},
    "Zerind":    {"Arad": 75,    "Oradea": 71},
    "Oradea":    {"Zerind": 71,  "Sibiu": 151},
    "Sibiu":     {"Arad": 140,   "Oradea": 151, "Fagaras": 99,  "Rimnicu": 80},
    "Timisoara": {"Arad": 118,   "Lugoj": 111},
    "Lugoj":     {"Timisoara": 111, "Mehadia": 70},
    "Mehadia":   {"Lugoj": 70,   "Dobreta": 75},
    "Dobreta":   {"Mehadia": 75, "Craiova": 120},
    "Craiova":   {"Dobreta": 120,"Rimnicu": 146, "Pitesti": 138},
    "Rimnicu":   {"Sibiu": 80,   "Craiova": 146, "Pitesti": 97},
    "Fagaras":   {"Sibiu": 99,   "Bucharest": 211},
    "Pitesti":   {"Rimnicu": 97, "Craiova": 138, "Bucharest": 101},
    "Bucharest": {"Fagaras": 211,"Pitesti": 101, "Giurgiu": 90,  "Urziceni": 85},
    "Giurgiu":   {"Bucharest": 90},
    "Urziceni":  {"Bucharest": 85,"Hirsova": 98, "Vaslui": 142},
    "Hirsova":   {"Urziceni": 98, "Eforie": 86},
    "Eforie":    {"Hirsova": 86},
    "Vaslui":    {"Urziceni": 142,"Iasi": 92},
    "Iasi":      {"Vaslui": 92,  "Neamt": 87},
    "Neamt":     {"Iasi": 87},
}

ROMANIA_SLD = {
    "Arad": 366, "Bucharest": 0,   "Craiova": 160, "Dobreta": 242,
    "Eforie": 161, "Fagaras": 176, "Giurgiu": 77,  "Hirsova": 151,
    "Iasi": 226,  "Lugoj": 244,   "Mehadia": 241, "Neamt": 234,
    "Oradea": 380,"Pitesti": 100, "Rimnicu": 193, "Sibiu": 253,
    "Timisoara": 329, "Urziceni": 80, "Vaslui": 199, "Zerind": 374,
}


# ═══════════════════════════════════════════════════════════════════════
# 2.  REFERENCE FIGURES  (self-contained — embed all needed data)
# ═══════════════════════════════════════════════════════════════════════

def show_grid_figure(rows=5, cols=5, start=(0, 0), goal=(4, 4),
                     wall_col=2):
    """
    Reference figure for the Grid Navigation problem.
    Shows a plain grid and a grid with a vertical wall at wall_col.
    """
    def _draw(ax, walls, title):
        ax.set_facecolor('#f0f4f8')
        for r in range(rows):
            for c in range(cols):
                s = (r, c)
                if s in walls:
                    fc, lbl, tc = '#555555', '\u2593', 'white'
                elif s == start:
                    fc, lbl, tc = '#27ae60', 'S', 'white'
                elif s == goal:
                    fc, lbl, tc = '#e74c3c', 'G', 'white'
                else:
                    fc, lbl, tc = '#ecf0f1', f'({r},{c})', '#aaaaaa'
                ax.add_patch(plt.Rectangle([c, rows-1-r], 1, 1,
                             facecolor=fc, edgecolor='#bdc3c7', lw=2))
                fs = 13 if s in (start, goal) or s in walls else 7
                ax.text(c+0.5, rows-1-r+0.5, lbl,
                        ha='center', va='center',
                        fontsize=fs, fontweight='bold', color=tc)
        ax.set_xlim(0, cols); ax.set_ylim(0, rows)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    walls_col = [(r, wall_col) for r in range(rows-1)]  # leave bottom row open
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))
    _draw(a1, set(), f"{rows}\u00d7{cols} Grid \u2014 No Walls\nS={start}  G={goal}")
    _draw(a2, set(walls_col),
          f"{rows}\u00d7{cols} Grid \u2014 Column {wall_col} Blocked\nRobot must navigate around the wall")
    fig.suptitle("Grid Navigation Problem   |   State = (row, col)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show()
    print(f"State space: {rows*cols} states ({rows}\u00d7{cols} grid)  |  "
          "Actions: up / down / left / right")
    print(f"Start: {start} \u2014 top-left corner     |  Goal:  {goal} \u2014 bottom-right corner")


def show_water_jug_figure(capacities=(3, 5), goal_litres=4):
    """
    Reference figure for the Water Pouring problem.
    Works for 2 or 3 jugs.
    capacities: tuple of jug capacities, e.g. (3, 5) or (3, 5, 9)
    goal_litres: target litres in any jug
    """
    n_jugs = len(capacities)
    colors = ['#2980b9', '#e67e22', '#8e44ad'][:n_jugs]
    labels = [chr(65+i) for i in range(n_jugs)]  # A, B, C ...

    fig, ax = plt.subplots(figsize=(max(8, n_jugs*3.5), 5.5))
    ax.set_facecolor('#f0f4f8'); ax.axis('off')
    ax.set_xlim(0, n_jugs*3.5 + 2); ax.set_ylim(0, 6.5)

    for i, (cap, col, lbl) in enumerate(zip(capacities, colors, labels)):
        jx = 1.2 + i * 3.2
        jug_h = cap * 0.65
        ax.add_patch(plt.Rectangle([jx, 0.8], 1.8, jug_h,
                                    facecolor='#dce8f5', edgecolor='#2c3e50', lw=2.5))
        for v in range(1, cap+1):
            vy = 0.8 + v * 0.65
            ax.plot([jx, jx+0.3], [vy, vy], color='#7f8c8d', lw=1.5)
            ax.text(jx-0.2, vy, str(v), ha='right', va='center',
                    fontsize=10, color='#555')
        ax.text(jx+0.9, 0.8+jug_h+0.3, f'{cap}L max', ha='center',
                fontsize=11, fontweight='bold', color='#2c3e50')
        ax.text(jx+0.9, 0.3, f'Jug {lbl} ({cap}L)', ha='center',
                fontsize=12, fontweight='bold', color=col)

    if n_jugs == 2:
        ax.annotate('', xy=(4.2, 3.5), xytext=(3.0, 3.5),
                    arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2.5))
        ax.text(3.6, 3.9, 'Pour A\u2194B', ha='center', fontsize=9,
                color='#e74c3c', fontweight='bold')

    ops_x = n_jugs * 3.2 + 0.5
    ax.text(ops_x, 5.5, "Operations:", ha='center', fontsize=11,
            fontweight='bold', color='#2c3e50')
    for i, (name, col, desc) in enumerate([
        ("Fill",  '#27ae60', "fill jug to capacity"),
        ("Empty", '#e74c3c', "empty jug completely"),
        ("Pour",  '#3498db', "pour i\u2192j until source empty or target full"),
    ]):
        ay = 4.5 - i*1.0
        ax.add_patch(plt.Rectangle([ops_x-1.5, ay-0.2], 1.0, 0.5,
                                    facecolor=col, alpha=0.8, edgecolor='none'))
        ax.text(ops_x-1.0, ay+0.05, name, ha='center', fontsize=9,
                fontweight='bold', color='white')
        ax.text(ops_x+0.1, ay+0.05, desc, ha='left', fontsize=9,
                color='#555', va='center')

    state_str = "(" + ", ".join(f"L in {l}" for l in labels) + ")"
    ax.set_title(f"Water Pouring Problem   |   State = {state_str}\n"
                 f"Initial: all empty     |   Goal: any jug contains exactly {goal_litres}L",
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout(); plt.show()
    state_size = 1
    for c in capacities:
        state_size *= (c+1)
    print(f"State space: at most {state_size} states  |  "
          "BFS finds the shortest solution path.")


def show_bfs_dfs_tree():
    """
    Reference figure: BFS vs DFS expansion order on a 4-level binary tree.
    Numbers show which node was expanded nth.
    """
    NODES = {
        0: (4.0, 4.5),
        1: (2.5, 3.5), 2: (5.5, 3.5),
        3: (1.5, 2.5), 4: (3.5, 2.5), 5: (4.5, 2.5), 6: (6.5, 2.5),
        7: (1.0, 1.4), 8: (2.0, 1.4), 9: (3.0, 1.4), 10: (4.0, 1.4),
        11: (4.5, 1.4), 12: (5.5, 1.4), 13: (6.0, 1.4), 14: (7.0, 1.4),
    }
    EDGES = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),
             (3,7),(3,8),(4,9),(4,10),(5,11),(5,12),(6,13),(6,14)]
    BFS_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    DFS_ORDER = [0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14]
    LEVEL_LABELS = [
        (-0.3, 4.5, 'Level 0\n(Start)'),
        (-0.3, 3.5, 'Level 1'),
        (-0.3, 2.5, 'Level 2'),
        (-0.3, 1.4, 'Level 3'),
    ]

    def _draw_tree(ax, order, color, title):
        for u, v in EDGES:
            xu, yu = NODES[u]; xv, yv = NODES[v]
            ax.plot([xu, xv], [yu, yv], color='#bdc3c7', lw=1.8, zorder=1)
        for nid, (x, y) in NODES.items():
            rank = order.index(nid) + 1 if nid in order else None
            fc   = color if rank else '#f0f0f0'
            alpha = 1.0 if rank else 0.5
            ax.add_patch(plt.Circle((x, y), 0.38, color=fc, alpha=alpha,
                                     ec='#95a5a6', lw=1.5, zorder=2))
            if rank:
                ax.text(x, y, str(rank), ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white', zorder=3)
            else:
                ax.text(x, y, '\u00b7', ha='center', va='center',
                        fontsize=16, color='#ccc', zorder=3)
        for lx, ly, lt in LEVEL_LABELS:
            ax.text(lx, ly, lt, ha='right', va='center', fontsize=7.5,
                    color='#95a5a6', style='italic')
        ax.set_xlim(-0.6, 8.5); ax.set_ylim(0.8, 5.3)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_facecolor('#fafafa')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    _draw_tree(ax1, BFS_ORDER, '#2980b9',
               "BFS \u2014 Level-by-Level\nExpand all nodes at depth d before depth d+1")
    _draw_tree(ax2, DFS_ORDER, '#e67e22',
               "DFS \u2014 Depth-First\nDive as deep as possible, then backtrack")
    fig.suptitle("Search Tree Expansion Order (numbers = which node expanded nth)\n"
                 "Same tree, same start \u2014 completely different strategy.",
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()


def show_romania_map(highlight_path=None, show_sld=False,
                     start_city='Arad', goal_city='Bucharest',
                     extra_highlight=None):
    """
    Reference figure: Romania Road Map (AIMA Figure 3.2).

    Parameters
    ----------
    highlight_path : list of city names to highlight (orange path)
    show_sld       : if True, also draw SLD-to-Bucharest bar chart on the right
    start_city     : highlighted green
    goal_city      : highlighted red
    extra_highlight: dict {city: color} for additional city highlights
    """
    hl = {start_city: '#27ae60', goal_city: '#e74c3c'}
    if extra_highlight:
        hl.update(extra_highlight)

    if show_sld:
        fig, (ax_map, ax_sld) = plt.subplots(
            1, 2, figsize=(17, 7.5), gridspec_kw={'width_ratios': [2.2, 1]})
        fig.patch.set_facecolor('#f5f0e8')
    else:
        fig, ax_map = plt.subplots(figsize=(14, 8))

    ax_map.set_facecolor('#f5f0e8')
    fig.patch.set_facecolor('#f5f0e8')

    path_set = set()
    path_edges = set()
    if highlight_path:
        path_set = set(highlight_path)
        for i in range(len(highlight_path)-1):
            path_edges.add(frozenset([highlight_path[i], highlight_path[i+1]]))

    drawn = set()
    for c1, nbrs in ROMANIA_ROADS.items():
        x1, y1 = ROMANIA_CITY_POS[c1]
        for c2, dist in nbrs.items():
            edge = frozenset([c1, c2])
            if edge in drawn: continue
            drawn.add(edge)
            x2, y2 = ROMANIA_CITY_POS[c2]
            in_path = edge in path_edges
            col = '#e67e22' if in_path else '#8B7355'
            lw  = 3.5     if in_path else 2.0
            ax_map.plot([x1, x2], [y1, y2], color=col, lw=lw, zorder=1+in_path,
                        solid_capstyle='round')
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax_map.text(mx, my, str(dist), fontsize=7, ha='center', va='center',
                        color='#5C4827', fontweight='bold', zorder=3,
                        bbox=dict(boxstyle='round,pad=0.12', facecolor='#f5f0e8',
                                  alpha=0.85, edgecolor='none'))

    for city, (x, y) in ROMANIA_CITY_POS.items():
        fc = hl.get(city, '#f39c12' if city in path_set else 'white')
        ax_map.add_patch(plt.Circle((x, y), 0.29, color=fc, ec='#8B7355',
                                     lw=2, zorder=4))
        ax_map.text(x, y+0.42, city, ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold', color='#2c3e50', zorder=5)

    # Compass rose
    ax_map.annotate('N', xy=(9.1, 9.2), fontsize=14, fontweight='bold',
                    ha='center', color='#2c3e50')
    ax_map.annotate('', xy=(9.1, 9.2), xytext=(9.1, 8.7),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    ax_map.set_xlim(-0.2, 9.8); ax_map.set_ylim(1.2, 10.0)
    ax_map.set_aspect('equal'); ax_map.axis('off')
    ax_map.set_title(
        "Romania Road Map   (road distances in km)\n"
        "Source: Russell & Norvig, Artificial Intelligence: A Modern Approach",
        fontsize=12, fontweight='bold', color='#2c3e50', pad=12)

    legend_items = [
        mpatches.Patch(color='#27ae60', label=f'{start_city} (common start)'),
        mpatches.Patch(color='#e74c3c', label=f'{goal_city} (common goal)'),
        mpatches.Patch(color='white',   label='Other cities', ec='#8B7355'),
    ]
    if extra_highlight:
        for city, color in extra_highlight.items():
            legend_items.append(mpatches.Patch(color=color, label=city))
    if highlight_path:
        legend_items.append(mpatches.Patch(color='#f39c12', label='Path cities'))
        legend_items.append(mpatches.Patch(color='#e67e22', label='Path roads'))
    ax_map.legend(handles=legend_items, loc='lower right', fontsize=9, framealpha=0.9)

    if show_sld:
        cities_sorted = sorted(ROMANIA_SLD.items(), key=lambda x: x[1])
        c_list  = [c for c, _ in cities_sorted]
        sld_list = [v for _, v in cities_sorted]
        bar_colors = [
            '#e74c3c' if c == goal_city else '#27ae60' if c == start_city
            else '#f39c12' if highlight_path and c in highlight_path
            else '#85c1e9'
            for c in c_list
        ]
        ax_sld.barh(range(len(c_list)), sld_list, color=bar_colors,
                    height=0.7, edgecolor='white', lw=1)
        ax_sld.set_yticks(range(len(c_list)))
        ax_sld.set_yticklabels(c_list, fontsize=8)
        ax_sld.set_xlabel('h(n) = SLD to Bucharest (km)', fontsize=9)
        ax_sld.set_title('Heuristic Values h(n)\n(SLD to Bucharest)',
                         fontsize=10, fontweight='bold')
        ax_sld.spines['top'].set_visible(False); ax_sld.spines['right'].set_visible(False)
        for i, (c, v) in enumerate(cities_sorted):
            ax_sld.text(v+3, i, str(v), va='center', fontsize=8)

    plt.tight_layout(); plt.show()


def show_8puzzle_figure():
    """
    Reference figure: 8-Puzzle problem — goal state and sample initial states.
    """
    def _draw_puzzle(ax, state, title):
        ax.set_facecolor('#f8f9fa')
        ax.set_aspect('equal'); ax.axis('off')
        for idx, val in enumerate(state):
            r, c = divmod(idx, 3)
            if val == 0:
                fc, txt = '#bdc3c7', ''
            else:
                intensity = val / 8
                fc  = (0.2 + 0.4*intensity, 0.5 + 0.2*(1-intensity), 0.85)
                txt = str(val)
            ax.add_patch(plt.Rectangle([c, 2-r], 1, 1,
                         facecolor=fc, edgecolor='white', lw=4, zorder=1))
            if txt:
                ax.text(c+0.5, 2-r+0.5, txt, ha='center', va='center',
                        fontsize=26, fontweight='bold', color='white', zorder=2)
            else:
                ax.add_patch(plt.Rectangle([c+0.1, 2-r+0.1], 0.8, 0.8,
                             facecolor='none', edgecolor='#95a5a6',
                             lw=1.5, linestyle='--', zorder=2))
                ax.text(c+0.5, 2-r+0.5, '\u25a1', ha='center', va='center',
                        fontsize=20, color='#95a5a6', zorder=2)
        ax.set_xlim(-0.1, 3.1); ax.set_ylim(-0.1, 3.1)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    configs = [
        ((1,2,3,4,5,6,7,8,0), "Goal State  \u2713\n(1-2-3 / 4-5-6 / 7-8-\u25a1)"),
        ((1,2,3,4,0,6,7,5,8), "Easy Initial State\n(~3 moves to goal)"),
        ((1,2,3,4,5,6,7,0,8), "Medium Initial State\n(~2 moves to goal)"),
        ((8,6,7,2,5,4,3,0,1), "Hard Initial State\n(~31 moves to goal)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (state, title) in zip(axes, configs):
        _draw_puzzle(ax, state, title)
    # Add position index labels on last panel
    for idx in range(9):
        r, c = divmod(idx, 3)
        axes[-1].text(c+0.85, 2-r+0.15, str(idx),
                      fontsize=8, color='#aaa', ha='right', va='bottom')
    fig.suptitle("The 8-Puzzle   |   State = 9-tuple  (0 = blank tile)\n"
                 "Actions: move blank UP / DOWN / LEFT / RIGHT\n"
                 "Position indices: 0=top-left, 8=bottom-right",
                 fontsize=11, fontweight='bold', y=1.04)
    plt.tight_layout(); plt.show()
    print("Goal state: (1, 2, 3, 4, 5, 6, 7, 8, 0)")
    print("Blank tile = 0  |  To move a tile: swap it with the blank")


def show_tictactoe_minimax_figure():
    """
    Reference figure: Tic-Tac-Toe board states + partial Minimax game tree.
    """
    fig = plt.figure(figsize=(15, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.08)
    ax_boards = fig.add_subplot(gs[0])
    ax_tree   = fig.add_subplot(gs[1])
    ax_boards.axis('off'); ax_tree.axis('off')

    # Left panel: board states
    def _mini_board(ax, cx, cy, sz, pieces, bg='#ecf0f1'):
        ax.add_patch(plt.Rectangle([cx, cy], sz, sz,
                     facecolor=bg, edgecolor='#bdc3c7', lw=1.5, zorder=2))
        for i in [1, 2]:
            ax.plot([cx, cx+sz], [cy+i*sz/3, cy+i*sz/3],
                    color='#95a5a6', lw=0.8, zorder=3)
            ax.plot([cx+i*sz/3, cx+i*sz/3], [cy, cy+sz],
                    color='#95a5a6', lw=0.8, zorder=3)
        for idx, p in enumerate(pieces):
            r, c = divmod(idx, 3)
            px = cx + (c+0.5)*sz/3
            py = cy + (2-r+0.5)*sz/3
            if p == 'X':
                ax.text(px, py, 'X', ha='center', va='center',
                        fontsize=sz*5.5, fontweight='bold', color='#2980b9', zorder=4)
            elif p == 'O':
                ax.text(px, py, 'O', ha='center', va='center',
                        fontsize=sz*5.5, fontweight='bold', color='#e74c3c', zorder=4)

    ax_boards.set_xlim(0, 6); ax_boards.set_ylim(0, 8.5)
    ax_boards.set_facecolor('white')
    boards_demo = [
        (1, 6.5, 'Empty Board\n(X moves first)', list('         '), 'white'),
        (1, 4.0, 'Mid-game\n(O to move)',        list('X  OX   O'), '#fef9f0'),
        (1, 1.5, 'X wins!\n(top row)',           list('XXX O O  '), '#eafaf1'),
    ]
    for bx, by, ttl, pieces, bg in boards_demo:
        _mini_board(ax_boards, bx, by, 2.0, pieces, bg)
        ax_boards.text(bx+1.0, by+2.2, ttl, ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color='#2c3e50')
    ax_boards.set_title("Board States  (state = 9-tuple)\nIndex layout:\n0|1|2 / 3|4|5 / 6|7|8",
                        fontsize=10, fontweight='bold', pad=6)

    # Right panel: minimax tree
    ax_tree.set_xlim(0, 12); ax_tree.set_ylim(-0.5, 8)
    ax_tree.set_facecolor('#fafafa')

    def _node(ax, x, y, val, fc, lbl_above="", lbl_below=""):
        ax.add_patch(plt.Circle((x, y), 0.38, color=fc, ec='#7f8c8d', lw=1.5, zorder=3))
        ax.text(x, y, str(val), ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=4)
        if lbl_above:
            ax.text(x, y+0.55, lbl_above, ha='center', va='bottom',
                    fontsize=7.5, color='#555', style='italic')
        if lbl_below:
            ax.text(x, y-0.55, lbl_below, ha='center', va='top',
                    fontsize=10, fontweight='bold', color='#333')

    def _edge(ax, x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color='#bdc3c7', lw=1.8, zorder=1)

    _node(ax_tree, 6.0, 7.0, 0, '#2980b9', "MAX (X)", "backed-up: 0")
    min_xs  = [2.5, 6.0, 9.5]
    min_vals = [0, -1, 0]
    for mx in min_xs:
        _edge(ax_tree, 6.0, 7.0, mx, 5.2)
    for mx, mv in zip(min_xs, min_vals):
        _node(ax_tree, mx, 5.2, mv, '#e74c3c', "MIN (O)")
    ax_tree.text(1.0, 5.2, "MIN (O)\npicks lowest",
                 ha='center', va='center', fontsize=8, color='#e74c3c', style='italic')
    leaf_data = [
        (1.0, 3.5, +1, '#27ae60'), (2.0, 3.5, -1, '#e74c3c'),
        (3.0, 3.5,  0, '#3498db'), (5.0, 3.5,  0, '#3498db'),
        (7.0, 3.5, -1, '#e74c3c'), (8.5, 3.5, +1, '#27ae60'),
        (10.0,3.5,  0, '#3498db'),
    ]
    leaf_parents = [2.5, 2.5, 2.5, 6.0, 6.0, 9.5, 9.5]
    for (lx, ly, lv, lc), px in zip(leaf_data, leaf_parents):
        _edge(ax_tree, px, 5.2, lx, ly)
        label = f'{lv:+d}' if lv != 0 else '0'
        _node(ax_tree, lx, ly, label, lc)
    ax_tree.text(6.0, 3.0,
                 "Leaf values:\n+1 = X wins   0 = draw   -1 = O wins",
                 ha='center', va='top', fontsize=9, color='#555',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    legend = [
        mpatches.Patch(color='#2980b9', label='MAX node (X to move \u2014 picks highest)'),
        mpatches.Patch(color='#e74c3c', label='MIN node (O to move \u2014 picks lowest)'),
        mpatches.Patch(color='#27ae60', label='Leaf +1 (X wins)'),
        mpatches.Patch(color='#e74c3c', label='Leaf \u22121 (O wins)'),
        mpatches.Patch(color='#3498db', label='Leaf 0 (draw)'),
    ]
    ax_tree.legend(handles=legend, loc='lower center', fontsize=8.5,
                   bbox_to_anchor=(0.5, -0.02), ncol=2, framealpha=0.9)
    ax_tree.set_title("Minimax Search Tree (partial \u2014 2 plies shown)\n"
                      "Backed-up values propagate from leaves to root",
                      fontsize=10, fontweight='bold', pad=8)
    fig.suptitle("Adversarial Search: Tic-Tac-Toe with Minimax",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.show()
    print("State representation: 9-tuple  ('X', 'O', or '.' for each cell)")
    print("Index layout:  0|1|2")
    print("               3|4|5")
    print("               6|7|8")


# ═══════════════════════════════════════════════════════════════════════
# 3.  GRID VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def plot_grid_path(problem, path, title="Grid Navigation"):
    """
    Draw the grid with the solution path highlighted.

    Parameters
    ----------
    problem : dict with keys 'initial', 'goal', 'grid_size' (or 'rows'+'cols'),
              and optional 'walls'
    path    : list of (row, col) state tuples
    title   : plot title
    """
    rows = cols = problem.get('grid_size', 5)
    rows = problem.get('rows', rows)
    cols = problem.get('cols', cols)
    walls    = set(problem.get('walls', []))
    start    = problem['initial']
    goal     = problem['goal']
    path_set = set(path) if path else set()

    fig, ax = plt.subplots(figsize=(5, 5))
    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            if state in walls:
                color = '#555555'
            elif state == start:
                color = '#2ecc71'
            elif state == goal:
                color = '#e74c3c'
            elif state in path_set:
                color = '#3498db'
            else:
                color = '#ecf0f1'
            ax.add_patch(plt.Rectangle([c, rows-1-r], 1, 1,
                         facecolor=color, edgecolor='white', linewidth=2))
            label = 'S' if state == start else 'G' if state == goal else ''
            ax.text(c+0.5, rows-1-r+0.5, label,
                    ha='center', va='center', fontsize=14,
                    fontweight='bold', color='white')

    if path and len(path) > 1:
        for i in range(len(path)-1):
            r1, c1 = path[i]; r2, c2 = path[i+1]
            ax.annotate("",
                xy=(c2+0.5, rows-1-r2+0.5),
                xytext=(c1+0.5, rows-1-r1+0.5),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))

    ax.set_xlim(0, cols); ax.set_ylim(0, rows)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    legend = [
        mpatches.Patch(color='#2ecc71', label='Start'),
        mpatches.Patch(color='#e74c3c', label='Goal'),
        mpatches.Patch(color='#3498db', label='Path'),
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=9)
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════
# 4.  WATER JUG VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def plot_jug_solution(path, capacities, title="Water Jug Solution",
                      is_goal_fn=None):
    """
    Visualize a water jug solution as a sequence of jug-pair bar charts.

    Parameters
    ----------
    path        : list of state tuples, e.g. [(0,0), (3,0), (3,2), ...]
    capacities  : tuple of jug capacities, e.g. (3, 5)
    title       : plot title
    is_goal_fn  : optional (state) -> bool, used to highlight the goal step
    """
    n = len(path)
    colors = ['#2980b9', '#e74c3c', '#8e44ad']
    empty_colors = ['#d6eaf8', '#fadbd8', '#e8daef']
    labels = [chr(65+i) for i in range(len(capacities))]
    max_cap = max(capacities)

    fig, axes = plt.subplots(1, n, figsize=(n*1.6 + 0.5, 4.5))
    if n == 1: axes = [axes]

    for i, (state, ax) in enumerate(zip(path, axes)):
        if not isinstance(state, (list, tuple)):
            state = (state,)
        goal_step = is_goal_fn(state) if is_goal_fn else False
        if goal_step:
            ax.set_facecolor('#eafaf1')

        for j, (level, cap) in enumerate(zip(state, capacities)):
            ax.bar([j], [level], color=colors[j % len(colors)], width=0.6)
            ax.bar([j], [cap-level], color=empty_colors[j % len(empty_colors)],
                   width=0.6, bottom=level)

        ax.set_xlim(-0.6, len(capacities)-0.4)
        ax.set_ylim(0, max_cap+0.6)
        ax.set_xticks(range(len(capacities)))
        ax.set_xticklabels([f'{l}\n({c}L)' for l, c in zip(labels, capacities)], fontsize=8)
        ax.set_title(f'Step {i}', fontsize=8)
        if i == 0:
            ax.set_ylabel('Litres', fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    legend = [mpatches.Patch(color=colors[i], label=f'Jug {labels[i]}')
              for i in range(len(capacities))]
    fig.legend(handles=legend, loc='lower center', ncol=len(capacities),
               fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()
    arrow = ' \u2192 '
    print("Solution: " + arrow.join(str(s) for s in path) + f"  ({len(path)-1} steps)")


# ═══════════════════════════════════════════════════════════════════════
# 5.  BFS / DFS EXPLORATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════

def plot_bfs_dfs_exploration(problem, get_actions_fn,
                              title_bfs="BFS Exploration Order",
                              title_dfs="DFS Exploration Order"):
    """
    Plot BFS vs DFS exploration order as side-by-side heatmaps on a grid.

    Parameters
    ----------
    problem        : dict with 'initial', 'goal', and 'grid_size' (or 'rows'/'cols')
                     and optional 'walls'
    get_actions_fn : callable(state) -> [(action, next_state), ...]
                     Pass a pre-bound lambda, e.g. lambda s: get_actions(problem, s)
    """
    rows = cols = problem.get('grid_size', 5)
    rows = problem.get('rows', rows)
    cols = problem.get('cols', cols)
    start = problem['initial']
    goal  = problem['goal']
    walls = set(map(tuple, problem.get('walls', [])))

    # Internal BFS order
    def _bfs_order():
        frontier = deque([start])
        visited  = {start}
        order    = {}
        count    = 0
        while frontier:
            state = frontier.popleft()
            order[state] = count; count += 1
            for _, ns in get_actions_fn(state):
                if ns not in visited:
                    visited.add(ns); frontier.append(ns)
        return order

    # Internal DFS order
    def _dfs_order():
        frontier = [start]
        visited  = set()
        order    = {}
        count    = 0
        while frontier:
            state = frontier.pop()
            if state in visited: continue
            visited.add(state)
            order[state] = count; count += 1
            for _, ns in get_actions_fn(state):
                if ns not in visited:
                    frontier.append(ns)
        return order

    def _plot_order(ax, order, cmap, title):
        max_v = max(order.values()) if order else 1
        grid  = np.full((rows, cols), np.nan)
        for (r, c), v in order.items():
            grid[r, c] = v
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=max_v, origin='upper', alpha=0.85)
        for r in range(rows):
            for c in range(cols):
                state = (r, c)
                if state in walls:
                    ax.add_patch(plt.Rectangle([c-0.5, r-0.5], 1, 1, color='#555'))
                    ax.text(c, r, '\u2593', ha='center', va='center',
                            color='white', fontsize=14)
                elif state == start:
                    ax.text(c, r, 'S', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='black')
                elif state == goal:
                    ax.text(c, r, 'G', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='black')
                elif state in order:
                    ax.text(c, r, str(order[state]), ha='center', va='center',
                            fontsize=9, color='black')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.grid(True, color='white', linewidth=2)

    bfs_ord = _bfs_order()
    dfs_ord = _dfs_order()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    _plot_order(ax1, bfs_ord, 'Blues',   title_bfs)
    _plot_order(ax2, dfs_ord, 'Oranges', title_dfs)
    fig.suptitle("BFS explores level by level (uniform rings)\n"
                 "DFS dives deep before backtracking",
                 fontsize=11, style='italic')
    plt.tight_layout(); plt.show()
    print(f"BFS explored {len(bfs_ord)} cells | DFS explored {len(dfs_ord)} cells")
    print("BFS guarantees shortest path.  DFS uses less memory in deep trees.")


# ═══════════════════════════════════════════════════════════════════════
# 6.  ROMANIA MAP VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def draw_romania_ax(ax, path=None, path_color='#e74c3c',
                    title='Romania Map', start='Arad', goal='Bucharest',
                    city_pos=None, road_map=None):
    """
    Draw Romania map on a given axes object.
    Used as a building block by plot_romania_algorithms().

    Parameters
    ----------
    ax         : matplotlib Axes
    path       : list of city name strings (highlighted route)
    path_color : color for highlighted edges
    title      : subplot title
    start/goal : start and goal cities (colored green/red)
    city_pos   : optional dict override; defaults to ROMANIA_CITY_POS
    road_map   : optional dict override; defaults to ROMANIA_ROADS
    """
    cp = city_pos if city_pos is not None else ROMANIA_CITY_POS
    rm = road_map if road_map is not None else ROMANIA_ROADS
    path_edges = set()
    if path and len(path) > 1:
        for i in range(len(path)-1):
            path_edges.add(frozenset([path[i], path[i+1]]))

    drawn = set()
    for c1, nbrs in rm.items():
        x1, y1 = cp[c1]
        for c2, dist in nbrs.items():
            edge = frozenset([c1, c2])
            if edge in drawn: continue
            drawn.add(edge)
            x2, y2 = cp[c2]
            in_path = edge in path_edges
            ax.plot([x1, x2], [y1, y2],
                    color=path_color if in_path else '#bdc3c7',
                    lw=3.5 if in_path else 1.0,
                    zorder=1 + in_path*2)
            ax.text((x1+x2)/2, (y1+y2)/2, str(dist),
                    fontsize=5.5, ha='center', va='center', color='#7f8c8d', zorder=2)

    for city, (x, y) in cp.items():
        if city == start:
            fc = '#27ae60'
        elif city == goal:
            fc = '#e74c3c'
        elif path and city in path:
            fc = '#f39c12'
        else:
            fc = '#d6eaf8'
        ax.add_patch(plt.Circle((x, y), 0.27, color=fc, ec='white', lw=1.5, zorder=4))
        ax.text(x, y+0.38, city, ha='center', va='bottom', fontsize=6,
                fontweight='bold', color='#2c3e50', zorder=5)
    ax.set_xlim(-0.3, 9.5); ax.set_ylim(1.0, 9.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)


def plot_romania_algorithms(results, road_map=None,
                             suptitle="Romania Road Map \u2014 Algorithm Comparison",
                             start='Arad', goal='Bucharest'):
    """
    Plot 2 or 3 algorithm results as side-by-side Romania maps.

    Parameters
    ----------
    results  : list of (label, path, color, cost_km) tuples
               e.g. [("UCS", ucs_path, '#2980b9', 418),
                     ("Greedy", greedy_path, '#8e44ad', 450),
                     ("A*", astar_path, '#e67e22', 418)]
    road_map : optional road-map dict override (defaults to built-in ROMANIA_ROADS).
               Pass the problem's graph dict here, e.g. romania_map.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6*n+1, 6))
    if n == 1: axes = [axes]

    for ax, (label, path, color, cost) in zip(axes, results):
        cost_str = f"{cost} km" if cost is not None else ""
        draw_romania_ax(ax, path, path_color=color,
                        title=f"{label}\n{cost_str}",
                        start=start, goal=goal,
                        road_map=road_map)

    legend = [
        mpatches.Patch(color='#27ae60', label=f'Start ({start})'),
        mpatches.Patch(color='#e74c3c', label=f'Goal ({goal})'),
        mpatches.Patch(color='#f39c12', label='On path'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.show()


def plot_algorithm_bar_chart(algos, costs, nodes, colors=None,
                              title="Algorithm Comparison"):
    """
    Two bar charts: path costs and nodes expanded for each algorithm.

    Parameters
    ----------
    algos  : list of algorithm name strings
    costs  : list of path costs (km)
    nodes  : list of nodes-expanded counts
    colors : list of bar colors (defaults to blue/purple/orange)
    title  : figure suptitle
    """
    if colors is None:
        colors = ['#2980b9', '#8e44ad', '#e67e22', '#27ae60'][:len(algos)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bars1 = ax1.bar(algos, costs, color=colors, width=0.5,
                    edgecolor='white', linewidth=1.5)
    ax1.set_ylabel("Path Cost (km)", fontsize=11)
    ax1.set_title("Path Cost (lower = better)", fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(costs)*1.3)
    for bar, val in zip(bars1, costs):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                 f'{val} km', ha='center', va='bottom', fontsize=11, fontweight='bold')
    if costs:
        ax1.axhline(y=min(costs), color='green', linestyle='--',
                    alpha=0.6, label=f'Optimal: {min(costs)} km')
        ax1.legend(fontsize=9)

    bars2 = ax2.bar(algos, nodes, color=colors, width=0.5,
                    edgecolor='white', linewidth=1.5)
    ax2.set_ylabel("Nodes Expanded", fontsize=11)
    ax2.set_title("Nodes Expanded (lower = more efficient)", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(nodes)*1.3)
    for bar, val in zip(bars2, nodes):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_facecolor('#fafafa')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_routes_comparison(routes_labels, algo_results,
                            algo_names=None, algo_colors=None,
                            title="Cost & Efficiency Comparison"):
    """
    Grouped bar chart comparing multiple algorithms across multiple routes.

    Parameters
    ----------
    routes_labels : list of route label strings
    algo_results  : list of (costs_list, nodes_list) per algorithm
    algo_names    : list of algorithm names
    algo_colors   : list of colors
    """
    n_algos  = len(algo_results)
    n_routes = len(routes_labels)
    if algo_names is None:
        algo_names = [f'Algo {i+1}' for i in range(n_algos)]
    if algo_colors is None:
        algo_colors = ['#2980b9', '#8e44ad', '#e67e22', '#27ae60'][:n_algos]

    x = np.arange(n_routes)
    w = 0.8 / n_algos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, ((costs, nodes), name, color) in enumerate(
            zip(algo_results, algo_names, algo_colors)):
        offset = (i - n_algos/2 + 0.5) * w
        b1 = ax1.bar(x+offset, costs, w, label=name, color=color,
                     edgecolor='white', lw=1.5)
        b2 = ax2.bar(x+offset, nodes, w, label=name, color=color,
                     edgecolor='white', lw=1.5)
        for bar, v in zip(b1, costs):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
                     str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, v in zip(b2, nodes):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

    for ax, ylabel, subtitle in [
        (ax1, 'Path Cost (km)',   'Path Cost \u2014 lower is optimal'),
        (ax2, 'Nodes Expanded', 'Nodes Expanded \u2014 lower is efficient'),
    ]:
        ax.set_xticks(x); ax.set_xticklabels(routes_labels, fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(subtitle, fontweight='bold')
        ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_facecolor('#fafafa')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════
# 7.  A*  /  f = g + h  VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_fgh(path, heuristic_fn, road_map=None):
    """
    Compute g(n), h(n), f(n) values along a path.

    Parameters
    ----------
    path         : list of city name strings
    heuristic_fn : callable(city) -> h value
    road_map     : optional dict override; defaults to ROMANIA_ROADS

    Returns
    -------
    list of (city, g, h, f) tuples
    """
    rm = road_map if road_map is not None else ROMANIA_ROADS
    result  = []
    cumcost = 0
    for i, city in enumerate(path):
        if i > 0:
            cumcost += rm[path[i-1]][city]
        h = heuristic_fn(city)
        result.append((city, cumcost, h, cumcost + h))
    return result


def plot_astar_path_map(astar_path, fgh_data, title=None,
                         start='Arad', goal='Bucharest',
                         city_pos=None, road_map=None):
    """
    Draw Romania map with A* optimal path and f=g+h annotations on each city.

    Parameters
    ----------
    astar_path : list of city name strings
    fgh_data   : list of (city, g, h, f) tuples (from compute_fgh())
    title      : optional plot title (auto-generated if None)
    """
    cp = city_pos  if city_pos  is not None else ROMANIA_CITY_POS
    rm = road_map  if road_map  is not None else ROMANIA_ROADS

    path_edges = set()
    for i in range(len(astar_path)-1):
        path_edges.add(frozenset([astar_path[i], astar_path[i+1]]))

    fig, ax = plt.subplots(figsize=(12, 7))
    drawn = set()
    for c1, nbrs in rm.items():
        x1, y1 = cp[c1]
        for c2, dist in nbrs.items():
            edge = frozenset([c1, c2])
            if edge in drawn: continue
            drawn.add(edge)
            x2, y2 = cp[c2]
            in_path = edge in path_edges
            ax.plot([x1, x2], [y1, y2],
                    color='#e67e22' if in_path else '#bdc3c7',
                    lw=3.5 if in_path else 1.0,
                    zorder=1 + in_path*2)
            ax.text((x1+x2)/2, (y1+y2)/2, str(dist),
                    fontsize=5.5, ha='center', va='center',
                    color='#7f8c8d', zorder=2)

    for city, (x, y) in cp.items():
        if city == start:       fc = '#27ae60'
        elif city == goal:      fc = '#e74c3c'
        elif city in astar_path: fc = '#f39c12'
        else:                   fc = '#d6eaf8'
        ax.add_patch(plt.Circle((x, y), 0.28, color=fc, ec='white',
                                 lw=1.5, zorder=4))
        ax.text(x, y+0.38, city, ha='center', va='bottom',
                fontsize=6.5, fontweight='bold', color='#2c3e50', zorder=5)

    fgh_dict = {city: (g, h, f) for city, g, h, f in fgh_data}
    for city in astar_path:
        x, y = cp[city]
        g, h, f = fgh_dict[city]
        ax.text(x, y-0.45, f'f={f}\ng={g}+h={h}',
                ha='center', va='top', fontsize=5.5, color='#34495e',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.85), zorder=6)

    total_cost = fgh_dict[astar_path[-1]][0] if astar_path else 0
    if title is None:
        _arrow = ' \u2192 '
        title = (f"A* Search: {start} \u2192 {goal}  |  "
                 f"Optimal cost={total_cost} km\n"
                 "Path: " + _arrow.join(astar_path))
    ax.set_xlim(-0.3, 9.5); ax.set_ylim(0.8, 9.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    legend = [
        mpatches.Patch(color='#27ae60', label='Start'),
        mpatches.Patch(color='#e74c3c', label='Goal'),
        mpatches.Patch(color='#f39c12', label='On A* path'),
        mpatches.Patch(color='#e67e22', label='A* road used'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=8, framealpha=0.9)
    plt.tight_layout(); plt.show()


def plot_fgh_decomposition(path, g_values, h_values, title=None):
    """
    Stacked bar chart showing f(n) = g(n) + h(n) along the A* path.

    Parameters
    ----------
    path     : list of city name strings
    g_values : list of g(n) values (path cost so far)
    h_values : list of h(n) values (heuristic estimate to goal)
    """
    f_values = [g + h for g, h in zip(g_values, h_values)]
    x = np.arange(len(path))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, g_values, label='g(n) \u2014 cost so far',
           color='#2980b9', width=0.5)
    ax.bar(x, h_values, label='h(n) \u2014 heuristic estimate',
           color='#e67e22', width=0.5, bottom=g_values)
    ax.plot(x, f_values, 'k--o', lw=2, ms=7, label='f(n) = g + h', zorder=5)

    for i, (g, h, f) in enumerate(zip(g_values, h_values, f_values)):
        ax.text(x[i], g/2,   str(g), ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        ax.text(x[i], g+h/2, str(h), ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        ax.text(x[i], f+12,  f'f={f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(path, fontsize=10)
    ax.set_ylabel('Cost (km)', fontsize=11)
    if title is None:
        title = ('A* Path: f(n) = g(n) + h(n) Decomposition\n'
                 '(g grows as we travel; h shrinks as we approach the goal)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_facecolor('#fafafa')
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════
# 8.  8-PUZZLE VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def plot_puzzle_heuristic_comparison(labels, misplaced_counts, manhattan_counts,
                                      title=None):
    """
    Grouped bar chart comparing Misplaced Tiles vs Manhattan Distance heuristics.

    Parameters
    ----------
    labels           : list of puzzle difficulty label strings
    misplaced_counts : list of nodes-expanded using misplaced-tiles heuristic
    manhattan_counts : list of nodes-expanded using manhattan-distance heuristic
    """
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x-w/2, misplaced_counts, w, label='Misplaced Tiles',
                color='#e74c3c', edgecolor='white', lw=1.5)
    b2 = ax.bar(x+w/2, manhattan_counts, w, label='Manhattan Distance',
                color='#27ae60', edgecolor='white', lw=1.5)
    for bar, v in zip(list(b1)+list(b2), misplaced_counts+manhattan_counts):
        y_offset = 200 if v > 1000 else 10
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+y_offset,
                f'{v:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Nodes Expanded', fontsize=11)
    if title is None:
        title = ('8-Puzzle: Nodes Expanded by Heuristic\n'
                 '(fewer nodes = stronger heuristic = faster search)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_facecolor('#fafafa')
    plt.tight_layout(); plt.show()

    for i, lbl in enumerate(labels):
        lbl_clean = lbl.replace('\n', ' ')
        ratio = misplaced_counts[i] / manhattan_counts[i] if manhattan_counts[i] else 1
        print(f"  {lbl_clean:25s}: "
              f"Misplaced={misplaced_counts[i]:6,}  "
              f"Manhattan={manhattan_counts[i]:6,}  "
              f"(Manhattan is {ratio:.1f}x better)")


# ═══════════════════════════════════════════════════════════════════════
# 9.  TIC-TAC-TOE VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def draw_ttt_board(board, title="Tic-Tac-Toe", ax=None, show=True):
    """
    Draw a Tic-Tac-Toe board.

    Parameters
    ----------
    board : tuple/list of 9 elements — 'X', 'O', or '.' (empty)
            OR a state dict with a 'board' key
    title : plot title
    ax    : optional matplotlib Axes (creates one if None)
    show  : if True and ax is None, calls plt.show()
    """
    if isinstance(board, dict) and 'board' in board:
        board = board['board']

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

    for i in [1, 2]:
        ax.axhline(i, color='#2c3e50', lw=3)
        ax.axvline(i, color='#2c3e50', lw=3)

    for idx, sym in enumerate(board):
        row, col = divmod(idx, 3)
        y = 2 - row  # flip so row 0 is at top
        bg = ('#eaf4fb' if sym == 'X'
              else '#fdf2f2' if sym == 'O'
              else '#f9f9f9')
        ax.add_patch(plt.Rectangle([col, y], 1, 1, facecolor=bg,
                                    edgecolor='#2c3e50', linewidth=3, zorder=0))
        if sym in ('X', 'O'):
            color = '#2980b9' if sym == 'X' else '#e74c3c'
            ax.text(col+0.5, y+0.5, sym, ha='center', va='center',
                    fontsize=28, fontweight='bold', color=color)

    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_aspect('equal')

    if own_fig and show:
        plt.tight_layout(); plt.show()


def draw_minimax_heatmap(board, is_max_turn, minimax_fn, apply_move_fn,
                          title="Minimax Values for Each Move"):
    """
    For a board state, compute the Minimax value for each empty cell
    and display as a heatmap overlay.

    Parameters
    ----------
    board          : tuple/list of 9 elements ('X', 'O', or '.')
                     OR a state dict with a 'board' key
    is_max_turn    : True if it is the MAX player's (X's) turn
    minimax_fn     : callable(state, is_maximizing) -> (value, best_move)
    apply_move_fn  : callable(state, position) -> new_state
    title          : plot title
    """
    if isinstance(board, dict):
        state = board
        board = board['board']
    else:
        state = {'board': board, 'to_move': 'X' if is_max_turn else 'O'}

    grid_vals = [None] * 9
    for pos in range(9):
        if board[pos] == '.':
            new_state = apply_move_fn(state, pos)
            val, _ = minimax_fn(new_state, is_maximizing=(not is_max_turn))
            grid_vals[pos] = val

    cmap_vals = [float(v) if v is not None else np.nan for v in grid_vals]
    grid = np.array(cmap_vals, dtype=float).reshape(3, 3)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(grid, cmap='RdYlGn', vmin=-1.1, vmax=1.1, aspect='equal')
    for i in [1, 2]:
        ax.axhline(i-0.5, color='white', lw=3)
        ax.axvline(i-0.5, color='white', lw=3)
    for idx, sym in enumerate(board):
        row, col = divmod(idx, 3)
        if sym in ('X', 'O'):
            ax.text(col, row, sym, ha='center', va='center', fontsize=20,
                    fontweight='bold',
                    color='#2980b9' if sym == 'X' else '#e74c3c')
        elif grid_vals[idx] is not None:
            v = grid_vals[idx]
            label = '+1\n(WIN)' if v >= 1 else '0\n(DRAW)' if v == 0 else '-1\n(LOSE)'
            ax.text(col, row, label, ha='center', va='center', fontsize=9,
                    fontweight='bold',
                    color='white' if abs(v) == 1 else 'black')
    ax.set_xticks([]); ax.set_yticks([])
    player_name = 'X (MAX)' if is_max_turn else 'O (MIN)'
    ax.set_title(f"{title}\n({player_name} to move)",
                 fontsize=10, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Lose (-1)', 'Draw (0)', 'Win (+1)'])
    plt.tight_layout(); plt.show()


def plot_minimax_vs_alphabeta(labels, mm_counts, ab_counts,
                               title="Minimax vs Alpha-Beta: Nodes Evaluated"):
    """
    Bar chart comparing Minimax vs Alpha-Beta node evaluation counts.

    Parameters
    ----------
    labels    : list of board-configuration label strings
    mm_counts : list of Minimax node counts
    ab_counts : list of Alpha-Beta node counts
    """
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x-w/2, mm_counts, w, label='Minimax',
                color='#e74c3c', edgecolor='white', lw=1.5)
    b2 = ax.bar(x+w/2, ab_counts, w, label='Alpha-Beta',
                color='#27ae60', edgecolor='white', lw=1.5)
    max_v = max(mm_counts) if mm_counts else 1
    for bar, v in zip(list(b1)+list(b2), mm_counts+ab_counts):
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + max_v*0.02,
                    f'{v:,}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Nodes Evaluated', fontsize=11)
    ax.set_title(f'{title}\n(fewer nodes = same answer, less work!)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_facecolor('#fafafa')
    plt.tight_layout(); plt.show()

    print("\n\U0001f4a1 Summary:")
    for i, lbl in enumerate(labels):
        if ab_counts[i] > 0:
            lbl_clean = lbl.replace('\n', ' ')
            ratio    = mm_counts[i] / ab_counts[i]
            savings  = 100 * (1 - ab_counts[i]/mm_counts[i]) if mm_counts[i] else 0
            print(f"  {lbl_clean:30s}: Minimax={mm_counts[i]:7,}  "
                  f"Alpha-Beta={ab_counts[i]:7,}  "
                  f"({savings:.0f}% reduction, {ratio:.1f}x speedup)")


# ═══════════════════════════════════════════════════════════════════════
# 10. MISSIONARIES & CANNIBALS — state and solution visualisations
# ═══════════════════════════════════════════════════════════════════════

_MC_COL_M    = "#4A90D9"   # missionary blue
_MC_COL_C    = "#E05C5C"   # cannibal red
_MC_COL_BANK = "#A8D5A2"   # grass green
_MC_COL_RIVER= "#B3D9F7"   # river blue
_MC_COL_BOAT = "#8B6914"   # wooden brown
_MC_COL_WARN = "#FF6B35"   # illegal state border
_MC_COL_SAFE = "#27AE60"   # safe state border
_MC_COL_BG   = "#FAFAFA"
_MC_RADIUS   = 0.18

def _mc_draw_bank(ax, cx, cy, n_m, n_c, label="", highlight=None):
    bw, bh = 1.4, 1.6
    r = mpatches.FancyBboxPatch((cx-bw/2, cy-bh/2), bw, bh,
        boxstyle="round,pad=0.05", facecolor=_MC_COL_BANK,
        edgecolor="white", linewidth=2, zorder=2)
    ax.add_patch(r)
    if highlight:
        col = _MC_COL_SAFE if highlight == "safe" else _MC_COL_WARN
        r2 = mpatches.FancyBboxPatch((cx-bw/2-0.05, cy-bh/2-0.05),
            bw+0.1, bh+0.1, boxstyle="round,pad=0.04",
            facecolor="none", edgecolor=col, linewidth=3, zorder=3)
        ax.add_patch(r2)
    for x in (np.linspace(cx-0.5, cx+0.5, max(n_m, 1))[:n_m]):
        c = plt.Circle((x, cy+0.35), _MC_RADIUS, color=_MC_COL_M, zorder=4)
        ax.add_patch(c)
        ax.text(x, cy+0.35, "M", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=5)
    for x in (np.linspace(cx-0.5, cx+0.5, max(n_c, 1))[:n_c]):
        c = plt.Circle((x, cy-0.20), _MC_RADIUS, color=_MC_COL_C, zorder=4)
        ax.add_patch(c)
        ax.text(x, cy-0.20, "C", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=5)
    if label:
        ax.text(cx, cy-bh/2-0.22, label,
                ha="center", va="top", fontsize=8.5, color="#333")

def _mc_draw_boat(ax, cx, cy, direction=None, people=""):
    bw, bh = 0.7, 0.32
    ax.add_patch(mpatches.FancyBboxPatch((cx-bw/2, cy-bh/2), bw, bh,
        boxstyle="round,pad=0.04", facecolor=_MC_COL_BOAT,
        edgecolor="#5A3A0A", linewidth=1.5, zorder=4))
    if direction == "right":
        ax.annotate("", xy=(cx+bw/2+0.15, cy), xytext=(cx+bw/2, cy),
            arrowprops=dict(arrowstyle="->", color="#5A3A0A", lw=1.5), zorder=5)
    elif direction == "left":
        ax.annotate("", xy=(cx-bw/2-0.15, cy), xytext=(cx-bw/2, cy),
            arrowprops=dict(arrowstyle="->", color="#5A3A0A", lw=1.5), zorder=5)
    if people:
        ax.text(cx, cy, people, ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=5)

def _mc_draw_state(ax, m_l, c_l, boat, title="", highlight=None,
                   caption="", lbl_l="", lbl_r="",
                   direction=None, boat_lbl=""):
    m_r, c_r = 3-m_l, 3-c_l
    ax.add_patch(mpatches.FancyBboxPatch((-0.55, -1.0), 1.1, 2.0,
        boxstyle="square,pad=0", facecolor=_MC_COL_RIVER,
        edgecolor="none", zorder=1))
    for y in np.linspace(-0.7, 0.7, 4):
        xs = np.linspace(-0.5, 0.5, 20)
        ax.plot(xs, y + 0.04*np.sin(np.linspace(0, 4*np.pi, 20)),
                color="#7EC8E3", lw=0.8, alpha=0.6, zorder=2)
    _mc_draw_bank(ax, -1.3, 0, m_l, c_l,
                  label=lbl_l or f"({m_l}M, {c_l}C)", highlight=highlight)
    _mc_draw_bank(ax,  1.3, 0, m_r, c_r,
                  label=lbl_r or f"({m_r}M, {c_r}C)")
    bx = -0.7 if boat == 0 else 0.7
    _mc_draw_boat(ax, bx, -0.4, direction=direction, people=boat_lbl)
    if title:
        ax.text(0, 1.1, title, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222")
    if caption:
        col = (_MC_COL_WARN if highlight == "illegal"
               else (_MC_COL_SAFE if highlight == "safe" else "#555"))
        ax.text(0, -1.25, caption, ha="center", va="top",
                fontsize=8, color=col, style="italic")
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-1.5, 1.4)
    ax.set_aspect("equal"); ax.axis("off")

_MC_LEGEND = [
    mpatches.Patch(facecolor=_MC_COL_M,     label="Missionary (M)"),
    mpatches.Patch(facecolor=_MC_COL_C,     label="Cannibal (C)"),
    mpatches.Patch(facecolor=_MC_COL_BOAT,  label="Boat"),
    mpatches.Patch(facecolor=_MC_COL_RIVER, label="River"),
]

def show_mc_state_diagram():
    """
    Figure 1 — State Representation.
    Shows Initial, a mid-game, and Goal state side by side.
    Called in Lab 3, Exercise 1.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor(_MC_COL_BG)
    cfg = [
        (3,3,0, "Initial State\n(3, 3, 0)",  None,      "Everyone starts on the left",
         "Left bank\n(3M, 3C)", "Right bank\n(0M, 0C)", None, ""),
        (1,1,1, "A Mid-Game State\n(1, 1, 1)","safe",   "Boat back to left\nafter ferrying",
         "Left bank\n(1M, 1C)", "Right bank\n(2M, 2C)", "left", "1M1C"),
        (0,0,1, "Goal State\n(0, 0, 1)",     "safe",    "Everyone crossed!\nGoal reached",
         "Left bank\n(0M, 0C)", "Right bank\n(3M, 3C)", None, ""),
    ]
    for ax, (ml,cl,bt,ti,hl,ca,ll,lr,di,bl) in zip(axes, cfg):
        ax.set_facecolor(_MC_COL_BG)
        _mc_draw_state(ax,ml,cl,bt, title=ti, highlight=hl, caption=ca,
                       lbl_l=ll, lbl_r=lr, direction=di, boat_lbl=bl)
    fig.suptitle("The Missionaries & Cannibals Problem — State Representation",
                 fontsize=12, fontweight="bold", y=1.0, color="#1A1A2E")
    fig.legend(handles=_MC_LEGEND, loc="lower center", ncol=4,
               framealpha=0.9, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(); plt.show()

def show_mc_safety_diagram():
    """
    Figure 2 — Legal vs Illegal States.
    Highlights a safe state, an illegal state, and another safe state.
    Called in Lab 3, Exercise 1.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor(_MC_COL_BG)
    cfg = [
        (2,2,0, "Safe State\n(2, 2, 0)",       "safe",    "2M >= 2C on both banks",
         "Left\n(2M, 2C)", "Right\n(1M, 1C)", None, ""),
        (1,2,0, "  Illegal State\n(1, 2, 0)",  "illegal",
         "2C > 1M on LEFT bank\nMissionaries get eaten!",
         "Left\n(1M, 2C)", "Right\n(2M, 1C)", "right", "1M1C"),
        (0,1,1, "Safe State\n(0, 1, 1)",        "safe",
         "Left: 0M so no danger\nRight: 3M >= 2C",
         "Left\n(0M, 1C)", "Right\n(3M, 2C)", None, ""),
    ]
    for ax, (ml,cl,bt,ti,hl,ca,ll,lr,di,bl) in zip(axes, cfg):
        ax.set_facecolor(_MC_COL_BG)
        _mc_draw_state(ax,ml,cl,bt, title=ti, highlight=hl, caption=ca,
                       lbl_l=ll, lbl_r=lr, direction=di, boat_lbl=bl)
    fig.suptitle("Which States Are Legal? — The Safety Constraint",
                 fontsize=12, fontweight="bold", y=1.0, color="#1A1A2E")
    fig.legend(handles=_MC_LEGEND, loc="lower center", ncol=4,
               framealpha=0.9, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(); plt.show()

def show_mc_solution_path(solution_states):
    """
    Figure 3 — BFS Solution Path.
    solution_states : list of (m_left, c_left, boat) tuples (length = steps+1)
    Called in Lab 3 Solution version after completing Exercise 1.
    """
    n = len(solution_states)
    cols, rows = 6, -(-n // 6)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*4.0))
    fig.patch.set_facecolor(_MC_COL_BG)
    axes = axes.flatten()
    for idx, (m_l, c_l, boat) in enumerate(solution_states):
        ax = axes[idx]; ax.set_facecolor(_MC_COL_BG)
        is_goal = (m_l == 0 and c_l == 0 and boat == 1)
        title = ("Step 0 - Start" if idx == 0
                 else f"Step {idx} - GOAL" if is_goal
                 else f"Step {idx}")
        di = None
        if idx < n - 1:
            _, _, nboat = solution_states[idx+1]
            di = "right" if nboat == 1 else "left"
        _mc_draw_state(ax, m_l, c_l, boat, title=title,
                       highlight="safe" if is_goal else None, direction=di)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"BFS Solution Path — {n} States ({n-1} moves)",
                 fontsize=12, fontweight="bold", y=1.01, color="#1A1A2E")
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════
# ALIASES — backward compatibility
# ═══════════════════════════════════════════════════════════════════════

# plot_algorithm_comparison is an alias for plot_algorithm_bar_chart
plot_algorithm_comparison = plot_algorithm_bar_chart
