"""
Visualize campus map to verify coordinates and graph structure match the PDF map.
Coordinates are rotated to match the PDF map's angle.

Edge types:
  - Gray solid lines: normal edges
  - Red solid lines: bridge crossing edges (edges connecting to bridge nodes)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Load campus map
with open('campus_map.json', 'r') as f:
    campus_data = json.load(f)

coords = campus_data['campus_coords']
graph = campus_data['campus_graph']
bridges = campus_data['campus_bridges']
regions = campus_data['regions']

west = set(regions['west'])
east = set(regions['east'])
bridge_set = set(regions['bridges'])

# Apply rotation transformation to match PDF map angle (~25 degrees)
def rotate_coords(x, y, angle_degrees=25):
    """Rotate coordinates by angle to match PDF map orientation"""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return x_rot, y_rot

# Transform all coordinates - adjust angle to match PDF map orientation
rotated_coords = {}
for loc, (x, y) in coords.items():
    x_rot, y_rot = rotate_coords(x, y, angle_degrees=22)
    rotated_coords[loc] = (x_rot, y_rot)

coords = rotated_coords

# Classify edges
def is_bridge_crossing(src, dst):
    """Check if edge involves a bridge node."""
    return src in bridge_set or dst in bridge_set

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))

# Draw edges (graph connections)
print("Drawing edges...")
drawn_pairs = set()
for src, neighbors in graph.items():
    x1, y1 = coords[src]
    for dst, cost in neighbors.items():
        x2, y2 = coords[dst]

        # Avoid drawing both directions of same edge
        pair = tuple(sorted([src, dst]))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)

        if is_bridge_crossing(src, dst):
            # Bridge crossing: red solid
            ax.plot([x1, x2], [y1, y2], color='#FF6B6B', linewidth=3,
                    zorder=2, alpha=0.6)
        else:
            # Normal edge: gray
            ax.plot([x1, x2], [y1, y2], color='#CCCCCC', linewidth=1,
                    zorder=1, alpha=0.6)

# Draw nodes (buildings)
print("Drawing nodes...")
for loc, (x, y) in coords.items():
    if loc in bridge_set:
        # Bridge zone nodes
        ax.plot(x, y, 'o', color='#FF6B6B', markersize=12, zorder=3,
               markeredgecolor='darkred', markeredgewidth=1.5)
        ax.text(x, y-16, loc.replace('_', '\n'), ha='center', fontsize=6, fontweight='bold')
    elif loc in west:
        # West campus buildings
        ax.plot(x, y, 'o', color='#4ECDC4', markersize=8, zorder=3)
        ax.text(x, y-16, loc.replace('_', '\n'), ha='center', fontsize=6.5)
    else:
        # East campus buildings
        ax.plot(x, y, 's', color='#FFD93D', markersize=8, zorder=3)
        ax.text(x, y-16, loc.replace('_', '\n'), ha='center', fontsize=6.5)

# Add legend
blue_circle = plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='#4ECDC4', markersize=8, label='West Campus')
yellow_square = plt.Line2D([0], [0], marker='s', color='w',
                          markerfacecolor='#FFD93D', markersize=8, label='East Campus')
red_bridge = plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#FF6B6B', markersize=10, label='Bridge Node')
gray_edge = plt.Line2D([0], [0], color='#CCCCCC', linewidth=2, label='Normal Edge')
red_edge = plt.Line2D([0], [0], color='#FF6B6B', linewidth=3, label='Bridge Crossing')

ax.legend(handles=[blue_circle, yellow_square, red_bridge, gray_edge, red_edge],
         loc='upper right', fontsize=9, framealpha=0.9)

# Title and labels
ax.set_xlabel('X Coordinate', fontsize=11)
ax.set_ylabel('Y Coordinate', fontsize=11)
ax.set_title('UNNC Campus Map: Building Locations and Connections\n'
            '(Red = bridge crossings with time-based congestion)',
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
# Set limits based on rotated coordinates
all_x = [c[0] for c in coords.values()]
all_y = [c[1] for c in coords.values()]
margin = 100
ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('campus_map_visualization.png', dpi=150, bbox_inches='tight')
print("Visualization saved to campus_map_visualization.png")
plt.close()

# Print statistics
print("\n" + "="*60)
print("CAMPUS MAP STATISTICS")
print("="*60)
print(f"Total buildings: {len(coords)}")
print(f"  - West campus: {len(regions['west'])}")
print(f"  - East campus: {len(regions['east'])}")
print(f"  - Bridge nodes: {len(regions['bridges'])}")

total_edges = sum(len(neighbors) for neighbors in graph.values())
bridge_edges = sum(
    1 for u in graph for v in graph[u]
    if u in bridge_set or v in bridge_set
)
print(f"\nTotal directed edges: {total_edges}")
print(f"  - Bridge-connected edges: {bridge_edges}")
print(f"  - Normal edges: {total_edges - bridge_edges}")

print(f"\nBridge congestion schedules:")
for bridge_name, bridge_info in bridges.items():
    print(f"  - {bridge_name}: {bridge_info['description']}")
    congestion = bridge_info.get('congestion', {})
    if congestion:
        hours = sorted(congestion.keys(), key=int)
        print(f"    Congested hours: {', '.join(hours)} (penalties: {[congestion[h] for h in hours]})")
    else:
        print(f"    No congestion schedule")

print("\n" + "="*60)
