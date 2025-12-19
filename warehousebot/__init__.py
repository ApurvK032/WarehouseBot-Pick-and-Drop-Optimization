"""
WarehouseBot: Pick and Drop Optimization

A modular package for optimizing warehouse robot task sequencing with
pickup/drop precedence constraints.

Modules:
    astar: A* pathfinding on 2D grids
    warehouse: Warehouse map definitions and generation
    cost_matrix: Precompute travel costs between all stops
    sequencing: Task sequencing with greedy, hill climbing, and simulated annealing
"""

__version__ = "1.0.0"

__all__ = []
