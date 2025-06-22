# src/python/sar_simulation/__init__.py

"""
SAR Simulation Package

A comprehensive Search and Rescue simulation system featuring:
- Distributed POMDP planning
- Custom probability maps
- 3D visualization
- Multiple scenario types
"""

__version__ = "1.0.0"
__author__ = "SAR Simulation Team"

# Import main components for easy access
from .core.entities import Drone, SearchTarget, Building, DroneStatus
from .planning.pomdp_planner import DistributedPOMDPSearchPlanner
from .planning.probability_maps import (
    ProbabilityMapGenerator, 
    PredefinedHypotheses, 
    SearchHypothesis,
    ProbabilityRegion,
    IncidentType
)

__all__ = [
    'Drone', 'SearchTarget', 'Building', 'DroneStatus',
    'DistributedPOMDPSearchPlanner',
    'ProbabilityMapGenerator', 'PredefinedHypotheses', 
    'SearchHypothesis', 'ProbabilityRegion', 'IncidentType'
]