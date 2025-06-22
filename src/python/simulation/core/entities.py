# src/python/sar_simulation/core/entities.py

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import deque

class DroneStatus(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    INVESTIGATING = "investigating"
    RETURNING = "returning"
    CHARGING = "charging"
    COORDINATING = "coordinating"

class ActionType(Enum):
    MOVE = "move"
    INVESTIGATE = "investigate"
    COMMUNICATE = "communicate"
    RETURN_BASE = "return_base"
    WAIT = "wait"

@dataclass
class BeliefState:
    """Represents belief about target locations and environment state"""
    target_probability_map: np.ndarray  # Probability distribution over grid cells
    coverage_map: np.ndarray  # How well each area has been searched
    communication_map: Dict[str, float]  # Last communication time with other drones
    time_step: int = 0

@dataclass
class Action:
    type: ActionType
    target_position: Optional[np.ndarray] = None
    duration: float = 1.0
    priority: float = 0.0

@dataclass
class Observation:
    """Sensor observation from a drone"""
    position: np.ndarray
    detected_targets: List[str]
    sensor_readings: Dict[str, float]  # Various sensor data
    timestamp: float
    confidence: float

@dataclass
class Drone:
    id: str
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [roll, pitch, yaw]
    battery_level: float  # 0.0 to 1.0
    status: DroneStatus
    sensor_range: float = 50.0
    speed: float = 10.0  # m/s
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0)
    
    # POMDP components
    belief_state: BeliefState = field(default_factory=lambda: None)
    current_action: Optional[Action] = None
    action_queue: deque = field(default_factory=deque)
    observations: List[Observation] = field(default_factory=list)
    
    # Communication and coordination
    communication_range: float = 100.0
    last_communication: Dict[str, float] = field(default_factory=dict)
    shared_information: Dict[str, any] = field(default_factory=dict)

@dataclass
class SearchTarget:
    id: str
    position: np.ndarray  # [x, y, z]
    size: float
    detected: bool = False
    confidence: float = 0.0
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    # POMDP additions
    true_detection_probability: float = 0.8  # How detectable this target actually is
    priority: float = 1.0

@dataclass
class Building:
    id: str
    vertices: np.ndarray
    faces: np.ndarray
    position: np.ndarray  # [x, y, z]
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)