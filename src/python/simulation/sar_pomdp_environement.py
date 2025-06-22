# src/python/simulation/sar_pomdp_environment.py

import numpy as np
import open3d as o3d
import trimesh
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import cv2
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import math
from collections import defaultdict, deque

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

class DistributedPOMDPSearchPlanner:
    """
    Distributed POMDP planner for SAR operations
    
    Each drone maintains its own belief state and plans actions based on:
    - Current belief about target locations
    - Information shared from other drones
    - Mission objectives and constraints
    """
    
    def __init__(self, grid_resolution: float = 10.0, planning_horizon: int = 5):
        self.grid_resolution = grid_resolution
        self.planning_horizon = planning_horizon
        self.information_decay_rate = 0.95  # How quickly old information becomes less reliable
        
    def initialize_belief_state(self, drone: Drone, bounds: Tuple[float, float, float, float]) -> BeliefState:
        """Initialize belief state for a drone"""
        min_x, min_y, max_x, max_y = bounds
        
        # Create grid for probability maps
        x_cells = int((max_x - min_x) / self.grid_resolution)
        y_cells = int((max_y - min_y) / self.grid_resolution)
        
        # Initialize uniform prior for target locations
        target_prob_map = np.ones((x_cells, y_cells)) / (x_cells * y_cells)
        
        # Initialize zero coverage
        coverage_map = np.zeros((x_cells, y_cells))
        
        return BeliefState(
            target_probability_map=target_prob_map,
            coverage_map=coverage_map,
            communication_map={},
            time_step=0
        )
    
    def update_belief_state(self, drone: Drone, observation: Observation, 
                          other_drones: Dict[str, Drone]) -> BeliefState:
        """Update belief state based on new observation and communication"""
        belief = drone.belief_state
        
        # Update coverage based on current observation
        self._update_coverage_map(belief, observation)
        
        # Update target probability based on detection/non-detection
        self._update_target_probabilities(belief, observation)
        
        # Incorporate information from other drones
        self._incorporate_shared_information(belief, drone, other_drones)
        
        # Apply temporal decay to old information
        self._apply_information_decay(belief)
        
        belief.time_step += 1
        return belief
    
    def _update_coverage_map(self, belief: BeliefState, observation: Observation):
        """Update coverage map based on sensor observation"""
        x_idx, y_idx = self._position_to_grid_index(observation.position, belief)
        
        # Circular sensor coverage pattern
        sensor_radius_cells = int(50.0 / self.grid_resolution)  # 50m sensor range
        
        for dx in range(-sensor_radius_cells, sensor_radius_cells + 1):
            for dy in range(-sensor_radius_cells, sensor_radius_cells + 1):
                if dx*dx + dy*dy <= sensor_radius_cells*sensor_radius_cells:
                    xi, yi = x_idx + dx, y_idx + dy
                    if (0 <= xi < belief.coverage_map.shape[0] and 
                        0 <= yi < belief.coverage_map.shape[1]):
                        # Increase coverage with distance-based weighting
                        distance = np.sqrt(dx*dx + dy*dy) / sensor_radius_cells
                        coverage_increase = observation.confidence * (1.0 - distance)
                        belief.coverage_map[xi, yi] += coverage_increase
                        belief.coverage_map[xi, yi] = min(1.0, belief.coverage_map[xi, yi])
    
    def _update_target_probabilities(self, belief: BeliefState, observation: Observation):
        """Update target probability map based on detection results"""
        x_idx, y_idx = self._position_to_grid_index(observation.position, belief)
        sensor_radius_cells = int(50.0 / self.grid_resolution)
        
        if len(observation.detected_targets) > 0:
            # Target detected - increase probability in detection area
            self._increase_target_probability(belief, x_idx, y_idx, sensor_radius_cells, 0.8)
        else:
            # No detection - decrease probability in searched area
            self._decrease_target_probability(belief, x_idx, y_idx, sensor_radius_cells, 0.1)
    
    def _increase_target_probability(self, belief: BeliefState, x_idx: int, y_idx: int, 
                                   radius: int, factor: float):
        """Increase target probability in a circular area"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    xi, yi = x_idx + dx, y_idx + dy
                    if (0 <= xi < belief.target_probability_map.shape[0] and 
                        0 <= yi < belief.target_probability_map.shape[1]):
                        belief.target_probability_map[xi, yi] *= (1 + factor)
        
        # Renormalize
        total_prob = np.sum(belief.target_probability_map)
        if total_prob > 0:
            belief.target_probability_map /= total_prob
    
    def _decrease_target_probability(self, belief: BeliefState, x_idx: int, y_idx: int, 
                                   radius: int, factor: float):
        """Decrease target probability in a circular area"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    xi, yi = x_idx + dx, y_idx + dy
                    if (0 <= xi < belief.target_probability_map.shape[0] and 
                        0 <= yi < belief.target_probability_map.shape[1]):
                        belief.target_probability_map[xi, yi] *= (1 - factor)
        
        # Renormalize
        total_prob = np.sum(belief.target_probability_map)
        if total_prob > 0:
            belief.target_probability_map /= total_prob
    
    def _incorporate_shared_information(self, belief: BeliefState, drone: Drone, 
                                      other_drones: Dict[str, Drone]):
        """Incorporate information shared from other drones"""
        for other_id, other_drone in other_drones.items():
            if other_id == drone.id:
                continue
                
            # Check if within communication range
            distance = np.linalg.norm(drone.position[:2] - other_drone.position[:2])
            if distance <= drone.communication_range:
                # Share information
                self._merge_belief_states(belief, other_drone.belief_state, weight=0.3)
                belief.communication_map[other_id] = time.time()
    
    def _merge_belief_states(self, belief1: BeliefState, belief2: BeliefState, weight: float):
        """Merge information from another drone's belief state"""
        if belief2 is None:
            return
            
        # Weighted average of probability maps
        belief1.target_probability_map = (
            (1 - weight) * belief1.target_probability_map + 
            weight * belief2.target_probability_map
        )
        
        # Take maximum coverage (conservative approach)
        belief1.coverage_map = np.maximum(belief1.coverage_map, belief2.coverage_map)
    
    def _apply_information_decay(self, belief: BeliefState):
        """Apply temporal decay to belief state"""
        # Slowly decay coverage to encourage re-searching old areas
        belief.coverage_map *= self.information_decay_rate
        
        # Slowly return target probabilities toward uniform distribution
        uniform_prob = 1.0 / belief.target_probability_map.size
        belief.target_probability_map = (
            0.99 * belief.target_probability_map + 
            0.01 * uniform_prob
        )
    
    def plan_action(self, drone: Drone, other_drones: Dict[str, Drone], 
                   bounds: Tuple[float, float, float, float]) -> Action:
        """Plan next action for a drone using POMDP planning"""
        
        if drone.battery_level < 0.3:
            return Action(type=ActionType.RETURN_BASE, priority=10.0)
        
        if drone.status == DroneStatus.INVESTIGATING:
            return self._plan_investigation_action(drone)
        
        # Main search planning
        return self._plan_search_action(drone, other_drones, bounds)
    
    def _plan_investigation_action(self, drone: Drone) -> Action:
        """Plan action for investigating a detected target"""
        # Simple investigation: hover and gather more data
        return Action(
            type=ActionType.INVESTIGATE,
            target_position=drone.position,
            duration=3.0,
            priority=8.0
        )
    
    def _plan_search_action(self, drone: Drone, other_drones: Dict[str, Drone], 
                          bounds: Tuple[float, float, float, float]) -> Action:
        """Plan search action using information-theoretic approach"""
        
        # Calculate information gain for potential positions
        candidate_positions = self._generate_candidate_positions(drone, bounds)
        best_position = None
        best_score = -float('inf')
        
        for pos in candidate_positions:
            score = self._calculate_information_gain(pos, drone, other_drones)
            if score > best_score:
                best_score = score
                best_position = pos
        
        if best_position is not None:
            return Action(
                type=ActionType.MOVE,
                target_position=best_position,
                priority=best_score
            )
        else:
            # Default action: random search
            return self._generate_random_search_action(drone, bounds)
    
    def _generate_candidate_positions(self, drone: Drone, bounds: Tuple[float, float, float, float], 
                                    num_candidates: int = 8) -> List[np.ndarray]:
        """Generate candidate positions for the drone to consider"""
        min_x, min_y, max_x, max_y = bounds
        candidates = []
        
        # Generate positions in a pattern around the drone
        current_pos = drone.position
        search_radius = drone.sensor_range * 1.5
        
        for i in range(num_candidates):
            angle = 2 * np.pi * i / num_candidates
            x = current_pos[0] + search_radius * np.cos(angle)
            y = current_pos[1] + search_radius * np.sin(angle)
            z = current_pos[2]  # Keep same altitude
            
            # Ensure within bounds
            x = np.clip(x, min_x + 50, max_x - 50)
            y = np.clip(y, min_y + 50, max_y - 50)
            z = np.clip(z, 20, 100)
            
            candidates.append(np.array([x, y, z]))
        
        return candidates
    
    def _calculate_information_gain(self, position: np.ndarray, drone: Drone, 
                                  other_drones: Dict[str, Drone]) -> float:
        """Calculate expected information gain from moving to a position"""
        belief = drone.belief_state
        x_idx, y_idx = self._position_to_grid_index(position, belief)
        
        # Information gain components:
        
        # 1. Target detection probability
        sensor_radius_cells = int(drone.sensor_range / self.grid_resolution)
        target_prob_gain = 0.0
        
        for dx in range(-sensor_radius_cells, sensor_radius_cells + 1):
            for dy in range(-sensor_radius_cells, sensor_radius_cells + 1):
                if dx*dx + dy*dy <= sensor_radius_cells*sensor_radius_cells:
                    xi, yi = x_idx + dx, y_idx + dy
                    if (0 <= xi < belief.target_probability_map.shape[0] and 
                        0 <= yi < belief.target_probability_map.shape[1]):
                        target_prob_gain += belief.target_probability_map[xi, yi]
        
        # 2. Coverage improvement (prefer unexplored areas)
        coverage_gain = 0.0
        for dx in range(-sensor_radius_cells, sensor_radius_cells + 1):
            for dy in range(-sensor_radius_cells, sensor_radius_cells + 1):
                if dx*dx + dy*dy <= sensor_radius_cells*sensor_radius_cells:
                    xi, yi = x_idx + dx, y_idx + dy
                    if (0 <= xi < belief.coverage_map.shape[0] and 
                        0 <= yi < belief.coverage_map.shape[1]):
                        coverage_gain += (1.0 - belief.coverage_map[xi, yi])
        
        # 3. Coordination penalty (avoid clustering with other drones)
        coordination_penalty = 0.0
        for other_drone in other_drones.values():
            if other_drone.id != drone.id:
                distance = np.linalg.norm(position[:2] - other_drone.position[:2])
                if distance < drone.sensor_range * 2:  # Too close
                    coordination_penalty += (drone.sensor_range * 2 - distance) / (drone.sensor_range * 2)
        
        # 4. Distance penalty (prefer closer positions)
        distance_penalty = np.linalg.norm(position - drone.position) / (drone.speed * 10)
        
        # Combine scores
        total_score = (
            2.0 * target_prob_gain +
            1.0 * coverage_gain -
            1.5 * coordination_penalty -
            0.5 * distance_penalty
        )
        
        return total_score
    
    def _generate_random_search_action(self, drone: Drone, bounds: Tuple[float, float, float, float]) -> Action:
        """Generate a random search action as fallback"""
        min_x, min_y, max_x, max_y = bounds
        
        # Random position within bounds
        target_pos = np.array([
            np.random.uniform(min_x + 50, max_x - 50),
            np.random.uniform(min_y + 50, max_y - 50),
            np.random.uniform(30, 80)
        ])
        
        return Action(
            type=ActionType.MOVE,
            target_position=target_pos,
            priority=1.0
        )
    
    def _position_to_grid_index(self, position: np.ndarray, belief: BeliefState) -> Tuple[int, int]:
        """Convert world position to grid indices"""
        # Assuming grid covers the full bounds starting from origin
        x_idx = int(position[0] / self.grid_resolution) + belief.target_probability_map.shape[0] // 2
        y_idx = int(position[1] / self.grid_resolution) + belief.target_probability_map.shape[1] // 2
        
        x_idx = np.clip(x_idx, 0, belief.target_probability_map.shape[0] - 1)
        y_idx = np.clip(y_idx, 0, belief.target_probability_map.shape[1] - 1)
        
        return x_idx, y_idx

class SARPOMDPEnvironment:
    """Enhanced SAR Environment with Distributed POMDP Search"""
    
    def __init__(self, bounds: Tuple[float, float, float, float] = (-200, -200, 200, 200)):
        self.bounds = bounds
        self.drones: Dict[str, Drone] = {}
        self.targets: Dict[str, SearchTarget] = {}
        self.buildings: Dict[str, Building] = {}
        
        # POMDP planner
        self.planner = DistributedPOMDPSearchPlanner()
        
        # Visualization components
        self.vis = None
        self.is_running = False
        self.last_update_time = time.time()
        self.update_interval = 0.5  # 2 FPS for more deliberate planning
        
        # Environment state
        self.terrain = self._generate_terrain()
        self.search_area = self._define_search_area()
        
        # Geometry tracking for updates
        self.geometry_objects = {}
        
        # Initialize 3D visualization
        self._setup_visualization()
    
    def _generate_terrain(self) -> trimesh.Trimesh:
        """Generate realistic terrain using noise"""
        min_x, min_y, max_x, max_y = self.bounds
        
        # Create terrain grid
        x = np.linspace(min_x, max_x, 100)
        y = np.linspace(min_y, max_y, 100)
        X, Y = np.meshgrid(x, y)
        
        # Generate height using Perlin-like noise
        Z = np.zeros_like(X)
        for i in range(3):
            freq = 0.01 * (2 ** i)
            amplitude = 10 / (2 ** i)
            Z += amplitude * np.sin(freq * X) * np.cos(freq * Y)
        
        # Create terrain mesh
        vertices = []
        faces = []
        
        for i in range(len(x)-1):
            for j in range(len(y)-1):
                # Create quad vertices
                v1 = [X[j,i], Y[j,i], Z[j,i]]
                v2 = [X[j,i+1], Y[j,i+1], Z[j,i+1]]
                v3 = [X[j+1,i+1], Y[j+1,i+1], Z[j+1,i+1]]
                v4 = [X[j+1,i], Y[j+1,i], Z[j+1,i]]
                
                base_idx = len(vertices)
                vertices.extend([v1, v2, v3, v4])
                
                # Create two triangles per quad
                faces.extend([
                    [base_idx, base_idx+1, base_idx+2],
                    [base_idx, base_idx+2, base_idx+3]
                ])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _define_search_area(self) -> Polygon:
        """Define the search area boundary"""
        min_x, min_y, max_x, max_y = self.bounds
        margin = 50
        return box(min_x + margin, min_y + margin, max_x - margin, max_y - margin)
    
    def _setup_visualization(self):
        """Initialize Open3D visualization"""
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="SAR POMDP Mission Simulation", width=1200, height=800)
        
        # Register key callbacks
        self.vis.register_key_callback(256, self._on_escape)  # ESC key
        
        # Add terrain
        terrain_o3d = o3d.geometry.TriangleMesh()
        terrain_o3d.vertices = o3d.utility.Vector3dVector(self.terrain.vertices)
        terrain_o3d.triangles = o3d.utility.Vector3iVector(self.terrain.faces)
        terrain_o3d.paint_uniform_color([0.4, 0.6, 0.2])  # Green terrain
        terrain_o3d.compute_vertex_normals()
        self.vis.add_geometry(terrain_o3d)
        self.geometry_objects['terrain'] = terrain_o3d
        
        # Add search area boundary
        self._add_search_area_boundary()
        
        # Set up camera
        self._setup_camera()
    
    def _on_escape(self, vis):
        """Handle escape key press"""
        self.is_running = False
        return False
    
    def _add_search_area_boundary(self):
        """Add search area boundary visualization"""
        coords = list(self.search_area.exterior.coords)
        points = []
        lines = []
        
        for i, (x, y) in enumerate(coords[:-1]):
            points.append([x, y, 20])
            if i < len(coords) - 2:
                lines.append([i, i + 1])
            else:
                lines.append([i, 0])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1.0, 1.0, 0.0])
        self.vis.add_geometry(line_set)
        self.geometry_objects['search_boundary'] = line_set
    
    def _setup_camera(self):
        """Setup optimal camera view"""
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_front([0.5, -0.5, -0.7])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
    
    def add_drone(self, drone_id: str, initial_position: np.ndarray = None) -> Drone:
        """Add a drone to the simulation"""
        if initial_position is None:
            min_x, min_y, max_x, max_y = self.bounds
            initial_position = np.array([
                np.random.uniform(min_x + 50, max_x - 50),
                np.random.uniform(min_y + 50, max_y - 50),
                50.0
            ])
        
        drone = Drone(
            id=drone_id,
            position=initial_position.copy(),
            orientation=np.array([0, 0, 0]),
            battery_level=1.0,
            status=DroneStatus.IDLE,
            color=(np.random.random(), np.random.random(), np.random.random())
        )
        
        # Initialize POMDP belief state
        drone.belief_state = self.planner.initialize_belief_state(drone, self.bounds)
        
        self.drones[drone_id] = drone
        self._add_drone_to_visualization(drone)
        return drone
    
    def _add_drone_to_visualization(self, drone: Drone):
        """Add drone mesh to visualization"""
        # Create drone representation
        drone_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=2, height=1)
        drone_mesh.translate(drone.position)
        drone_mesh.paint_uniform_color(drone.color)
        
        # Add sensor range as wireframe
        sensor_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=drone.sensor_range, resolution=20)
        sensor_sphere.translate(drone.position)
        sensor_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(sensor_sphere)
        sensor_wireframe.paint_uniform_color([0.5, 0.5, 0.8])
        
        # Store references
        drone._mesh = drone_mesh
        drone._sensor_sphere = sensor_wireframe
        drone._last_position = drone.position.copy()
        
        self.vis.add_geometry(drone_mesh)
        self.vis.add_geometry(sensor_wireframe)
        
        self.geometry_objects[f'drone_{drone.id}'] = drone_mesh
        self.geometry_objects[f'sensor_{drone.id}'] = sensor_wireframe
    
    def add_target(self, target_id: str, position: np.ndarray = None) -> SearchTarget:
        """Add a search target to the simulation"""
        if position is None:
            min_x, min_y, max_x, max_y = self.search_area.bounds
            position = np.array([
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y),
                np.random.uniform(0, 5)
            ])
        
        target = SearchTarget(
            id=target_id,
            position=position.copy(),
            size=np.random.uniform(1, 3),
            true_detection_probability=np.random.uniform(0.6, 0.95)
        )
        
        self.targets[target_id] = target
        self._add_target_to_visualization(target)
        return target
    
    def _add_target_to_visualization(self, target: SearchTarget):
        """Add target to visualization"""
        target_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=target.size)
        target_mesh.translate(target.position)
        target_mesh.paint_uniform_color(target.color)
        
        target._mesh = target_mesh
        self.vis.add_geometry(target_mesh)
        self.geometry_objects[f'target_{target.id}'] = target_mesh
    
    def add_building(self, building_id: str, position: np.ndarray, size: np.ndarray):
        """Add a building to the simulation"""
        building_mesh = o3d.geometry.TriangleMesh.create_box(
            width=size[0], height=size[1], depth=size[2]
        )
        building_mesh.translate(position)
        building_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        building = Building(
            id=building_id,
            vertices=np.asarray(building_mesh.vertices),
            faces=np.asarray(building_mesh.triangles),
            position=position.copy()
        )
        
        self.buildings[building_id] = building
        building._mesh = building_mesh
        self.vis.add_geometry(building_mesh)
        self.geometry_objects[f'building_{building_id}'] = building_mesh
        
        return building
    
    def update_drone_position(self, drone_id: str, new_position: np.ndarray):
        """Update drone position"""
        if drone_id not in self.drones:
            return
        
        drone = self.drones[drone_id]
        old_position = drone._last_position
        drone.position = new_position.copy()
        
        if hasattr(drone, '_mesh'):
            translation = new_position - old_position
            drone._mesh.translate(translation)
            drone._sensor_sphere.translate(translation)
            
            self.vis.update_geometry(drone._mesh)
            self.vis.update_geometry(drone._sensor_sphere)
            
        drone._last_position = new_position.copy()
    
    def simulate_sensor_detection(self, drone_id: str) -> Observation:
        """Simulate sensor detection and return observation"""
        if drone_id not in self.drones:
            return None
        
        drone = self.drones[drone_id]
        detected_targets = []
        
        for target_id, target in self.targets.items():
            if target.detected:
                continue
                
            distance = np.linalg.norm(drone.position - target.position)
            
            if distance <= drone.sensor_range:
                # POMDP-style detection with uncertainty
                detection_prob = (
                    target.true_detection_probability * 
                    (1 - distance / drone.sensor_range) * 
                    (target.size / 3)
                )
                detection_prob = np.clip(detection_prob, 0, 1)
                
                if np.random.random() < detection_prob:
                    target.detected = True
                    target.confidence = detection_prob
                    target.color = (0.0, 1.0, 0.0)  # Green when detected
                    
                    # Update visualization
                    if hasattr(target, '_mesh'):
                        target._mesh.paint_uniform_color(target.color)
                        self.vis.update_geometry(target._mesh)
                    
                    detected_targets.append(target_id)
        
        # Create observation
        observation = Observation(
            position=drone.position.copy(),
            detected_targets=detected_targets,
            sensor_readings={'range': drone.sensor_range, 'coverage': 1.0},
            timestamp=time.time(),
            confidence=0.9  # Base sensor confidence
        )
        
        # Store observation history
        drone.observations.append(observation)
        if len(drone.observations) > 50:  # Keep last 50 observations
            drone.observations.pop(0)
        
        return observation
    
    def _simulation_step(self):
        """Enhanced simulation step with POMDP planning"""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # POMDP planning and execution for each drone
        for drone in self.drones.values():
            if drone.status == DroneStatus.SEARCHING or drone.status == DroneStatus.IDLE:
                # Get sensor observation
                observation = self.simulate_sensor_detection(drone.id)
                
                # Update belief state
                drone.belief_state = self.planner.update_belief_state(
                    drone, observation, self.drones
                )
                
                # Plan next action
                if drone.current_action is None or self._action_completed(drone):
                    action = self.planner.plan_action(drone, self.drones, self.bounds)
                    drone.current_action = action
                    drone.status = DroneStatus.SEARCHING
                    
                    print(f"Drone {drone.id}: Planning {action.type.value} to {action.target_position}")
                
                # Execute current action
                self._execute_action(drone)
                
                # Check for target detections
                if len(observation.detected_targets) > 0:
                    print(f"🎯 Drone {drone.id} detected targets: {observation.detected_targets}")
                    drone.status = DroneStatus.INVESTIGATING
                    # Investigation will be handled in next iteration
            
            elif drone.status == DroneStatus.INVESTIGATING:
                # Continue investigation
                if drone.current_action and drone.current_action.type == ActionType.INVESTIGATE:
                    if self._action_completed(drone):
                        print(f"Drone {drone.id}: Investigation complete, resuming search")
                        drone.status = DroneStatus.SEARCHING
                        drone.current_action = None
                else:
                    # Start investigation
                    investigate_action = Action(
                        type=ActionType.INVESTIGATE,
                        target_position=drone.position.copy(),
                        duration=5.0,
                        priority=8.0
                    )
                    drone.current_action = investigate_action
                    drone.current_action.start_time = current_time
        
        # Update battery levels and handle low battery
        for drone in self.drones.values():
            if drone.status != DroneStatus.CHARGING:
                if drone.status == DroneStatus.INVESTIGATING:
                    drone.battery_level -= 0.002  # Investigation uses more power
                else:
                    drone.battery_level -= 0.001  # Normal drain
                
                if drone.battery_level <= 0.2:
                    print(f"🔋 Drone {drone.id}: Low battery, returning to base")
                    drone.status = DroneStatus.RETURNING
                    return_action = Action(
                        type=ActionType.RETURN_BASE,
                        target_position=np.array([0, 0, 50]),  # Base at origin
                        priority=10.0
                    )
                    drone.current_action = return_action
                    drone.current_action.start_time = current_time
    
    def _action_completed(self, drone: Drone) -> bool:
        """Check if current action is completed"""
        if drone.current_action is None:
            return True
        
        action = drone.current_action
        current_time = time.time()
        
        if action.type == ActionType.MOVE:
            # Check if reached target position
            if action.target_position is not None:
                distance = np.linalg.norm(drone.position - action.target_position)
                return distance < 5.0  # Within 5 meters
        
        elif action.type == ActionType.INVESTIGATE:
            # Check if investigation duration elapsed
            if hasattr(action, 'start_time'):
                return current_time - action.start_time >= action.duration
            else:
                action.start_time = current_time
                return False
        
        elif action.type == ActionType.RETURN_BASE:
            # Check if reached base
            if action.target_position is not None:
                distance = np.linalg.norm(drone.position - action.target_position)
                if distance < 10.0:
                    drone.status = DroneStatus.CHARGING
                    drone.battery_level = 1.0  # Instant recharge for demo
                    return True
        
        return False
    
    def _execute_action(self, drone: Drone):
        """Execute the current action for a drone"""
        if drone.current_action is None:
            return
        
        action = drone.current_action
        
        if action.type == ActionType.MOVE:
            if action.target_position is not None:
                # Move towards target position
                direction = action.target_position - drone.position
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    # Normalize direction and apply speed
                    direction = direction / distance
                    move_distance = min(drone.speed * self.update_interval, distance)
                    new_position = drone.position + direction * move_distance
                    
                    # Ensure within bounds
                    min_x, min_y, max_x, max_y = self.bounds
                    new_position[0] = np.clip(new_position[0], min_x + 20, max_x - 20)
                    new_position[1] = np.clip(new_position[1], min_y + 20, max_y - 20)
                    new_position[2] = np.clip(new_position[2], 20, 100)
                    
                    self.update_drone_position(drone.id, new_position)
        
        elif action.type == ActionType.INVESTIGATE:
            # Hover in place during investigation
            pass
        
        elif action.type == ActionType.RETURN_BASE:
            if action.target_position is not None:
                # Move towards base
                direction = action.target_position - drone.position
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    direction = direction / distance
                    move_distance = drone.speed * self.update_interval * 1.5  # Faster return
                    new_position = drone.position + direction * move_distance
                    self.update_drone_position(drone.id, new_position)
    
    def get_search_statistics(self) -> Dict[str, float]:
        """Get statistics about search progress"""
        total_targets = len(self.targets)
        detected_targets = sum(1 for t in self.targets.values() if t.detected)
        
        # Calculate total coverage
        total_coverage = 0.0
        coverage_count = 0
        for drone in self.drones.values():
            if drone.belief_state:
                total_coverage += np.mean(drone.belief_state.coverage_map)
                coverage_count += 1
        
        avg_coverage = total_coverage / max(coverage_count, 1)
        
        # Calculate drone coordination (average distance between drones)
        distances = []
        drone_list = list(self.drones.values())
        for i in range(len(drone_list)):
            for j in range(i + 1, len(drone_list)):
                dist = np.linalg.norm(drone_list[i].position - drone_list[j].position)
                distances.append(dist)
        
        avg_drone_distance = np.mean(distances) if distances else 0.0
        
        return {
            'detection_rate': detected_targets / max(total_targets, 1),
            'coverage': avg_coverage,
            'drone_coordination': avg_drone_distance,
            'active_drones': sum(1 for d in self.drones.values() 
                               if d.status in [DroneStatus.SEARCHING, DroneStatus.INVESTIGATING])
        }
    
    def start_simulation(self):
        """Start the POMDP simulation"""
        self.is_running = True
        
        # Register animation callback
        self.vis.register_animation_callback(self._animation_callback)
        
        print("🧠 POMDP SAR Simulation started")
        print("Features:")
        print("  - Distributed belief state maintenance")
        print("  - Information-theoretic search planning")
        print("  - Drone coordination and communication")
        print("  - Probabilistic target detection")
        print("\nPress ESC to exit.")
        
        # Start visualization
        self.vis.run()
    
    def _animation_callback(self, vis):
        """Animation callback for Open3D"""
        if not self.is_running:
            return False
        
        self._simulation_step()
        
        # Print statistics every 10 seconds
        if hasattr(self, '_last_stats_time'):
            if time.time() - self._last_stats_time > 10:
                stats = self.get_search_statistics()
                print(f"📊 Detection: {stats['detection_rate']:.1%}, "
                      f"Coverage: {stats['coverage']:.1%}, "
                      f"Active: {stats['active_drones']} drones")
                self._last_stats_time = time.time()
        else:
            self._last_stats_time = time.time()
        
        return True
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        print("Stopping POMDP simulation...")
        
        if self.vis:
            try:
                self.vis.close()
            except:
                pass

def create_pomdp_demo_scenario():
    """Create a demonstration scenario for POMDP SAR"""
    print("🧠 Creating POMDP SAR Simulation Environment...")
    
    # Create environment with larger bounds for more complex search
    env = SARPOMDPEnvironment(bounds=(-400, -400, 400, 400))
    
    # Add buildings to create complex search environment
    print("🏢 Adding complex urban environment...")
    building_positions = [
        (np.array([100, 100, 0]), np.array([40, 60, 30])),
        (np.array([-120, 150, 0]), np.array([30, 30, 35])),
        (np.array([200, -180, 0]), np.array([25, 80, 25])),
        (np.array([-200, -100, 0]), np.array([50, 40, 20])),
        (np.array([50, -250, 0]), np.array([35, 35, 40])),
        (np.array([-300, 200, 0]), np.array([45, 25, 30])),
    ]
    
    for i, (pos, size) in enumerate(building_positions):
        env.add_building(f"building_{i+1}", pos, size)
    
    # Deploy drone fleet with POMDP capabilities
    print("🚁 Deploying intelligent drone fleet...")
    drone_start_positions = [
        np.array([-100, -100, 60]),
        np.array([100, -100, 60]),
        np.array([-100, 100, 60]),
        np.array([100, 100, 60]),
        np.array([0, 0, 80]),  # Central coordinator
    ]
    
    for i, start_pos in enumerate(drone_start_positions):
        drone_id = f"drone_{i+1}"
        drone = env.add_drone(drone_id, start_pos)
        drone.status = DroneStatus.SEARCHING
        drone.sensor_range = 60.0 if i == 4 else 50.0  # Central drone has larger range
        print(f"   Deployed {drone_id} at {drone.position} with {drone.sensor_range}m range")
    
    # Place search targets in challenging locations
    print("🎯 Placing search targets in complex environment...")
    target_positions = [
        np.array([80, 120, 2]),    # Near building
        np.array([-150, -80, 1]),  # In building shadow
        np.array([180, -200, 3]),  # Remote area
        np.array([-180, 180, 1]),  # Corner area
        np.array([20, -230, 2]),   # Near building cluster
        np.array([-280, 150, 1]),  # Edge of search area
        np.array([0, 0, 1]),       # Central area
        np.array([250, 250, 2]),   # Far corner
    ]
    
    for i, pos in enumerate(target_positions):
        target_id = f"target_{i+1}"
        target = env.add_target(target_id, pos)
        # Vary detection difficulty
        target.true_detection_probability = 0.6 + 0.3 * np.random.random()
        print(f"   Target {target_id} at {target.position} "
              f"(detection_prob: {target.true_detection_probability:.2f})")
    
    print(f"\n🚀 Starting POMDP SAR Mission...")
    print(f"Mission Parameters:")
    print(f"  - Search Area: {env.bounds}")
    print(f"  - {len(env.drones)} autonomous drones")
    print(f"  - {len(env.targets)} targets to locate")
    print(f"  - {len(env.buildings)} buildings creating obstacles")
    print(f"\nWatch the drones coordinate their search using:")
    print(f"  - Belief state updates")
    print(f"  - Information-theoretic planning")
    print(f"  - Distributed coordination")
    
    return env

if __name__ == "__main__":
    # Create and run POMDP demo
    env = create_pomdp_demo_scenario()
    
    try:
        env.start_simulation()
    except KeyboardInterrupt:
        print("\n🛑 POMDP Simulation stopped by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        env.is_running = False
        print("✅ POMDP Simulation ended")