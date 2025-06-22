# src/python/sar_simulation/planning/pomdp_planner.py

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from ..core.entities import (
    Drone, SearchTarget, BeliefState, Action, ActionType, 
    DroneStatus, Observation
)

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
            z = 25.0  # Fixed optimal altitude for ground search
            
            # Ensure within bounds
            x = np.clip(x, min_x + 50, max_x - 50)
            y = np.clip(y, min_y + 50, max_y - 50)
            z = np.clip(z, 15, 35)  # Optimal ground search altitude
            
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
        
        # Random position within bounds at optimal search altitude
        target_pos = np.array([
            np.random.uniform(min_x + 50, max_x - 50),
            np.random.uniform(min_y + 50, max_y - 50),
            np.random.uniform(20, 30)  # Optimal ground search altitude range
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