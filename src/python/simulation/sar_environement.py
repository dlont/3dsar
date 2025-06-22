# src/python/simulation/sar_environment.py

import numpy as np
import open3d as o3d
import trimesh
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import cv2
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

class DroneStatus(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    INVESTIGATING = "investigating"
    RETURNING = "returning"
    CHARGING = "charging"

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

@dataclass
class SearchTarget:
    id: str
    position: np.ndarray  # [x, y, z]
    size: float
    detected: bool = False
    confidence: float = 0.0
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)

@dataclass
class Building:
    id: str
    vertices: np.ndarray
    faces: np.ndarray
    position: np.ndarray  # [x, y, z]
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)

class SARSimulationEnvironment:
    """
    3D Simulation Environment for Search and Rescue Operations
    
    Features:
    - 3D terrain and building visualization
    - Multiple drone fleet simulation
    - Search target placement and detection
    - Real-time sensor coverage visualization
    - Mission area definition
    """
    
    def __init__(self, bounds: Tuple[float, float, float, float] = (-200, -200, 200, 200)):
        """
        Initialize SAR simulation environment
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) in meters
        """
        self.bounds = bounds
        self.drones: Dict[str, Drone] = {}
        self.targets: Dict[str, SearchTarget] = {}
        self.buildings: Dict[str, Building] = {}
        
        # Visualization components
        self.vis = None
        self.is_running = False
        self.update_thread = None
        
        # Environment state
        self.terrain = self._generate_terrain()
        self.search_area = self._define_search_area()
        
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
        # Create search area smaller than full bounds
        margin = 50
        return box(min_x + margin, min_y + margin, max_x - margin, max_y - margin)
    
    def _setup_visualization(self):
        """Initialize Open3D visualization"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="SAR Mission Simulation", width=1200, height=800)
        
        # Add terrain
        terrain_o3d = o3d.geometry.TriangleMesh()
        terrain_o3d.vertices = o3d.utility.Vector3dVector(self.terrain.vertices)
        terrain_o3d.triangles = o3d.utility.Vector3iVector(self.terrain.faces)
        terrain_o3d.paint_uniform_color([0.4, 0.6, 0.2])  # Green terrain
        terrain_o3d.compute_vertex_normals()
        self.vis.add_geometry(terrain_o3d)
        
        # Add search area boundary
        self._add_search_area_boundary()
        
        # Set up camera
        self._setup_camera()
    
    def _add_search_area_boundary(self):
        """Add search area boundary visualization"""
        coords = list(self.search_area.exterior.coords)
        points = []
        lines = []
        
        for i, (x, y) in enumerate(coords[:-1]):  # Skip last point (same as first)
            points.append([x, y, 20])  # Height of 20m
            if i < len(coords) - 2:
                lines.append([i, i + 1])
            else:
                lines.append([i, 0])  # Close the loop
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow boundary
        self.vis.add_geometry(line_set)
    
    def _setup_camera(self):
        """Setup optimal camera view"""
        ctr = self.vis.get_view_control()
        # Set camera to isometric view
        ctr.set_zoom(0.3)
        ctr.set_front([0.5, -0.5, -0.7])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
    
    def add_drone(self, drone_id: str, initial_position: np.ndarray = None) -> Drone:
        """Add a drone to the simulation"""
        if initial_position is None:
            # Random position within bounds
            min_x, min_y, max_x, max_y = self.bounds
            initial_position = np.array([
                np.random.uniform(min_x + 50, max_x - 50),
                np.random.uniform(min_y + 50, max_y - 50),
                50.0  # 50m altitude
            ])
        
        drone = Drone(
            id=drone_id,
            position=initial_position.copy(),
            orientation=np.array([0, 0, 0]),
            battery_level=1.0,
            status=DroneStatus.IDLE,
            color=(np.random.random(), np.random.random(), np.random.random())
        )
        
        self.drones[drone_id] = drone
        self._add_drone_to_visualization(drone)
        return drone
    
    def _add_drone_to_visualization(self, drone: Drone):
        """Add drone mesh to visualization"""
        # Create simple drone representation (cylinder + rotors)
        drone_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=2, height=1)
        drone_mesh.translate(drone.position)
        drone_mesh.paint_uniform_color(drone.color)
        
        # Add sensor range sphere (wireframe)
        sensor_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=drone.sensor_range)
        sensor_sphere.translate(drone.position)
        # sensor_sphere.paint_uniform_color([*drone.color, 0.1])  # Transparent
        
        # Store references for updates
        setattr(drone, '_mesh', drone_mesh)
        setattr(drone, '_sensor_sphere', sensor_sphere)
        
        self.vis.add_geometry(drone_mesh)
        self.vis.add_geometry(sensor_sphere)
    
    def add_target(self, target_id: str, position: np.ndarray = None) -> SearchTarget:
        """Add a search target to the simulation"""
        if position is None:
            # Random position within search area
            min_x, min_y, max_x, max_y = self.search_area.bounds
            position = np.array([
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y),
                np.random.uniform(0, 5)  # Ground level to 5m
            ])
        
        target = SearchTarget(
            id=target_id,
            position=position.copy(),
            size=np.random.uniform(1, 3)
        )
        
        self.targets[target_id] = target
        self._add_target_to_visualization(target)
        return target
    
    def _add_target_to_visualization(self, target: SearchTarget):
        """Add target to visualization"""
        target_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=target.size)
        target_mesh.translate(target.position)
        target_mesh.paint_uniform_color(target.color)
        
        setattr(target, '_mesh', target_mesh)
        self.vis.add_geometry(target_mesh)
    
    def add_building(self, building_id: str, position: np.ndarray, size: np.ndarray):
        """Add a building to the simulation"""
        # Create simple box building
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
        setattr(building, '_mesh', building_mesh)
        self.vis.add_geometry(building_mesh)
        
        return building
    
    def update_drone_position(self, drone_id: str, new_position: np.ndarray):
        """Update drone position"""
        if drone_id not in self.drones:
            return
        
        drone = self.drones[drone_id]
        old_position = drone.position.copy()
        drone.position = new_position.copy()
        
        # Update visualization
        if hasattr(drone, '_mesh'):
            translation = new_position - old_position
            drone._mesh.translate(translation)
            drone._sensor_sphere.translate(translation)
    
    def simulate_sensor_detection(self, drone_id: str) -> List[str]:
        """Simulate sensor detection for a drone"""
        if drone_id not in self.drones:
            return []
        
        drone = self.drones[drone_id]
        detected_targets = []
        
        for target_id, target in self.targets.items():
            if target.detected:
                continue
                
            # Calculate distance
            distance = np.linalg.norm(drone.position - target.position)
            
            # Check if within sensor range
            if distance <= drone.sensor_range:
                # Simulate detection probability based on distance and target size
                detection_prob = (1 - distance / drone.sensor_range) * target.size / 3
                detection_prob = np.clip(detection_prob, 0, 1)
                
                if np.random.random() < detection_prob:
                    target.detected = True
                    target.confidence = detection_prob
                    target.color = (0.0, 1.0, 0.0)  # Green when detected
                    
                    # Update visualization
                    if hasattr(target, '_mesh'):
                        target._mesh.paint_uniform_color(target.color)
                    
                    detected_targets.append(target_id)
        
        return detected_targets
    
    def get_coverage_map(self) -> np.ndarray:
        """Generate coverage heatmap based on drone positions"""
        min_x, min_y, max_x, max_y = self.bounds
        
        # Create grid
        resolution = 5  # 5m resolution
        x = np.arange(min_x, max_x, resolution)
        y = np.arange(min_y, max_y, resolution)
        X, Y = np.meshgrid(x, y)
        
        coverage = np.zeros_like(X)
        
        for drone in self.drones.values():
            # Calculate coverage for each grid point
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    point = np.array([X[i,j], Y[i,j], 0])
                    distance = np.linalg.norm(drone.position[:2] - point[:2])
                    
                    if distance <= drone.sensor_range:
                        coverage[i,j] += 1 - (distance / drone.sensor_range)
        
        return np.clip(coverage, 0, 1)
    
    def start_simulation(self):
        """Start the simulation loop"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._simulation_loop)
        self.update_thread.start()
        
        # Start visualization
        self.vis.run()
    
    def _simulation_loop(self):
        """Main simulation update loop"""
        while self.is_running:
            # Update drone positions (simple random walk for demo)
            for drone in self.drones.values():
                if drone.status == DroneStatus.SEARCHING:
                    # Random movement within bounds
                    direction = np.random.uniform(-1, 1, 3)
                    direction[2] *= 0.1  # Less vertical movement
                    new_position = drone.position + direction * drone.speed * 0.1
                    
                    # Keep within bounds
                    min_x, min_y, max_x, max_y = self.bounds
                    new_position[0] = np.clip(new_position[0], min_x, max_x)
                    new_position[1] = np.clip(new_position[1], min_y, max_y)
                    new_position[2] = np.clip(new_position[2], 20, 100)  # Altitude limits
                    
                    self.update_drone_position(drone.id, new_position)
                    
                    # Simulate sensor detection
                    detected = self.simulate_sensor_detection(drone.id)
                    if detected:
                        print(f"Drone {drone.id} detected targets: {detected}")
                        drone.status = DroneStatus.INVESTIGATING
            
            # Update battery levels
            for drone in self.drones.values():
                if drone.status != DroneStatus.CHARGING:
                    drone.battery_level -= 0.001  # Drain battery
                    if drone.battery_level <= 0.2:
                        drone.status = DroneStatus.RETURNING
            
            # Update visualization
            self.vis.poll_events()
            self.vis.update_renderer()
            
            time.sleep(0.1)  # 10 FPS
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        self.vis.close()

def create_demo_scenario():
    """Create a demo SAR scenario"""
    print("🚁 Creating SAR Simulation Environment...")
    
    # Create environment
    env = SARSimulationEnvironment(bounds=(-300, -300, 300, 300))
    
    # Add buildings for realistic urban search scenario
    print("🏢 Adding buildings...")
    env.add_building("building_1", np.array([50, 50, 0]), np.array([30, 40, 25]))
    env.add_building("building_2", np.array([-80, 100, 0]), np.array([25, 25, 30]))
    env.add_building("building_3", np.array([120, -150, 0]), np.array([20, 50, 20]))
    
    # Add search and rescue drones
    print("🚁 Deploying drone fleet...")
    for i in range(4):
        drone_id = f"drone_{i+1}"
        drone = env.add_drone(drone_id)
        drone.status = DroneStatus.SEARCHING
        print(f"   Deployed {drone_id} at position {drone.position}")
    
    # Add search targets
    print("🎯 Placing search targets...")
    for i in range(6):
        target_id = f"target_{i+1}"
        target = env.add_target(target_id)
        print(f"   Target {target_id} at position {target.position}")
    
    print("\n🚀 Starting simulation...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - ESC: Exit simulation")
    print("\nWatch as drones search for targets (red spheres turn green when detected)")
    
    return env

if __name__ == "__main__":
    # Create and run demo
    env = create_demo_scenario()
    
    try:
        env.start_simulation()
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    finally:
        env.stop_simulation()
        print("✅ Simulation ended")