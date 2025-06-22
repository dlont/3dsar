import open3d as o3d
import trimesh
from shapely.geometry import box
import time
import numpy as np
import matplotlib
try:
    matplotlib.use('Qt5Agg')
    print("✅ Using Qt5Agg backend for interactive plots")
    INTERACTIVE_PLOTS = True
except:
    try:
        matplotlib.use('TkAgg')
        print("✅ Using TkAgg backend for interactive plots")
        INTERACTIVE_PLOTS = True
    except:
        matplotlib.use('Agg')
        print("⚠️  Using Agg backend - plots will be saved as files")
        INTERACTIVE_PLOTS = False
import matplotlib.pyplot as plt

# Import from our modular components
from simulation.core.entities import (
    Drone, SearchTarget, Building, DroneStatus, Action, ActionType, Observation, BeliefState
)
from simulation.planning.pomdp_planner import DistributedPOMDPSearchPlanner
from simulation.planning.probability_maps import (
    ProbabilityMapGenerator, PredefinedHypotheses, 
    SearchHypothesis, ProbabilityRegion, IncidentType
)

class EnhancedSAREnvironment:
    """Enhanced SAR Environment with custom probability map integration and 3D visualization"""
    
    def __init__(self, bounds=(-200, -200, 200, 200), initial_hypothesis=None):
        self.bounds = bounds
        self.drones = {}
        self.targets = {}
        self.buildings = {}
        
        # Initialize probability map generator
        self.prob_map_generator = ProbabilityMapGenerator(bounds, resolution=10.0)
        self.initial_hypothesis = initial_hypothesis
        self.master_probability_map = None
        
        if initial_hypothesis:
            self.master_probability_map = self.prob_map_generator.generate_probability_map(initial_hypothesis)
        
        # POMDP planner
        self.planner = DistributedPOMDPSearchPlanner()
        
        # Visualization components
        self.vis = None
        self.is_running = False
        self.last_update_time = time.time()
        self.update_interval = 0.5  # 2 FPS for deliberate planning
        
        # Environment state
        self.terrain = self._generate_terrain()
        self.search_area = self._define_search_area()
        
        # Geometry tracking for updates
        self.geometry_objects = {}
        
        # Initialize 3D visualization
        self._setup_visualization()
    
    def _generate_terrain(self) -> trimesh.Trimesh:
        """Generate realistic terrain using noise with probability-based coloring"""
        min_x, min_y, max_x, max_y = self.bounds
        
        # Create terrain grid (store for later probability mapping)
        self.terrain_resolution = 100
        x = np.linspace(min_x, max_x, self.terrain_resolution)
        y = np.linspace(min_y, max_y, self.terrain_resolution)
        self.terrain_x_coords = x
        self.terrain_y_coords = y
        X, Y = np.meshgrid(x, y)
        self.terrain_X = X
        self.terrain_Y = Y
        
        # Generate height using Perlin-like noise
        Z = np.zeros_like(X)
        for i in range(3):
            freq = 0.01 * (2 ** i)
            amplitude = 10 / (2 ** i)
            Z += amplitude * np.sin(freq * X) * np.cos(freq * Y)
        
        self.terrain_Z = Z
        
        # Create terrain mesh
        vertices = []
        faces = []
        vertex_colors = []
        
        for i in range(len(x)-1):
            for j in range(len(y)-1):
                # Create quad vertices
                v1 = [X[j,i], Y[j,i], Z[j,i]]
                v2 = [X[j,i+1], Y[j,i+1], Z[j,i+1]]
                v3 = [X[j+1,i+1], Y[j+1,i+1], Z[j+1,i+1]]
                v4 = [X[j+1,i], Y[j+1,i], Z[j+1,i]]
                
                base_idx = len(vertices)
                vertices.extend([v1, v2, v3, v4])
                
                # Initialize with default green terrain colors
                default_color = [0.4, 0.6, 0.2]  # Green terrain
                vertex_colors.extend([default_color] * 4)
                
                # Create two triangles per quad
                faces.extend([
                    [base_idx, base_idx+1, base_idx+2],
                    [base_idx, base_idx+2, base_idx+3]
                ])
        
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        terrain_mesh.visual.vertex_colors = np.array(vertex_colors)
        
        return terrain_mesh
    
    def _define_search_area(self):
        """Define the search area boundary"""
        min_x, min_y, max_x, max_y = self.bounds
        margin = 50
        return box(min_x + margin, min_y + margin, max_x - margin, max_y - margin)
    
    def _setup_visualization(self):
        """Initialize Open3D visualization"""
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Enhanced SAR POMDP Mission Simulation", width=1200, height=800)
        
        # Register key callbacks
        self.vis.register_key_callback(256, self._on_escape)  # ESC key
        
        # Add terrain with probability coloring
        terrain_o3d = o3d.geometry.TriangleMesh()
        terrain_o3d.vertices = o3d.utility.Vector3dVector(self.terrain.vertices)
        terrain_o3d.triangles = o3d.utility.Vector3iVector(self.terrain.faces)
        
        # Apply initial probability coloring if available
        if hasattr(self.terrain, 'visual') and hasattr(self.terrain.visual, 'vertex_colors'):
            terrain_o3d.vertex_colors = o3d.utility.Vector3dVector(
                self.terrain.visual.vertex_colors[:, :3] / 255.0
            )
        else:
            terrain_o3d.paint_uniform_color([0.4, 0.6, 0.2])  # Default green
        
        terrain_o3d.compute_vertex_normals()
        self.vis.add_geometry(terrain_o3d)
        self.geometry_objects['terrain'] = terrain_o3d
        
        # Update terrain colors with initial hypothesis if available
        if self.master_probability_map is not None:
            self._update_terrain_colors_with_probability_map(self.master_probability_map)
        
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
    
    def _update_terrain_colors_with_probability_map(self, probability_map: np.ndarray):
        """Update terrain colors based on probability map"""
        try:
            if not hasattr(self, 'terrain_X') or not hasattr(self, 'terrain_Y'):
                return
            
            # Get terrain mesh
            terrain_o3d = self.geometry_objects.get('terrain')
            if terrain_o3d is None:
                return
            
            # Create new vertex colors array
            vertices = np.asarray(terrain_o3d.vertices)
            num_vertices = len(vertices)
            new_colors = np.zeros((num_vertices, 3))
            
            # Check if probability map is valid
            if probability_map is None or probability_map.size == 0:
                # Use default green coloring
                new_colors[:] = [0.4, 0.6, 0.2]
                terrain_o3d.vertex_colors = o3d.utility.Vector3dVector(new_colors)
                self.vis.update_geometry(terrain_o3d)
                return
            
            # Map probability values to terrain vertices
            for i, vertex in enumerate(vertices):
                x, y = vertex[0], vertex[1]
                
                # Find corresponding probability map indices
                prob_x_idx = int((x - self.bounds[0]) / (self.bounds[2] - self.bounds[0]) * (probability_map.shape[1] - 1))
                prob_y_idx = int((y - self.bounds[1]) / (self.bounds[3] - self.bounds[1]) * (probability_map.shape[0] - 1))
                
                # Clamp indices to valid range
                prob_x_idx = max(0, min(prob_x_idx, probability_map.shape[1] - 1))
                prob_y_idx = max(0, min(prob_y_idx, probability_map.shape[0] - 1))
                
                # Get probability value (note: probability_map uses (y, x) indexing)
                prob_value = probability_map[prob_y_idx, prob_x_idx]
                
                # Normalize probability for coloring (enhance contrast)
                normalized_prob = np.clip(prob_value * 1000, 0, 1)  # Scale up for visibility
                
                # Color scheme: Green (low) -> Yellow -> Red (high probability)
                if normalized_prob < 0.5:
                    # Green to yellow transition
                    red = normalized_prob * 2
                    green = 0.6
                    blue = 0.2 * (1 - normalized_prob * 2)
                else:
                    # Yellow to red transition
                    red = 1.0
                    green = 0.6 * (2 - normalized_prob * 2)
                    blue = 0.0
                
                new_colors[i] = [red, green, blue]
            
            # Update terrain colors
            terrain_o3d.vertex_colors = o3d.utility.Vector3dVector(new_colors)
            self.vis.update_geometry(terrain_o3d)
            print('Colors updated')
            
        except Exception as e:
            print(f"Warning: Could not update terrain colors: {e}")
            # Continue with default green terrain if coloring fails
    
    def _update_terrain_with_combined_belief_states(self):
        """Update terrain colors with combined belief states from all drones"""
        try:
            if not self.drones:
                return
            
            # Combine belief states from all drones
            combined_belief_map = None
            valid_beliefs = 0
            
            for drone in self.drones.values():
                if (hasattr(drone, 'belief_state') and drone.belief_state and 
                    hasattr(drone.belief_state, 'target_probability_map') and 
                    drone.belief_state.target_probability_map is not None):
                    
                    if combined_belief_map is None:
                        combined_belief_map = drone.belief_state.target_probability_map.copy()
                    else:
                        # Weighted average of belief states
                        combined_belief_map += drone.belief_state.target_probability_map
                    valid_beliefs += 1
            
            if combined_belief_map is not None and valid_beliefs > 0:
                # Average the combined beliefs
                combined_belief_map /= valid_beliefs
                
                # Update terrain colors
                self._update_terrain_colors_with_probability_map(combined_belief_map)
                
        except Exception as e:
            print(f"Warning: Could not update terrain with belief states: {e}")
            # Continue simulation without terrain updates
        
        # Update all existing drone belief states with this prior
        for drone in self.drones.values():
            self._initialize_drone_belief_with_prior(drone)
    
    def _initialize_drone_belief_with_prior(self, drone):
        """Initialize drone's belief state with the master probability map"""
        if self.master_probability_map is not None:
            # Initialize with custom probability map
            drone.belief_state = self.planner.initialize_belief_state(drone, self.bounds)
            drone.belief_state.target_probability_map = self.master_probability_map.copy()
        else:
            # Use uniform distribution
            drone.belief_state = self.planner.initialize_belief_state(drone, self.bounds)
    
    def add_drone_with_prior(self, drone_id: str, initial_position: np.ndarray = None):
        """Add drone and initialize with probability prior"""
        drone = self.add_drone(drone_id, initial_position)
        
        # Initialize with master probability map if available
        if self.master_probability_map is not None:
            drone.belief_state.target_probability_map = self.master_probability_map.copy()
        
        return drone
    
    def add_drone(self, drone_id: str, initial_position: np.ndarray = None) -> Drone:
        """Add a drone to the simulation"""
        if initial_position is None:
            min_x, min_y, max_x, max_y = self.bounds
            initial_position = np.array([
                np.random.uniform(min_x + 50, max_x - 50),
                np.random.uniform(min_y + 50, max_y - 50),
                25.0  # Optimal altitude for ground search
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
        self._initialize_drone_belief_with_prior(drone)
        
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
        
        # Update terrain colors with combined belief states every few seconds
        if not hasattr(self, '_last_terrain_update'):
            self._last_terrain_update = current_time
        elif current_time - self._last_terrain_update > 3.0:  # Update every 3 seconds
            self._update_terrain_with_combined_belief_states()
            self._last_terrain_update = current_time
        
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
                        target_position=np.array([0, 0, 25]),
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
                    
                    # Ensure within bounds and optimal altitude for ground search
                    min_x, min_y, max_x, max_y = self.bounds
                    new_position[0] = np.clip(new_position[0], min_x + 20, max_x - 20)
                    new_position[1] = np.clip(new_position[1], min_y + 20, max_y - 20)
                    new_position[2] = np.clip(new_position[2], 15, 35)  # Optimal altitude: 15-35m above ground
                    
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
    
    def get_search_statistics(self) -> dict:
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
        """Start the POMDP simulation with 3D rendering"""
        self.is_running = True
        
        # Register animation callback
        self.vis.register_animation_callback(self._animation_callback)
        
        print("🧠 Enhanced SAR POMDP Simulation started with 3D rendering")
        print("Features:")
        print("  - Custom probability map hypotheses")
        print("  - Distributed belief state maintenance")
        print("  - Information-theoretic search planning")
        print("  - Drone coordination and communication")
        print("  - Probabilistic target detection")
        print("  - Real-time 3D visualization")
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
    
    def visualize_initial_hypothesis(self):
        """Visualize the initial probability hypothesis"""
        if self.master_probability_map is not None:
            plt.figure(figsize=(12, 10))
            
            min_x, min_y, max_x, max_y = self.bounds
            im = plt.imshow(self.master_probability_map, 
                           extent=[min_x, max_x, min_y, max_y], 
                           origin='lower', cmap='hot', alpha=0.8)
            
            plt.colorbar(im, label='Target Probability Density')
            plt.xlabel('X Coordinate (m)')
            plt.ylabel('Y Coordinate (m)')
            plt.title(f'Initial Search Hypothesis: {self.initial_hypothesis.description}')
            plt.grid(True, alpha=0.3)
            
            # Add drone positions if any
            for drone in self.drones.values():
                plt.plot(drone.position[0], drone.position[1], 'bo', 
                        markersize=8, label=f'Drone {drone.id}')
            
            # Add target positions if any (for validation)
            for target in self.targets.values():
                plt.plot(target.position[0], target.position[1], 'r*', 
                        markersize=12, label=f'Target {target.id}')
            
            if self.drones or self.targets:
                plt.legend()
            
            if INTERACTIVE_PLOTS:
                plt.show()
            else:
                filename = f"hypothesis_{self.initial_hypothesis.incident_type.value}.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"💾 Hypothesis visualization saved as: {filename}")
                plt.close()
        else:
            print("No initial hypothesis set. Use set_initial_hypothesis() first.")

