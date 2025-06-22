# src/python/simulation/sar_with_custom_maps.py

# Import the probability map components
from sar_probability_maps import (
    ProbabilityMapGenerator, 
    PredefinedHypotheses, 
    SearchHypothesis,
    ProbabilityRegion,
    IncidentType
)
from sar_pomdp_environment import SARPOMDPEnvironment, DistributedPOMDPSearchPlanner
import numpy as np
import matplotlib.pyplot as plt

class EnhancedSAREnvironment(SARPOMDPEnvironment):
    """SAR Environment with custom initial probability maps"""
    
    def __init__(self, bounds=(-200, -200, 200, 200), initial_hypothesis=None):
        super().__init__(bounds)
        
        # Initialize probability map generator
        self.prob_map_generator = ProbabilityMapGenerator(bounds, resolution=10.0)
        self.initial_hypothesis = initial_hypothesis
        
        # Store the master probability map
        self.master_probability_map = None
        if initial_hypothesis:
            self.master_probability_map = self.prob_map_generator.generate_probability_map(initial_hypothesis)
    
    def set_initial_hypothesis(self, hypothesis: SearchHypothesis):
        """Set the initial search hypothesis and generate probability map"""
        self.initial_hypothesis = hypothesis
        self.master_probability_map = self.prob_map_generator.generate_probability_map(hypothesis)
        
        # Update all existing drone belief states with this prior
        for drone in self.drones.values():
            self._initialize_drone_belief_with_prior(drone)
    
    def _initialize_drone_belief_with_prior(self, drone):
        """Initialize drone's belief state with the master probability map"""
        if self.master_probability_map is not None:
            # Copy the master probability map as initial belief
            drone.belief_state.target_probability_map = self.master_probability_map.copy()
        else:
            # Use uniform distribution if no hypothesis provided
            super()._initialize_drone_belief_state(drone)
    
    def add_drone_with_prior(self, drone_id: str, initial_position: np.ndarray = None):
        """Add drone and initialize with probability prior"""
        drone = super().add_drone(drone_id, initial_position)
        
        # Initialize with master probability map if available
        if self.master_probability_map is not None:
            drone.belief_state.target_probability_map = self.master_probability_map.copy()
        
        return drone
    
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
            
            plt.show()
        else:
            print("No initial hypothesis set. Use set_initial_hypothesis() first.")

def create_aircraft_crash_scenario():
    """Create realistic aircraft crash SAR scenario"""
    print("✈️ Creating Aircraft Crash SAR Scenario")
    print("="*50)
    
    # Scenario parameters
    bounds = (-1000, -1000, 1000, 1000)  # 2km x 2km search area
    last_known_position = (-200, 100)
    flight_heading = np.pi/6  # 30 degrees northeast
    uncertainty_radius = 500
    
    # Create aircraft crash hypothesis
    hypothesis = PredefinedHypotheses.aircraft_crash_hypothesis(
        last_known_position=last_known_position,
        heading=flight_heading,
        uncertainty_radius=uncertainty_radius
    )
    
    # Create enhanced environment
    env = EnhancedSAREnvironment(bounds=bounds, initial_hypothesis=hypothesis)
    
    # Add buildings to simulate terrain obstacles
    print("🏢 Adding terrain obstacles...")
    env.add_building("forest_1", np.array([300, 300, 0]), np.array([200, 150, 15]))
    env.add_building("hill_1", np.array([-400, -200, 0]), np.array([150, 200, 25]))
    env.add_building("rocky_area", np.array([100, -400, 0]), np.array([180, 120, 10]))
    
    # Deploy specialized SAR drones
    print("🚁 Deploying SAR drone fleet...")
    drone_positions = [
        np.array([-500, -500, 25]),  # Southwest corner
        np.array([500, -500, 25]),   # Southeast corner
        np.array([-500, 500, 25]),   # Northwest corner
        np.array([500, 500, 25]),    # Northeast corner
        np.array([0, 0, 30]),        # Central command drone
        np.array([-200, 100, 25]),   # Near last known position
    ]
    
    for i, pos in enumerate(drone_positions):
        drone_id = f"SAR_{i+1}"
        drone = env.add_drone_with_prior(drone_id, pos)
        drone.sensor_range = 75.0 if i == 4 else 60.0  # Command drone has larger range
        drone.status = drone.status.SEARCHING
        print(f"   Deployed {drone_id} at {pos[:2]} with {drone.sensor_range}m sensor range")
    
    # Place targets based on crash scenario physics
    print("🎯 Placing crash debris and survivors...")
    
    # Main wreckage near projected impact
    impact_x = last_known_position[0] + 300 * np.cos(flight_heading)
    impact_y = last_known_position[1] + 300 * np.sin(flight_heading)
    main_wreckage = np.array([impact_x, impact_y, 2])
    
    # Debris scattered by impact
    debris_positions = [
        main_wreckage,  # Main wreckage
        main_wreckage + np.array([50, 30, -1]),   # Debris field
        main_wreckage + np.array([-30, 60, 0]),  # More debris
        main_wreckage + np.array([80, -20, 1]),  # Scattered parts
        # Survivors potentially ejected or walked away
        main_wreckage + np.array([200, 150, 0]), # Survivor 1
        main_wreckage + np.array([-100, 200, 0]), # Survivor 2
    ]
    
    target_descriptions = [
        "Main wreckage", "Debris field", "Engine parts", 
        "Wing section", "Survivor Alpha", "Survivor Beta"
    ]
    
    for i, (pos, desc) in enumerate(zip(debris_positions, target_descriptions)):
        target_id = f"target_{i+1}"
        target = env.add_target(target_id, pos)
        target.size = 3.0 if "Survivor" in desc else 2.0
        target.true_detection_probability = 0.9 if "wreckage" in desc else 0.7
        print(f"   Placed {desc} at {pos[:2]}")
    
    return env

def create_missing_hiker_scenario():
    """Create missing hiker SAR scenario"""
    print("🥾 Creating Missing Hiker SAR Scenario")
    print("="*50)
    
    bounds = (-800, -800, 800, 800)
    
    # Define hiking trail
    trail_points = [
        (-600, -400), (-400, -300), (-200, -200), 
        (0, -100), (200, 0), (400, 100), (600, 200)
    ]
    last_seen_position = (100, -50)  # Last seen near middle of trail
    
    # Create missing hiker hypothesis
    hypothesis = PredefinedHypotheses.missing_hiker_hypothesis(
        trail_points=trail_points,
        last_seen=last_seen_position
    )
    
    # Create environment
    env = EnhancedSAREnvironment(bounds=bounds, initial_hypothesis=hypothesis)
    
    # Add natural obstacles
    print("🌲 Adding natural terrain...")
    env.add_building("dense_forest", np.array([300, 300, 0]), np.array([250, 200, 20]))
    env.add_building("rocky_ridge", np.array([-300, 200, 0]), np.array([150, 300, 30]))
    env.add_building("steep_slope", np.array([100, -400, 0]), np.array([200, 150, 25]))
    
    # Deploy search and rescue teams
    print("🚁 Deploying mountain rescue drones...")
    
    # Position drones along trail and off-trail areas
    drone_positions = [
        np.array([-400, -250, 25]),  # Trail start area
        np.array([100, -50, 25]),    # Last seen position
        np.array([400, 100, 25]),    # Trail end area
        np.array([0, 200, 25]),      # Off-trail north
        np.array([0, -300, 25]),     # Off-trail south
    ]
    
    for i, pos in enumerate(drone_positions):
        drone_id = f"MR_{i+1}"  # Mountain Rescue
        drone = env.add_drone_with_prior(drone_id, pos)
        drone.sensor_range = 50.0
        drone.status = drone.status.SEARCHING
        print(f"   Deployed {drone_id} at {pos[:2]}")
    
    # Place missing hiker scenarios
    print("🎯 Placing potential hiker locations...")
    
    # Hiker could be:
    hiker_scenarios = [
        (np.array([150, -30, 0]), "On trail - injured"),
        (np.array([80, 50, 0]), "Off trail - lost"),
        (np.array([300, -100, 0]), "Sought shelter"),
        (np.array([-50, 150, 0]), "Following stream"),
    ]
    
    for i, (pos, scenario) in enumerate(hiker_scenarios):
        target_id = f"hiker_scenario_{i+1}"
        target = env.add_target(target_id, pos)
        target.size = 1.5  # Person-sized
        target.true_detection_probability = 0.8
        print(f"   Scenario {i+1}: {scenario} at {pos[:2]}")
    
    return env

def create_urban_collapse_scenario():
    """Create urban building collapse SAR scenario"""
    print("🏢 Creating Urban Collapse SAR Scenario")
    print("="*50)
    
    bounds = (-300, -300, 300, 300)
    
    # Define collapsed building
    building_center = (50, 50)
    building_size = (60, 40)
    wind_direction = np.pi/4  # Northeast wind affecting debris
    
    # Create urban collapse hypothesis
    hypothesis = PredefinedHypotheses.urban_collapse_hypothesis(
        building_location=building_center,
        building_size=building_size,
        wind_direction=wind_direction
    )
    
    # Create environment
    env = EnhancedSAREnvironment(bounds=bounds, initial_hypothesis=hypothesis)
    
    # Add surrounding buildings
    print("🏙️ Adding urban environment...")
    env.add_building("adjacent_building_1", np.array([150, 50, 0]), np.array([40, 50, 35]))
    env.add_building("adjacent_building_2", np.array([50, 150, 0]), np.array([50, 40, 30]))
    env.add_building("parking_structure", np.array([-100, -100, 0]), np.array([80, 60, 15]))
    
    # Deploy urban SAR teams
    print("🚁 Deploying urban SAR drones...")
    
    # Close-quarters urban search pattern
    drone_positions = [
        np.array([0, 0, 25]),      # Central coordination
        np.array([100, 100, 20]),  # Northeast quadrant
        np.array([-100, 100, 20]), # Northwest quadrant
        np.array([100, -100, 20]), # Southeast quadrant
        np.array([-100, -100, 20]), # Southwest quadrant
    ]
    
    for i, pos in enumerate(drone_positions):
        drone_id = f"USAR_{i+1}"  # Urban Search and Rescue
        drone = env.add_drone_with_prior(drone_id, pos)
        drone.sensor_range = 40.0  # Shorter range for urban environment
        drone.status = drone.status.SEARCHING
        print(f"   Deployed {drone_id} at {pos[:2]}")
    
    # Place survivors and victims in collapse scenario
    print("🎯 Placing survivors in collapse scenario...")
    
    # Realistic collapse survivor distribution
    survivor_locations = [
        (np.array([45, 55, 1]), "Survivor in void space"),
        (np.array([60, 45, 0]), "Victim under debris"),
        (np.array([80, 70, 0]), "Evacuee in debris field"),
        (np.array([30, 80, 0]), "Person near building edge"),
    ]
    
    for i, (pos, description) in enumerate(survivor_locations):
        target_id = f"person_{i+1}"
        target = env.add_target(target_id, pos)
        target.size = 1.8  # Human-sized
        target.true_detection_probability = 0.6 if "debris" in description else 0.8
        print(f"   {description} at {pos[:2]}")
    
    return env

def run_scenario_comparison():
    """Run and compare different SAR scenarios"""
    print("🎯 SAR Scenario Comparison")
    print("="*50)
    
    # Create all scenarios
    scenarios = {
        "Aircraft Crash": create_aircraft_crash_scenario(),
        "Missing Hiker": create_missing_hiker_scenario(), 
        "Urban Collapse": create_urban_collapse_scenario()
    }
    
    print("\n📊 Scenario Statistics:")
    for name, env in scenarios.items():
        print(f"\n{name}:")
        print(f"  - Search area: {env.bounds}")
        print(f"  - Drones: {len(env.drones)}")
        print(f"  - Targets: {len(env.targets)}")
        print(f"  - Buildings: {len(env.buildings)}")
        print(f"  - Hypothesis: {env.initial_hypothesis.description}")
        print(f"  - Confidence: {env.initial_hypothesis.confidence:.1%}")
    
    # Visualize initial hypotheses
    print("\n🗺️ Visualizing initial probability maps...")
    for name, env in scenarios.items():
        print(f"Showing {name} probability map...")
        env.visualize_initial_hypothesis()
    
    return scenarios

if __name__ == "__main__":
    print("🚁 Enhanced SAR Simulation with Custom Probability Maps")
    print("="*60)
    
    # Choose scenario to run
    scenario_choice = input("\nChoose scenario:\n1. Aircraft Crash\n2. Missing Hiker\n3. Urban Collapse\n4. Compare All\nEnter choice (1-4): ")
    
    if scenario_choice == "1":
        env = create_aircraft_crash_scenario()
        print("\n🚀 Starting Aircraft Crash SAR simulation...")
        env.visualize_initial_hypothesis()
        env.start_simulation()
        
    elif scenario_choice == "2":
        env = create_missing_hiker_scenario()
        print("\n🚀 Starting Missing Hiker SAR simulation...")
        env.visualize_initial_hypothesis()
        env.start_simulation()
        
    elif scenario_choice == "3":
        env = create_urban_collapse_scenario()
        print("\n🚀 Starting Urban Collapse SAR simulation...")
        env.visualize_initial_hypothesis()
        env.start_simulation()
        
    elif scenario_choice == "4":
        scenarios = run_scenario_comparison()
        
        # Let user choose which to simulate
        choice = input("\nWhich scenario would you like to simulate? (aircraft/hiker/urban): ").lower()
        if choice.startswith('a'):
            scenarios["Aircraft Crash"].start_simulation()
        elif choice.startswith('h'):
            scenarios["Missing Hiker"].start_simulation()
        elif choice.startswith('u'):
            scenarios["Urban Collapse"].start_simulation()
    
    else:
        print("Invalid choice. Running aircraft crash scenario by default...")
        env = create_aircraft_crash_scenario()
        env.start_simulation()