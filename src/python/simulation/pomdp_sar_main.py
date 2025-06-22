# src/python/sar_simulation/integrated_sar_main.py

"""
Integrated SAR Main Application

Combines the POMDP environment, probability maps, and 3D visualization
for complete Search and Rescue simulation with intelligent drone coordination.

This file imports from the modular components and creates integrated scenarios.
"""

import sys
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
import open3d as o3d
import trimesh
from shapely.geometry import box

# Configure matplotlib backend for interactive/headless mode
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
    Drone, SearchTarget, Building, DroneStatus, BeliefState, Action, ActionType, Observation
)
from simulation.planning.pomdp_planner import DistributedPOMDPSearchPlanner
from simulation.planning.probability_maps import (
    ProbabilityMapGenerator, PredefinedHypotheses, 
    SearchHypothesis, ProbabilityRegion, IncidentType
)

from simulation.sarenvironement.sar_pomdp_environement import EnhancedSAREnvironment


def create_aircraft_crash_scenario():
    """Create realistic aircraft crash SAR scenario with custom probability map"""
    print("✈️ Creating Aircraft Crash SAR Scenario with Custom Probability Map")
    print("="*70)
    
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
    
    # Create enhanced environment with hypothesis
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
        drone.status = DroneStatus.SEARCHING
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
    """Create missing hiker SAR scenario with trail-based probability map"""
    print("🥾 Creating Missing Hiker SAR Scenario with Trail-Based Probability Map")
    print("="*70)
    
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
        drone.status = DroneStatus.SEARCHING
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
    """Create urban building collapse SAR scenario with debris pattern probability map"""
    print("🏢 Creating Urban Collapse SAR Scenario with Debris Pattern Probability Map")
    print("="*75)
    
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
        drone.status = DroneStatus.SEARCHING
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
    """Run and compare different SAR scenarios with their probability maps"""
    print("🎯 SAR Scenario Comparison with Custom Probability Maps")
    print("="*60)
    
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

def create_custom_multi_region_scenario():
    """Create a custom scenario with multiple probability regions"""
    print("🎯 Creating Custom Multi-Region SAR Scenario")
    print("="*50)
    
    bounds = (-500, -500, 500, 500)
    
    # Create custom hypothesis with multiple probability regions
    custom_hypothesis = SearchHypothesis(
        incident_type=IncidentType.AIRCRAFT_CRASH,
        regions=[
            # Primary crash site
            ProbabilityRegion(
                center=(100, 100),
                shape='circle',
                parameters={'radius': 80},
                probability_weight=0.4,
                decay_type='gaussian',
                description="Primary impact site"
            ),
            
            # Secondary debris field
            ProbabilityRegion(
                center=(200, 150),
                shape='ellipse',
                parameters={
                    'semi_major': 120,
                    'semi_minor': 60,
                    'angle': np.pi/4
                },
                probability_weight=0.3,
                decay_type='linear',
                description="Debris scatter"
            ),
            
            # Evacuation route
            ProbabilityRegion(
                center=(0, 0),  # Not used for line
                shape='line',
                parameters={
                    'points': [(100, 120), (150, 200), (200, 300)],
                    'width': 40
                },
                probability_weight=0.2,
                decay_type='gaussian',
                description="Escape route"
            ),
            
            # Search area polygon
            ProbabilityRegion(
                center=(-200, -200),
                shape='polygon',
                parameters={
                    'vertices': [(-300, -300), (-100, -300), (-150, -100), (-250, -100)],
                    'smoothing': 3.0
                },
                probability_weight=0.1,
                decay_type='gaussian',
                description="Secondary search area"
            )
        ],
        base_probability=0.0001,
        description="Custom multi-region search hypothesis",
        confidence=0.75
    )
    
    # Create environment with custom hypothesis
    env = EnhancedSAREnvironment(bounds=bounds, initial_hypothesis=custom_hypothesis)
    
    # Add mixed environment
    print("🏗️ Adding mixed environment...")
    env.add_building("facility_1", np.array([0, -200, 0]), np.array([100, 50, 20]))
    env.add_building("warehouse", np.array([-200, 0, 0]), np.array([80, 80, 15]))
    
    # Deploy diverse drone fleet
    print("🚁 Deploying diverse drone fleet...")
    drone_positions = [
        np.array([150, 150, 25]),   # Near primary site
        np.array([-150, -150, 25]), # Opposite corner
        np.array([0, 250, 25]),     # North coverage
        np.array([250, 0, 25]),     # East coverage
        np.array([0, 0, 30]),       # Central coordinator
    ]
    
    for i, pos in enumerate(drone_positions):
        drone_id = f"MULTI_{i+1}"
        drone = env.add_drone_with_prior(drone_id, pos)
        drone.sensor_range = 55.0
        drone.status = DroneStatus.SEARCHING
        print(f"   Deployed {drone_id} at {pos[:2]}")
    
    # Add targets across all regions
    print("🎯 Placing targets across multiple regions...")
    target_locations = [
        (np.array([95, 105, 1]), "Primary site target"),
        (np.array([180, 160, 2]), "Debris field target"),
        (np.array([175, 275, 0]), "Escape route target"),
        (np.array([-175, -175, 1]), "Secondary area target"),
        (np.array([0, 0, 0]), "Central target"),
    ]
    
    for i, (pos, description) in enumerate(target_locations):
        target_id = f"multi_target_{i+1}"
        target = env.add_target(target_id, pos)
        target.size = 2.0
        target.true_detection_probability = 0.75
        print(f"   {description} at {pos[:2]}")
    
    return env

def main():
    """Main function to run integrated SAR scenarios"""
    
    print("🚁 Integrated SAR Simulation with POMDP and Custom Probability Maps")
    print("="*80)
    print("Combining modular components:")
    print("  - Core entities from sar_simulation.core.entities")
    print("  - POMDP planner from sar_simulation.planning.pomdp_planner")
    print("  - Probability maps from sar_simulation.planning.probability_maps")
    print("  - 3D SAR environment integration with real-time rendering")
    print()
    
    try:
        choice = input("Choose integrated scenario to run:\n"
                      "1. Aircraft Crash with Flight Path Probability\n"
                      "2. Missing Hiker with Trail-Based Search\n"
                      "3. Urban Collapse with Debris Pattern\n"
                      "4. Custom Multi-Region Scenario\n"
                      "5. Compare All Scenarios\n"
                      "Enter choice (1-5): ").strip()
        
        if choice == "1":
            env = create_aircraft_crash_scenario()
            print("\n🗺️ Visualizing probability hypothesis...")
            env.visualize_initial_hypothesis()
            print("\n🚀 Starting aircraft crash simulation with 3D rendering...")
            env.start_simulation()
            
        elif choice == "2":
            env = create_missing_hiker_scenario()
            print("\n🗺️ Visualizing probability hypothesis...")
            env.visualize_initial_hypothesis()
            print("\n🚀 Starting missing hiker simulation with 3D rendering...")
            env.start_simulation()
            
        elif choice == "3":
            env = create_urban_collapse_scenario()
            print("\n🗺️ Visualizing probability hypothesis...")
            env.visualize_initial_hypothesis()
            print("\n🚀 Starting urban collapse simulation with 3D rendering...")
            env.start_simulation()
            
        elif choice == "4":
            env = create_custom_multi_region_scenario()
            print("\n🗺️ Visualizing probability hypothesis...")
            env.visualize_initial_hypothesis()
            print("\n🚀 Starting custom multi-region simulation with 3D rendering...")
            env.start_simulation()
            
        elif choice == "5":
            scenarios = run_scenario_comparison()
            
            # Let user choose which to simulate
            choice = input("\nWhich scenario would you like to simulate with 3D rendering? (aircraft/hiker/urban): ").lower()
            if choice.startswith('a'):
                scenarios["Aircraft Crash"].start_simulation()
            elif choice.startswith('h'):
                scenarios["Missing Hiker"].start_simulation()
            elif choice.startswith('u'):
                scenarios["Urban Collapse"].start_simulation()
            else:
                print("No simulation selected.")
        
        else:
            print("Invalid choice. Running aircraft crash scenario by default...")
            env = create_aircraft_crash_scenario()
            env.visualize_initial_hypothesis()
            env.start_simulation()
        
        print("\n✅ Integrated SAR simulation complete!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Custom probability map generation")
        print("  ✅ POMDP-based intelligent planning")
        print("  ✅ Modular component integration")
        print("  ✅ Scenario-specific drone deployment")
        print("  ✅ Real-time probability visualization")
        print("  ✅ 3D rendering with drone fleet coordination")
        print("  ✅ Interactive simulation environment")
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()