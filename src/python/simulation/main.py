# src/python/sar_simulation/main.py

"""
Main SAR Simulation Runner

This file integrates all components and provides scenario examples.
Run this file to see the complete SAR simulation with probability maps.
"""

import sys
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from simulation.planning.probability_maps import (
    ProbabilityMapGenerator, 
    PredefinedHypotheses, 
    SearchHypothesis,
    ProbabilityRegion,
    IncidentType
)

def create_simple_aircraft_scenario():
    """Create a simple aircraft crash scenario to test the system"""
    
    print("✈️ Creating Aircraft Crash Scenario")
    print("="*40)
    
    # Define search area
    bounds = (-1000, -1000, 1000, 1000)  # 2km x 2km
    
    # Create probability map generator
    generator = ProbabilityMapGenerator(bounds, resolution=20.0)
    
    # Create aircraft crash hypothesis
    last_known_position = (-200, 100)
    flight_heading = np.pi/6  # 30 degrees northeast
    
    hypothesis = PredefinedHypotheses.aircraft_crash_hypothesis(
        last_known_position=last_known_position,
        heading=flight_heading,
        uncertainty_radius=500
    )
    
    # Generate probability map
    prob_map = generator.generate_probability_map(hypothesis)
    
    # Visualize the probability map
    print("📊 Generating probability map visualization...")
    
    plt.figure(figsize=(12, 10))
    
    min_x, min_y, max_x, max_y = bounds
    im = plt.imshow(prob_map, extent=[min_x, max_x, min_y, max_y], 
                   origin='lower', cmap='hot', alpha=0.8)
    
    plt.colorbar(im, label='Target Probability Density')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('Aircraft Crash Search Probability Map')
    plt.grid(True, alpha=0.3)
    
    # Mark key locations
    plt.plot(last_known_position[0], last_known_position[1], 'bo', 
            markersize=10, label='Last Known Position')
    
    # Show projected crash site
    projected_x = last_known_position[0] + 300 * np.cos(flight_heading)
    projected_y = last_known_position[1] + 300 * np.sin(flight_heading)
    plt.plot(projected_x, projected_y, 'r*', markersize=15, label='Projected Impact')
    
    plt.legend()
    plt.show()
    
    return generator, hypothesis, prob_map

def create_hiker_scenario():
    """Create a missing hiker scenario"""
    
    print("🥾 Creating Missing Hiker Scenario")
    print("="*40)
    
    # Define search area
    bounds = (-800, -800, 800, 800)
    
    # Create probability map generator
    generator = ProbabilityMapGenerator(bounds, resolution=15.0)
    
    # Define hiking trail
    trail_points = [
        (-600, -400), (-400, -300), (-200, -200), 
        (0, -100), (200, 0), (400, 100), (600, 200)
    ]
    last_seen_position = (100, -50)
    
    # Create hiker hypothesis
    hypothesis = PredefinedHypotheses.missing_hiker_hypothesis(
        trail_points=trail_points,
        last_seen=last_seen_position
    )
    
    # Generate probability map
    prob_map = generator.generate_probability_map(hypothesis)
    
    # Visualize
    print("📊 Generating hiker probability map...")
    
    plt.figure(figsize=(12, 10))
    
    min_x, min_y, max_x, max_y = bounds
    im = plt.imshow(prob_map, extent=[min_x, max_x, min_y, max_y], 
                   origin='lower', cmap='hot', alpha=0.8)
    
    plt.colorbar(im, label='Target Probability Density')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('Missing Hiker Search Probability Map')
    plt.grid(True, alpha=0.3)
    
    # Draw hiking trail
    trail_x = [p[0] for p in trail_points]
    trail_y = [p[1] for p in trail_points]
    plt.plot(trail_x, trail_y, 'g-', linewidth=3, label='Hiking Trail')
    
    # Mark last seen location
    plt.plot(last_seen_position[0], last_seen_position[1], 'bo', 
            markersize=10, label='Last Seen')
    
    plt.legend()
    plt.show()
    
    return generator, hypothesis, prob_map

def create_urban_scenario():
    """Create an urban collapse scenario"""
    
    print("🏢 Creating Urban Collapse Scenario")
    print("="*40)
    
    # Define search area
    bounds = (-300, -300, 300, 300)
    
    # Create probability map generator
    generator = ProbabilityMapGenerator(bounds, resolution=10.0)
    
    # Define collapsed building
    building_location = (50, 50)
    building_size = (60, 40)
    wind_direction = np.pi/4  # Northeast wind
    
    # Create urban collapse hypothesis
    hypothesis = PredefinedHypotheses.urban_collapse_hypothesis(
        building_location=building_location,
        building_size=building_size,
        wind_direction=wind_direction
    )
    
    # Generate probability map
    prob_map = generator.generate_probability_map(hypothesis)
    
    # Visualize
    print("📊 Generating urban collapse probability map...")
    
    plt.figure(figsize=(12, 10))
    
    min_x, min_y, max_x, max_y = bounds
    im = plt.imshow(prob_map, extent=[min_x, max_x, min_y, max_y], 
                   origin='lower', cmap='hot', alpha=0.8)
    
    plt.colorbar(im, label='Target Probability Density')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('Urban Building Collapse Probability Map')
    plt.grid(True, alpha=0.3)
    
    # Draw building outline
    bldg_x, bldg_y = building_location
    width, height = building_size
    building_corners = [
        (bldg_x - width/2, bldg_y - height/2),
        (bldg_x + width/2, bldg_y - height/2),
        (bldg_x + width/2, bldg_y + height/2),
        (bldg_x - width/2, bldg_y + height/2),
        (bldg_x - width/2, bldg_y - height/2)  # Close the square
    ]
    
    corner_x = [c[0] for c in building_corners]
    corner_y = [c[1] for c in building_corners]
    plt.plot(corner_x, corner_y, 'k-', linewidth=3, label='Collapsed Building')
    
    # Show wind direction
    wind_end_x = bldg_x + 100 * np.cos(wind_direction)
    wind_end_y = bldg_y + 100 * np.sin(wind_direction)
    plt.arrow(bldg_x, bldg_y, wind_end_x - bldg_x, wind_end_y - bldg_y,
              head_width=10, head_length=15, fc='blue', ec='blue', 
              label='Wind Direction')
    
    plt.legend()
    plt.show()
    
    return generator, hypothesis, prob_map

def create_custom_hypothesis_example():
    """Create a custom hypothesis with multiple regions"""
    
    print("🎯 Creating Custom Multi-Region Hypothesis")
    print("="*45)
    
    bounds = (-500, -500, 500, 500)
    generator = ProbabilityMapGenerator(bounds, resolution=12.0)
    
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
    
    # Generate and visualize
    prob_map = generator.generate_probability_map(custom_hypothesis)
    
    plt.figure(figsize=(14, 10))
    
    min_x, min_y, max_x, max_y = bounds
    im = plt.imshow(prob_map, extent=[min_x, max_x, min_y, max_y], 
                   origin='lower', cmap='hot', alpha=0.8)
    
    plt.colorbar(im, label='Target Probability Density')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('Custom Multi-Region Search Hypothesis')
    plt.grid(True, alpha=0.3)
    
    # Annotate regions
    for i, region in enumerate(custom_hypothesis.regions):
        if region.shape != 'line':
            plt.plot(region.center[0], region.center[1], 'wo', markersize=8)
            plt.annotate(f'{i+1}', region.center, xytext=(5, 5), 
                        textcoords='offset points', color='white', fontweight='bold')
    
    plt.show()
    
    return generator, custom_hypothesis, prob_map

def save_hypothesis_example():
    """Example of saving and loading hypotheses"""
    
    print("💾 Demonstrating Hypothesis Save/Load")
    print("="*40)
    
    # Create aircraft hypothesis
    hypothesis = PredefinedHypotheses.aircraft_crash_hypothesis(
        last_known_position=(0, 0),
        heading=0,
        uncertainty_radius=1000
    )
    
    # Convert to serializable format
    data = {
        'incident_type': hypothesis.incident_type.value,
        'description': hypothesis.description,
        'confidence': hypothesis.confidence,
        'base_probability': hypothesis.base_probability,
        'regions': []
    }
    
    for region in hypothesis.regions:
        region_data = {
            'center': region.center,
            'shape': region.shape,
            'parameters': region.parameters,
            'probability_weight': region.probability_weight,
            'decay_type': region.decay_type,
            'description': region.description
        }
        data['regions'].append(region_data)
    
    # Save to JSON
    import json
    filename = 'example_aircraft_hypothesis.json'
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Hypothesis saved to {filename}")
    
    # Show the JSON structure
    print("\n📄 JSON Structure:")
    print(json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data, indent=2)) > 500 else json.dumps(data, indent=2))
    
    return filename

def main():
    """Main function to run SAR probability map examples"""
    
    print("🗺️ SAR Probability Map Generator")
    print("="*50)
    print("Demonstrating custom probability maps for Search and Rescue operations")
    print()
    
    try:
        choice = input("Choose scenario to demonstrate:\n"
                      "1. Aircraft Crash\n"
                      "2. Missing Hiker\n"
                      "3. Urban Collapse\n"
                      "4. Custom Multi-Region\n"
                      "5. Save/Load Example\n"
                      "6. All Scenarios\n"
                      "Enter choice (1-6): ").strip()
        
        if choice == "1":
            create_simple_aircraft_scenario()
        elif choice == "2":
            create_hiker_scenario()
        elif choice == "3":
            create_urban_scenario()
        elif choice == "4":
            create_custom_hypothesis_example()
        elif choice == "5":
            save_hypothesis_example()
        elif choice == "6":
            print("\n🎬 Running all scenario demonstrations...")
            create_simple_aircraft_scenario()
            create_hiker_scenario()
            create_urban_scenario()
            create_custom_hypothesis_example()
            save_hypothesis_example()
        else:
            print("Invalid choice. Running aircraft crash scenario by default...")
            create_simple_aircraft_scenario()
        
        print("\n✅ Demonstration complete!")
        print("\nNext steps:")
        print("1. Integrate these probability maps with the POMDP environment")
        print("2. Use the generated maps as initial drone beliefs")
        print("3. Run the complete SAR simulation with intelligent search")
        
    except KeyboardInterrupt:
        print("\n🛑 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()