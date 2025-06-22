# src/python/sar_simulation/planning/probability_maps.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
from shapely.geometry import Point, Polygon, LineString, box
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter

class IncidentType(Enum):
    """Types of SAR incidents with different search patterns"""
    AIRCRAFT_CRASH = "aircraft_crash"
    MISSING_HIKER = "missing_hiker"
    MARITIME_ACCIDENT = "maritime_accident"
    AVALANCHE = "avalanche"
    URBAN_COLLAPSE = "urban_collapse"
    WILDFIRE_EVACUATION = "wildfire_evacuation"
    FLOOD_RESCUE = "flood_rescue"

@dataclass
class ProbabilityRegion:
    """Define a region with associated probability"""
    center: Tuple[float, float]  # (x, y) center point
    shape: str  # 'circle', 'ellipse', 'polygon', 'line'
    parameters: Dict  # Shape-specific parameters
    probability_weight: float  # Relative probability (0.0 to 1.0)
    decay_type: str = 'gaussian'  # 'gaussian', 'linear', 'exponential', 'uniform'
    description: str = ""

@dataclass
class SearchHypothesis:
    """Complete search hypothesis with multiple probability regions"""
    incident_type: IncidentType
    regions: List[ProbabilityRegion]
    base_probability: float = 0.001  # Background probability
    description: str = ""
    confidence: float = 1.0  # Overall confidence in this hypothesis

class ProbabilityMapGenerator:
    """Generator for creating initial probability maps based on SAR hypotheses"""
    
    def __init__(self, bounds: Tuple[float, float, float, float], resolution: float = 10.0):
        self.bounds = bounds
        self.resolution = resolution
        
        # Create coordinate grids
        min_x, min_y, max_x, max_y = bounds
        self.x_cells = int((max_x - min_x) / resolution)
        self.y_cells = int((max_y - min_y) / resolution)
        
        # Create coordinate arrays
        x = np.linspace(min_x, max_x, self.x_cells)
        y = np.linspace(min_y, max_y, self.y_cells)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Store coordinate vectors for later use
        self.x_coords = x
        self.y_coords = y
    
    def generate_probability_map(self, hypothesis: SearchHypothesis) -> np.ndarray:
        """Generate probability map from search hypothesis"""
        # Initialize with base probability
        prob_map = np.full((self.y_cells, self.x_cells), hypothesis.base_probability)
        
        # Add each probability region
        for region in hypothesis.regions:
            region_prob = self._generate_region_probability(region)
            prob_map += region_prob * region.probability_weight
        
        # Normalize to ensure probabilities sum to 1
        total_prob = np.sum(prob_map)
        if total_prob > 0:
            prob_map = prob_map / total_prob
        
        return prob_map
    
    def _generate_region_probability(self, region: ProbabilityRegion) -> np.ndarray:
        """Generate probability distribution for a single region"""
        
        if region.shape == 'circle':
            return self._generate_circle_probability(region)
        elif region.shape == 'ellipse':
            return self._generate_ellipse_probability(region)
        elif region.shape == 'polygon':
            return self._generate_polygon_probability(region)
        elif region.shape == 'line':
            return self._generate_line_probability(region)
        elif region.shape == 'gaussian':
            return self._generate_gaussian_probability(region)
        else:
            raise ValueError(f"Unknown region shape: {region.shape}")
    
    def _generate_circle_probability(self, region: ProbabilityRegion) -> np.ndarray:
        """Generate circular probability distribution"""
        center_x, center_y = region.center
        radius = region.parameters.get('radius', 50.0)
        
        # Calculate distance from center
        distances = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)
        
        # Apply decay function
        if region.decay_type == 'gaussian':
            sigma = radius / 3  # 99.7% within radius
            probs = np.exp(-0.5 * (distances / sigma)**2)
        elif region.decay_type == 'linear':
            probs = np.maximum(0, 1 - distances / radius)
        elif region.decay_type == 'exponential':
            probs = np.exp(-distances / (radius / 3))
        elif region.decay_type == 'uniform':
            probs = (distances <= radius).astype(float)
        else:
            probs = np.exp(-0.5 * (distances / (radius / 3))**2)
        
        return probs
    
    def _generate_ellipse_probability(self, region: ProbabilityRegion) -> np.ndarray:
        """Generate elliptical probability distribution"""
        center_x, center_y = region.center
        a = region.parameters.get('semi_major', 100.0)  # Semi-major axis
        b = region.parameters.get('semi_minor', 50.0)   # Semi-minor axis
        angle = region.parameters.get('angle', 0.0)     # Rotation angle in radians
        
        # Rotate coordinates
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        x_rot = (self.X - center_x) * cos_angle + (self.Y - center_y) * sin_angle
        y_rot = -(self.X - center_x) * sin_angle + (self.Y - center_y) * cos_angle
        
        # Calculate elliptical distance
        ellipse_dist = np.sqrt((x_rot / a)**2 + (y_rot / b)**2)
        
        # Apply decay function
        if region.decay_type == 'gaussian':
            sigma = 1.0 / 3  # 99.7% within ellipse
            probs = np.exp(-0.5 * (ellipse_dist / sigma)**2)
        elif region.decay_type == 'linear':
            probs = np.maximum(0, 1 - ellipse_dist)
        elif region.decay_type == 'uniform':
            probs = (ellipse_dist <= 1.0).astype(float)
        else:
            probs = np.exp(-0.5 * (ellipse_dist / (1.0 / 3))**2)
        
        return probs
    
    def _generate_line_probability(self, region: ProbabilityRegion) -> np.ndarray:
        """Generate probability along a line (e.g., hiking trail, flight path)"""
        points = region.parameters.get('points', [])
        width = region.parameters.get('width', 20.0)
        
        if len(points) < 2:
            return np.zeros_like(self.X)
        
        # Create line geometry
        line = LineString(points)
        
        # Calculate distance to line for each grid point
        distances = np.zeros_like(self.X)
        
        for i in range(self.x_cells):
            for j in range(self.y_cells):
                point = Point(self.X[j, i], self.Y[j, i])
                distances[j, i] = line.distance(point)
        
        # Apply decay function based on distance to line
        if region.decay_type == 'gaussian':
            sigma = width / 3
            probs = np.exp(-0.5 * (distances / sigma)**2)
        elif region.decay_type == 'linear':
            probs = np.maximum(0, 1 - distances / width)
        elif region.decay_type == 'uniform':
            probs = (distances <= width).astype(float)
        else:
            probs = np.exp(-0.5 * (distances / (width / 3))**2)
        
        return probs
    
    def _generate_polygon_probability(self, region: ProbabilityRegion) -> np.ndarray:
        """Generate probability within a polygon region"""
        vertices = region.parameters.get('vertices', [])
        
        if len(vertices) < 3:
            return np.zeros_like(self.X)
        
        # Create polygon
        polygon = Polygon(vertices)
        
        # Check which grid points are inside polygon
        probs = np.zeros_like(self.X)
        
        for i in range(self.x_cells):
            for j in range(self.y_cells):
                point = Point(self.X[j, i], self.Y[j, i])
                if polygon.contains(point) or polygon.touches(point):
                    probs[j, i] = 1.0
        
        # Apply smoothing if requested
        if region.decay_type == 'gaussian':
            sigma = region.parameters.get('smoothing', 2.0)
            probs = gaussian_filter(probs, sigma=sigma)
        
        return probs

class PredefinedHypotheses:
    """Collection of predefined search hypotheses for common SAR scenarios"""
    
    @staticmethod
    def aircraft_crash_hypothesis(last_known_position: Tuple[float, float], 
                                heading: float, uncertainty_radius: float = 2000) -> SearchHypothesis:
        """Create hypothesis for aircraft crash scenario"""
        lkp_x, lkp_y = last_known_position
        
        regions = [
            # High probability near last known position
            ProbabilityRegion(
                center=last_known_position,
                shape='circle',
                parameters={'radius': uncertainty_radius * 0.3},
                probability_weight=0.5,
                decay_type='gaussian',
                description="Last known position"
            ),
            
            # Medium probability in direction of travel
            ProbabilityRegion(
                center=(lkp_x + uncertainty_radius * 0.6 * np.cos(heading),
                       lkp_y + uncertainty_radius * 0.6 * np.sin(heading)),
                shape='ellipse',
                parameters={
                    'semi_major': uncertainty_radius * 0.8,
                    'semi_minor': uncertainty_radius * 0.4,
                    'angle': heading
                },
                probability_weight=0.3,
                decay_type='gaussian',
                description="Projected flight path"
            ),
            
            # Lower probability in wider area
            ProbabilityRegion(
                center=last_known_position,
                shape='circle',
                parameters={'radius': uncertainty_radius},
                probability_weight=0.2,
                decay_type='exponential',
                description="Extended search area"
            )
        ]
        
        return SearchHypothesis(
            incident_type=IncidentType.AIRCRAFT_CRASH,
            regions=regions,
            base_probability=0.0001,
            description="Aircraft crash based on last known position and heading",
            confidence=0.8
        )
    
    @staticmethod
    def missing_hiker_hypothesis(trail_points: List[Tuple[float, float]], 
                               last_seen: Tuple[float, float]) -> SearchHypothesis:
        """Create hypothesis for missing hiker scenario"""
        regions = [
            # High probability near last seen location
            ProbabilityRegion(
                center=last_seen,
                shape='circle',
                parameters={'radius': 500},
                probability_weight=0.4,
                decay_type='gaussian',
                description="Last seen location"
            ),
            
            # Medium probability along trail
            ProbabilityRegion(
                center=(0, 0),  # Not used for line shape
                shape='line',
                parameters={
                    'points': trail_points,
                    'width': 200
                },
                probability_weight=0.35,
                decay_type='gaussian',
                description="Hiking trail"
            ),
            
            # Lower probability in off-trail areas
            ProbabilityRegion(
                center=last_seen,
                shape='circle',
                parameters={'radius': 2000},
                probability_weight=0.25,
                decay_type='exponential',
                description="Off-trail search area"
            )
        ]
        
        return SearchHypothesis(
            incident_type=IncidentType.MISSING_HIKER,
            regions=regions,
            base_probability=0.0001,
            description="Missing hiker based on trail and last known position",
            confidence=0.7
        )
    
    @staticmethod
    def urban_collapse_hypothesis(building_location: Tuple[float, float],
                                building_size: Tuple[float, float],
                                wind_direction: float = 0.0) -> SearchHypothesis:
        """Create hypothesis for urban building collapse"""
        bldg_x, bldg_y = building_location
        width, height = building_size
        
        regions = [
            # Highest probability in building footprint
            ProbabilityRegion(
                center=building_location,
                shape='polygon',
                parameters={
                    'vertices': [
                        (bldg_x - width/2, bldg_y - height/2),
                        (bldg_x + width/2, bldg_y - height/2),
                        (bldg_x + width/2, bldg_y + height/2),
                        (bldg_x - width/2, bldg_y + height/2)
                    ],
                    'smoothing': 1.0
                },
                probability_weight=0.6,
                decay_type='gaussian',
                description="Building footprint"
            ),
            
            # Medium probability in debris field (wind direction)
            ProbabilityRegion(
                center=(bldg_x + 50 * np.cos(wind_direction),
                       bldg_y + 50 * np.sin(wind_direction)),
                shape='ellipse',
                parameters={
                    'semi_major': max(width, height) * 1.5,
                    'semi_minor': max(width, height) * 0.8,
                    'angle': wind_direction
                },
                probability_weight=0.3,
                decay_type='linear',
                description="Debris field"
            ),
            
            # Lower probability in surrounding area
            ProbabilityRegion(
                center=building_location,
                shape='circle',
                parameters={'radius': max(width, height) * 2},
                probability_weight=0.1,
                decay_type='exponential',
                description="Extended search area"
            )
        ]
        
        return SearchHypothesis(
            incident_type=IncidentType.URBAN_COLLAPSE,
            regions=regions,
            base_probability=0.0001,
            description="Urban collapse with debris pattern",
            confidence=0.9
        )