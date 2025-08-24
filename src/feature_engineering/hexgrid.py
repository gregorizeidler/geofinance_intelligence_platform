"""
Hexagonal Grid System for Geospatial Analysis
=============================================

This module provides a comprehensive hexagonal grid framework using H3 (Hierarchical Hexagons)
for spatial analysis and feature aggregation. The system supports multi-resolution analysis
and efficient spatial operations across the Porto Alegre metropolitan area.

Author: Geo-Financial Intelligence Platform
"""

import h3
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from typing import List, Tuple, Dict, Optional, Union
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """Configuration for hexagonal grid generation"""
    resolution: int = 9  # H3 resolution (9 ≈ 174m average edge length)
    buffer_km: float = 5.0  # Buffer around the main area
    center_lat: float = -30.0331  # Porto Alegre center
    center_lon: float = -51.2300
    

class HexagonalGrid:
    """
    Advanced hexagonal grid system for geospatial analysis.
    
    This class provides:
    - H3-based hexagonal grid generation
    - Multi-resolution spatial indexing
    - Efficient spatial joins and aggregations
    - Geographic boundary management
    """
    
    def __init__(self, config: GridConfig = None):
        """
        Initialize the hexagonal grid system.
        
        Args:
            config: Grid configuration parameters
        """
        self.config = config or GridConfig()
        self.grid_gdf = None
        self.boundary = None
        logger.info(f"Initialized HexagonalGrid with resolution {self.config.resolution}")
    
    def create_porto_alegre_boundary(self) -> gpd.GeoDataFrame:
        """
        Create a boundary for Porto Alegre metropolitan area.
        In a real implementation, this would use official boundaries from IBGE.
        For this demo, we create a realistic boundary approximation.
        
        Returns:
            GeoDataFrame with the metropolitan boundary
        """
        # Porto Alegre approximate boundary (simplified for demonstration)
        # In production, you'd load this from IBGE municipal boundaries
        center_point = Point(self.config.center_lon, self.config.center_lat)
        
        # Create a buffer around the center (approximately metropolitan area)
        buffer_degrees = self.config.buffer_km / 111.0  # Rough km to degrees conversion
        boundary_polygon = center_point.buffer(buffer_degrees)
        
        # Create a more realistic boundary shape (elongated north-south)
        coords = list(boundary_polygon.exterior.coords)
        # Stretch the shape to better approximate Porto Alegre's geography
        stretched_coords = []
        for lon, lat in coords:
            # Stretch north-south by 1.5x, east-west by 1.2x
            new_lat = self.config.center_lat + (lat - self.config.center_lat) * 1.5
            new_lon = self.config.center_lon + (lon - self.config.center_lon) * 1.2
            stretched_coords.append((new_lon, new_lat))
        
        boundary_polygon = Polygon(stretched_coords)
        
        boundary_gdf = gpd.GeoDataFrame(
            {'region': ['Porto Alegre Metropolitan Area']}, 
            geometry=[boundary_polygon],
            crs='EPSG:4326'
        )
        
        self.boundary = boundary_gdf
        logger.info("Created Porto Alegre metropolitan boundary")
        return boundary_gdf
    
    def generate_hexagonal_grid(self, boundary: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """
        Generate hexagonal grid covering the specified boundary.
        
        Args:
            boundary: Optional boundary GeoDataFrame. If None, uses Porto Alegre boundary.
            
        Returns:
            GeoDataFrame with hexagonal cells
        """
        if boundary is None:
            boundary = self.create_porto_alegre_boundary()
        
        # Get the bounds of the boundary
        bounds = boundary.total_bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        logger.info(f"Generating H3 grid for bounds: {bounds}")
        
        # Generate H3 hexagons covering the area
        hexagons = h3.polyfill(
            boundary.geometry.iloc[0].__geo_interface__,
            res=self.config.resolution,
            geo_json_conformant=True
        )
        
        logger.info(f"Generated {len(hexagons)} hexagons at resolution {self.config.resolution}")
        
        # Convert H3 indices to polygons
        hex_polygons = []
        hex_indices = []
        
        for hex_id in hexagons:
            # Get hex boundary coordinates
            hex_boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
            
            # Create polygon (H3 returns coordinates in [lng, lat] format)
            hex_polygon = Polygon(hex_boundary)
            hex_polygons.append(hex_polygon)
            hex_indices.append(hex_id)
        
        # Create GeoDataFrame
        grid_gdf = gpd.GeoDataFrame({
            'hex_id': hex_indices,
            'area_km2': [self._calculate_hex_area(hex_id) for hex_id in hex_indices],
            'centroid_lat': [h3.h3_to_geo(hex_id)[0] for hex_id in hex_indices],
            'centroid_lon': [h3.h3_to_geo(hex_id)[1] for hex_id in hex_indices],
            'resolution': [self.config.resolution] * len(hex_indices)
        }, geometry=hex_polygons, crs='EPSG:4326')
        
        # Add additional grid metadata
        grid_gdf['grid_version'] = '1.0'
        grid_gdf['created_at'] = pd.Timestamp.now()
        
        self.grid_gdf = grid_gdf
        logger.info(f"Created hexagonal grid with {len(grid_gdf)} cells")
        
        return grid_gdf
    
    def _calculate_hex_area(self, hex_id: str) -> float:
        """Calculate the area of a hexagon in km²"""
        # H3 provides exact area calculation
        return h3.hex_area(hex_id, unit='km^2')
    
    def get_neighbors(self, hex_id: str, k: int = 1) -> List[str]:
        """
        Get neighboring hexagons within k rings.
        
        Args:
            hex_id: H3 hexagon identifier
            k: Number of rings (1 = immediate neighbors)
            
        Returns:
            List of neighboring hexagon IDs
        """
        return h3.k_ring(hex_id, k)
    
    def point_to_hex(self, lat: float, lon: float) -> str:
        """
        Convert a point to its containing hexagon ID.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            H3 hexagon identifier
        """
        return h3.geo_to_h3(lat, lon, self.config.resolution)
    
    def aggregate_to_parent_resolution(self, resolution: int) -> gpd.GeoDataFrame:
        """
        Aggregate current grid to a parent resolution (lower number = larger hexagons).
        
        Args:
            resolution: Target resolution (must be < current resolution)
            
        Returns:
            GeoDataFrame with aggregated hexagons
        """
        if resolution >= self.config.resolution:
            raise ValueError("Parent resolution must be smaller than current resolution")
        
        if self.grid_gdf is None:
            raise ValueError("Grid not yet generated. Call generate_hexagonal_grid() first.")
        
        # Map each hex to its parent
        parent_mapping = {}
        for hex_id in self.grid_gdf['hex_id']:
            parent_id = h3.h3_to_parent(hex_id, resolution)
            if parent_id not in parent_mapping:
                parent_mapping[parent_id] = []
            parent_mapping[parent_id].append(hex_id)
        
        # Create parent hexagons
        parent_polygons = []
        parent_indices = []
        parent_areas = []
        parent_centroids = []
        
        for parent_id in parent_mapping.keys():
            hex_boundary = h3.h3_to_geo_boundary(parent_id, geo_json=True)
            parent_polygon = Polygon(hex_boundary)
            parent_lat, parent_lon = h3.h3_to_geo(parent_id)
            
            parent_polygons.append(parent_polygon)
            parent_indices.append(parent_id)
            parent_areas.append(h3.hex_area(parent_id, unit='km^2'))
            parent_centroids.append((parent_lat, parent_lon))
        
        parent_gdf = gpd.GeoDataFrame({
            'hex_id': parent_indices,
            'area_km2': parent_areas,
            'centroid_lat': [c[0] for c in parent_centroids],
            'centroid_lon': [c[1] for c in parent_centroids],
            'resolution': [resolution] * len(parent_indices),
            'child_count': [len(parent_mapping[pid]) for pid in parent_indices]
        }, geometry=parent_polygons, crs='EPSG:4326')
        
        logger.info(f"Aggregated to resolution {resolution}: {len(parent_gdf)} hexagons")
        return parent_gdf
    
    def spatial_join_points(self, points_gdf: gpd.GeoDataFrame, 
                          aggregation_func: Dict[str, str] = None) -> gpd.GeoDataFrame:
        """
        Perform spatial join of points with hexagonal grid and aggregate.
        
        Args:
            points_gdf: GeoDataFrame with point geometries
            aggregation_func: Dict mapping column names to aggregation functions
                             e.g., {'value': 'sum', 'count': 'count', 'price': 'mean'}
        
        Returns:
            Grid GeoDataFrame with aggregated point data
        """
        if self.grid_gdf is None:
            raise ValueError("Grid not yet generated. Call generate_hexagonal_grid() first.")
        
        # Ensure both GDFs have the same CRS
        if points_gdf.crs != self.grid_gdf.crs:
            points_gdf = points_gdf.to_crs(self.grid_gdf.crs)
        
        # Perform spatial join
        joined = gpd.sjoin(points_gdf, self.grid_gdf, how='inner', op='within')
        
        # Default aggregation
        if aggregation_func is None:
            aggregation_func = {'geometry': 'count'}  # Count points per hex
        
        # Group by hex_id and aggregate
        agg_data = joined.groupby('hex_id').agg(aggregation_func).reset_index()
        
        # Merge back with grid
        result = self.grid_gdf.merge(agg_data, on='hex_id', how='left')
        
        # Fill NaN values with 0 for count-based aggregations
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        result[numeric_columns] = result[numeric_columns].fillna(0)
        
        logger.info(f"Spatial join completed: {len(result)} hexagons with aggregated data")
        return result
    
    def export_grid(self, filepath: str, format: str = 'geojson'):
        """
        Export the hexagonal grid to file.
        
        Args:
            filepath: Output file path
            format: Export format ('geojson', 'shapefile', 'parquet')
        """
        if self.grid_gdf is None:
            raise ValueError("Grid not yet generated. Call generate_hexagonal_grid() first.")
        
        if format.lower() == 'geojson':
            self.grid_gdf.to_file(filepath, driver='GeoJSON')
        elif format.lower() == 'shapefile':
            self.grid_gdf.to_file(filepath, driver='ESRI Shapefile')
        elif format.lower() == 'parquet':
            self.grid_gdf.to_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Grid exported to {filepath} in {format} format")
    
    def get_grid_stats(self) -> Dict:
        """Get summary statistics about the current grid."""
        if self.grid_gdf is None:
            return {"error": "Grid not generated"}
        
        return {
            "total_hexagons": len(self.grid_gdf),
            "total_area_km2": self.grid_gdf['area_km2'].sum(),
            "avg_hex_area_km2": self.grid_gdf['area_km2'].mean(),
            "resolution": self.config.resolution,
            "bounds": self.grid_gdf.total_bounds.tolist(),
            "center_lat": self.config.center_lat,
            "center_lon": self.config.center_lon
        }


def create_porto_alegre_grid(resolution: int = 9, export_path: str = None) -> HexagonalGrid:
    """
    Convenience function to create a hexagonal grid for Porto Alegre.
    
    Args:
        resolution: H3 resolution level
        export_path: Optional path to export the grid
        
    Returns:
        Configured HexagonalGrid instance
    """
    config = GridConfig(resolution=resolution)
    grid = HexagonalGrid(config)
    grid.generate_hexagonal_grid()
    
    if export_path:
        grid.export_grid(export_path)
    
    return grid


if __name__ == "__main__":
    # Example usage
    print("Creating Porto Alegre hexagonal grid...")
    
    # Create grid
    grid = create_porto_alegre_grid(resolution=9)
    
    # Print statistics
    stats = grid.get_grid_stats()
    print(f"Grid Statistics:")
    print(f"- Total hexagons: {stats['total_hexagons']:,}")
    print(f"- Total area: {stats['total_area_km2']:.1f} km²")
    print(f"- Average hex area: {stats['avg_hex_area_km2']:.3f} km²")
    print(f"- Resolution: {stats['resolution']}")