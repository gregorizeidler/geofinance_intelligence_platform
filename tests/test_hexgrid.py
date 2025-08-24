"""
Unit Tests for Hexagonal Grid System
====================================

Test suite for the H3-based hexagonal grid generation and spatial operations.
"""

import unittest
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from feature_engineering.hexgrid import HexagonalGrid, GridConfig, create_porto_alegre_grid


class TestHexagonalGrid(unittest.TestCase):
    """Test cases for HexagonalGrid class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = GridConfig(resolution=10)  # Smaller resolution for faster tests
        self.hex_grid = HexagonalGrid(self.config)
    
    def test_grid_initialization(self):
        """Test hexagonal grid initialization"""
        self.assertIsNotNone(self.hex_grid)
        self.assertEqual(self.hex_grid.config.resolution, 10)
        self.assertIsNone(self.hex_grid.grid_gdf)
    
    def test_boundary_creation(self):
        """Test boundary creation for Porto Alegre"""
        boundary = self.hex_grid.create_porto_alegre_boundary()
        
        self.assertIsInstance(boundary, gpd.GeoDataFrame)
        self.assertEqual(len(boundary), 1)
        self.assertIn('region', boundary.columns)
        self.assertIn('geometry', boundary.columns)
        self.assertEqual(boundary.crs, 'EPSG:4326')
    
    def test_grid_generation(self):
        """Test hexagonal grid generation"""
        grid_gdf = self.hex_grid.generate_hexagonal_grid()
        
        # Basic structure tests
        self.assertIsInstance(grid_gdf, gpd.GeoDataFrame)
        self.assertGreater(len(grid_gdf), 0)
        self.assertEqual(grid_gdf.crs, 'EPSG:4326')
        
        # Required columns
        required_cols = ['hex_id', 'area_km2', 'centroid_lat', 'centroid_lon', 'resolution']
        for col in required_cols:
            self.assertIn(col, grid_gdf.columns)
        
        # Data validation
        self.assertTrue(all(grid_gdf['resolution'] == 10))
        self.assertTrue(all(grid_gdf['area_km2'] > 0))
        self.assertTrue(all(grid_gdf['hex_id'].str.len() > 10))  # H3 IDs are long strings
    
    def test_hex_neighbors(self):
        """Test neighbor identification"""
        # Generate a small grid first
        self.hex_grid.generate_hexagonal_grid()
        
        if len(self.hex_grid.grid_gdf) > 0:
            # Test neighbor finding
            hex_id = self.hex_grid.grid_gdf.iloc[0]['hex_id']
            neighbors = self.hex_grid.get_neighbors(hex_id, k=1)
            
            self.assertIsInstance(neighbors, list)
            self.assertIn(hex_id, neighbors)  # Should include self
            self.assertLessEqual(len(neighbors), 7)  # Max 6 neighbors + self
    
    def test_point_to_hex_conversion(self):
        """Test point to hex conversion"""
        # Porto Alegre center coordinates
        lat, lon = -30.0331, -51.2300
        hex_id = self.hex_grid.point_to_hex(lat, lon)
        
        self.assertIsInstance(hex_id, str)
        self.assertGreater(len(hex_id), 10)
    
    def test_spatial_join(self):
        """Test spatial join functionality"""
        # Create test grid
        self.hex_grid.generate_hexagonal_grid()
        
        if len(self.hex_grid.grid_gdf) > 0:
            # Create test points
            test_points = []
            for _, row in self.hex_grid.grid_gdf.head(5).iterrows():
                test_points.append({
                    'id': f'point_{len(test_points)}',
                    'value': np.random.random(),
                    'geometry': Point(row['centroid_lon'], row['centroid_lat'])
                })
            
            points_gdf = gpd.GeoDataFrame(test_points, crs='EPSG:4326')
            
            # Perform spatial join
            result = self.hex_grid.spatial_join_points(points_gdf, {'value': 'sum'})
            
            self.assertIsInstance(result, gpd.GeoDataFrame)
            self.assertEqual(len(result), len(self.hex_grid.grid_gdf))
    
    def test_grid_stats(self):
        """Test grid statistics calculation"""
        self.hex_grid.generate_hexagonal_grid()
        stats = self.hex_grid.get_grid_stats()
        
        self.assertIsInstance(stats, dict)
        
        required_stats = ['total_hexagons', 'total_area_km2', 'resolution', 'bounds']
        for stat in required_stats:
            self.assertIn(stat, stats)
        
        if 'total_hexagons' in stats:
            self.assertGreater(stats['total_hexagons'], 0)
            self.assertGreater(stats['total_area_km2'], 0)
    
    def test_create_porto_alegre_grid_function(self):
        """Test the convenience function"""
        grid = create_porto_alegre_grid(resolution=10)
        
        self.assertIsInstance(grid, HexagonalGrid)
        self.assertIsNotNone(grid.grid_gdf)
        self.assertGreater(len(grid.grid_gdf), 0)


class TestGridConfig(unittest.TestCase):
    """Test cases for GridConfig class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = GridConfig()
        
        self.assertEqual(config.resolution, 9)
        self.assertEqual(config.buffer_km, 5.0)
        self.assertEqual(config.center_lat, -30.0331)
        self.assertEqual(config.center_lon, -51.2300)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = GridConfig(
            resolution=8,
            buffer_km=10.0,
            center_lat=-29.0,
            center_lon=-50.0
        )
        
        self.assertEqual(config.resolution, 8)
        self.assertEqual(config.buffer_km, 10.0)
        self.assertEqual(config.center_lat, -29.0)
        self.assertEqual(config.center_lon, -50.0)


class TestGridPerformance(unittest.TestCase):
    """Performance tests for grid operations"""
    
    def test_grid_generation_performance(self):
        """Test grid generation performance"""
        import time
        
        config = GridConfig(resolution=10)  # Smaller for performance test
        hex_grid = HexagonalGrid(config)
        
        start_time = time.time()
        grid_gdf = hex_grid.generate_hexagonal_grid()
        generation_time = time.time() - start_time
        
        # Should complete reasonably quickly
        self.assertLess(generation_time, 30.0)  # 30 seconds max
        self.assertGreater(len(grid_gdf), 0)
        
        print(f"Grid generation time: {generation_time:.2f}s for {len(grid_gdf)} hexagons")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)