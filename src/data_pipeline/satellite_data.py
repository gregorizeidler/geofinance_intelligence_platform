"""
Satellite Data Processing Module
===============================

This module handles the integration and processing of satellite imagery data
for enhanced spatial analysis.
"""

import os
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
import ee
from shapely.geometry import box
import pandas as pd

class SatelliteDataProcessor:
    """Processes satellite imagery data from various sources."""
    
    def __init__(self, config):
        """Initialize the satellite data processor.
        
        Args:
            config: Configuration dictionary with satellite data settings
        """
        self.config = config
        self.initialize_earth_engine()
        
    def initialize_earth_engine(self):
        """Initialize Google Earth Engine connection."""
        try:
            ee.Initialize()
        except Exception as e:
            print(f"Warning: Earth Engine initialization failed: {e}")
            print("Some satellite features may not be available")
    
    def get_landsat_features(self, bounds):
        """Extract Landsat imagery features for the given bounds.
        
        Args:
            bounds: Tuple of (minx, miny, maxx, maxy) in WGS84
            
        Returns:
            DataFrame with satellite-derived features per hexagon
        """
        try:
            # Define area of interest
            roi = ee.Geometry.Rectangle(bounds)
            
            # Get Landsat 8 collection
            collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                        .filterBounds(roi)
                        .filterDate('2024-01-01', '2025-01-01')
                        .sort('CLOUD_COVER')
                        .first())
            
            if collection:
                # Extract common indices
                ndvi = collection.normalizedDifference(['SR_B5', 'SR_B4'])
                ndbi = collection.normalizedDifference(['SR_B6', 'SR_B5'])
                
                # Calculate urban features
                urban_features = {
                    'urban_density': ndbi,
                    'vegetation_index': ndvi,
                    'built_up_index': collection.select('SR_B6'),
                }
                
                return self._extract_features_to_df(urban_features, roi)
            
        except Exception as e:
            print(f"Warning: Landsat feature extraction failed: {e}")
        
        return pd.DataFrame()
    
    def get_sentinel_features(self, bounds):
        """Extract Sentinel-2 imagery features for the given bounds.
        
        Args:
            bounds: Tuple of (minx, miny, maxx, maxy) in WGS84
            
        Returns:
            DataFrame with satellite-derived features per hexagon
        """
        try:
            # Define area of interest
            roi = ee.Geometry.Rectangle(bounds)
            
            # Get Sentinel-2 collection
            collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                        .filterBounds(roi)
                        .filterDate('2024-01-01', '2025-01-01')
                        .sort('CLOUDY_PIXEL_PERCENTAGE')
                        .first())
            
            if collection:
                # Calculate indices
                ndvi = collection.normalizedDifference(['B8', 'B4'])
                ndwi = collection.normalizedDifference(['B3', 'B8'])
                
                # Extract features
                sentinel_features = {
                    'vegetation_density': ndvi,
                    'water_index': ndwi,
                    'urban_change': collection.select('B12'),
                }
                
                return self._extract_features_to_df(sentinel_features, roi)
            
        except Exception as e:
            print(f"Warning: Sentinel feature extraction failed: {e}")
        
        return pd.DataFrame()
    
    def _extract_features_to_df(self, features, roi):
        """Helper method to extract features to DataFrame format.
        
        Args:
            features: Dictionary of feature name to ee.Image
            roi: Earth Engine geometry defining region of interest
            
        Returns:
            DataFrame with features per hexagon
        """
        feature_data = {}
        
        for name, image in features.items():
            try:
                # Get feature values
                values = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=roi,
                    scale=30,
                    maxPixels=1e9
                ).getInfo()
                
                feature_data[name] = values
                
            except Exception as e:
                print(f"Warning: Failed to extract {name}: {e}")
        
        return pd.DataFrame(feature_data)
