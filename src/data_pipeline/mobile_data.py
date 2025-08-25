"""
Mobile Data Processing Module
===========================

This module handles the integration and processing of mobile data
for enhanced spatial analysis.
"""

import os
import pandas as pd
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from datetime import datetime, timedelta

class MobileDataProcessor:
    """Processes mobile data for spatial analysis."""
    
    def __init__(self, config):
        """Initialize the mobile data processor.
        
        Args:
            config: Configuration dictionary with mobile data settings
        """
        self.config = config
        self.cache_dir = config.get('cache_dir', 'data/raw/mobile')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_mobility_features(self, hex_grid):
        """Extract mobility features from mobile data.
        
        Args:
            hex_grid: H3 hexagonal grid for the region
            
        Returns:
            DataFrame with mobility features per hexagon
        """
        try:
            # Load and process mobility data
            mobility_data = self._load_mobility_data()
            
            if mobility_data.empty:
                return pd.DataFrame()
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(
                mobility_data,
                geometry=[Point(xy) for xy in zip(mobility_data.longitude, mobility_data.latitude)],
                crs="EPSG:4326"
            )
            
            # Join with hexagons and calculate features
            hex_features = self._calculate_hex_features(gdf, hex_grid)
            
            return hex_features
            
        except Exception as e:
            print(f"Warning: Mobility feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _load_mobility_data(self):
        """Load mobile data from various sources.
        
        Returns:
            DataFrame with processed mobile data
        """
        try:
            # Implement data loading from your mobile data source
            # This is a placeholder that should be replaced with actual data loading
            
            # Example structure:
            data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1H'),
                'latitude': np.random.uniform(-30.2, -29.9, 1000),
                'longitude': np.random.uniform(-51.3, -51.1, 1000),
                'user_id': np.random.randint(1, 1000, 1000),
                'activity_type': np.random.choice(['home', 'work', 'leisure', 'transit'], 1000),
            })
            
            return data
            
        except Exception as e:
            print(f"Warning: Failed to load mobility data: {e}")
            return pd.DataFrame()
    
    def _calculate_hex_features(self, gdf, hex_grid):
        """Calculate features for each hexagon.
        
        Args:
            gdf: GeoDataFrame with mobile data points
            hex_grid: H3 hexagonal grid
            
        Returns:
            DataFrame with features per hexagon
        """
        features = {}
        
        try:
            # Join points with hexagons
            joined = gpd.sjoin(gdf, hex_grid, how='left', op='within')
            
            # Group by hexagon and calculate features
            grouped = joined.groupby('hex_id').agg({
                'user_id': ['nunique', 'count'],
                'activity_type': lambda x: x.value_counts().to_dict()
            }).reset_index()
            
            # Calculate time-based features
            time_features = self._calculate_time_features(joined)
            
            # Combine all features
            features = pd.merge(grouped, time_features, on='hex_id', how='outer')
            
            # Add derived features
            features['activity_density'] = features['user_id']['count'] / hex_grid['area_km2']
            features['user_diversity'] = features['user_id']['nunique'] / features['user_id']['count']
            
            return features
            
        except Exception as e:
            print(f"Warning: Failed to calculate hex features: {e}")
            return pd.DataFrame()
    
    def _calculate_time_features(self, joined):
        """Calculate time-based features from mobile data.
        
        Args:
            joined: DataFrame with points joined to hexagons
            
        Returns:
            DataFrame with time-based features per hexagon
        """
        try:
            # Extract hour from timestamp
            joined['hour'] = joined['timestamp'].dt.hour
            
            # Define time periods
            time_periods = {
                'morning': (6, 10),
                'midday': (10, 14),
                'afternoon': (14, 18),
                'evening': (18, 22),
                'night': (22, 6)
            }
            
            # Calculate activity by time period
            time_features = {}
            for period, (start, end) in time_periods.items():
                mask = (joined['hour'] >= start) & (joined['hour'] < end)
                time_features[f'{period}_activity'] = joined[mask].groupby('hex_id').size()
            
            return pd.DataFrame(time_features)
            
        except Exception as e:
            print(f"Warning: Failed to calculate time features: {e}")
            return pd.DataFrame()
