"""
Multi-Source Geospatial Data Pipeline
====================================

This module provides a comprehensive data ingestion and processing pipeline for the
Geo-Financial Intelligence Platform. It handles data from multiple sources:
- IBGE (Brazilian Census and Economic Data)
- OpenStreetMap (POI and Infrastructure Data)
- Synthetic Financial Transaction Data
- Urban Infrastructure and Transportation Networks

Author: Geo-Financial Intelligence Platform
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import requests
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data source connections and processing"""
    cache_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    ibge_api_base: str = "https://servicodados.ibge.gov.br/api/v1"
    osm_network_type: str = "drive"  # drive, walk, bike, all
    synthetic_data_seed: int = 42
    

class IBGEDataProcessor:
    """
    Processor for IBGE (Brazilian Institute of Geography and Statistics) data.
    Handles census data, economic indicators, and administrative boundaries.
    """
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir) / "ibge"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_census_data_by_tract(self, city_code: str = "4314902") -> gpd.GeoDataFrame:
        """
        Fetch IBGE census data by census tract for Porto Alegre.
        
        Args:
            city_code: IBGE city code (4314902 = Porto Alegre)
            
        Returns:
            GeoDataFrame with census tract polygons and demographic data
        """
        # Note: In a real implementation, you would use the actual IBGE API
        # This creates realistic synthetic census data for demonstration
        
        logger.info(f"Fetching census data for city code: {city_code}")
        
        # Generate synthetic census tracts for Porto Alegre area
        tracts_data = self._generate_synthetic_census_tracts()
        
        logger.info(f"Retrieved {len(tracts_data)} census tracts")
        return tracts_data
    
    def _generate_synthetic_census_tracts(self) -> gpd.GeoDataFrame:
        """Generate realistic synthetic census tract data for Porto Alegre"""
        np.random.seed(self.config.synthetic_data_seed)
        
        # Define Porto Alegre bounds
        center_lat, center_lon = -30.0331, -51.2300
        
        # Generate approximately 200 census tracts
        n_tracts = 200
        tract_data = []
        
        for i in range(n_tracts):
            # Random location within Porto Alegre bounds
            lat_offset = np.random.normal(0, 0.05)  # ~5km radius variation
            lon_offset = np.random.normal(0, 0.05)
            
            tract_lat = center_lat + lat_offset
            tract_lon = center_lon + lon_offset
            
            # Create small polygon for each tract (~1km²)
            size = 0.005  # degrees
            tract_polygon = Polygon([
                (tract_lon - size/2, tract_lat - size/2),
                (tract_lon + size/2, tract_lat - size/2),
                (tract_lon + size/2, tract_lat + size/2),
                (tract_lon - size/2, tract_lat + size/2)
            ])
            
            # Generate realistic demographic data
            base_population = np.random.lognormal(7.5, 0.5)  # ~2000 avg population
            
            tract_info = {
                'tract_id': f"430149020000{i:03d}",
                'population_total': int(base_population),
                'households': int(base_population / np.random.uniform(2.5, 3.5)),
                'avg_income_brl': np.random.lognormal(8.5, 0.6),  # ~5000 BRL avg
                'education_higher_pct': np.random.beta(3, 7) * 100,  # Skewed toward lower
                'age_0_14_pct': np.random.uniform(15, 25),
                'age_15_64_pct': np.random.uniform(60, 70),
                'age_65_plus_pct': np.random.uniform(10, 20),
                'unemployment_rate': np.random.beta(2, 8) * 20,  # 0-20% range
                'density_pop_km2': base_population / (size * 111 * size * 111),  # Rough calculation
                'geometry': tract_polygon
            }
            
            # Ensure age percentages sum to ~100%
            total_age = tract_info['age_0_14_pct'] + tract_info['age_15_64_pct'] + tract_info['age_65_plus_pct']
            for age_col in ['age_0_14_pct', 'age_15_64_pct', 'age_65_plus_pct']:
                tract_info[age_col] = (tract_info[age_col] / total_age) * 100
            
            tract_data.append(tract_info)
        
        return gpd.GeoDataFrame(tract_data, crs='EPSG:4326')
    
    def get_municipal_boundaries(self) -> gpd.GeoDataFrame:
        """Get municipal boundaries for the metropolitan area"""
        # Synthetic boundary for Porto Alegre and surrounding municipalities
        municipalities = [
            {"name": "Porto Alegre", "code": "4314902", "lat": -30.0331, "lon": -51.2300, "size": 0.15},
            {"name": "Canoas", "code": "4304606", "lat": -29.9177, "lon": -51.1844, "size": 0.08},
            {"name": "Gravataí", "code": "4309209", "lat": -29.9444, "lon": -50.9919, "size": 0.06},
            {"name": "Viamão", "code": "4323002", "lat": -30.0811, "lon": -51.0236, "size": 0.12},
            {"name": "Alvorada", "code": "4301206", "lat": -29.9897, "lon": -51.0831, "size": 0.05}
        ]
        
        municipal_data = []
        for muni in municipalities:
            size = muni["size"]
            polygon = Polygon([
                (muni["lon"] - size/2, muni["lat"] - size/2),
                (muni["lon"] + size/2, muni["lat"] - size/2),
                (muni["lon"] + size/2, muni["lat"] + size/2),
                (muni["lon"] - size/2, muni["lat"] + size/2)
            ])
            
            municipal_data.append({
                'municipality': muni["name"],
                'ibge_code': muni["code"],
                'area_km2': size * 111 * size * 111,  # Rough area calculation
                'geometry': polygon
            })
        
        return gpd.GeoDataFrame(municipal_data, crs='EPSG:4326')


class OSMDataProcessor:
    """
    Processor for OpenStreetMap data including Points of Interest (POI),
    road networks, and urban infrastructure.
    """
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir) / "osm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_pois_by_category(self, boundary: gpd.GeoDataFrame, 
                           categories: List[str] = None) -> gpd.GeoDataFrame:
        """
        Fetch Points of Interest (POI) from OpenStreetMap within the boundary.
        
        Args:
            boundary: Geographic boundary to query
            categories: List of POI categories to fetch
            
        Returns:
            GeoDataFrame with POI locations and attributes
        """
        if categories is None:
            categories = [
                'restaurant', 'cafe', 'bank', 'atm', 'pharmacy', 'hospital',
                'school', 'university', 'shopping_centre', 'supermarket',
                'fuel', 'parking', 'hotel', 'tourist_attraction'
            ]
        
        logger.info(f"Fetching OSM POIs for {len(categories)} categories")
        
        # Get boundary polygon for querying
        if len(boundary) > 1:
            # Union multiple polygons
            query_polygon = boundary.geometry.unary_union
        else:
            query_polygon = boundary.geometry.iloc[0]
        
        all_pois = []
        
        for category in categories:
            try:
                # Use OSMnx to fetch POIs
                pois = ox.geometries_from_polygon(
                    query_polygon,
                    tags={category: True}
                )
                
                if not pois.empty:
                    # Convert to points (in case we get polygons/lines)
                    pois = pois.copy()
                    pois['poi_category'] = category
                    pois['geometry'] = pois.geometry.centroid
                    
                    # Select relevant columns
                    poi_cols = ['poi_category', 'name', 'geometry']
                    available_cols = [col for col in poi_cols if col in pois.columns]
                    pois = pois[available_cols]
                    
                    all_pois.append(pois)
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Could not fetch {category} POIs: {e}")
                # Generate synthetic POIs for demonstration
                synthetic_pois = self._generate_synthetic_pois(query_polygon, category, 20)
                if not synthetic_pois.empty:
                    all_pois.append(synthetic_pois)
        
        if all_pois:
            combined_pois = gpd.GeoDataFrame(pd.concat(all_pois, ignore_index=True))
            combined_pois.crs = 'EPSG:4326'
            logger.info(f"Retrieved {len(combined_pois)} POIs total")
            return combined_pois
        else:
            logger.warning("No POIs found, generating synthetic data")
            return self._generate_synthetic_pois(query_polygon, "mixed", 500)
    
    def _generate_synthetic_pois(self, polygon: Polygon, category: str, count: int) -> gpd.GeoDataFrame:
        """Generate synthetic POI data within a polygon"""
        np.random.seed(self.config.synthetic_data_seed)
        
        bounds = polygon.bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        pois_data = []
        generated = 0
        attempts = 0
        max_attempts = count * 10
        
        while generated < count and attempts < max_attempts:
            # Random point within bounds
            lon = np.random.uniform(min_lon, max_lon)
            lat = np.random.uniform(min_lat, max_lat)
            point = Point(lon, lat)
            
            # Check if point is within polygon
            if polygon.contains(point):
                pois_data.append({
                    'poi_category': category,
                    'name': f"{category.title()} {generated + 1}",
                    'geometry': point
                })
                generated += 1
            
            attempts += 1
        
        return gpd.GeoDataFrame(pois_data, crs='EPSG:4326')
    
    def get_road_network(self, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Fetch road network data from OpenStreetMap.
        
        Args:
            boundary: Geographic boundary to query
            
        Returns:
            GeoDataFrame with road network
        """
        logger.info("Fetching road network from OSM")
        
        try:
            # Get boundary polygon
            query_polygon = boundary.geometry.unary_union if len(boundary) > 1 else boundary.geometry.iloc[0]
            
            # Fetch road network using OSMnx
            G = ox.graph_from_polygon(
                query_polygon,
                network_type=self.config.osm_network_type,
                simplify=True,
                retain_all=False
            )
            
            # Convert to GeoDataFrame
            edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
            edges_gdf.reset_index(inplace=True)
            
            # Simplify columns
            network_cols = ['u', 'v', 'key', 'highway', 'length', 'geometry']
            available_cols = [col for col in network_cols if col in edges_gdf.columns]
            edges_gdf = edges_gdf[available_cols]
            
            logger.info(f"Retrieved {len(edges_gdf)} road segments")
            return edges_gdf
            
        except Exception as e:
            logger.warning(f"Could not fetch road network: {e}")
            # Return empty GeoDataFrame with correct structure
            return gpd.GeoDataFrame(columns=['highway', 'length', 'geometry'], crs='EPSG:4326')


class SyntheticFinancialData:
    """
    Generator for realistic synthetic financial transaction data.
    Used for demonstration purposes in the absence of real financial data.
    """
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        np.random.seed(config.synthetic_data_seed)
    
    def generate_merchant_locations(self, boundary: gpd.GeoDataFrame, 
                                   n_merchants: int = 1000) -> gpd.GeoDataFrame:
        """
        Generate synthetic merchant locations with realistic spatial clustering.
        
        Args:
            boundary: Geographic boundary for merchant placement
            n_merchants: Number of merchants to generate
            
        Returns:
            GeoDataFrame with merchant locations and attributes
        """
        logger.info(f"Generating {n_merchants} synthetic merchant locations")
        
        # Get boundary polygon
        polygon = boundary.geometry.unary_union if len(boundary) > 1 else boundary.geometry.iloc[0]
        bounds = polygon.bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        merchants = []
        
        # Define merchant categories with relative frequencies
        categories = {
            'restaurant': 0.20,
            'retail': 0.25,
            'services': 0.15,
            'grocery': 0.10,
            'pharmacy': 0.08,
            'gas_station': 0.05,
            'other': 0.17
        }
        
        # Business size distribution
        size_distribution = {
            'micro': 0.60,    # < 10 employees
            'small': 0.30,    # 10-49 employees
            'medium': 0.08,   # 50-249 employees
            'large': 0.02     # 250+ employees
        }
        
        generated = 0
        attempts = 0
        max_attempts = n_merchants * 10
        
        while generated < n_merchants and attempts < max_attempts:
            # Generate location with some clustering (80% clustered, 20% random)
            if np.random.random() < 0.8:
                # Clustered around commercial centers
                center_lon = np.random.uniform(min_lon + 0.01, max_lon - 0.01)
                center_lat = np.random.uniform(min_lat + 0.01, max_lat - 0.01)
                
                # Small radius around center
                lon = np.random.normal(center_lon, 0.005)
                lat = np.random.normal(center_lat, 0.005)
            else:
                # Random location
                lon = np.random.uniform(min_lon, max_lon)
                lat = np.random.uniform(min_lat, max_lat)
            
            point = Point(lon, lat)
            
            # Check if within boundary
            if polygon.contains(point):
                # Select category and size
                category = np.random.choice(
                    list(categories.keys()),
                    p=list(categories.values())
                )
                
                size = np.random.choice(
                    list(size_distribution.keys()),
                    p=list(size_distribution.values())
                )
                
                # Generate business attributes based on size
                if size == 'micro':
                    monthly_ttv = np.random.lognormal(8.5, 0.8)  # ~5K BRL
                    employees = np.random.randint(1, 10)
                elif size == 'small':
                    monthly_ttv = np.random.lognormal(10.5, 0.6)  # ~40K BRL
                    employees = np.random.randint(10, 50)
                elif size == 'medium':
                    monthly_ttv = np.random.lognormal(12.0, 0.5)  # ~150K BRL
                    employees = np.random.randint(50, 250)
                else:  # large
                    monthly_ttv = np.random.lognormal(13.5, 0.4)  # ~700K BRL
                    employees = np.random.randint(250, 1000)
                
                # Business age (affects risk profile)
                years_operating = np.random.exponential(5)  # Average 5 years
                
                # Risk factors
                risk_score = self._calculate_merchant_risk(category, size, years_operating, monthly_ttv)
                
                merchants.append({
                    'merchant_id': f"M{generated + 1:06d}",
                    'category': category,
                    'size_category': size,
                    'employees': employees,
                    'monthly_ttv_brl': monthly_ttv,
                    'years_operating': years_operating,
                    'risk_score': risk_score,
                    'lat': lat,
                    'lon': lon,
                    'geometry': point
                })
                
                generated += 1
            
            attempts += 1
        
        merchants_gdf = gpd.GeoDataFrame(merchants, crs='EPSG:4326')
        logger.info(f"Generated {len(merchants_gdf)} synthetic merchants")
        
        return merchants_gdf
    
    def _calculate_merchant_risk(self, category: str, size: str, years_operating: float, 
                               monthly_ttv: float) -> float:
        """Calculate synthetic risk score based on business characteristics"""
        base_risk = 0.5
        
        # Category risk adjustments
        category_risk = {
            'restaurant': 0.1, 'retail': 0.0, 'services': -0.05,
            'grocery': -0.1, 'pharmacy': -0.15, 'gas_station': 0.05, 'other': 0.02
        }
        
        # Size risk (larger = lower risk)
        size_risk = {
            'micro': 0.2, 'small': 0.1, 'medium': -0.05, 'large': -0.15
        }
        
        # Operating history (longer = lower risk)
        history_factor = max(-0.2, -0.05 * years_operating)
        
        # TTV factor (higher volume = lower risk, but with diminishing returns)
        ttv_factor = -0.1 * np.log(monthly_ttv / 1000) / 10
        
        risk = base_risk + category_risk.get(category, 0) + size_risk.get(size, 0) + \
               history_factor + ttv_factor
        
        # Add some noise and bound between 0 and 1
        risk += np.random.normal(0, 0.1)
        return max(0.0, min(1.0, risk))


class DataPipeline:
    """
    Main data pipeline orchestrator that coordinates data ingestion from all sources.
    """
    
    def __init__(self, config: DataSourceConfig = None):
        """Initialize the data pipeline with all processors"""
        self.config = config or DataSourceConfig()
        
        # Initialize processors
        self.ibge_processor = IBGEDataProcessor(self.config)
        self.osm_processor = OSMDataProcessor(self.config)
        self.synthetic_processor = SyntheticFinancialData(self.config)
        
        # Create output directories
        Path(self.config.processed_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Data pipeline initialized")
    
    def run_full_pipeline(self, boundary: gpd.GeoDataFrame = None) -> Dict[str, gpd.GeoDataFrame]:
        """
        Run the complete data ingestion pipeline.
        
        Args:
            boundary: Optional boundary for data collection
            
        Returns:
            Dictionary of processed datasets
        """
        logger.info("Starting full data pipeline execution")
        
        # If no boundary provided, use IBGE municipal boundaries
        if boundary is None:
            boundary = self.ibge_processor.get_municipal_boundaries()
        
        datasets = {}
        
        # 1. Census and demographic data
        logger.info("Processing IBGE census data...")
        datasets['census_tracts'] = self.ibge_processor.get_census_data_by_tract()
        
        # 2. Municipal boundaries
        datasets['municipalities'] = self.ibge_processor.get_municipal_boundaries()
        
        # 3. Points of Interest
        logger.info("Processing OpenStreetMap POI data...")
        datasets['pois'] = self.osm_processor.get_pois_by_category(boundary)
        
        # 4. Road network
        logger.info("Processing road network data...")
        datasets['road_network'] = self.osm_processor.get_road_network(boundary)
        
        # 5. Synthetic merchant data
        logger.info("Generating synthetic financial data...")
        datasets['merchants'] = self.synthetic_processor.generate_merchant_locations(boundary)
        
        # Save processed datasets
        self._save_datasets(datasets)
        
        logger.info("Data pipeline execution completed")
        return datasets
    
    def _save_datasets(self, datasets: Dict[str, gpd.GeoDataFrame]):
        """Save processed datasets to files"""
        output_dir = Path(self.config.processed_dir)
        
        for name, gdf in datasets.items():
            if not gdf.empty:
                output_path = output_dir / f"{name}.geojson"
                gdf.to_file(output_path, driver='GeoJSON')
                logger.info(f"Saved {name}: {len(gdf)} records to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Initializing Geo-Financial Data Pipeline...")
    
    # Create pipeline
    pipeline = DataPipeline()
    
    # Run full pipeline
    datasets = pipeline.run_full_pipeline()
    
    # Print summary
    print("\nData Pipeline Results:")
    for name, gdf in datasets.items():
        print(f"- {name.title().replace('_', ' ')}: {len(gdf):,} records")