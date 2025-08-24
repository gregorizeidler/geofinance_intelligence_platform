"""
Comprehensive Spatial Feature Engineering System
===============================================

This module creates the core spatial intelligence layer by generating dozens of geospatial
features for each hexagon in the grid. These features capture the socioeconomic,
commercial, infrastructural, and financial characteristics of geographic areas.

Author: Geo-Financial Intelligence Platform
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
from shapely.geometry import Point
from geopy.distance import geodesic
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import warnings

# Local imports
from .hexgrid import HexagonalGrid
from ..data_pipeline.data_sources import DataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering parameters"""
    neighbor_radius: int = 2  # k-ring radius for neighborhood analysis
    poi_categories: List[str] = None
    standardize_features: bool = True
    min_merchants_for_competition: int = 3
    
    def __post_init__(self):
        if self.poi_categories is None:
            self.poi_categories = [
                'restaurant', 'cafe', 'bank', 'atm', 'pharmacy', 'hospital',
                'school', 'university', 'shopping_centre', 'supermarket', 'fuel'
            ]


class SpatialFeatureEngine:
    """
    Advanced spatial feature engineering system that creates comprehensive
    geospatial intelligence features for financial applications.
    """
    
    def __init__(self, hex_grid: HexagonalGrid, config: FeatureConfig = None):
        """
        Initialize the spatial feature engine.
        
        Args:
            hex_grid: Hexagonal grid system
            config: Feature engineering configuration
        """
        self.hex_grid = hex_grid
        self.config = config or FeatureConfig()
        self.feature_datasets = {}
        self.engineered_features = None
        
        logger.info("Spatial Feature Engine initialized")
    
    def load_datasets(self, datasets: Dict[str, gpd.GeoDataFrame]):
        """Load all required datasets for feature engineering"""
        self.feature_datasets = datasets
        logger.info(f"Loaded {len(datasets)} datasets for feature engineering")
    
    def generate_all_features(self) -> gpd.GeoDataFrame:
        """
        Generate comprehensive spatial features for all hexagons.
        
        Returns:
            GeoDataFrame with hexagons and all engineered features
        """
        logger.info("Starting comprehensive feature generation")
        
        if self.hex_grid.grid_gdf is None:
            raise ValueError("Hexagonal grid not generated. Call hex_grid.generate_hexagonal_grid() first")
        
        # Start with base hexagonal grid
        features_gdf = self.hex_grid.grid_gdf.copy()
        
        # Generate different categories of features
        socioeconomic_features = self._generate_socioeconomic_features()
        commercial_features = self._generate_commercial_ecosystem_features()
        infrastructure_features = self._generate_infrastructure_features()
        financial_features = self._generate_financial_activity_features()
        spatial_features = self._generate_spatial_relationship_features()
        
        # Merge all feature sets
        feature_sets = [
            socioeconomic_features,
            commercial_features,
            infrastructure_features,
            financial_features,
            spatial_features
        ]
        
        for feature_set in feature_sets:
            if feature_set is not None and not feature_set.empty:
                features_gdf = features_gdf.merge(
                    feature_set.drop(columns=['geometry'], errors='ignore'),
                    on='hex_id',
                    how='left'
                )
        
        # Handle missing values
        features_gdf = self._handle_missing_values(features_gdf)
        
        # Standardize features if requested
        if self.config.standardize_features:
            features_gdf = self._standardize_features(features_gdf)
        
        self.engineered_features = features_gdf
        logger.info(f"Feature generation completed: {len(features_gdf)} hexagons with {len(features_gdf.columns)} features")
        
        return features_gdf
    
    def _generate_socioeconomic_features(self) -> gpd.GeoDataFrame:
        """Generate socioeconomic features from census data"""
        logger.info("Generating socioeconomic features")
        
        if 'census_tracts' not in self.feature_datasets:
            logger.warning("No census data available, skipping socioeconomic features")
            return gpd.GeoDataFrame()
        
        census_data = self.feature_datasets['census_tracts']
        hex_grid = self.hex_grid.grid_gdf
        
        # Spatial join census tracts with hexagons
        hex_census = gpd.sjoin(hex_grid, census_data, how='left', op='intersects')
        
        # Aggregate census data by hexagon (area-weighted average)
        socioeconomic_features = hex_census.groupby('hex_id').agg({
            'population_total': 'sum',
            'households': 'sum',
            'avg_income_brl': 'mean',
            'education_higher_pct': 'mean',
            'age_0_14_pct': 'mean',
            'age_15_64_pct': 'mean',
            'age_65_plus_pct': 'mean',
            'unemployment_rate': 'mean',
            'density_pop_km2': 'mean'
        }).reset_index()
        
        # Calculate derived features
        socioeconomic_features['income_diversity_index'] = self._calculate_income_diversity(hex_census)
        socioeconomic_features['age_diversity_index'] = self._calculate_age_diversity(socioeconomic_features)
        socioeconomic_features['economic_vulnerability_score'] = self._calculate_economic_vulnerability(socioeconomic_features)
        
        logger.info(f"Generated socioeconomic features for {len(socioeconomic_features)} hexagons")
        return socioeconomic_features
    
    def _generate_commercial_ecosystem_features(self) -> gpd.GeoDataFrame:
        """Generate commercial ecosystem features from POI and merchant data"""
        logger.info("Generating commercial ecosystem features")
        
        hex_grid = self.hex_grid.grid_gdf
        commercial_features_list = []
        
        # POI-based features
        if 'pois' in self.feature_datasets:
            poi_features = self._analyze_poi_ecosystem(self.feature_datasets['pois'])
            commercial_features_list.append(poi_features)
        
        # Merchant-based features
        if 'merchants' in self.feature_datasets:
            merchant_features = self._analyze_merchant_ecosystem(self.feature_datasets['merchants'])
            commercial_features_list.append(merchant_features)
        
        # Combine all commercial features
        if commercial_features_list:
            commercial_features = commercial_features_list[0]
            for additional_features in commercial_features_list[1:]:
                commercial_features = commercial_features.merge(
                    additional_features, on='hex_id', how='outer'
                )
        else:
            commercial_features = pd.DataFrame({'hex_id': hex_grid['hex_id']})
        
        logger.info(f"Generated commercial features for {len(commercial_features)} hexagons")
        return commercial_features
    
    def _analyze_poi_ecosystem(self, pois_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Analyze Points of Interest ecosystem"""
        hex_grid = self.hex_grid.grid_gdf
        
        # Spatial join POIs with hexagons
        pois_hex = gpd.sjoin(pois_gdf, hex_grid, how='left', op='within')
        
        # Count POIs by category and hexagon
        poi_counts = pois_hex.groupby(['hex_id', 'poi_category']).size().unstack(fill_value=0)
        poi_counts = poi_counts.add_prefix('poi_count_')
        
        # Total POI metrics
        poi_features = poi_counts.copy()
        poi_features['poi_total_count'] = poi_counts.sum(axis=1)
        poi_features['poi_diversity_index'] = self._calculate_diversity_index(poi_counts)
        
        # Commercial density and accessibility
        poi_features['commercial_density'] = poi_features['poi_total_count'] / hex_grid.set_index('hex_id')['area_km2']
        
        # Service accessibility score
        essential_services = ['bank', 'atm', 'pharmacy', 'hospital', 'supermarket']
        available_services = [f'poi_count_{service}' for service in essential_services if f'poi_count_{service}' in poi_features.columns]
        poi_features['essential_services_score'] = poi_features[available_services].sum(axis=1)
        
        return poi_features.reset_index()
    
    def _analyze_merchant_ecosystem(self, merchants_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Analyze merchant ecosystem and business environment"""
        hex_grid = self.hex_grid.grid_gdf
        
        # Spatial join merchants with hexagons
        merchants_hex = gpd.sjoin(merchants_gdf, hex_grid, how='left', op='within')
        
        # Aggregate merchant data by hexagon
        merchant_features = merchants_hex.groupby('hex_id').agg({
            'merchant_id': 'count',
            'monthly_ttv_brl': ['sum', 'mean', 'std'],
            'employees': 'sum',
            'years_operating': 'mean',
            'risk_score': 'mean'
        }).round(2)
        
        # Flatten column names
        merchant_features.columns = [
            'merchant_count', 'total_ttv_brl', 'avg_ttv_brl', 'ttv_volatility',
            'total_employees', 'avg_business_age', 'avg_risk_score'
        ]
        
        # Calculate derived metrics
        merchant_features['ttv_per_merchant'] = merchant_features['total_ttv_brl'] / merchant_features['merchant_count']
        merchant_features['employees_per_merchant'] = merchant_features['total_employees'] / merchant_features['merchant_count']
        merchant_features['business_maturity_score'] = np.minimum(merchant_features['avg_business_age'] / 10, 1.0)
        
        # Competition analysis
        category_counts = merchants_hex.groupby(['hex_id', 'category']).size().unstack(fill_value=0)
        merchant_features['category_diversity'] = self._calculate_diversity_index(category_counts)
        merchant_features['max_category_concentration'] = (category_counts.max(axis=1) / category_counts.sum(axis=1)).fillna(0)
        
        # Market opportunity score (inverse of competition for certain categories)
        high_demand_categories = ['restaurant', 'retail', 'services']
        for category in high_demand_categories:
            if category in category_counts.columns:
                col_name = f'{category}_opportunity_score'
                # Higher score for areas with demand but low competition
                merchant_features[col_name] = 1 / (1 + category_counts[category])
        
        return merchant_features.reset_index()
    
    def _generate_infrastructure_features(self) -> gpd.GeoDataFrame:
        """Generate infrastructure and accessibility features"""
        logger.info("Generating infrastructure features")
        
        hex_grid = self.hex_grid.grid_gdf
        infrastructure_features = pd.DataFrame({'hex_id': hex_grid['hex_id']})
        
        # Road network analysis
        if 'road_network' in self.feature_datasets:
            road_features = self._analyze_road_network(self.feature_datasets['road_network'])
            infrastructure_features = infrastructure_features.merge(road_features, on='hex_id', how='left')
        
        # Transportation accessibility
        transport_features = self._calculate_transport_accessibility()
        infrastructure_features = infrastructure_features.merge(transport_features, on='hex_id', how='left')
        
        # Urban centrality measures
        centrality_features = self._calculate_urban_centrality()
        infrastructure_features = infrastructure_features.merge(centrality_features, on='hex_id', how='left')
        
        logger.info(f"Generated infrastructure features for {len(infrastructure_features)} hexagons")
        return infrastructure_features
    
    def _analyze_road_network(self, road_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Analyze road network connectivity and accessibility"""
        hex_grid = self.hex_grid.grid_gdf
        
        if road_gdf.empty:
            # Return empty features if no road data
            return pd.DataFrame({
                'hex_id': hex_grid['hex_id'],
                'road_density_km': 0,
                'highway_access_score': 0,
                'connectivity_index': 0
            })
        
        # Spatial join roads with hexagons
        roads_hex = gpd.sjoin(road_gdf, hex_grid, how='left', op='intersects')
        
        # Calculate road density
        road_features = roads_hex.groupby('hex_id').agg({
            'length': 'sum',
            'highway': lambda x: (x == 'primary').sum()  # Count major roads
        }).reset_index()
        
        road_features.columns = ['hex_id', 'total_road_length', 'major_road_count']
        
        # Calculate density (km of roads per km²)
        hex_areas = hex_grid.set_index('hex_id')['area_km2']
        road_features['road_density_km'] = road_features['total_road_length'] / 1000 / hex_areas[road_features['hex_id']].values
        
        # Highway access score
        road_features['highway_access_score'] = np.minimum(road_features['major_road_count'] / 5, 1.0)
        
        # Connectivity index (simplified)
        road_features['connectivity_index'] = (
            road_features['road_density_km'] * 0.7 + 
            road_features['highway_access_score'] * 0.3
        )
        
        return road_features[['hex_id', 'road_density_km', 'highway_access_score', 'connectivity_index']]
    
    def _calculate_transport_accessibility(self) -> pd.DataFrame:
        """Calculate public transportation accessibility scores"""
        hex_grid = self.hex_grid.grid_gdf
        
        # For demonstration, create synthetic transport accessibility scores
        # In real implementation, this would use GTFS data or transit station locations
        
        # Create some synthetic "metro stations" and "bus stops"
        np.random.seed(42)
        transport_scores = []
        
        for _, hex_row in hex_grid.iterrows():
            hex_center = Point(hex_row['centroid_lon'], hex_row['centroid_lat'])
            
            # Simulate metro accessibility (distance to nearest metro)
            # Porto Alegre has a small metro system, so most areas have low scores
            metro_score = np.random.beta(1, 5)  # Skewed toward low values
            
            # Bus accessibility (more comprehensive)
            bus_score = np.random.beta(3, 2)  # Skewed toward higher values
            
            # Walking accessibility to commercial centers
            commercial_walkability = np.random.beta(2, 3)
            
            transport_scores.append({
                'hex_id': hex_row['hex_id'],
                'metro_accessibility_score': metro_score,
                'bus_accessibility_score': bus_score,
                'walkability_score': commercial_walkability,
                'overall_transport_score': (metro_score * 0.3 + bus_score * 0.5 + commercial_walkability * 0.2)
            })
        
        return pd.DataFrame(transport_scores)
    
    def _calculate_urban_centrality(self) -> pd.DataFrame:
        """Calculate urban centrality and spatial position metrics"""
        hex_grid = self.hex_grid.grid_gdf
        
        # Calculate distance from city center (downtown Porto Alegre)
        city_center = Point(-51.2300, -30.0331)
        
        centrality_features = []
        
        for _, hex_row in hex_grid.iterrows():
            hex_center = Point(hex_row['centroid_lon'], hex_row['centroid_lat'])
            
            # Distance to city center
            distance_to_center = geodesic(
                (city_center.y, city_center.x),
                (hex_center.y, hex_center.x)
            ).kilometers
            
            # Centrality score (inverse of distance, normalized)
            centrality_score = 1 / (1 + distance_to_center / 20)  # 20km normalization
            
            # Edge vs. center classification
            is_urban_core = distance_to_center < 5  # Within 5km of center
            is_suburban = 5 <= distance_to_center < 15
            is_peripheral = distance_to_center >= 15
            
            centrality_features.append({
                'hex_id': hex_row['hex_id'],
                'distance_to_center_km': distance_to_center,
                'centrality_score': centrality_score,
                'is_urban_core': int(is_urban_core),
                'is_suburban': int(is_suburban),
                'is_peripheral': int(is_peripheral)
            })
        
        return pd.DataFrame(centrality_features)
    
    def _generate_financial_activity_features(self) -> pd.DataFrame:
        """Generate financial activity and transaction-based features"""
        logger.info("Generating financial activity features")
        
        hex_grid = self.hex_grid.grid_gdf
        
        # Financial activity features from merchant data
        financial_features = pd.DataFrame({'hex_id': hex_grid['hex_id']})
        
        if 'merchants' in self.feature_datasets:
            merchants_gdf = self.feature_datasets['merchants']
            
            # Spatial join merchants with hexagons
            merchants_hex = gpd.sjoin(merchants_gdf, hex_grid, how='left', op='within')
            
            # Transaction volume analysis
            ttv_features = merchants_hex.groupby('hex_id').agg({
                'monthly_ttv_brl': ['sum', 'mean', 'count'],
                'risk_score': ['mean', 'std']
            }).round(2)
            
            ttv_features.columns = [
                'total_monthly_ttv', 'avg_monthly_ttv', 'active_merchants',
                'avg_risk_score', 'risk_volatility'
            ]
            
            # Calculate payment ecosystem health
            ttv_features['payment_ecosystem_score'] = self._calculate_payment_ecosystem_score(ttv_features)
            
            # Market penetration potential
            ttv_features['market_penetration_score'] = self._calculate_market_penetration(ttv_features)
            
            financial_features = financial_features.merge(ttv_features.reset_index(), on='hex_id', how='left')
        
        # Add synthetic financial behavior indicators
        financial_features = self._add_synthetic_financial_indicators(financial_features)
        
        logger.info(f"Generated financial features for {len(financial_features)} hexagons")
        return financial_features
    
    def _generate_spatial_relationship_features(self) -> pd.DataFrame:
        """Generate features based on spatial relationships and neighborhood effects"""
        logger.info("Generating spatial relationship features")
        
        hex_grid = self.hex_grid.grid_gdf
        spatial_features = pd.DataFrame({'hex_id': hex_grid['hex_id']})
        
        # Neighborhood influence features
        neighborhood_features = self._calculate_neighborhood_effects()
        spatial_features = spatial_features.merge(neighborhood_features, on='hex_id', how='left')
        
        # Spatial clustering and hotspots
        clustering_features = self._identify_spatial_clusters()
        spatial_features = spatial_features.merge(clustering_features, on='hex_id', how='left')
        
        logger.info(f"Generated spatial relationship features for {len(spatial_features)} hexagons")
        return spatial_features
    
    def _calculate_neighborhood_effects(self) -> pd.DataFrame:
        """Calculate neighborhood influence and spillover effects"""
        hex_grid = self.hex_grid.grid_gdf
        
        neighborhood_features = []
        
        for _, hex_row in hex_grid.iterrows():
            hex_id = hex_row['hex_id']
            
            # Get k-ring neighbors
            neighbors = self.hex_grid.get_neighbors(hex_id, k=self.config.neighbor_radius)
            neighbors.remove(hex_id)  # Remove self
            
            # Calculate neighborhood averages (if we have existing features)
            neighborhood_data = {
                'hex_id': hex_id,
                'neighbor_count': len(neighbors)
            }
            
            # For demonstration, add some synthetic neighborhood effects
            if neighbors:
                # Simulate neighborhood economic spillover
                neighbor_income_effect = np.random.normal(0, 0.1)  # Small spillover effect
                neighbor_commercial_effect = np.random.normal(0, 0.05)
                
                neighborhood_data.update({
                    'neighborhood_income_spillover': neighbor_income_effect,
                    'neighborhood_commercial_spillover': neighbor_commercial_effect,
                    'spatial_isolation_score': 1 / (1 + len(neighbors))  # Lower = less isolated
                })
            else:
                neighborhood_data.update({
                    'neighborhood_income_spillover': 0,
                    'neighborhood_commercial_spillover': 0,
                    'spatial_isolation_score': 1
                })
            
            neighborhood_features.append(neighborhood_data)
        
        return pd.DataFrame(neighborhood_features)
    
    def _identify_spatial_clusters(self) -> pd.DataFrame:
        """Identify spatial clusters and business hotspots"""
        hex_grid = self.hex_grid.grid_gdf
        
        # For demonstration, create synthetic cluster assignments
        np.random.seed(42)
        
        # Use spatial coordinates for clustering
        coordinates = np.column_stack([
            hex_grid['centroid_lat'].values,
            hex_grid['centroid_lon'].values
        ])
        
        # K-means clustering to identify spatial regions
        n_clusters = min(8, len(hex_grid) // 10)  # Adaptive cluster count
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)
        else:
            cluster_labels = np.zeros(len(hex_grid))
        
        clustering_features = pd.DataFrame({
            'hex_id': hex_grid['hex_id'],
            'spatial_cluster_id': cluster_labels,
            'cluster_centrality': np.random.random(len(hex_grid))  # Distance to cluster center
        })
        
        # Business hotspot identification (synthetic)
        hotspot_scores = np.random.beta(2, 5, len(hex_grid))  # Skewed toward low values
        clustering_features['business_hotspot_score'] = hotspot_scores
        clustering_features['is_business_hotspot'] = (hotspot_scores > np.percentile(hotspot_scores, 80)).astype(int)
        
        return clustering_features
    
    # Helper methods for feature calculations
    
    def _calculate_diversity_index(self, category_counts: pd.DataFrame) -> pd.Series:
        """Calculate Shannon diversity index for categories"""
        # Avoid log(0) by adding small constant
        props = category_counts.div(category_counts.sum(axis=1) + 1e-10, axis=0)
        props = props.replace(0, 1e-10)  # Replace exact zeros
        
        diversity = -1 * (props * np.log(props)).sum(axis=1)
        return diversity.fillna(0)
    
    def _calculate_income_diversity(self, hex_census: gpd.GeoDataFrame) -> pd.Series:
        """Calculate income diversity within each hexagon"""
        # Placeholder calculation - in real implementation, would use income brackets
        income_diversity = hex_census.groupby('hex_id')['avg_income_brl'].std().fillna(0)
        return income_diversity / income_diversity.max()  # Normalize
    
    def _calculate_age_diversity(self, socioeconomic_features: pd.DataFrame) -> pd.Series:
        """Calculate age group diversity"""
        age_cols = ['age_0_14_pct', 'age_15_64_pct', 'age_65_plus_pct']
        age_data = socioeconomic_features[age_cols].div(100)  # Convert to proportions
        
        # Shannon diversity for age groups
        age_data = age_data.replace(0, 1e-10)  # Avoid log(0)
        diversity = -1 * (age_data * np.log(age_data)).sum(axis=1)
        return diversity.fillna(0)
    
    def _calculate_economic_vulnerability(self, socioeconomic_features: pd.DataFrame) -> pd.Series:
        """Calculate economic vulnerability score"""
        # Composite score based on income, education, and unemployment
        income_score = 1 - (socioeconomic_features['avg_income_brl'] / socioeconomic_features['avg_income_brl'].max())
        education_score = 1 - (socioeconomic_features['education_higher_pct'] / 100)
        unemployment_score = socioeconomic_features['unemployment_rate'] / 100
        
        vulnerability = (income_score * 0.4 + education_score * 0.3 + unemployment_score * 0.3)
        return vulnerability.fillna(0.5)  # Default to medium vulnerability
    
    def _calculate_payment_ecosystem_score(self, ttv_features: pd.DataFrame) -> pd.Series:
        """Calculate payment ecosystem health score"""
        # Higher TTV and more merchants = better ecosystem
        ttv_norm = ttv_features['total_monthly_ttv'] / ttv_features['total_monthly_ttv'].max()
        merchant_norm = ttv_features['active_merchants'] / ttv_features['active_merchants'].max()
        risk_norm = 1 - ttv_features['avg_risk_score']  # Lower risk = better
        
        ecosystem_score = (ttv_norm * 0.4 + merchant_norm * 0.4 + risk_norm * 0.2)
        return ecosystem_score.fillna(0)
    
    def _calculate_market_penetration(self, ttv_features: pd.DataFrame) -> pd.Series:
        """Calculate market penetration potential score"""
        # Areas with moderate activity but room for growth
        ttv_norm = ttv_features['total_monthly_ttv'] / ttv_features['total_monthly_ttv'].max()
        
        # Sweet spot: moderate current activity with growth potential
        penetration_score = 4 * ttv_norm * (1 - ttv_norm)  # Inverted U-curve
        return penetration_score.fillna(0)
    
    def _add_synthetic_financial_indicators(self, financial_features: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic financial behavior indicators"""
        np.random.seed(42)
        n_hexes = len(financial_features)
        
        # Digital payment adoption rate
        financial_features['digital_payment_adoption'] = np.random.beta(3, 2, n_hexes)
        
        # Credit demand intensity
        financial_features['credit_demand_score'] = np.random.gamma(2, 0.3, n_hexes)
        
        # Financial inclusion index
        financial_features['financial_inclusion_index'] = np.random.beta(2, 1, n_hexes)
        
        # Payment method diversity
        financial_features['payment_method_diversity'] = np.random.beta(2, 3, n_hexes)
        
        return financial_features
    
    def _handle_missing_values(self, features_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Handle missing values in feature dataset"""
        # Get numeric columns only
        numeric_cols = features_gdf.select_dtypes(include=[np.number]).columns
        
        # Fill with appropriate values
        for col in numeric_cols:
            if col.endswith('_count') or col.endswith('_score') or col.endswith('_index'):
                features_gdf[col] = features_gdf[col].fillna(0)
            elif col.endswith('_pct') or col.endswith('_rate'):
                features_gdf[col] = features_gdf[col].fillna(features_gdf[col].median())
            else:
                features_gdf[col] = features_gdf[col].fillna(features_gdf[col].mean())
        
        return features_gdf
    
    def _standardize_features(self, features_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardize numeric features for machine learning"""
        # Identify feature columns (exclude ID and geometry columns)
        exclude_cols = ['hex_id', 'area_km2', 'centroid_lat', 'centroid_lon', 
                       'resolution', 'grid_version', 'created_at', 'geometry']
        feature_cols = [col for col in features_gdf.columns if col not in exclude_cols]
        
        # Only standardize numeric columns
        numeric_feature_cols = features_gdf[feature_cols].select_dtypes(include=[np.number]).columns
        
        if len(numeric_feature_cols) > 0:
            scaler = StandardScaler()
            features_gdf[numeric_feature_cols] = scaler.fit_transform(features_gdf[numeric_feature_cols])
            
            logger.info(f"Standardized {len(numeric_feature_cols)} numeric features")
        
        return features_gdf
    
    def get_feature_summary(self) -> Dict:
        """Get summary statistics of generated features"""
        if self.engineered_features is None:
            return {"error": "Features not yet generated"}
        
        # Identify feature columns
        exclude_cols = ['hex_id', 'area_km2', 'centroid_lat', 'centroid_lon', 
                       'resolution', 'grid_version', 'created_at', 'geometry']
        feature_cols = [col for col in self.engineered_features.columns if col not in exclude_cols]
        
        numeric_features = self.engineered_features[feature_cols].select_dtypes(include=[np.number])
        
        return {
            "total_hexagons": len(self.engineered_features),
            "total_features": len(feature_cols),
            "numeric_features": len(numeric_features.columns),
            "feature_categories": {
                "socioeconomic": len([c for c in feature_cols if any(x in c for x in ['income', 'age', 'education', 'population'])]),
                "commercial": len([c for c in feature_cols if any(x in c for x in ['poi', 'merchant', 'commercial', 'business'])]),
                "infrastructure": len([c for c in feature_cols if any(x in c for x in ['road', 'transport', 'accessibility'])]),
                "financial": len([c for c in feature_cols if any(x in c for x in ['ttv', 'risk', 'payment', 'financial'])]),
                "spatial": len([c for c in feature_cols if any(x in c for x in ['cluster', 'neighbor', 'hotspot', 'centrality'])])
            },
            "data_quality": {
                "completeness": (1 - numeric_features.isnull().sum().sum() / (len(numeric_features) * len(numeric_features.columns))) * 100,
                "feature_ranges": numeric_features.describe().to_dict()
            }
        }


def create_comprehensive_features(hex_grid: HexagonalGrid, datasets: Dict[str, gpd.GeoDataFrame],
                                config: FeatureConfig = None) -> gpd.GeoDataFrame:
    """
    Convenience function to create comprehensive spatial features.
    
    Args:
        hex_grid: Hexagonal grid system
        datasets: Dictionary of processed datasets
        config: Feature engineering configuration
        
    Returns:
        GeoDataFrame with comprehensive spatial features
    """
    feature_engine = SpatialFeatureEngine(hex_grid, config)
    feature_engine.load_datasets(datasets)
    return feature_engine.generate_all_features()


if __name__ == "__main__":
    # Example usage
    print("Testing Spatial Feature Engineering System...")
    
    from ..data_pipeline.data_sources import DataPipeline
    from .hexgrid import create_porto_alegre_grid
    
    # Create components
    hex_grid = create_porto_alegre_grid(resolution=9)
    data_pipeline = DataPipeline()
    
    # Run pipeline
    datasets = data_pipeline.run_full_pipeline()
    
    # Generate features
    features = create_comprehensive_features(hex_grid, datasets)
    
    # Print summary
    feature_engine = SpatialFeatureEngine(hex_grid)
    feature_engine.engineered_features = features
    summary = feature_engine.get_feature_summary()
    
    print(f"Feature Engineering Results:")
    print(f"- Total hexagons: {summary['total_hexagons']:,}")
    print(f"- Total features: {summary['total_features']}")
    print(f"- Data completeness: {summary['data_quality']['completeness']:.1f}%")
    print(f"- Feature categories: {summary['feature_categories']}")