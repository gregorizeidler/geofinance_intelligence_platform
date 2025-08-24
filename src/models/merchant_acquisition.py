"""
Intelligent Merchant Acquisition & Opportunity Scoring System
===========================================================

This module implements an advanced merchant acquisition strategy system that identifies
optimal locations and opportunities for business expansion and payment service adoption.
It combines market analysis, competition intelligence, and spatial optimization.

Author: Geo-Financial Intelligence Platform
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
import lightgbm as lgb

# Optimization
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Local imports
from ..feature_engineering.spatial_features import SpatialFeatureEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class AcquisitionConfig:
    """Configuration for merchant acquisition modeling"""
    target_categories: List[str] = None
    min_opportunity_score: float = 0.3
    max_competition_density: float = 0.8
    roi_threshold: float = 2.0  # Minimum ROI multiplier
    market_penetration_weight: float = 0.3
    competition_weight: float = 0.25
    accessibility_weight: float = 0.2
    economic_potential_weight: float = 0.25
    
    def __post_init__(self):
        if self.target_categories is None:
            self.target_categories = ['restaurant', 'retail', 'services', 'grocery']


class MarketOpportunityAnalyzer:
    """
    Analyzes market opportunities by identifying demand-supply gaps,
    underserved areas, and optimal expansion locations.
    """
    
    def __init__(self, config: AcquisitionConfig = None):
        self.config = config or AcquisitionConfig()
        self.opportunity_scores = None
        self.market_clusters = None
        self.expansion_recommendations = None
        
        logger.info("Market Opportunity Analyzer initialized")
    
    def analyze_market_opportunities(self, features_gdf: gpd.GeoDataFrame,
                                   merchants_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """
        Comprehensive market opportunity analysis for each hexagon.
        
        Args:
            features_gdf: GeoDataFrame with spatial features
            merchants_gdf: Optional merchant data for competition analysis
            
        Returns:
            GeoDataFrame with opportunity scores and recommendations
        """
        logger.info("Starting comprehensive market opportunity analysis")
        
        # Start with spatial features
        opportunity_gdf = features_gdf.copy()
        
        # Calculate market potential components
        opportunity_gdf = self._calculate_demand_indicators(opportunity_gdf)
        opportunity_gdf = self._calculate_supply_competition(opportunity_gdf, merchants_gdf)
        opportunity_gdf = self._calculate_accessibility_factors(opportunity_gdf)
        opportunity_gdf = self._calculate_economic_potential(opportunity_gdf)
        
        # Composite opportunity scoring
        opportunity_gdf = self._calculate_composite_opportunity_score(opportunity_gdf)
        
        # Market segmentation and clustering
        opportunity_gdf = self._perform_market_segmentation(opportunity_gdf)
        
        # Generate specific recommendations
        opportunity_gdf = self._generate_expansion_recommendations(opportunity_gdf)
        
        self.opportunity_scores = opportunity_gdf
        logger.info(f"Market analysis completed for {len(opportunity_gdf)} locations")
        
        return opportunity_gdf
    
    def _calculate_demand_indicators(self, features_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate demand-side indicators for each location"""
        logger.info("Calculating demand indicators")
        
        demand_gdf = features_gdf.copy()
        
        # Population-based demand
        if 'population_total' in demand_gdf.columns:
            demand_gdf['population_demand'] = demand_gdf['population_total'] / demand_gdf['population_total'].max()
        else:
            demand_gdf['population_demand'] = 0.5  # Default medium demand
        
        # Income-based purchasing power
        if 'avg_income_brl' in demand_gdf.columns:
            demand_gdf['purchasing_power'] = demand_gdf['avg_income_brl'] / demand_gdf['avg_income_brl'].max()
        else:
            demand_gdf['purchasing_power'] = 0.5
        
        # Age demographics demand (working age preference)
        if 'age_15_64_pct' in demand_gdf.columns:
            demand_gdf['demographic_demand'] = demand_gdf['age_15_64_pct'] / 100
        else:
            demand_gdf['demographic_demand'] = 0.65  # Average working age percentage
        
        # Employment-based demand (inverse unemployment)
        if 'unemployment_rate' in demand_gdf.columns:
            demand_gdf['employment_demand'] = 1 - (demand_gdf['unemployment_rate'] / 100)
        else:
            demand_gdf['employment_demand'] = 0.9
        
        # Education-based demand (higher education correlates with service demand)
        if 'education_higher_pct' in demand_gdf.columns:
            demand_gdf['education_demand'] = demand_gdf['education_higher_pct'] / 100
        else:
            demand_gdf['education_demand'] = 0.3
        
        # Composite demand score
        demand_gdf['total_demand_score'] = (
            demand_gdf['population_demand'] * 0.3 +
            demand_gdf['purchasing_power'] * 0.25 +
            demand_gdf['demographic_demand'] * 0.2 +
            demand_gdf['employment_demand'] * 0.15 +
            demand_gdf['education_demand'] * 0.1
        )
        
        return demand_gdf
    
    def _calculate_supply_competition(self, features_gdf: gpd.GeoDataFrame,
                                    merchants_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """Calculate supply-side competition and market saturation"""
        logger.info("Calculating supply and competition metrics")
        
        supply_gdf = features_gdf.copy()
        
        # Existing merchant density (competition)
        if 'merchant_count' in supply_gdf.columns:
            max_merchants = supply_gdf['merchant_count'].max()
            supply_gdf['merchant_density'] = supply_gdf['merchant_count'] / max_merchants if max_merchants > 0 else 0
        else:
            supply_gdf['merchant_density'] = np.random.beta(2, 5, len(supply_gdf))  # Synthetic competition
        
        # POI-based service density
        poi_service_cols = [col for col in supply_gdf.columns if col.startswith('poi_count_')]
        if poi_service_cols:
            supply_gdf['poi_service_density'] = supply_gdf[poi_service_cols].sum(axis=1)
            supply_gdf['poi_service_density'] = supply_gdf['poi_service_density'] / supply_gdf['poi_service_density'].max()
        else:
            supply_gdf['poi_service_density'] = np.random.beta(2, 3, len(supply_gdf))
        
        # Category-specific competition
        for category in self.config.target_categories:
            col_name = f'poi_count_{category}'
            competition_col = f'{category}_competition'
            
            if col_name in supply_gdf.columns:
                supply_gdf[competition_col] = supply_gdf[col_name] / (supply_gdf[col_name].max() + 1)
            else:
                # Synthetic competition based on category
                if category == 'restaurant':
                    supply_gdf[competition_col] = np.random.beta(3, 2, len(supply_gdf))  # High competition
                elif category == 'retail':
                    supply_gdf[competition_col] = np.random.beta(2, 3, len(supply_gdf))  # Medium competition
                else:
                    supply_gdf[competition_col] = np.random.beta(1, 4, len(supply_gdf))  # Lower competition
        
        # Market saturation index (inverse opportunity)
        supply_gdf['market_saturation'] = (
            supply_gdf['merchant_density'] * 0.6 +
            supply_gdf['poi_service_density'] * 0.4
        )
        
        # Competition intensity score
        competition_cols = [col for col in supply_gdf.columns if col.endswith('_competition')]
        if competition_cols:
            supply_gdf['competition_intensity'] = supply_gdf[competition_cols].mean(axis=1)
        else:
            supply_gdf['competition_intensity'] = supply_gdf['market_saturation']
        
        return supply_gdf
    
    def _calculate_accessibility_factors(self, features_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate accessibility and infrastructure factors"""
        logger.info("Calculating accessibility factors")
        
        access_gdf = features_gdf.copy()
        
        # Transportation accessibility
        transport_cols = [col for col in access_gdf.columns if 'transport' in col.lower() or 'accessibility' in col.lower()]
        if transport_cols:
            access_gdf['transport_accessibility'] = access_gdf[transport_cols].mean(axis=1)
        else:
            access_gdf['transport_accessibility'] = np.random.beta(2, 2, len(access_gdf))  # Moderate accessibility
        
        # Road network connectivity
        if 'connectivity_index' in access_gdf.columns:
            access_gdf['road_connectivity'] = access_gdf['connectivity_index']
        else:
            access_gdf['road_connectivity'] = np.random.beta(2, 3, len(access_gdf))
        
        # Urban centrality
        if 'centrality_score' in access_gdf.columns:
            access_gdf['urban_centrality'] = access_gdf['centrality_score']
        else:
            access_gdf['urban_centrality'] = np.random.beta(1.5, 3, len(access_gdf))  # Skewed toward periphery
        
        # Walkability and foot traffic potential
        if 'walkability_score' in access_gdf.columns:
            access_gdf['walkability'] = access_gdf['walkability_score']
        else:
            access_gdf['walkability'] = np.random.beta(2, 3, len(access_gdf))
        
        # Composite accessibility score
        access_gdf['total_accessibility_score'] = (
            access_gdf['transport_accessibility'] * 0.3 +
            access_gdf['road_connectivity'] * 0.25 +
            access_gdf['urban_centrality'] * 0.25 +
            access_gdf['walkability'] * 0.2
        )
        
        return access_gdf
    
    def _calculate_economic_potential(self, features_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate economic growth potential and business environment"""
        logger.info("Calculating economic potential")
        
        econ_gdf = features_gdf.copy()
        
        # Transaction volume potential
        if 'total_monthly_ttv' in econ_gdf.columns:
            ttv_potential = econ_gdf['total_monthly_ttv'] / econ_gdf['total_monthly_ttv'].max()
        else:
            ttv_potential = np.random.lognormal(0, 0.8, len(econ_gdf))  # Log-normal distribution
            ttv_potential = ttv_potential / ttv_potential.max()
        
        econ_gdf['ttv_potential'] = ttv_potential
        
        # Business growth environment
        if 'avg_business_age' in econ_gdf.columns:
            # Newer businesses suggest growing market
            business_growth = 1 - (econ_gdf['avg_business_age'] / econ_gdf['avg_business_age'].max())
        else:
            business_growth = np.random.beta(3, 2, len(econ_gdf))  # Favor growth areas
        
        econ_gdf['business_growth_potential'] = business_growth
        
        # Risk environment (lower risk = higher potential)
        if 'avg_risk_score' in econ_gdf.columns:
            risk_environment = 1 - econ_gdf['avg_risk_score']
        else:
            risk_environment = np.random.beta(3, 2, len(econ_gdf))  # Generally favorable
        
        econ_gdf['risk_environment'] = risk_environment
        
        # Digital adoption potential
        if 'digital_payment_adoption' in econ_gdf.columns:
            digital_potential = econ_gdf['digital_payment_adoption']
        else:
            # Higher in more educated, urban areas
            if 'education_higher_pct' in econ_gdf.columns and 'urban_centrality' in econ_gdf.columns:
                digital_potential = (
                    econ_gdf['education_higher_pct'] / 100 * 0.6 +
                    econ_gdf['urban_centrality'] * 0.4
                )
            else:
                digital_potential = np.random.beta(2, 2, len(econ_gdf))
        
        econ_gdf['digital_adoption_potential'] = digital_potential
        
        # Composite economic potential
        econ_gdf['total_economic_potential'] = (
            econ_gdf['ttv_potential'] * 0.3 +
            econ_gdf['business_growth_potential'] * 0.25 +
            econ_gdf['risk_environment'] * 0.2 +
            econ_gdf['digital_adoption_potential'] * 0.25
        )
        
        return econ_gdf
    
    def _calculate_composite_opportunity_score(self, features_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate final composite opportunity scores"""
        logger.info("Calculating composite opportunity scores")
        
        opportunity_gdf = features_gdf.copy()
        
        # Market penetration score (high demand, low competition)
        market_penetration = (
            opportunity_gdf['total_demand_score'] * 0.6 +
            (1 - opportunity_gdf['competition_intensity']) * 0.4
        )
        
        # Normalize all components to 0-1 scale
        components = {
            'market_penetration': market_penetration,
            'competition': 1 - opportunity_gdf['competition_intensity'],
            'accessibility': opportunity_gdf['total_accessibility_score'],
            'economic_potential': opportunity_gdf['total_economic_potential']
        }
        
        # Apply weights from configuration
        weights = {
            'market_penetration': self.config.market_penetration_weight,
            'competition': self.config.competition_weight,
            'accessibility': self.config.accessibility_weight,
            'economic_potential': self.config.economic_potential_weight
        }
        
        # Calculate weighted composite score
        composite_score = sum(
            components[component] * weights[component]
            for component in components.keys()
        )
        
        opportunity_gdf['opportunity_score'] = composite_score
        
        # Create opportunity categories
        opportunity_gdf['opportunity_category'] = pd.cut(
            opportunity_gdf['opportunity_score'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Exceptional'],
            include_lowest=True
        )
        
        # Store individual component scores
        for component, score in components.items():
            opportunity_gdf[f'{component}_score'] = score
        
        return opportunity_gdf
    
    def _perform_market_segmentation(self, opportunity_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Perform market segmentation to identify similar market conditions"""
        logger.info("Performing market segmentation")
        
        # Features for clustering
        clustering_features = [
            'total_demand_score', 'competition_intensity', 
            'total_accessibility_score', 'total_economic_potential'
        ]
        
        # Ensure features exist
        available_features = [f for f in clustering_features if f in opportunity_gdf.columns]
        
        if len(available_features) >= 2:
            # Prepare data for clustering
            cluster_data = opportunity_gdf[available_features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)
            
            # K-means clustering
            n_clusters = min(6, len(opportunity_gdf) // 20 + 2)  # Adaptive cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(cluster_data_scaled)
            
            opportunity_gdf['market_segment'] = cluster_labels
            
            # Characterize each segment
            segment_characteristics = self._characterize_market_segments(
                opportunity_gdf, cluster_labels, available_features
            )
            
            self.market_clusters = {
                'n_clusters': n_clusters,
                'characteristics': segment_characteristics,
                'features_used': available_features
            }
        else:
            # Fallback simple segmentation
            opportunity_gdf['market_segment'] = pd.qcut(
                opportunity_gdf['opportunity_score'], 
                q=4, 
                labels=['Segment_1', 'Segment_2', 'Segment_3', 'Segment_4']
            )
        
        return opportunity_gdf
    
    def _characterize_market_segments(self, opportunity_gdf: gpd.GeoDataFrame,
                                    cluster_labels: np.ndarray, features: List[str]) -> Dict:
        """Characterize each market segment"""
        segments = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_data = opportunity_gdf[cluster_labels == cluster_id]
            
            segment_profile = {
                'size': len(cluster_data),
                'avg_opportunity_score': cluster_data['opportunity_score'].mean(),
                'feature_profile': {}
            }
            
            # Feature averages for this segment
            for feature in features:
                segment_profile['feature_profile'][feature] = cluster_data[feature].mean()
            
            # Segment interpretation
            if segment_profile['avg_opportunity_score'] > 0.7:
                segment_profile['interpretation'] = 'High-opportunity market'
            elif segment_profile['avg_opportunity_score'] > 0.5:
                segment_profile['interpretation'] = 'Moderate-opportunity market'
            elif segment_profile['avg_opportunity_score'] > 0.3:
                segment_profile['interpretation'] = 'Emerging market'
            else:
                segment_profile['interpretation'] = 'Challenging market'
            
            segments[f'Segment_{cluster_id}'] = segment_profile
        
        return segments
    
    def _generate_expansion_recommendations(self, opportunity_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Generate specific expansion recommendations for each location"""
        logger.info("Generating expansion recommendations")
        
        recommendation_gdf = opportunity_gdf.copy()
        
        recommendations = []
        priorities = []
        strategies = []
        
        for _, row in recommendation_gdf.iterrows():
            opportunity_score = row['opportunity_score']
            
            # Determine recommendation based on opportunity score and characteristics
            if opportunity_score >= 0.8:
                recommendation = "High Priority - Immediate Expansion"
                priority = "High"
                strategy = "Direct merchant acquisition with premium incentives"
            
            elif opportunity_score >= 0.6:
                recommendation = "Medium Priority - Strategic Expansion"
                priority = "Medium"
                
                if row.get('competition_intensity', 0) > 0.7:
                    strategy = "Competitive differentiation strategy with unique value proposition"
                else:
                    strategy = "Standard acquisition with competitive pricing"
            
            elif opportunity_score >= 0.4:
                recommendation = "Low Priority - Selective Expansion"
                priority = "Low"
                strategy = "Partnership-based entry with local players"
            
            else:
                recommendation = "Monitor - Future Consideration"
                priority = "Monitor"
                strategy = "Market development and education initiatives"
            
            # Category-specific recommendations
            category_suggestions = self._get_category_specific_suggestions(row)
            
            recommendations.append(recommendation)
            priorities.append(priority)
            strategies.append(strategy)
        
        recommendation_gdf['expansion_recommendation'] = recommendations
        recommendation_gdf['priority_level'] = priorities
        recommendation_gdf['acquisition_strategy'] = strategies
        
        # Calculate expected ROI
        recommendation_gdf['expected_roi'] = self._calculate_expected_roi(recommendation_gdf)
        
        return recommendation_gdf
    
    def _get_category_specific_suggestions(self, location_row: pd.Series) -> Dict[str, str]:
        """Generate category-specific expansion suggestions"""
        suggestions = {}
        
        for category in self.config.target_categories:
            competition_col = f'{category}_competition'
            
            if competition_col in location_row:
                competition_level = location_row[competition_col]
                
                if competition_level < 0.3:
                    suggestions[category] = "High opportunity - low competition"
                elif competition_level < 0.6:
                    suggestions[category] = "Moderate opportunity - compete on service"
                else:
                    suggestions[category] = "Saturated market - consider differentiation"
            else:
                suggestions[category] = "Assessment needed"
        
        return suggestions
    
    def _calculate_expected_roi(self, opportunity_gdf: gpd.GeoDataFrame) -> pd.Series:
        """Calculate expected ROI based on opportunity characteristics"""
        
        # Base ROI calculation
        base_roi = 1.5  # 150% minimum expected return
        
        # Opportunity score multiplier
        opportunity_multiplier = 1 + opportunity_gdf['opportunity_score'] * 2
        
        # Risk adjustment
        if 'risk_environment' in opportunity_gdf.columns:
            risk_adjustment = opportunity_gdf['risk_environment']
        else:
            risk_adjustment = 0.8  # Moderate risk assumption
        
        # Market size adjustment
        if 'total_demand_score' in opportunity_gdf.columns:
            market_size_adjustment = 0.5 + opportunity_gdf['total_demand_score'] * 0.5
        else:
            market_size_adjustment = 0.75
        
        expected_roi = (
            base_roi * 
            opportunity_multiplier * 
            risk_adjustment * 
            market_size_adjustment
        )
        
        return expected_roi.round(2)
    
    def get_top_opportunities(self, n_top: int = 20, min_score: float = None) -> gpd.GeoDataFrame:
        """Get top expansion opportunities"""
        if self.opportunity_scores is None:
            raise ValueError("Market analysis not performed. Call analyze_market_opportunities() first.")
        
        min_score = min_score or self.config.min_opportunity_score
        
        # Filter by minimum score
        filtered_opportunities = self.opportunity_scores[
            self.opportunity_scores['opportunity_score'] >= min_score
        ].copy()
        
        # Sort by opportunity score and expected ROI
        top_opportunities = filtered_opportunities.nlargest(n_top, ['opportunity_score', 'expected_roi'])
        
        logger.info(f"Identified {len(top_opportunities)} top opportunities")
        return top_opportunities
    
    def generate_market_report(self) -> str:
        """Generate comprehensive market analysis report"""
        if self.opportunity_scores is None:
            return "Market analysis not performed yet."
        
        total_locations = len(self.opportunity_scores)
        high_opp = len(self.opportunity_scores[self.opportunity_scores['opportunity_score'] >= 0.7])
        medium_opp = len(self.opportunity_scores[
            (self.opportunity_scores['opportunity_score'] >= 0.4) & 
            (self.opportunity_scores['opportunity_score'] < 0.7)
        ])
        
        avg_opportunity = self.opportunity_scores['opportunity_score'].mean()
        avg_roi = self.opportunity_scores['expected_roi'].mean()
        
        report = f"""
# Market Opportunity Analysis Report

## Overview
- **Total Locations Analyzed**: {total_locations:,}
- **High Opportunity Locations**: {high_opp:,} ({high_opp/total_locations*100:.1f}%)
- **Medium Opportunity Locations**: {medium_opp:,} ({medium_opp/total_locations*100:.1f}%)
- **Average Opportunity Score**: {avg_opportunity:.3f}
- **Average Expected ROI**: {avg_roi:.2f}x

## Top Recommendations
"""
        
        # Top 5 opportunities
        top_5 = self.get_top_opportunities(n_top=5)
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            report += f"**{idx}. Hex {row['hex_id'][:8]}...**\n"
            report += f"   - Opportunity Score: {row['opportunity_score']:.3f}\n"
            report += f"   - Expected ROI: {row['expected_roi']:.2f}x\n"
            report += f"   - Strategy: {row['acquisition_strategy']}\n\n"
        
        if self.market_clusters:
            report += "\n## Market Segments\n"
            for segment_name, segment_data in self.market_clusters['characteristics'].items():
                report += f"**{segment_name}**: {segment_data['interpretation']} ({segment_data['size']} locations)\n"
        
        return report


class MerchantAcquisitionOptimizer:
    """
    Optimization engine for strategic merchant acquisition planning.
    Determines optimal resource allocation and expansion sequencing.
    """
    
    def __init__(self, config: AcquisitionConfig = None):
        self.config = config or AcquisitionConfig()
        self.optimization_results = None
        
    def optimize_expansion_plan(self, opportunity_gdf: gpd.GeoDataFrame,
                              budget_constraint: float = 1000000,  # 1M BRL budget
                              time_horizon_months: int = 12) -> Dict:
        """
        Optimize merchant acquisition plan under budget and time constraints.
        
        Args:
            opportunity_gdf: GeoDataFrame with opportunity analysis
            budget_constraint: Total available budget in BRL
            time_horizon_months: Planning time horizon
            
        Returns:
            Dictionary with optimized expansion plan
        """
        logger.info(f"Optimizing expansion plan with {budget_constraint:,.0f} BRL budget")
        
        # Filter viable opportunities
        viable_opportunities = opportunity_gdf[
            (opportunity_gdf['opportunity_score'] >= self.config.min_opportunity_score) &
            (opportunity_gdf['expected_roi'] >= self.config.roi_threshold)
        ].copy()
        
        if len(viable_opportunities) == 0:
            logger.warning("No viable opportunities found with current criteria")
            return {"error": "No viable opportunities"}
        
        # Estimate costs and returns for each opportunity
        viable_opportunities = self._estimate_acquisition_costs(viable_opportunities)
        viable_opportunities = self._estimate_revenue_potential(viable_opportunities)
        
        # Perform optimization
        optimization_result = self._solve_acquisition_optimization(
            viable_opportunities, budget_constraint, time_horizon_months
        )
        
        self.optimization_results = optimization_result
        return optimization_result
    
    def _estimate_acquisition_costs(self, opportunities_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Estimate acquisition costs for each opportunity"""
        
        # Base acquisition cost
        base_cost = 5000  # 5K BRL base cost per merchant
        
        # Cost adjustments based on market characteristics
        cost_adjustments = opportunities_gdf.copy()
        
        # Competition increases costs
        competition_multiplier = 1 + cost_adjustments.get('competition_intensity', 0) * 0.5
        
        # Urban centrality increases costs
        centrality_multiplier = 1 + cost_adjustments.get('urban_centrality', 0.5) * 0.3
        
        # Market maturity affects costs
        if 'market_saturation' in cost_adjustments.columns:
            saturation_multiplier = 1 + cost_adjustments['market_saturation'] * 0.2
        else:
            saturation_multiplier = 1.1
        
        estimated_cost = (
            base_cost * 
            competition_multiplier * 
            centrality_multiplier * 
            saturation_multiplier
        )
        
        opportunities_gdf['estimated_acquisition_cost'] = estimated_cost.round(0)
        
        return opportunities_gdf
    
    def _estimate_revenue_potential(self, opportunities_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Estimate monthly revenue potential for each opportunity"""
        
        # Base monthly revenue per merchant
        base_monthly_revenue = 800  # 800 BRL base monthly revenue
        
        # Revenue adjustments
        revenue_adjustments = opportunities_gdf.copy()
        
        # Demand drives revenue
        demand_multiplier = 1 + revenue_adjustments.get('total_demand_score', 0.5) * 1.5
        
        # Economic potential affects revenue
        economic_multiplier = 1 + revenue_adjustments.get('total_economic_potential', 0.5) * 1.0
        
        # Accessibility affects transaction volume
        accessibility_multiplier = 1 + revenue_adjustments.get('total_accessibility_score', 0.5) * 0.5
        
        estimated_revenue = (
            base_monthly_revenue *
            demand_multiplier *
            economic_multiplier *
            accessibility_multiplier
        )
        
        opportunities_gdf['estimated_monthly_revenue'] = estimated_revenue.round(0)
        
        # Calculate payback period
        opportunities_gdf['payback_period_months'] = (
            opportunities_gdf['estimated_acquisition_cost'] / 
            opportunities_gdf['estimated_monthly_revenue']
        ).round(1)
        
        return opportunities_gdf
    
    def _solve_acquisition_optimization(self, opportunities_gdf: gpd.GeoDataFrame,
                                      budget: float, time_horizon: int) -> Dict:
        """Solve the merchant acquisition optimization problem"""
        
        n_opportunities = len(opportunities_gdf)
        
        # Decision variables: binary selection for each opportunity
        costs = opportunities_gdf['estimated_acquisition_cost'].values
        revenues = opportunities_gdf['estimated_monthly_revenue'].values * time_horizon
        
        # Simple greedy optimization (ROI-based selection)
        # In production, would use more sophisticated optimization
        
        # Calculate ROI for each opportunity
        roi_values = (revenues - costs) / costs
        
        # Sort by ROI descending
        sorted_indices = np.argsort(roi_values)[::-1]
        
        # Greedy selection within budget
        selected_opportunities = []
        total_cost = 0
        total_revenue = 0
        
        for idx in sorted_indices:
            opportunity_cost = costs[idx]
            
            if total_cost + opportunity_cost <= budget:
                selected_opportunities.append(idx)
                total_cost += opportunity_cost
                total_revenue += revenues[idx]
                
                # Stop if we have enough good opportunities
                if len(selected_opportunities) >= min(n_opportunities, 50):  # Max 50 acquisitions
                    break
        
        # Create optimization results
        selected_gdf = opportunities_gdf.iloc[selected_opportunities].copy()
        selected_gdf = selected_gdf.sort_values('opportunity_score', ascending=False)
        
        optimization_result = {
            'selected_opportunities': selected_gdf,
            'total_selected': len(selected_opportunities),
            'total_cost': total_cost,
            'total_expected_revenue': total_revenue,
            'expected_profit': total_revenue - total_cost,
            'average_roi': (total_revenue - total_cost) / total_cost if total_cost > 0 else 0,
            'budget_utilization': total_cost / budget,
            'summary_stats': {
                'avg_acquisition_cost': selected_gdf['estimated_acquisition_cost'].mean(),
                'avg_monthly_revenue': selected_gdf['estimated_monthly_revenue'].mean(),
                'avg_payback_months': selected_gdf['payback_period_months'].mean(),
                'avg_opportunity_score': selected_gdf['opportunity_score'].mean()
            }
        }
        
        logger.info(f"Optimization completed: {len(selected_opportunities)} opportunities selected")
        logger.info(f"Total cost: {total_cost:,.0f} BRL, Expected ROI: {optimization_result['average_roi']:.2f}x")
        
        return optimization_result


def run_complete_merchant_acquisition_analysis(features_gdf: gpd.GeoDataFrame,
                                             merchants_gdf: gpd.GeoDataFrame = None,
                                             budget: float = 1000000) -> Tuple[MarketOpportunityAnalyzer, MerchantAcquisitionOptimizer]:
    """
    Run the complete merchant acquisition analysis pipeline.
    
    Args:
        features_gdf: GeoDataFrame with spatial features
        merchants_gdf: Optional existing merchant data
        budget: Available budget for expansion
        
    Returns:
        Tuple of (MarketOpportunityAnalyzer, MerchantAcquisitionOptimizer)
    """
    logger.info("Starting complete merchant acquisition analysis")
    
    # Market opportunity analysis
    market_analyzer = MarketOpportunityAnalyzer()
    opportunity_analysis = market_analyzer.analyze_market_opportunities(features_gdf, merchants_gdf)
    
    # Generate market report
    market_report = market_analyzer.generate_market_report()
    print(market_report)
    
    # Expansion optimization
    optimizer = MerchantAcquisitionOptimizer()
    optimization_results = optimizer.optimize_expansion_plan(opportunity_analysis, budget)
    
    # Print optimization summary
    if 'error' not in optimization_results:
        print(f"\n## Expansion Optimization Results")
        print(f"Selected Opportunities: {optimization_results['total_selected']}")
        print(f"Total Investment: {optimization_results['total_cost']:,.0f} BRL")
        print(f"Expected Annual ROI: {optimization_results['average_roi']:.2f}x")
        print(f"Budget Utilization: {optimization_results['budget_utilization']:.1%}")
    
    logger.info("Merchant acquisition analysis completed")
    return market_analyzer, optimizer


if __name__ == "__main__":
    # Example usage
    print("Testing Merchant Acquisition & Opportunity Scoring System...")
    
    from ..feature_engineering.hexgrid import create_porto_alegre_grid
    from ..data_pipeline.data_sources import DataPipeline
    from ..feature_engineering.spatial_features import create_comprehensive_features
    
    # Create components
    hex_grid = create_porto_alegre_grid(resolution=10)  # Smaller for testing
    data_pipeline = DataPipeline()
    datasets = data_pipeline.run_full_pipeline()
    
    # Generate features
    features = create_comprehensive_features(hex_grid, datasets)
    
    # Run merchant acquisition analysis
    market_analyzer, optimizer = run_complete_merchant_acquisition_analysis(
        features, datasets.get('merchants'), budget=500000
    )
    
    print("\nMerchant Acquisition Analysis Completed!")
    print(f"Top opportunities identified: {len(market_analyzer.get_top_opportunities())}")