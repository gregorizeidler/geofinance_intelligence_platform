#!/usr/bin/env python3
"""
API Example - Geo-Financial Intelligence Platform
================================================

Example REST API implementation for production deployment of the 
Geo-Financial Intelligence Platform using FastAPI.

This demonstrates how to expose the platform's capabilities as web services
for integration with financial applications.

Usage:
    python api_example.py
    # API available at http://localhost:8000
    # Interactive docs at http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Geo-Financial Intelligence Platform API",
    description="""
    Advanced geospatial data science API for financial technology applications.
    
    ## Features
    * **Spatial Analysis**: H3-based hexagonal grid analysis
    * **Credit Risk Scoring**: ML-enhanced risk assessment with spatial features
    * **Market Intelligence**: Opportunity scoring and merchant acquisition optimization
    * **Real-time Processing**: Sub-second feature enrichment and scoring
    
    ## Use Cases
    * Credit risk assessment for SME loans
    * Strategic merchant acquisition planning
    * Market expansion optimization
    * Geospatial business intelligence
    """,
    version="1.0.0",
    contact={
        "name": "Geo-Financial Intelligence Platform Team",
        "email": "contact@geo-financial-platform.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class LocationRequest(BaseModel):
    """Request model for location-based analysis"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    
class CreditRiskRequest(BaseModel):
    """Request model for credit risk assessment"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    loan_amount_brl: float = Field(..., gt=0, description="Loan amount in Brazilian Reais")
    business_type: str = Field(..., description="Business category")
    business_age_years: float = Field(..., ge=0, description="Years in business")
    annual_revenue_brl: float = Field(..., gt=0, description="Annual revenue in BRL")
    credit_score: Optional[float] = Field(None, ge=300, le=850, description="Traditional credit score")
    
class MarketAnalysisRequest(BaseModel):
    """Request model for market opportunity analysis"""
    bounds: List[float] = Field(..., min_items=4, max_items=4, 
                                description="Bounding box [min_lon, min_lat, max_lon, max_lat]")
    business_categories: Optional[List[str]] = Field(None, description="Target business categories")
    min_opportunity_score: Optional[float] = Field(0.3, ge=0, le=1, description="Minimum opportunity threshold")

class SpatialFeatureResponse(BaseModel):
    """Response model for spatial features"""
    hex_id: str
    latitude: float
    longitude: float
    features: Dict[str, float]
    feature_categories: Dict[str, int]

class CreditRiskResponse(BaseModel):
    """Response model for credit risk assessment"""
    risk_probability: float = Field(..., ge=0, le=1, description="Default probability")
    risk_score: int = Field(..., ge=0, le=1000, description="Risk score (0-1000)")
    risk_category: str = Field(..., description="Risk category (Low, Medium, High, Very High)")
    spatial_factors: Dict[str, float] = Field(..., description="Spatial risk contributors")
    recommendation: str = Field(..., description="Risk assessment recommendation")

class MarketOpportunityResponse(BaseModel):
    """Response model for market opportunities"""
    hex_id: str
    latitude: float
    longitude: float
    opportunity_score: float = Field(..., ge=0, le=1)
    expected_roi: float = Field(..., gt=0)
    priority_level: str
    acquisition_strategy: str
    market_characteristics: Dict[str, float]

class PlatformStatusResponse(BaseModel):
    """Response model for platform status"""
    status: str
    version: str
    uptime_seconds: float
    models_loaded: bool
    last_updated: datetime

# Global variables for loaded models (in production, use proper dependency injection)
_models_loaded = False
_hex_grid = None
_features_gdf = None
_credit_model = None
_market_analyzer = None
_startup_time = datetime.now()

async def initialize_platform():
    """Initialize platform components (call on startup)"""
    global _models_loaded, _hex_grid, _features_gdf, _credit_model, _market_analyzer
    
    if _models_loaded:
        return
    
    try:
        logger.info("Initializing Geo-Financial Intelligence Platform...")
        
        # Import platform components
        from feature_engineering.hexgrid import create_porto_alegre_grid
        from data_pipeline.data_sources import DataPipeline
        from feature_engineering.spatial_features import create_comprehensive_features
        from models.credit_risk_model import GeoCreditRiskModel, CreditRiskDataGenerator
        from models.merchant_acquisition import MarketOpportunityAnalyzer
        
        # Initialize components
        _hex_grid = create_porto_alegre_grid(resolution=9)
        
        data_pipeline = DataPipeline()
        datasets = data_pipeline.run_full_pipeline()
        
        _features_gdf = create_comprehensive_features(_hex_grid, datasets)
        
        # Initialize and train models (in production, load pre-trained models)
        _credit_model = GeoCreditRiskModel()
        data_generator = CreditRiskDataGenerator()
        loans_df = data_generator.generate_synthetic_loan_data(_features_gdf, n_loans_per_hex=2)
        X, y = _credit_model.prepare_training_data(_features_gdf, loans_df)
        _credit_model.train_model(X, y)
        
        _market_analyzer = MarketOpportunityAnalyzer()
        _market_analyzer.analyze_market_opportunities(_features_gdf, datasets.get('merchants'))
        
        _models_loaded = True
        logger.info("Platform initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Platform initialization failed: {str(e)}")
        raise

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize platform on startup"""
    await initialize_platform()

@app.get("/", response_model=PlatformStatusResponse)
async def get_platform_status():
    """Get platform status and health information"""
    uptime = (datetime.now() - _startup_time).total_seconds()
    
    return PlatformStatusResponse(
        status="active" if _models_loaded else "initializing",
        version="1.0.0",
        uptime_seconds=uptime,
        models_loaded=_models_loaded,
        last_updated=datetime.now()
    )

@app.post("/spatial-features", response_model=SpatialFeatureResponse)
async def get_spatial_features(location: LocationRequest):
    """Get comprehensive spatial features for a location"""
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Platform still initializing")
    
    try:
        # Find hexagon containing the point
        hex_id = _hex_grid.point_to_hex(location.latitude, location.longitude)
        
        # Get features for this hexagon
        hex_features = _features_gdf[_features_gdf['hex_id'] == hex_id]
        
        if len(hex_features) == 0:
            raise HTTPException(status_code=404, detail="Location outside analysis area")
        
        features_row = hex_features.iloc[0]
        
        # Extract numerical features
        feature_cols = [col for col in _features_gdf.columns 
                       if col not in ['hex_id', 'geometry', 'area_km2', 'centroid_lat', 'centroid_lon']]
        
        features_dict = {}
        for col in feature_cols:
            if col in features_row:
                value = features_row[col]
                if not pd.isna(value):
                    features_dict[col] = float(value)
        
        # Categorize features
        feature_categories = {
            'socioeconomic': len([c for c in features_dict if any(x in c.lower() 
                                 for x in ['income', 'age', 'education', 'population'])]),
            'commercial': len([c for c in features_dict if any(x in c.lower() 
                              for x in ['poi', 'merchant', 'business', 'ttv'])]),
            'infrastructure': len([c for c in features_dict if any(x in c.lower() 
                                  for x in ['road', 'transport', 'accessibility'])]),
            'financial': len([c for c in features_dict if any(x in c.lower() 
                             for x in ['risk', 'payment', 'financial'])]),
            'spatial': len([c for c in features_dict if any(x in c.lower() 
                           for x in ['cluster', 'neighbor', 'spatial'])])
        }
        
        return SpatialFeatureResponse(
            hex_id=hex_id,
            latitude=location.latitude,
            longitude=location.longitude,
            features=features_dict,
            feature_categories=feature_categories
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

@app.post("/credit-risk", response_model=CreditRiskResponse)
async def assess_credit_risk(request: CreditRiskRequest):
    """Assess credit risk using spatial intelligence"""
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Platform still initializing")
    
    try:
        import pandas as pd
        
        # Find hexagon and get spatial features
        hex_id = _hex_grid.point_to_hex(request.latitude, request.longitude)
        hex_features = _features_gdf[_features_gdf['hex_id'] == hex_id]
        
        if len(hex_features) == 0:
            raise HTTPException(status_code=404, detail="Location outside analysis area")
        
        # Create loan application data
        loan_data = {
            'hex_id': hex_id,
            'loan_amount_brl': request.loan_amount_brl,
            'business_type': request.business_type,
            'business_age_years': request.business_age_years,
            'annual_revenue_brl': request.annual_revenue_brl,
            'credit_score': request.credit_score or 650,  # Default if not provided
            'debt_to_income_ratio': 0.3,  # Default value
            'existing_credit_products': 1,  # Default value
            'days_since_last_payment': 15  # Default value
        }
        
        loan_df = pd.DataFrame([loan_data])
        
        # Merge with spatial features
        loan_features = loan_df.merge(hex_features.drop(columns=['geometry'], errors='ignore'), 
                                     on='hex_id', how='left')
        
        # Predict risk
        risk_prediction = _credit_model.predict_risk_score(loan_features)
        risk_result = risk_prediction.iloc[0]
        
        # Get spatial risk factors (top contributing spatial features)
        if _credit_model.feature_importance:
            spatial_features = _credit_model.feature_importance['feature_level']
            spatial_features = spatial_features[
                spatial_features['category'].isin(['Spatial', 'Infrastructure', 'Commercial'])
            ].head(5)
            
            spatial_factors = {}
            for _, row in spatial_features.iterrows():
                feature_name = row['feature']
                if feature_name in loan_features.columns:
                    spatial_factors[feature_name] = float(loan_features.iloc[0][feature_name])
        else:
            spatial_factors = {"spatial_analysis": "not_available"}
        
        # Generate recommendation
        risk_prob = risk_result['risk_probability']
        if risk_prob < 0.1:
            recommendation = "APPROVE: Low risk application with strong spatial indicators"
        elif risk_prob < 0.3:
            recommendation = "APPROVE: Moderate risk, spatial factors are favorable"
        elif risk_prob < 0.7:
            recommendation = "REVIEW: Higher risk, consider additional spatial mitigants"
        else:
            recommendation = "DECLINE: High risk with poor spatial characteristics"
        
        return CreditRiskResponse(
            risk_probability=float(risk_prob),
            risk_score=int(risk_result['risk_score']),
            risk_category=risk_result['risk_category'],
            spatial_factors=spatial_factors,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@app.post("/market-opportunities", response_model=List[MarketOpportunityResponse])
async def analyze_market_opportunities(request: MarketAnalysisRequest):
    """Analyze market opportunities in a geographic area"""
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Platform still initializing")
    
    try:
        # Filter opportunities within bounds
        min_lon, min_lat, max_lon, max_lat = request.bounds
        
        opportunities = _market_analyzer.opportunity_scores
        
        # Filter by geographic bounds
        in_bounds = (
            (opportunities['centroid_lat'] >= min_lat) &
            (opportunities['centroid_lat'] <= max_lat) &
            (opportunities['centroid_lon'] >= min_lon) &
            (opportunities['centroid_lon'] <= max_lon)
        )
        
        filtered_opportunities = opportunities[in_bounds]
        
        # Filter by opportunity score
        if request.min_opportunity_score:
            filtered_opportunities = filtered_opportunities[
                filtered_opportunities['opportunity_score'] >= request.min_opportunity_score
            ]
        
        # Limit results for API performance
        filtered_opportunities = filtered_opportunities.nlargest(50, 'opportunity_score')
        
        # Format response
        results = []
        for _, row in filtered_opportunities.iterrows():
            # Get market characteristics
            market_chars = {
                'demand_score': row.get('total_demand_score', 0.5),
                'competition_level': row.get('competition_intensity', 0.5),
                'accessibility': row.get('total_accessibility_score', 0.5),
                'economic_potential': row.get('total_economic_potential', 0.5)
            }
            
            results.append(MarketOpportunityResponse(
                hex_id=row['hex_id'],
                latitude=row['centroid_lat'],
                longitude=row['centroid_lon'],
                opportunity_score=row['opportunity_score'],
                expected_roi=row['expected_roi'],
                priority_level=row.get('priority_level', 'Medium'),
                acquisition_strategy=row.get('acquisition_strategy', 'Standard approach'),
                market_characteristics=market_chars
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

# Additional utility endpoints
@app.get("/metrics")
async def get_metrics():
    """Get platform performance metrics"""
    if not _models_loaded:
        return {"status": "initializing"}
    
    return {
        "total_hexagons": len(_features_gdf) if _features_gdf is not None else 0,
        "features_per_location": len([col for col in _features_gdf.columns 
                                     if col not in ['hex_id', 'geometry']]) if _features_gdf is not None else 0,
        "model_performance": {
            "credit_auc": _credit_model.model_performance.get('test_auc', 0) if _credit_model and hasattr(_credit_model, 'model_performance') else 0,
            "high_opportunities": len(_market_analyzer.opportunity_scores[
                _market_analyzer.opportunity_scores['opportunity_score'] >= 0.7
            ]) if _market_analyzer and hasattr(_market_analyzer, 'opportunity_scores') else 0
        },
        "uptime_seconds": (datetime.now() - _startup_time).total_seconds()
    }

if __name__ == "__main__":
    print("""
🚀 Starting Geo-Financial Intelligence Platform API
==================================================

API Features:
• Spatial feature extraction
• Credit risk assessment  
• Market opportunity analysis
• Real-time processing

Access the API:
• Base URL: http://localhost:8000
• Interactive docs: http://localhost:8000/docs
• OpenAPI spec: http://localhost:8000/openapi.json

    """)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )