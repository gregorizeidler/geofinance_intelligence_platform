"""
Advanced Geo-Enhanced Credit Risk Prediction Model
=================================================

This module implements a sophisticated credit risk prediction system that leverages
spatial features and geospatial intelligence to enhance traditional credit scoring.
The model uses XGBoost with spatial features to predict SME default probability.

Author: Geo-Financial Intelligence Platform
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
import warnings

# Machine Learning Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from ..feature_engineering.spatial_features import SpatialFeatureEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class ModelConfig:
    """Configuration for credit risk model training and evaluation"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    hyperparameter_tuning: bool = True
    feature_selection: bool = True
    shap_analysis: bool = True
    model_output_dir: str = "models/credit_risk"


class CreditRiskDataGenerator:
    """
    Generates realistic synthetic credit risk data for demonstration purposes.
    In production, this would be replaced with actual loan performance data.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_synthetic_loan_data(self, features_gdf: gpd.GeoDataFrame, 
                                   n_loans_per_hex: int = 5) -> pd.DataFrame:
        """
        Generate synthetic loan performance data based on spatial features.
        
        Args:
            features_gdf: GeoDataFrame with spatial features
            n_loans_per_hex: Average number of loans per hexagon
            
        Returns:
            DataFrame with synthetic loan data
        """
        logger.info(f"Generating synthetic loan data for {len(features_gdf)} hexagons")
        
        loans_data = []
        loan_id = 1
        
        for _, hex_row in features_gdf.iterrows():
            # Number of loans varies by hexagon characteristics
            base_loans = n_loans_per_hex
            
            # More loans in commercial areas
            if 'merchant_count' in hex_row and hex_row['merchant_count'] > 0:
                loan_multiplier = 1 + (hex_row['merchant_count'] / hex_row['merchant_count'].max() if hasattr(hex_row['merchant_count'], 'max') else 1) * 2
            else:
                loan_multiplier = 1
            
            n_loans = max(1, int(np.random.poisson(base_loans * loan_multiplier)))
            
            for _ in range(n_loans):
                loan_data = self._generate_single_loan(hex_row, loan_id)
                loans_data.append(loan_data)
                loan_id += 1
        
        loans_df = pd.DataFrame(loans_data)
        logger.info(f"Generated {len(loans_df)} synthetic loans")
        
        return loans_df
    
    def _generate_single_loan(self, hex_features: pd.Series, loan_id: int) -> Dict:
        """Generate a single synthetic loan record"""
        
        # Base default probability influenced by spatial features
        base_default_prob = 0.08  # 8% base default rate
        
        # Spatial risk adjustments
        spatial_risk_adjustment = 0
        
        # Economic vulnerability increases risk
        if 'economic_vulnerability_score' in hex_features:
            spatial_risk_adjustment += hex_features['economic_vulnerability_score'] * 0.15
        
        # Low income areas increase risk
        if 'avg_income_brl' in hex_features and not pd.isna(hex_features['avg_income_brl']):
            income_factor = 1 - min(hex_features['avg_income_brl'] / 10000, 1)  # Normalize to ~10k
            spatial_risk_adjustment += income_factor * 0.10
        
        # High risk merchant concentration increases risk
        if 'avg_risk_score' in hex_features and not pd.isna(hex_features['avg_risk_score']):
            spatial_risk_adjustment += hex_features['avg_risk_score'] * 0.08
        
        # Poor accessibility increases risk
        if 'overall_transport_score' in hex_features and not pd.isna(hex_features['overall_transport_score']):
            spatial_risk_adjustment += (1 - hex_features['overall_transport_score']) * 0.05
        
        # Calculate final default probability
        default_probability = min(0.5, base_default_prob + spatial_risk_adjustment)
        
        # Generate individual loan characteristics
        loan_amount = np.random.lognormal(10, 0.8)  # Mean ~22k BRL, varies widely
        business_age_years = np.random.exponential(4) + 0.5  # 6 months minimum
        annual_revenue = loan_amount * np.random.uniform(3, 12)  # 3-12x loan amount
        
        # Business type influences risk
        business_categories = ['retail', 'restaurant', 'services', 'manufacturing', 'agriculture']
        business_weights = [0.3, 0.25, 0.25, 0.15, 0.05]  # Urban focus
        business_type = np.random.choice(business_categories, p=business_weights)
        
        # Business type risk adjustment
        type_risk = {
            'restaurant': 0.03, 'retail': 0.0, 'services': -0.02,
            'manufacturing': -0.01, 'agriculture': 0.05
        }
        default_probability += type_risk.get(business_type, 0)
        
        # Generate actual default based on probability
        is_default = np.random.random() < default_probability
        
        # Additional features that correlate with default
        credit_score = np.random.normal(650, 100)  # Base credit score
        if is_default:
            credit_score -= np.random.uniform(50, 150)  # Lower scores for defaults
        credit_score = max(300, min(850, credit_score))  # Bound realistic range
        
        # Debt-to-income ratio
        debt_to_income = np.random.beta(2, 5) * 0.6  # Skewed toward lower ratios
        if is_default:
            debt_to_income += np.random.uniform(0.1, 0.3)  # Higher for defaults
        debt_to_income = min(debt_to_income, 0.8)
        
        # Number of existing credit products
        existing_credit_products = np.random.poisson(2)
        
        # Days since last payment (for existing customers)
        days_since_last_payment = np.random.exponential(30) if not is_default else np.random.exponential(60)
        
        return {
            'loan_id': f"L{loan_id:08d}",
            'hex_id': hex_features['hex_id'],
            'loan_amount_brl': loan_amount,
            'business_age_years': business_age_years,
            'annual_revenue_brl': annual_revenue,
            'business_type': business_type,
            'credit_score': credit_score,
            'debt_to_income_ratio': debt_to_income,
            'existing_credit_products': existing_credit_products,
            'days_since_last_payment': days_since_last_payment,
            'is_default': int(is_default),
            'default_probability_true': default_probability  # Hidden ground truth for evaluation
        }


class GeoCreditRiskModel:
    """
    Advanced credit risk prediction model that combines traditional financial features
    with comprehensive spatial intelligence features.
    """
    
    def __init__(self, config: ModelConfig = None):
        """Initialize the geo-enhanced credit risk model"""
        self.config = config or ModelConfig()
        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_importance = None
        self.shap_explainer = None
        self.model_performance = {}
        
        # Create output directory
        Path(self.config.model_output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Geo-Enhanced Credit Risk Model initialized")
    
    def prepare_training_data(self, features_gdf: gpd.GeoDataFrame, 
                            loans_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare comprehensive training dataset by merging spatial features with loan data.
        
        Args:
            features_gdf: GeoDataFrame with spatial features
            loans_df: DataFrame with loan performance data
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Preparing training data with spatial feature enrichment")
        
        # Merge loans with spatial features
        features_for_ml = features_gdf.drop(columns=['geometry'], errors='ignore')
        training_data = loans_df.merge(features_for_ml, on='hex_id', how='left')
        
        logger.info(f"Training data shape: {training_data.shape}")
        
        # Separate target variable
        target = training_data['is_default']
        
        # Remove target and identifier columns
        exclude_cols = [
            'loan_id', 'hex_id', 'is_default', 'default_probability_true',
            'geometry', 'grid_version', 'created_at'
        ]
        feature_data = training_data.drop(columns=exclude_cols, errors='ignore')
        
        # Handle categorical variables
        categorical_cols = feature_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            feature_data[col] = le.fit_transform(feature_data[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.median())
        
        self.feature_columns = feature_data.columns.tolist()
        logger.info(f"Prepared {len(self.feature_columns)} features for training")
        
        return feature_data, target
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the XGBoost credit risk model with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target variable (default indicator)
            
        Returns:
            Dictionary with training results and performance metrics
        """
        logger.info("Training Geo-Enhanced Credit Risk Model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=y
        )
        
        # Feature scaling for certain algorithms (not needed for XGBoost but good practice)
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Train base XGBoost model
        if self.config.hyperparameter_tuning:
            self.model = self._train_with_hyperparameter_tuning(X_train, y_train)
        else:
            self.model = self._train_baseline_model(X_train, y_train)
        
        # Evaluate model
        performance = self._evaluate_model(X_test, y_test, X_train, y_train)
        self.model_performance = performance
        
        # Feature importance analysis
        self._analyze_feature_importance(X_train)
        
        # SHAP analysis for explainability
        if self.config.shap_analysis:
            self._generate_shap_analysis(X_train.sample(min(1000, len(X_train))))
        
        # Save model
        self._save_model()
        
        logger.info("Model training completed")
        return performance
    
    def _train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """Train baseline XGBoost model with default parameters"""
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config.random_state,
            eval_metric='auc'
        )
        
        model.fit(X_train, y_train)
        return model
    
    def _train_with_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """Train XGBoost with hyperparameter optimization"""
        logger.info("Performing hyperparameter tuning")
        
        # Define hyperparameter search space
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        # Base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.config.random_state,
            eval_metric='auc'
        )
        
        # Randomized search
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_distributions,
            n_iter=50,
            cv=self.config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.config.random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series,
                       X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Comprehensive model evaluation"""
        logger.info("Evaluating model performance")
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        # Performance metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        performance = {
            'test_auc': test_auc,
            'train_auc': train_auc,
            'overfitting_score': train_auc - test_auc,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'test_size': len(y_test),
            'default_rate': y_test.mean()
        }
        
        logger.info(f"Model Performance - Test AUC: {test_auc:.4f}, Train AUC: {train_auc:.4f}")
        
        return performance
    
    def _analyze_feature_importance(self, X_train: pd.DataFrame):
        """Analyze and store feature importance"""
        logger.info("Analyzing feature importance")
        
        # XGBoost feature importance
        importance_scores = self.model.feature_importances_
        
        # Create importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Categorize features
        feature_categories = self._categorize_features(feature_importance_df['feature'])
        feature_importance_df['category'] = feature_categories
        
        # Calculate category-level importance
        category_importance = feature_importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
        
        self.feature_importance = {
            'feature_level': feature_importance_df,
            'category_level': category_importance,
            'top_10_features': feature_importance_df.head(10)
        }
        
        logger.info("Top 5 most important features:")
        for idx, row in feature_importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def _categorize_features(self, features: pd.Series) -> List[str]:
        """Categorize features into interpretable groups"""
        categories = []
        
        for feature in features:
            feature_lower = feature.lower()
            
            if any(x in feature_lower for x in ['income', 'age', 'education', 'population', 'unemployment']):
                categories.append('Socioeconomic')
            elif any(x in feature_lower for x in ['poi', 'merchant', 'commercial', 'business', 'ttv']):
                categories.append('Commercial')
            elif any(x in feature_lower for x in ['road', 'transport', 'accessibility', 'centrality']):
                categories.append('Infrastructure')
            elif any(x in feature_lower for x in ['risk', 'payment', 'financial', 'credit']):
                categories.append('Financial')
            elif any(x in feature_lower for x in ['cluster', 'neighbor', 'hotspot', 'spatial']):
                categories.append('Spatial')
            elif any(x in feature_lower for x in ['loan_amount', 'business_age', 'revenue', 'debt']):
                categories.append('Loan_Characteristics')
            else:
                categories.append('Other')
        
        return categories
    
    def _generate_shap_analysis(self, X_sample: pd.DataFrame):
        """Generate SHAP explanations for model interpretability"""
        logger.info("Generating SHAP analysis for model interpretability")
        
        try:
            # Create SHAP explainer
            self.shap_explainer = shap.TreeExplainer(self.model)
            
            # Calculate SHAP values for sample
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Store SHAP analysis results
            self.shap_analysis = {
                'explainer': self.shap_explainer,
                'sample_shap_values': shap_values,
                'sample_data': X_sample,
                'feature_names': X_sample.columns.tolist()
            }
            
            logger.info("SHAP analysis completed")
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            self.shap_analysis = None
    
    def predict_risk_score(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict credit risk scores for new loan applications.
        
        Args:
            features: DataFrame with loan and spatial features
            
        Returns:
            DataFrame with risk scores and predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare features (same preprocessing as training)
        features_processed = features.copy()
        
        # Handle categorical variables
        for col, encoder in self.label_encoders.items():
            if col in features_processed.columns:
                features_processed[col] = encoder.transform(features_processed[col].astype(str))
        
        # Ensure all training features are present
        for col in self.feature_columns:
            if col not in features_processed.columns:
                features_processed[col] = 0  # Default value for missing features
        
        # Select and order columns as in training
        features_processed = features_processed[self.feature_columns]
        
        # Handle missing values
        features_processed = features_processed.fillna(features_processed.median())
        
        # Predict
        risk_probabilities = self.model.predict_proba(features_processed)[:, 1]
        risk_predictions = (risk_probabilities > 0.5).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'risk_probability': risk_probabilities,
            'risk_prediction': risk_predictions,
            'risk_score': (risk_probabilities * 1000).astype(int),  # Convert to 0-1000 score
            'risk_category': pd.cut(risk_probabilities, 
                                  bins=[0, 0.1, 0.3, 0.7, 1.0], 
                                  labels=['Low', 'Medium', 'High', 'Very High'])
        })
        
        return results
    
    def _save_model(self):
        """Save trained model and associated objects"""
        model_path = Path(self.config.model_output_dir)
        
        # Save model
        with open(model_path / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save preprocessing objects
        with open(model_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(model_path / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save feature columns
        pd.Series(self.feature_columns).to_json(model_path / 'feature_columns.json')
        
        # Save performance metrics
        pd.Series(self.model_performance).to_json(model_path / 'performance_metrics.json')
        
        logger.info(f"Model saved to {model_path}")
    
    def generate_model_report(self) -> str:
        """Generate comprehensive model performance report"""
        if not self.model_performance:
            return "Model not trained yet."
        
        report = f"""
# Geo-Enhanced Credit Risk Model Report

## Model Performance
- **Test AUC**: {self.model_performance['test_auc']:.4f}
- **Train AUC**: {self.model_performance['train_auc']:.4f}
- **Overfitting Score**: {self.model_performance['overfitting_score']:.4f}
- **Default Rate**: {self.model_performance['default_rate']:.2%}

## Classification Metrics
- **Precision**: {self.model_performance['classification_report']['1']['precision']:.4f}
- **Recall**: {self.model_performance['classification_report']['1']['recall']:.4f}
- **F1-Score**: {self.model_performance['classification_report']['1']['f1-score']:.4f}

## Feature Importance (Top 10)
"""
        
        if self.feature_importance:
            for _, row in self.feature_importance['top_10_features'].iterrows():
                report += f"- **{row['feature']}**: {row['importance']:.4f} ({row['category']})\n"
            
            report += "\n## Feature Category Importance\n"
            for category, importance in self.feature_importance['category_level'].items():
                report += f"- **{category}**: {importance:.4f}\n"
        
        return report


def run_complete_credit_risk_modeling(features_gdf: gpd.GeoDataFrame) -> GeoCreditRiskModel:
    """
    Run the complete credit risk modeling pipeline.
    
    Args:
        features_gdf: GeoDataFrame with comprehensive spatial features
        
    Returns:
        Trained GeoCreditRiskModel instance
    """
    logger.info("Starting complete credit risk modeling pipeline")
    
    # Generate synthetic loan data
    data_generator = CreditRiskDataGenerator()
    loans_df = data_generator.generate_synthetic_loan_data(features_gdf)
    
    # Initialize and train model
    model = GeoCreditRiskModel()
    X, y = model.prepare_training_data(features_gdf, loans_df)
    performance = model.train_model(X, y)
    
    # Generate report
    report = model.generate_model_report()
    print(report)
    
    logger.info("Credit risk modeling pipeline completed")
    return model


if __name__ == "__main__":
    # Example usage
    print("Testing Geo-Enhanced Credit Risk Model...")
    
    # This would typically be called with real spatial features
    # For testing, we'll create a minimal example
    
    from ..feature_engineering.hexgrid import create_porto_alegre_grid
    from ..data_pipeline.data_sources import DataPipeline
    from ..feature_engineering.spatial_features import create_comprehensive_features
    
    # Create components
    hex_grid = create_porto_alegre_grid(resolution=10)  # Smaller resolution for testing
    data_pipeline = DataPipeline()
    datasets = data_pipeline.run_full_pipeline()
    
    # Generate features
    features = create_comprehensive_features(hex_grid, datasets)
    
    # Run credit risk modeling
    credit_model = run_complete_credit_risk_modeling(features)
    
    print("\nCredit Risk Model Training Completed!")
    print(f"Model AUC: {credit_model.model_performance['test_auc']:.4f}")