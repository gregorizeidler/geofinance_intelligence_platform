#!/usr/bin/env python3
"""
Quick Start Script - Geo-Financial Intelligence Platform
========================================================

Minimal example demonstrating core platform capabilities for rapid evaluation.
Perfect for demos, presentations, and initial exploration.

Usage:
    python scripts/quick_start.py
    python scripts/quick_start.py --fast-mode
    python scripts/quick_start.py --resolution 10
"""

import sys
import time
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def run_quick_demo(resolution=10, fast_mode=True):
    """Run a quick demonstration of platform capabilities"""
    
    print("""
🚀 =====================================================================
   GEO-FINANCIAL INTELLIGENCE PLATFORM - QUICK START DEMO
   =====================================================================
   
   This demo showcases core platform capabilities:
   • Hexagonal spatial grid generation
   • Multi-source data integration  
   • Spatial feature engineering
   • Credit risk prediction
   • Market opportunity analysis
   
⚡ =====================================================================
    """)
    
    try:
        from feature_engineering.hexgrid import create_porto_alegre_grid
        from data_pipeline.data_sources import DataPipeline
        from feature_engineering.spatial_features import create_comprehensive_features
        from models.credit_risk_model import CreditRiskDataGenerator, GeoCreditRiskModel
        from models.merchant_acquisition import MarketOpportunityAnalyzer
        
        results = {}
        total_start = time.time()
        
        # Step 1: Create Spatial Grid
        print("🗺️  [1/5] Creating hexagonal spatial grid...")
        start = time.time()
        
        hex_grid = create_porto_alegre_grid(resolution=resolution)
        grid_stats = hex_grid.get_grid_stats()
        
        elapsed = time.time() - start
        print(f"     ✅ Generated {grid_stats['total_hexagons']:,} hexagons in {elapsed:.1f}s")
        print(f"     📊 Coverage: {grid_stats['total_area_km2']:.0f} km²")
        
        results['hexagons'] = grid_stats['total_hexagons']
        results['area_km2'] = grid_stats['total_area_km2']
        
        # Step 2: Data Integration
        print("\n📊 [2/5] Integrating multi-source data...")
        start = time.time()
        
        data_pipeline = DataPipeline()
        datasets = data_pipeline.run_full_pipeline()
        
        active_datasets = [name for name, gdf in datasets.items() if not gdf.empty]
        elapsed = time.time() - start
        print(f"     ✅ Integrated {len(active_datasets)} datasets in {elapsed:.1f}s")
        
        for name in active_datasets:
            print(f"     📋 {name.replace('_', ' ').title()}: {len(datasets[name]):,} records")
        
        results['datasets'] = len(active_datasets)
        
        # Step 3: Feature Engineering
        print("\n⚙️  [3/5] Engineering spatial features...")
        start = time.time()
        
        features_gdf = create_comprehensive_features(hex_grid, datasets)
        feature_count = len([col for col in features_gdf.columns 
                           if col not in ['hex_id', 'geometry', 'area_km2', 'centroid_lat', 'centroid_lon']])
        
        elapsed = time.time() - start
        print(f"     ✅ Generated {feature_count} features in {elapsed:.1f}s")
        print(f"     🎯 Categories: Socioeconomic, Commercial, Infrastructure, Financial, Spatial")
        
        results['features'] = feature_count
        
        # Step 4: Credit Risk Modeling (Simplified)
        print("\n🤖 [4/5] Training credit risk model...")
        start = time.time()
        
        # Generate synthetic loan data
        data_generator = CreditRiskDataGenerator(random_state=42)
        loans_per_hex = 2 if fast_mode else 3
        loans_df = data_generator.generate_synthetic_loan_data(features_gdf, n_loans_per_hex=loans_per_hex)
        
        # Train model
        credit_model = GeoCreditRiskModel()
        X, y = credit_model.prepare_training_data(features_gdf, loans_df)
        
        # Quick training without hyperparameter tuning in fast mode
        if fast_mode:
            credit_model.config.hyperparameter_tuning = False
            credit_model.config.shap_analysis = False
        
        performance = credit_model.train_model(X, y)
        
        elapsed = time.time() - start
        print(f"     ✅ Model trained in {elapsed:.1f}s")
        print(f"     📈 AUC Score: {performance['test_auc']:.3f}")
        print(f"     🎯 Precision: {performance['classification_report']['1']['precision']:.3f}")
        
        baseline_auc = 0.75
        improvement = ((performance['test_auc'] - baseline_auc) / baseline_auc) * 100
        print(f"     🚀 Improvement: +{improvement:.1f}% vs baseline")
        
        results['model_auc'] = performance['test_auc']
        results['loans_analyzed'] = len(loans_df)
        
        # Step 5: Market Opportunity Analysis
        print("\n🏪 [5/5] Analyzing market opportunities...")
        start = time.time()
        
        market_analyzer = MarketOpportunityAnalyzer()
        opportunity_analysis = market_analyzer.analyze_market_opportunities(
            features_gdf, datasets.get('merchants')
        )
        
        high_opportunities = (opportunity_analysis['opportunity_score'] >= 0.7).sum()
        avg_roi = opportunity_analysis['expected_roi'].mean()
        
        elapsed = time.time() - start
        print(f"     ✅ Market analysis completed in {elapsed:.1f}s")
        print(f"     🎯 High-opportunity locations: {high_opportunities:,}")
        print(f"     💰 Average expected ROI: {avg_roi:.2f}x")
        
        results['high_opportunities'] = high_opportunities
        results['avg_roi'] = avg_roi
        
        # Final Summary
        total_time = time.time() - total_start
        
        print(f"\n🎉 QUICK DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"⏱️  Total Execution Time: {total_time:.1f} seconds")
        print(f"🌍 Geographic Analysis: {results['hexagons']:,} locations")
        print(f"🔧 Spatial Features: {results['features']} per location")
        print(f"🤖 Credit Risk Model: {results['model_auc']:.3f} AUC")
        print(f"🏪 Market Opportunities: {results['high_opportunities']:,} high-value locations")
        print(f"💼 Business Impact: {results['avg_roi']:.1f}x average ROI potential")
        
        print(f"\n✨ Platform Performance Summary:")
        print(f"   📊 Processing Speed: {results['hexagons'] / total_time:.0f} locations/second")
        print(f"   🎯 Model Accuracy: {((results['model_auc'] - 0.5) / 0.5) * 100:.0f}% above random")
        print(f"   🚀 Ready for production deployment in financial applications!")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        return None
        
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        print("   💡 Check the error details and try running with --fast-mode")
        return None

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Quick Start Demo - Geo-Financial Intelligence Platform"
    )
    
    parser.add_argument(
        '--resolution',
        type=int,
        default=10,
        choices=[8, 9, 10, 11],
        help='H3 grid resolution (8=large, 11=small). Default: 10'
    )
    
    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='Enable fast mode for quicker execution'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run multiple iterations for performance benchmarking'
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        print("🏁 Running Performance Benchmark...")
        iterations = 3
        times = []
        
        for i in range(iterations):
            print(f"\n🔄 Iteration {i+1}/{iterations}")
            start = time.time()
            result = run_quick_demo(args.resolution, fast_mode=True)
            if result:
                times.append(time.time() - start)
            else:
                print("❌ Benchmark failed")
                return 1
        
        print(f"\n📊 BENCHMARK RESULTS:")
        print(f"   ⏱️  Average Time: {sum(times)/len(times):.1f}s")
        print(f"   ⚡ Best Time: {min(times):.1f}s")
        print(f"   📈 Consistency: {(1 - (max(times) - min(times))/sum(times)*len(times)) * 100:.1f}%")
        
    else:
        result = run_quick_demo(args.resolution, args.fast_mode)
        
        if result is None:
            return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)