#!/usr/bin/env python3
"""
Test Runner for Geo-Financial Intelligence Platform
==================================================

Comprehensive test suite runner with coverage reporting and performance monitoring.
"""

import unittest
import sys
import time
from pathlib import Path
import subprocess
import argparse

def print_banner():
    """Print test runner banner"""
    banner = """
🧪 =====================================================================
   GEO-FINANCIAL INTELLIGENCE PLATFORM - TEST SUITE
   =====================================================================
   
   Running comprehensive tests for all platform components:
   • Hexagonal Grid System
   • Data Pipeline
   • Feature Engineering
   • ML Models
   • Integration Tests
   
🔬 =====================================================================
    """
    print(banner)

def run_tests(test_pattern='test_*.py', verbosity=2, fast_mode=False):
    """
    Run the test suite with specified parameters.
    
    Args:
        test_pattern: Pattern to match test files
        verbosity: Test output verbosity (0-2)
        fast_mode: Skip slow integration tests
    """
    print_banner()
    
    # Add src to Python path
    src_path = Path(__file__).parent / 'src'
    sys.path.insert(0, str(src_path))
    
    # Discover and run tests
    start_time = time.time()
    
    try:
        # Create test loader
        loader = unittest.TestLoader()
        
        # Discover tests in the tests directory
        test_dir = Path(__file__).parent / 'tests'
        if not test_dir.exists():
            test_dir.mkdir()
            print("📁 Created tests directory")
        
        # Load tests
        suite = loader.discover(str(test_dir), pattern=test_pattern)
        
        # Count tests
        test_count = suite.countTestCases()
        
        if test_count == 0:
            print("⚠️  No tests found matching pattern:", test_pattern)
            return False
        
        print(f"🔍 Discovered {test_count} tests")
        
        if fast_mode:
            print("⚡ Running in FAST mode (skipping slow tests)")
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            stream=sys.stdout,
            buffer=True
        )
        
        print(f"\n🚀 Running tests...\n")
        result = runner.run(suite)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Print results summary
        print(f"\n📊 TEST EXECUTION SUMMARY")
        print("=" * 50)
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"✅ Tests Run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"💥 Errors: {len(result.errors)}")
        print(f"⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        # Success rate
        if result.testsRun > 0:
            success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Print detailed failures/errors if any
        if result.failures:
            print(f"\n❌ FAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"   • {test}")
                if verbosity > 1:
                    print(f"     {traceback.split('\\n')[-2]}")
        
        if result.errors:
            print(f"\n💥 ERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"   • {test}")
                if verbosity > 1:
                    print(f"     {traceback.split('\\n')[-2]}")
        
        # Overall result
        if result.wasSuccessful():
            print(f"\n🎉 ALL TESTS PASSED! Platform is ready for deployment.")
            return True
        else:
            print(f"\n⚠️  SOME TESTS FAILED. Please review and fix issues before deployment.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        return False

def run_quick_smoke_tests():
    """Run quick smoke tests to verify basic functionality"""
    print("🔥 Running Quick Smoke Tests...")
    
    try:
        # Test 1: Import core modules
        print("   📦 Testing imports...", end=" ")
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        
        from feature_engineering.hexgrid import HexagonalGrid
        from data_pipeline.data_sources import DataPipeline
        from feature_engineering.spatial_features import SpatialFeatureEngine
        print("✅")
        
        # Test 2: Create minimal hexagonal grid
        print("   🗺️  Testing hexagonal grid...", end=" ")
        from feature_engineering.hexgrid import GridConfig
        config = GridConfig(resolution=11)  # Very small for speed
        grid = HexagonalGrid(config)
        boundary = grid.create_porto_alegre_boundary()
        assert len(boundary) > 0
        print("✅")
        
        # Test 3: Data pipeline initialization
        print("   🔄 Testing data pipeline...", end=" ")
        pipeline = DataPipeline()
        assert pipeline.config is not None
        print("✅")
        
        print("\n🎉 Smoke tests passed! Core functionality is working.")
        return True
        
    except Exception as e:
        print(f"❌\n💥 Smoke test failed: {str(e)}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking Dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'geopandas', 'shapely', 'h3', 
        'sklearn', 'xgboost', 'matplotlib', 'seaborn', 'folium'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 All dependencies are installed!")
        return True

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Geo-Financial Intelligence Platform Test Suite")
    
    parser.add_argument('--pattern', default='test_*.py', 
                       help='Test file pattern (default: test_*.py)')
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=2,
                       help='Test output verbosity (default: 2)')
    parser.add_argument('--fast', action='store_true',
                       help='Run in fast mode (skip slow tests)')
    parser.add_argument('--smoke', action='store_true',
                       help='Run only quick smoke tests')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    # Check dependencies first
    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies first.")
        sys.exit(1)
    
    # Run smoke tests if requested
    if args.smoke:
        success = run_quick_smoke_tests()
        sys.exit(0 if success else 1)
    
    # Run full test suite
    success = run_tests(
        test_pattern=args.pattern,
        verbosity=args.verbosity,
        fast_mode=args.fast
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()