#!/usr/bin/env python3
"""
Setup Script for Geo-Financial Intelligence Platform
===================================================

Installation and setup utilities for the Geo-Financial Intelligence Platform.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="geo-financial-intelligence-platform",
    version="1.0.0",
    description="Advanced geospatial data science platform for financial technology applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Geo-Financial Intelligence Platform Team",
    author_email="contact@geo-financial-platform.com",
    url="https://github.com/your-username/geo-financial-platform",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencies
    install_requires=requirements,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Extra dependencies for development
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
            "pydantic>=1.10.0",
        ]
    },
    
    # Entry points for command line tools
    entry_points={
        "console_scripts": [
            "geo-financial-platform=main:main",
            "geo-platform-test=run_tests:main",
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.md", "*.txt"],
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    keywords=[
        "geospatial", "fintech", "machine-learning", "spatial-analysis", 
        "credit-risk", "merchant-acquisition", "h3", "gis", "financial-intelligence"
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/your-username/geo-financial-platform/issues",
        "Source": "https://github.com/your-username/geo-financial-platform",
        "Documentation": "https://geo-financial-platform.readthedocs.io/",
    },
)