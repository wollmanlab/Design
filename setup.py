#!/usr/bin/env python
"""Setup script for CIPHER package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="cipher",
    version="1.0.0",
    description="Cell Identity Projection using Hybridization Encoding Rules - A deep learning framework for designing multiplexed ISH probe sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Zachary Hemminger",
    author_email="zehemminger@gmail.com",
    url="https://github.com/zehemminger/Design",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.0.0",
        "anndata>=0.8.0",
        "ipykernel>=6.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="bioinformatics, in situ hybridization, probe design, deep learning, cell type identification",
)

