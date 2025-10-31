"""
Climate Policy Dataset Processing Package

This package provides tools for processing, analyzing, and evaluating 
climate policy documents and datasets, specifically focused on COP29 data.
"""

__version__ = "1.0.0"
__author__ = "Climate Policy Research Team"
__email__ = "research@climatepolicy.org"

from .data_loader import DataLoader
from .preprocessor import DocumentPreprocessor
from .evaluator import PolicyEvaluator

__all__ = [
    "DataLoader",
    "DocumentPreprocessor", 
    "PolicyEvaluator"
]