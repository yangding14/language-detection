"""
Language Detection Models Package

This package contains different language detection model implementations.
Each model is in its own file for better organization and maintainability.
"""

from .base_model import BaseLanguageModel
from .xlm_roberta_detector import XLMRobertaLanguageDetector
from .songjun import PlaceholderModel1
from .placeholder_model_2 import PlaceholderModel2
from .placeholder_model_3 import PlaceholderModel3

__all__ = [
    'BaseLanguageModel',
    'XLMRobertaLanguageDetector',
    'PlaceholderModel1',
    'PlaceholderModel2', 
    'PlaceholderModel3'
] 