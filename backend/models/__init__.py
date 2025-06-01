"""
Language Detection Models Module

This module provides the base model interface and all available language detection models.
Models are organized by architecture (A: XLM-RoBERTa, B: BERT) and training dataset (A: standard, B: enhanced).
"""

from .base_model import BaseLanguageModel
from .model_config import (
    get_model_config, 
    get_all_model_configs, 
    get_supported_languages, 
    get_language_name,
    LANGUAGE_MAPPINGS
)

# Import all model implementations
from .model_a_dataset_a import ModelADatasetA
from .model_b_dataset_a import ModelBDatasetA  
from .model_a_dataset_b import ModelADatasetB
from .model_b_dataset_b import ModelBDatasetB

__all__ = [
    'BaseLanguageModel',
    'ModelADatasetA',
    'ModelBDatasetA', 
    'ModelADatasetB',
    'ModelBDatasetB',
    'get_model_config',
    'get_all_model_configs',
    'get_supported_languages',
    'get_language_name',
    'LANGUAGE_MAPPINGS'
] 