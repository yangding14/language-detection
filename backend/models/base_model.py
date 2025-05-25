"""
Base Language Model Abstract Class

This module defines the interface that all language detection models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseLanguageModel(ABC):
    """
    Abstract base class for language detection models.
    
    All language detection models must inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the language of the given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict containing prediction results with structure:
            {
                'predictions': [
                    {
                        'language_code': str,
                        'confidence': float
                    },
                    ...
                ],
                'text_length': int,
                'model_version': str,
                'model_type': str
            }
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict containing model metadata and description with structure:
            {
                'name': str,
                'description': str,
                'accuracy': str,
                'model_size': str,
                'languages_supported': str,
                'training_details': str,
                'use_cases': str,
                'strengths': str,
                'limitations': str
            }
        """
        pass 