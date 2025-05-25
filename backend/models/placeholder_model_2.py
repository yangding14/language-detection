"""
Placeholder Model 2

This is a template for implementing a third language detection model.
Replace this implementation with your actual Hugging Face model.
"""

from typing import Dict, List, Any
from .base_model import BaseLanguageModel


class PlaceholderModel2(BaseLanguageModel):
    """
    Placeholder for third language detection model.
    
    This class serves as a template for implementing additional language
    detection models. Replace this with your actual Hugging Face model
    implementation.
    
    TODO: Replace with actual model implementation
    Examples:
    - microsoft/DialoGPT-medium
    - distilbert-base-multilingual-cased
    - Any other suitable language detection model
    """
    
    def __init__(self):
        """Initialize the placeholder model."""
        self.model_info = {
            "name": "Language Model 3 (Coming Soon)",
            "description": "This model slot is reserved for a third language detection model. Implementation coming soon. Consider using models optimized for specific use cases or language families.",
            "accuracy": "TBD",
            "model_size": "TBD",
            "languages_supported": "TBD", 
            "training_details": "TBD",
            "use_cases": "TBD",
            "strengths": "TBD",
            "limitations": "TBD"
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict method - not yet implemented.
        
        Args:
            text (str): Input text to analyze
            
        Raises:
            NotImplementedError: This model is not yet implemented
        """
        raise NotImplementedError(
            "This model is not yet implemented. Please select a different model or "
            "implement this placeholder with your actual model."
        )
    
    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages - empty for placeholder.
        
        Returns:
            Empty list since this is a placeholder
        """
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for the placeholder.
        
        Returns:
            Dict containing placeholder model information
        """
        return self.model_info.copy() 