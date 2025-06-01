"""
Model B Dataset A - BERT Language Detection

This module implements the BERT based language detection model
fine-tuned on Dataset A (standard multilingual language detection dataset).

Model Architecture: BERT (Model B)
Training Dataset: Dataset A (standard multilingual)  
Performance: 96.17% accuracy across 100+ languages
"""

import logging
from typing import Dict, List, Any

from .base_model import BaseLanguageModel
from .model_config import get_model_config, get_supported_languages, get_language_name

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("Transformers library not available. Please install with: pip install transformers torch")


class ModelBDatasetA(BaseLanguageModel):
    """
    BERT based language detection model (Model B) trained on Dataset A.
    
    This model represents the BERT architecture fine-tuned on a standard
    multilingual language detection dataset, achieving 96.17% accuracy with
    optimized efficiency and broad language coverage across 100+ languages.
    
    Architecture: BERT (Model B)
    Dataset: Dataset A (standard multilingual)
    Base Model: bert-base-multilingual-cased
    Accuracy: 96.17%
    Parameters: 178M
    """
    
    def __init__(self):
        """Initialize the Model B Dataset A language detector."""
        self.model_key = "model-b-dataset-a"
        self.config = get_model_config(self.model_key)
        self.model_name = self.config["huggingface_model"]
        
        # Check if transformers library is available
        if not HF_AVAILABLE:
            raise ImportError(
                "Transformers library required for Model B Dataset A. "
                "Install with: pip install transformers torch"
            )
        
        # Initialize the model pipeline
        try:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=0,
                top_k=None  # Return all scores
            )
            logging.info(f"Successfully loaded {self.config['display_name']} ({self.model_name})")
        except Exception as e:
            logging.error(f"Failed to load {self.config['display_name']}: {e}")
            raise RuntimeError(f"Could not initialize Model B Dataset A: {str(e)}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict language using Model B Dataset A (BERT).
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict with predictions, metadata, and model information
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        try:
            # Run the model prediction
            results = self.classifier(text)
            
            # Handle the format returned by the pipeline
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # Nested list format: [[{'label': 'en', 'score': 0.99}, ...]]
                    prediction_list = results[0]
                else:
                    # Direct list format: [{'label': 'en', 'score': 0.99}, ...]
                    prediction_list = results
            else:
                raise ValueError("Unexpected pipeline output format")
            
            # Sort predictions by confidence score (descending)
            predictions = [
                {
                    'language_code': result['label'].lower(),
                    'confidence': result['score']
                }
                for result in sorted(prediction_list, key=lambda x: x['score'], reverse=True)
            ]
            
            return {
                'predictions': predictions,
                'text_length': len(text),
                'model_version': self.model_name,
                'model_type': f"{self.config['architecture'].lower()}-{self.config['dataset'].lower().replace(' ', '-')}"
            }
            
        except Exception as e:
            logging.error(f"Model B Dataset A prediction failed: {e}")
            raise RuntimeError(f"Model prediction failed: {str(e)}")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for Model B Dataset A.
        
        Returns:
            List of ISO 639-1 language codes supported by the model
        """
        return get_supported_languages(self.model_key)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about Model B Dataset A.
        
        Returns:
            Dict containing comprehensive model metadata
        """
        # Build comprehensive model info from centralized config
        model_info = {
            "name": self.config["display_name"],
            "description": self.config["description"],
            "accuracy": self.config["accuracy"],
            "model_size": self.config["model_size"],
            "architecture": self.config["architecture"],
            "base_model": self.config["base_model"],
            "dataset": self.config["dataset"],
            "languages_supported": f"{self.config['languages_supported']}+ languages",
            "training_details": self.config["training_details"],
            "use_cases": self.config["use_cases"],
            "strengths": self.config["strengths"],
            "limitations": self.config["limitations"]
        }
        
        return model_info 