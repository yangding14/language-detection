"""
Model A Dataset B - XLM-RoBERTa Language Detection

This module implements the XLM-RoBERTa based language detection model
fine-tuned on Dataset B (enhanced/specialized language detection dataset).

Model Architecture: XLM-RoBERTa (Model A)
Training Dataset: Dataset B (enhanced/specialized)
Performance: 99.72% accuracy across 100+ languages
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


class ModelADatasetB(BaseLanguageModel):
    """
    XLM-RoBERTa based language detection model (Model A) trained on Dataset B.
    
    This model represents the XLM-RoBERTa architecture fine-tuned on an enhanced
    language detection dataset, achieving exceptional 99.72% accuracy with
    state-of-the-art performance across 100+ languages.
    
    Architecture: XLM-RoBERTa (Model A)
    Dataset: Dataset B (enhanced/specialized)
    Base Model: xlm-roberta-base
    Accuracy: 99.72%
    Parameters: 278M
    Training Loss: 0.0176
    """
    
    def __init__(self):
        """Initialize the Model A Dataset B language detector."""
        self.model_key = "model-a-dataset-b"
        self.config = get_model_config(self.model_key)
        self.model_name = self.config["huggingface_model"]
        
        # Check if transformers library is available
        if not HF_AVAILABLE:
            raise ImportError(
                "Transformers library required for Model A Dataset B. "
                "Install with: pip install transformers torch"
            )
        
        # Initialize the model pipeline
        try:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=-1,  # Use CPU (-1) for compatibility; change to 0 for GPU
                top_k=None  # Return all scores
            )
            logging.info(f"Successfully loaded {self.config['display_name']} ({self.model_name})")
        except Exception as e:
            logging.error(f"Failed to load {self.config['display_name']}: {e}")
            raise RuntimeError(f"Could not initialize Model A Dataset B: {str(e)}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict language using Model A Dataset B (XLM-RoBERTa enhanced).
        
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
            logging.error(f"Model A Dataset B prediction failed: {e}")
            raise RuntimeError(f"Model prediction failed: {str(e)}")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for Model A Dataset B.
        
        Returns:
            List of ISO 639-1 language codes supported by the model
        """
        return get_supported_languages(self.model_key)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about Model A Dataset B.
        
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
            "training_loss": f"{self.config.get('training_loss', 'N/A')}",
            "use_cases": self.config["use_cases"],
            "strengths": self.config["strengths"],
            "limitations": self.config["limitations"]
        }
        
        return model_info 