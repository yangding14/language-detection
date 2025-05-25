"""
XLM-RoBERTa Language Detection Model

This module implements the ZheYu03/xlm-r-langdetect-model from Hugging Face.
High-accuracy multilingual language detection based on XLM-RoBERTa.
"""

import logging
from typing import Dict, List, Any

from .base_model import BaseLanguageModel

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("Transformers library not available. Please install with: pip install transformers torch")


class XLMRobertaLanguageDetector(BaseLanguageModel):
    """
    XLM-RoBERTa based language detection model by ZheYu03.
    
    This model is fine-tuned on language detection task with 97.9% accuracy.
    Based on xlm-roberta-base and trained for 10 epochs with high performance
    across multiple languages.
    
    Model Info:
    - Hugging Face: https://huggingface.co/ZheYu03/xlm-r-langdetect-model
    - Base Model: xlm-roberta-base
    - Accuracy: 97.9%
    - Parameters: 278M
    """
    
    def __init__(self):
        """Initialize the XLM-RoBERTa language detector."""
        self.model_name = "ZheYu03/xlm-r-langdetect-model"
        
        # Model metadata and description
        self.model_info = {
            "name": "XLM-RoBERTa Language Detector",
            "description": "Fine-tuned XLM-RoBERTa model for language detection with 97.9% accuracy. Based on xlm-roberta-base and trained for 10 epochs with comprehensive multilingual support.",
            "accuracy": "97.9%",
            "model_size": "278M parameters",
            "languages_supported": "60+ languages including major European, Asian, and other language families",
            "training_details": "Trained with AdamW optimizer, learning rate 2e-05, batch size 64, 10 epochs with linear learning rate scheduling",
            "use_cases": "General purpose language detection for multilingual text, content classification, preprocessing for NLP pipelines",
            "strengths": "High accuracy, robust performance across different languages, well-tested on diverse datasets",
            "limitations": "Larger model size requires more computational resources, may struggle with very short texts or mixed-language content"
        }
        
        # Check if transformers library is available
        if not HF_AVAILABLE:
            raise ImportError(
                "Transformers library required for XLM-RoBERTa model. "
                "Install with: pip install transformers torch"
            )
        
        # Initialize the model pipeline
        try:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=0,  # Use CPU (-1) or GPU (0, 1, etc.)
                top_k=None  # Return all scores (replaces deprecated return_all_scores=True)
            )
            logging.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Could not initialize XLM-RoBERTa model: {str(e)}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict language using XLM-RoBERTa model.
        
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
            # When return_all_scores=True, results is a list containing a list of dicts
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
                'model_type': 'xlm-roberta'
            }
            
        except Exception as e:
            logging.error(f"XLM-RoBERTa prediction failed: {e}")
            raise RuntimeError(f"Model prediction failed: {str(e)}")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for the XLM-RoBERTa model.
        
        Returns:
            List of ISO 639-1 language codes supported by the model
        """
        # Based on XLM-RoBERTa's multilingual training
        # This list may need to be updated based on the specific model's outputs
        return [
            'af', 'ar', 'bg', 'bn', 'ca', 'cs', 'cy', 'da', 'de', 'el',
            'en', 'es', 'et', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hr',
            'hu', 'id', 'it', 'ja', 'kn', 'ko', 'lt', 'lv', 'mk', 'ml',
            'mr', 'ne', 'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'sk',
            'sl', 'so', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr',
            'uk', 'ur', 'vi', 'zh'
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the XLM-RoBERTa model.
        
        Returns:
            Dict containing comprehensive model metadata
        """
        return self.model_info.copy()  # Return a copy to prevent external modification 