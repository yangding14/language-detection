"""
Zues0102 BERT Multilingual Language Detection Model

This module implements the zues0102/bert-base-multilingual-cased from Hugging Face.
Exceptional ultra high-accuracy multilingual language detection based on BERT 
with outstanding performance achieving 99.85% accuracy.
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


class PlaceholderModel3(BaseLanguageModel):
    """
    Zues0102 BERT Multilingual based language detection model.
    
    This model is based on bert-base-multilingual-cased achieving exceptional 
    99.85% accuracy on language detection tasks. It represents the highest 
    performing model in the collection with state-of-the-art results.
    
    Model Info:
    - Hugging Face: https://huggingface.co/zues0102/bert-base-multilingual-cased
    - Base Model: bert-base-multilingual-cased
    - Accuracy: 99.85%
    - Parameters: ~178M
    - Loss: 0.0125
    - Languages: 20 carefully selected high-performance languages
    """
    
    def __init__(self):
        """Initialize the Zues0102 BERT Multilingual language detector."""
        self.model_name = "zues0102/bert-base-multilingual-cased"
        
        # Model metadata and description based on the Hugging Face model card
        self.model_info = {
            "name": "Zues0102 BERT Multilingual Language Detector",
            "description": "State-of-the-art BERT-based language detection model with 99.85% accuracy. Fine-tuned from bert-base-multilingual-cased with exceptional performance on 20 carefully selected languages. This model represents the pinnacle of language detection accuracy.",
            "accuracy": "99.85%",
            "model_size": "~178M parameters",
            "languages_supported": "20 high-performance languages: Arabic, Bulgarian, German, Greek, English, Spanish, French, Hindi, Italian, Japanese, Dutch, Polish, Portuguese, Russian, Swahili, Thai, Turkish, Urdu, Vietnamese, Chinese",
            "training_details": "Fine-tuned with specialized training achieving loss: 0.0125, optimized for maximum accuracy on core world languages",
            "use_cases": "Ultra-high precision language detection, research applications, critical production systems requiring maximum accuracy, multilingual content processing for supported languages",
            "strengths": "Highest accuracy (99.85%), exceptional performance on 20 core languages, state-of-the-art BERT architecture, extremely low loss (0.0125)",
            "limitations": "Limited to 20 languages (vs 100+ in other models), may require more computational resources, specialized for specific language set"
        }
        
        # Check if transformers library is available
        if not HF_AVAILABLE:
            raise ImportError(
                "Transformers library required for Zues0102 BERT Multilingual model. "
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
            logging.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Could not initialize Zues0102 BERT Multilingual model: {str(e)}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict language using Zues0102 BERT Multilingual model.
        
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
            # When top_k=None, results is a list containing a list of dicts
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
                'model_type': 'bert-multilingual'
            }
            
        except Exception as e:
            logging.error(f"Zues0102 BERT Multilingual prediction failed: {e}")
            raise RuntimeError(f"Model prediction failed: {str(e)}")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for the Zues0102 BERT Multilingual model.
        
        Returns:
            List of ISO 639-1 language codes supported by the model
        """
        # Based on the model configuration showing 20 specifically supported languages
        # These are the languages the model was fine-tuned on for maximum accuracy
        return [
            'ar',  # Arabic
            'bg',  # Bulgarian
            'de',  # German
            'el',  # Greek
            'en',  # English
            'es',  # Spanish
            'fr',  # French
            'hi',  # Hindi
            'it',  # Italian
            'ja',  # Japanese
            'nl',  # Dutch
            'pl',  # Polish
            'pt',  # Portuguese
            'ru',  # Russian
            'sw',  # Swahili
            'th',  # Thai
            'tr',  # Turkish
            'ur',  # Urdu
            'vi',  # Vietnamese
            'zh'   # Chinese
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Zues0102 BERT Multilingual model.
        
        Returns:
            Dict containing comprehensive model metadata
        """
        return self.model_info.copy()  # Return a copy to prevent external modification 