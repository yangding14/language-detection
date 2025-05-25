"""
Language Detection Backend

This module provides the main LanguageDetector class and ModelRegistry
for managing multiple language detection models.
"""

import logging
from typing import Dict, List, Any

from .models import (
    BaseLanguageModel,
    XLMRobertaLanguageDetector,
    PlaceholderModel1,
    PlaceholderModel2,
    PlaceholderModel3
)


class ModelRegistry:
    """
    Registry for managing available language detection models.
    
    This class handles the registration and creation of language detection models.
    Add new models here by importing them and adding them to the models dictionary.
    """
    
    def __init__(self):
        """Initialize the model registry with available models."""
        self.models = {
            "xlm-roberta-langdetect": {
                "class": XLMRobertaLanguageDetector,
                "display_name": "XLM-RoBERTa Language Detector",
                "description": "High-accuracy multilingual language detection (97.9%)",
                "status": "available"
            },
            "model-2": {
                "class": PlaceholderModel1,
                "display_name": "SongJuNN XLM-R Language Detector",
                "description": "Fine-tuned XLM-RoBERTa model for language detection (96.17% accuracy)",
                "status": "available"
            },
            "model-3": {
                "class": PlaceholderModel2,
                "display_name": "Language Model 3", 
                "description": "Coming soon - Additional language detection model",
                "status": "coming_soon"
            },
            "model-4": {
                "class": PlaceholderModel3,
                "display_name": "Language Model 4",
                "description": "Coming soon - Additional language detection model", 
                "status": "coming_soon"
            }
        }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered models.
        
        Returns:
            Dict containing all model information
        """
        return self.models.copy()
    
    def create_model(self, model_key: str) -> BaseLanguageModel:
        """
        Create an instance of the specified model.
        
        Args:
            model_key (str): Key of the model to create
            
        Returns:
            BaseLanguageModel: Instance of the requested model
            
        Raises:
            ValueError: If the model key is not found
        """
        if model_key not in self.models:
            available_keys = list(self.models.keys())
            raise ValueError(f"Unknown model: {model_key}. Available models: {available_keys}")
        
        model_class = self.models[model_key]["class"]
        return model_class()


class LanguageDetector:
    """
    Main language detection class that orchestrates model predictions.
    
    This class provides a unified interface for language detection using
    different models. It handles model switching and provides consistent
    output formatting.
    """
    
    def __init__(self, model_key: str = "xlm-roberta-langdetect"):
        """
        Initialize the language detector.
        
        Args:
            model_key (str): Key of the model to use from the registry
        """
        self.registry = ModelRegistry()
        self.current_model_key = model_key
        self.model = self.registry.create_model(model_key)
        
        # Comprehensive language code to name mapping
        self.language_names = {
            'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'as': 'Assamese', 
            'az': 'Azerbaijani', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bengali',
            'br': 'Breton', 'bs': 'Bosnian', 'ca': 'Catalan', 'cs': 'Czech', 
            'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'dz': 'Dzongkha',
            'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish',
            'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish', 
            'fr': 'French', 'fy': 'Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic',
            'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'he': 'Hebrew', 
            'hi': 'Hindi', 'hr': 'Croatian', 'ht': 'Haitian Creole', 'hu': 'Hungarian',
            'hy': 'Armenian', 'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian', 
            'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kk': 'Kazakh',
            'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'ku': 'Kurdish',
            'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lo': 'Lao',
            'lt': 'Lithuanian', 'lv': 'Latvian', 'mg': 'Malagasy', 'mk': 'Macedonian', 
            'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay',
            'mt': 'Maltese', 'my': 'Myanmar (Burmese)', 'nb': 'Norwegian Bokmål',
            'ne': 'Nepali', 'nl': 'Dutch', 'nn': 'Norwegian Nynorsk', 'no': 'Norwegian',
            'oc': 'Occitan', 'or': 'Odia', 'pa': 'Punjabi', 'pl': 'Polish',
            'ps': 'Pashto', 'pt': 'Portuguese', 'qu': 'Quechua', 'ro': 'Romanian', 
            'ru': 'Russian', 'rw': 'Kinyarwanda', 'se': 'Northern Sami', 'si': 'Sinhala',
            'sk': 'Slovak', 'sl': 'Slovenian', 'so': 'Somali', 'sq': 'Albanian',
            'sr': 'Serbian', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 
            'te': 'Telugu', 'th': 'Thai', 'tl': 'Filipino', 'tr': 'Turkish',
            'ug': 'Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese',
            'vo': 'Volapük', 'wa': 'Walloon', 'xh': 'Xhosa', 'yi': 'Yiddish',
            'yo': 'Yoruba', 'zh': 'Chinese', 'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)', 'zh-hans': 'Chinese (Simplified)',
            'zh-hant': 'Chinese (Traditional)', 'zu': 'Zulu'
        }
    
    def switch_model(self, model_key: str):
        """
        Switch to a different model.
        
        Args:
            model_key (str): Key of the new model to use
            
        Raises:
            Exception: If model switching fails
        """
        try:
            self.model = self.registry.create_model(model_key)
            self.current_model_key = model_key
            logging.info(f"Successfully switched to model: {model_key}")
        except Exception as e:
            logging.error(f"Failed to switch to model {model_key}: {e}")
            raise
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently selected model.
        
        Returns:
            Dict containing current model information
        """
        return self.model.get_model_info()
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available models.
        
        Returns:
            Dict containing all available models
        """
        return self.registry.get_available_models()
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict containing:
                - language: Main predicted language name
                - language_code: Main predicted language code
                - confidence: Confidence score for main prediction
                - top_predictions: List of top 5 predictions with details
                - metadata: Additional information about the prediction
                
        Raises:
            ValueError: If input text is empty
            RuntimeError: If model prediction fails
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Get predictions from the current model
        model_result = self.model.predict(text.strip())
        predictions = model_result['predictions']
        
        if not predictions:
            raise RuntimeError("Model returned no predictions")
        
        # Extract main prediction
        top_prediction = predictions[0]
        main_language_code = top_prediction['language_code']
        main_confidence = top_prediction['confidence']
        
        # Get human-readable language name
        main_language_name = self.language_names.get(
            main_language_code, 
            f"Unknown ({main_language_code})"
        )
        
        # Format top predictions (limit to 5)
        top_predictions = []
        for pred in predictions[:5]:
            lang_code = pred['language_code']
            lang_name = self.language_names.get(lang_code, f"Unknown ({lang_code})")
            top_predictions.append({
                'language': lang_name,
                'language_code': lang_code,
                'confidence': pred['confidence']
            })
        
        # Prepare metadata
        metadata = {
            'text_length': model_result.get('text_length', len(text)),
            'model_name': model_result.get('model_version', 'unknown'),
            'model_type': model_result.get('model_type', 'unknown'),
            'current_model_key': self.current_model_key,
            'model_info': self.get_current_model_info()
        }
        
        return {
            'language': main_language_name,
            'language_code': main_language_code,
            'confidence': main_confidence,
            'top_predictions': top_predictions,
            'metadata': metadata
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get dictionary of supported language codes and names.
        
        Returns:
            Dict mapping language codes to language names
        """
        supported_codes = self.model.get_supported_languages()
        return {
            code: self.language_names.get(code, f"Unknown ({code})")
            for code in supported_codes
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector with default model
    detector = LanguageDetector()
    
    # Test with sample texts
    test_texts = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?", 
        "Hola, ¿cómo estás?",
        "Guten Tag, wie geht es Ihnen?"
    ]
    
    print("Language Detection Test")
    print("=" * 50)
    
    for text in test_texts:
        try:
            result = detector.detect_language(text)
            print(f"Text: {text}")
            print(f"Detected: {result['language']} ({result['language_code']}) - {result['confidence']:.3f}")
            print("---")
        except Exception as e:
            print(f"Error detecting language for '{text}': {e}")
            print("---")
    
    # Show available models
    print("\nAvailable Models:")
    models = detector.get_available_models()
    for key, info in models.items():
        status = "✅" if info["status"] == "available" else "🚧"
        print(f"{status} {info['display_name']} ({key}): {info['description']}") 