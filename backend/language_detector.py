"""
Language Detection Backend

This module provides the main LanguageDetector class and ModelRegistry
for managing multiple language detection models organized by architecture and dataset.

Model Architecture:
- Model A: XLM-RoBERTa based architectures  
- Model B: BERT based architectures

Training Datasets:
- Dataset A: Standard multilingual language detection dataset
- Dataset B: Enhanced/specialized language detection dataset
"""

import logging
from typing import Dict, List, Any

from .models import (
    BaseLanguageModel,
    ModelADatasetA,
    ModelBDatasetA,
    ModelADatasetB, 
    ModelBDatasetB,
    get_all_model_configs,
    get_language_name,
    LANGUAGE_MAPPINGS
)


class ModelRegistry:
    """
    Registry for managing available language detection models.
    
    This class handles the registration and creation of language detection models
    organized by model architecture (A: XLM-RoBERTa, B: BERT) and training 
    dataset (A: standard, B: enhanced).
    """
    
    def __init__(self):
        """Initialize the model registry with available models."""
        # Get model configurations from centralized config
        self.model_configs = get_all_model_configs()
        
        # Map model keys to their implementation classes
        self.model_classes = {
            "model-a-dataset-a": ModelADatasetA,      # XLM-RoBERTa + Dataset A
            "model-b-dataset-a": ModelBDatasetA,      # BERT + Dataset A  
            "model-a-dataset-b": ModelADatasetB,      # XLM-RoBERTa + Dataset B
            "model-b-dataset-b": ModelBDatasetB,      # BERT + Dataset B
        }
        
        # Build models registry by combining configs with classes
        self.models = {}
        
        # Add the new organized models
        for model_key, config in self.model_configs.items():
            if model_key in self.model_classes:
                self.models[model_key] = {
                    "class": self.model_classes[model_key],
                    "display_name": config["display_name"],
                    "description": config["description"],
                    "status": config["status"]
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
    different model architectures and training datasets. It handles model 
    switching and provides consistent output formatting.
    """
    
    def __init__(self, model_key: str = "model-a-dataset-a"):
        """
        Initialize the language detector.
        
        Args:
            model_key (str): Key of the model to use from the registry
                - "model-a-dataset-a": XLM-RoBERTa + standard dataset
                - "model-b-dataset-a": BERT + standard dataset  
                - "model-a-dataset-b": XLM-RoBERTa + enhanced dataset
                - "model-b-dataset-b": BERT + enhanced dataset
        """
        self.registry = ModelRegistry()
        self.current_model_key = model_key
        self.model = self.registry.create_model(model_key)
        
        # Use centralized language mappings
        self.language_names = LANGUAGE_MAPPINGS
    
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
        
        # Get human-readable language name using centralized function
        main_language_name = get_language_name(main_language_code)
        
        # Format top predictions (limit to 5)
        top_predictions = []
        for pred in predictions[:5]:
            lang_code = pred['language_code']
            lang_name = get_language_name(lang_code)
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
            code: get_language_name(code)
            for code in supported_codes
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector with default model (Model A Dataset A)
    detector = LanguageDetector()
    
    # Test with sample texts
    test_texts = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?", 
        "Hola, ¿cómo estás?",
        "Guten Tag, wie geht es Ihnen?"
    ]
    
    print("Language Detection Test - Model A Dataset A")
    print("=" * 60)
    
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