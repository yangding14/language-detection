"""
Centralized Model Configuration

This module contains the configuration for all language detection models
organized by the model architecture and training dataset combinations.

Model Architecture:
- Model A: XLM-RoBERTa based architectures
- Model B: BERT based architectures

Training Datasets:
- Dataset A: Standard multilingual language detection dataset
- Dataset B: Enhanced/specialized language detection dataset
"""

from typing import Dict, Any, List

# Model configurations organized by architecture and dataset
MODEL_CONFIGURATIONS = {
    "model-a-dataset-a": {
        "huggingface_model": "ZheYu03/xlm-r-langdetect-model",
        "display_name": "XLM-RoBERTa Model A Dataset A",
        "short_name": "Model A Dataset A",
        "architecture": "XLM-RoBERTa",
        "base_model": "xlm-roberta-base",
        "dataset": "Dataset A",
        "accuracy": "97.9%",
        "model_size": "278M parameters",
        "training_epochs": 10,
        "languages_supported": 60,
        "description": "High-performance XLM-RoBERTa based language detection model fine-tuned on standard multilingual dataset. Delivers reliable 97.9% accuracy across 60+ languages with robust cross-lingual capabilities.",
        "training_details": "Fine-tuned XLM-RoBERTa base model with AdamW optimizer, 10 epochs training on comprehensive multilingual language detection dataset",
        "use_cases": "General-purpose language detection, multilingual content processing, cross-lingual text analysis",
        "strengths": "Excellent multilingual performance, robust cross-lingual transfer, proven reliability",
        "limitations": "Higher computational requirements, moderate inference speed",
        "status": "available"
    },
    
    "model-b-dataset-a": {
        "huggingface_model": "SongJuNN/xlm-r-langdetect-model",
        "display_name": "BERT Model B Dataset A", 
        "short_name": "Model B Dataset A",
        "architecture": "BERT",
        "base_model": "bert-base-multilingual-cased",
        "dataset": "Dataset A",
        "accuracy": "96.17%",
        "model_size": "178M parameters",
        "training_epochs": 10,
        "languages_supported": 100,
        "description": "Efficient BERT-based language detection model trained on standard multilingual dataset. Optimized for speed and broad language coverage with 96.17% accuracy across 100+ languages.",
        "training_details": "BERT multilingual model fine-tuned with AdamW optimizer (lr=2e-05), mixed precision training, optimized for efficiency",
        "use_cases": "High-throughput language detection, real-time applications, resource-constrained environments",
        "strengths": "Fast inference speed, lower memory usage, broad language support, efficient processing",
        "limitations": "Slightly lower accuracy compared to XLM-RoBERTa variants",
        "status": "available"
    },
    
    "model-a-dataset-b": {
        "huggingface_model": "zues0102/xlmr-papluca-model",
        "display_name": "XLM-RoBERTa Model A Dataset B",
        "short_name": "Model A Dataset B", 
        "architecture": "XLM-RoBERTa",
        "base_model": "xlm-roberta-base",
        "dataset": "Dataset B",
        "accuracy": "99.72%",
        "model_size": "278M parameters", 
        "training_epochs": 10,
        "training_loss": 0.0176,
        "languages_supported": 20,
        "description": "Ultra high-accuracy XLM-RoBERTa model fine-tuned on enhanced dataset. Achieves exceptional 99.72% accuracy with support for 20 carefully selected high-performance languages and state-of-the-art performance.",
        "training_details": "Advanced fine-tuning of XLM-RoBERTa on enhanced dataset with specialized training procedures, achieving loss of 0.0176",
        "use_cases": "Research applications, high-precision language detection, critical accuracy requirements",
        "strengths": "Exceptional accuracy (99.72%), focused language support, state-of-the-art results",
        "limitations": "Higher computational requirements, limited to 20 languages",
        "status": "available"
    },
    
    "model-b-dataset-b": {
        "huggingface_model": "zues0102/bert-base-multilingual-cased",
        "display_name": "BERT Model B Dataset B",
        "short_name": "Model B Dataset B",
        "architecture": "BERT", 
        "base_model": "bert-base-multilingual-cased",
        "dataset": "Dataset B",
        "accuracy": "99.85%",
        "model_size": "178M parameters",
        "training_epochs": 10,
        "training_loss": 0.0125,
        "languages_supported": 20,
        "description": "State-of-the-art BERT model achieving highest accuracy (99.85%) through specialized training on enhanced dataset. Optimized for 20 carefully selected high-performance languages.",
        "training_details": "Precision-optimized BERT training on enhanced dataset achieving ultra-low loss of 0.0125, specialized for maximum accuracy",
        "use_cases": "Maximum precision applications, research requiring highest accuracy, critical language identification",
        "strengths": "Highest accuracy (99.85%), ultra-low training loss, optimized precision, efficient architecture",
        "limitations": "Limited to 20 languages, specialized for specific language set",
        "status": "available"
    }
}

# Language mappings - comprehensive set
LANGUAGE_MAPPINGS = {
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

# Model-specific language support
MODEL_LANGUAGE_SUPPORT = {
    "model-a-dataset-a": [
        'af', 'ar', 'bg', 'bn', 'ca', 'cs', 'cy', 'da', 'de', 'el',
        'en', 'es', 'et', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hr',
        'hu', 'id', 'it', 'ja', 'kn', 'ko', 'lt', 'lv', 'mk', 'ml',
        'mr', 'ne', 'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'sk',
        'sl', 'so', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr',
        'uk', 'ur', 'vi', 'zh'
    ],
    
    "model-b-dataset-a": [
        'af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 
        'cs', 'cy', 'da', 'de', 'dz', 'el', 'en', 'eo', 'es', 'et', 'eu', 
        'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 
        'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 
        'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 
        'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'nb', 'ne', 'nl', 'nn', 
        'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'qu', 'ro', 'ru', 'rw', 
        'se', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 
        'th', 'tl', 'tr', 'ug', 'uk', 'ur', 'vi', 'vo', 'wa', 'xh', 'yi', 
        'yo', 'zh', 'zu'
    ],
    
    "model-a-dataset-b": [
        'ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'it', 'ja',
        'nl', 'pl', 'pt', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'
    ],
    
    "model-b-dataset-b": [
        'ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'it', 'ja',
        'nl', 'pl', 'pt', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'
    ]
}

def get_model_config(model_key: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODEL_CONFIGURATIONS.get(model_key, {})

def get_all_model_configs() -> Dict[str, Dict[str, Any]]:
    """Get all model configurations."""
    return MODEL_CONFIGURATIONS.copy()

def get_supported_languages(model_key: str) -> List[str]:
    """Get supported languages for a specific model."""
    return MODEL_LANGUAGE_SUPPORT.get(model_key, [])

def get_language_name(language_code: str) -> str:
    """Get human-readable language name from code."""
    return LANGUAGE_MAPPINGS.get(language_code.lower(), f"Unknown ({language_code})") 