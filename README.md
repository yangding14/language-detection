# 🌍 Language Detection App

A powerful and elegant language detection application built with Gradio frontend and a modular backend featuring multiple state-of-the-art ML models.

## ✨ Features

- **Clean Gradio Interface**: Simple, intuitive web interface for language detection
- **Multiple Models**: Choose between four different high-performance models
- **Modular Backend**: Easy-to-extend architecture for plugging in your own ML models
- **Real-time Detection**: Instant language detection with confidence scores
- **Multiple Predictions**: Shows top 5 language predictions with confidence levels
- **100+ Languages**: Support for major world languages (varies by model)
- **Example Texts**: Pre-loaded examples in various languages for testing
- **Model Switching**: Seamlessly switch between different models
- **Extensible**: Abstract base class for implementing custom models

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Backend

```bash
# Run tests to verify everything works
python test_app.py

# Test specific models
python test_zues_model.py
python test_bert_model.py
```

### 3. Launch the App

```bash
# Start the Gradio app
python app.py
```

The app will be available at `http://localhost:7860`

## 🧩 Modular Architecture

The backend is now organized into a clean, modular structure:

### 🏗️ Core Components

- **`BaseLanguageModel`**: Abstract interface that all models must implement
- **`ModelRegistry`**: Manages model registration and creation
- **`LanguageDetector`**: Main orchestrator for language detection

### 🤖 Available Models

1. **XLM-RoBERTa Language Detector** ✅
   - **Model**: [`ZheYu03/xlm-r-langdetect-model`](https://huggingface.co/ZheYu03/xlm-r-langdetect-model)
   - **Accuracy**: 97.9%
   - **Size**: 278M parameters
   - **Base**: xlm-roberta-base
   - **Languages**: 60+ languages
   - **Status**: Available and working

2. **SongJuNN XLM-R Language Detector** ✅
   - **Model**: [`SongJuNN/xlm-r-langdetect-model`](https://huggingface.co/SongJuNN/xlm-r-langdetect-model)
   - **Accuracy**: 96.17%
   - **Size**: 178M parameters
   - **Base**: bert-base-multilingual-cased
   - **Languages**: 100+ languages
   - **Status**: Available and working

3. **Zues0102 XLM-R Papluca Language Detector** ✅
   - **Model**: [`zues0102/xlmr-papluca-model`](https://huggingface.co/zues0102/xlmr-papluca-model)
   - **Accuracy**: 99.72%
   - **Size**: 278M parameters
   - **Base**: xlm-roberta-base
   - **Languages**: 100+ languages
   - **Status**: Available and working

4. **Zues0102 BERT Multilingual Language Detector** ✅
   - **Model**: [`zues0102/bert-base-multilingual-cased`](https://huggingface.co/zues0102/bert-base-multilingual-cased)
   - **Accuracy**: 99.85%
   - **Size**: ~178M parameters
   - **Base**: bert-base-multilingual-cased
   - **Languages**: 20 carefully selected high-performance languages
   - **Status**: Available and working

### 🔧 Adding New Models

To add a new model, simply:

1. Create a new file in `backend/models/`
2. Inherit from `BaseLanguageModel`
3. Implement the required methods
4. Register it in `ModelRegistry`

Example:
```python
# backend/models/your_new_model.py
from .base_model import BaseLanguageModel

class YourNewModel(BaseLanguageModel):
    def __init__(self):
        # Initialize your model
        pass
    
    def predict(self, text: str) -> Dict[str, Any]:
        # Implement prediction logic
        pass
    
    def get_supported_languages(self) -> List[str]:
        # Return supported language codes
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        # Return model metadata
        pass
```

Then add it to the registry in `language_detector.py`:
```python
from .models import YourNewModel

# In ModelRegistry.__init__:
"your-model-key": {
    "class": YourNewModel,
    "display_name": "Your Model Name",
    "description": "Your model description",
    "status": "available"
}
```

## 🧪 Testing

The project includes comprehensive test suites:

- **`test_app.py`**: General app functionality tests
- **`test_zues_model.py`**: Specific tests for the Zues0102 models with model comparison
- **`test_bert_model.py`**: Comprehensive tests for the BERT model with all 4 models comparison
- **Model accuracy tests**: Automated testing with multiple languages
- **Model switching tests**: Verify seamless model switching

## 🌐 Supported Languages

The models support different language sets:

- **XLM-RoBERTa & Zues0102 XLM-R Papluca**: 100+ languages including major European, Asian, African, and other world languages
- **SongJuNN XLM-R**: 100+ languages with comprehensive multilingual support
- **Zues0102 BERT Multilingual**: 20 carefully selected high-performance languages (Arabic, Bulgarian, German, Greek, English, Spanish, French, Hindi, Italian, Japanese, Dutch, Polish, Portuguese, Russian, Swahili, Thai, Turkish, Urdu, Vietnamese, Chinese)

## 📊 Model Comparison

| Feature | ZheYu03 XLM-RoBERTa | SongJuNN XLM-R | Zues0102 XLM-R Papluca | Zues0102 BERT Multilingual |
|---------|---------------------|----------------|------------------------|----------------------------|
| **Accuracy** | 97.9% | 96.17% | 99.72% | **99.85%** 🏆 |
| **Model Size** | 278M parameters | 178M parameters | 278M parameters | ~178M parameters |
| **Base Model** | xlm-roberta-base | bert-base-multilingual-cased | xlm-roberta-base | bert-base-multilingual-cased |
| **Languages** | 60+ | 100+ | 100+ | 20 (curated) |
| **Speed** | Moderate | **Faster** | Moderate | **Faster** |
| **Memory Usage** | Higher | **Lower** | Higher | **Lower** |
| **Training Loss** | N/A | N/A | 0.0176 | **0.0125** |
| **Architecture** | XLM-RoBERTa | XLM-RoBERTa | XLM-RoBERTa | **BERT** |
| **Best For** | Balanced performance | Speed & broad coverage | Ultra-high accuracy | **Maximum precision** |

### 🎯 Model Selection Guide

- **🏆 Zues0102 BERT Multilingual**: Choose for maximum accuracy on supported languages (20 languages)
- **🔬 Zues0102 XLM-R Papluca**: Choose for ultra-high accuracy with broad language support (100+ languages)
- **⚖️ ZheYu03 XLM-RoBERTa**: Choose for balanced performance and reliability
- **⚡ SongJuNN XLM-R**: Choose for faster inference and lower memory usage

## 🔧 Configuration

You can configure the default model and other settings in the code:

```python
# Default model selection
detector = LanguageDetector(model_key="xlm-roberta-langdetect")  # Balanced
detector = LanguageDetector(model_key="model-2")  # Fast (SongJuNN)
detector = LanguageDetector(model_key="model-3")  # Ultra-high accuracy (Zues0102 XLM-R)
detector = LanguageDetector(model_key="model-4")  # Maximum precision (Zues0102 BERT)

# GPU usage (change device parameter in model files)
device=0    # Use GPU
device=-1   # Use CPU (default for compatibility)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Implement your model following the `BaseLanguageModel` interface
4. Add tests for your implementation
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and model hosting
- **ZheYu03** for the XLM-RoBERTa language detection model
- **SongJuNN** for the fine-tuned XLM-R language detection model
- **Zues0102** for both the ultra high-accuracy XLM-R Papluca and BERT Multilingual language detection models
- **Gradio** for the excellent web interface framework