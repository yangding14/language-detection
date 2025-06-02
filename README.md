# 🌍 Language Detection App

A powerful and elegant language detection application built with Gradio frontend and a modular backend featuring multiple state-of-the-art ML models organized by architecture and training dataset.

## ✨ Features

- **Clean Gradio Interface**: Simple, intuitive web interface for language detection
- **Multiple Model Architectures**: Choose between XLM-RoBERTa (Model A) and BERT (Model B) architectures
- **Multiple Training Datasets**: Models trained on standard (Dataset A) and enhanced (Dataset B) datasets
- **Centralized Configuration**: All model configurations and settings in one place
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

# Test specific model combinations
python test_model_a_dataset_a.py
python test_model_b_dataset_b.py
```

### 3. Launch the App

```bash
# Start the Gradio app
python app.py
```

The app will be available at `http://localhost:7860`

## 🧩 Model Architecture

The system is organized around two dimensions:

### 🏗️ Model Architectures
- **Model A**: XLM-RoBERTa based architectures - Excellent cross-lingual transfer capabilities
- **Model B**: BERT based architectures - Efficient and fast processing

### 📊 Training Datasets  
- **Dataset A**: Standard multilingual language detection dataset - Broad language coverage
- **Dataset B**: Enhanced/specialized language detection dataset - Ultra-high accuracy focus

### 🤖 Available Model Combinations

1. **Model A Dataset A** - XLM-RoBERTa + Standard Dataset ✅
   - **Architecture**: XLM-RoBERTa (Model A)
   - **Training**: Dataset A (standard multilingual)
   - **Accuracy**: 97.9%
   - **Size**: 278M parameters
   - **Languages**: 60+ languages
   - **Strengths**: Balanced performance, robust cross-lingual capabilities
   - **Use Cases**: General-purpose language detection, multilingual content processing

2. **Model B Dataset A** - BERT + Standard Dataset ✅
   - **Architecture**: BERT (Model B)
   - **Training**: Dataset A (standard multilingual)
   - **Accuracy**: 96.17%
   - **Size**: 178M parameters
   - **Languages**: 100+ languages
   - **Strengths**: Fast inference, broad language support, efficient processing
   - **Use Cases**: High-throughput detection, real-time applications, resource-constrained environments

3. **Model A Dataset B** - XLM-RoBERTa + Enhanced Dataset ✅
   - **Architecture**: XLM-RoBERTa (Model A)
   - **Training**: Dataset B (enhanced/specialized)
   - **Accuracy**: 99.72%
   - **Size**: 278M parameters
   - **Training Loss**: 0.0176
   - **Languages**: 20 carefully selected languages
   - **Strengths**: Exceptional accuracy, focused language support, state-of-the-art results
   - **Use Cases**: Research applications, high-precision detection, critical accuracy requirements

4. **Model B Dataset B** - BERT + Enhanced Dataset ✅
   - **Architecture**: BERT (Model B)
   - **Training**: Dataset B (enhanced/specialized)
   - **Accuracy**: 99.85%
   - **Size**: 178M parameters
   - **Training Loss**: 0.0125
   - **Languages**: 20 carefully selected languages
   - **Strengths**: Highest accuracy, ultra-low training loss, precision-optimized
   - **Use Cases**: Maximum precision applications, research requiring highest accuracy

### 🏗️ Core Components

- **`BaseLanguageModel`**: Abstract interface that all models must implement
- **`ModelRegistry`**: Manages model registration and creation with centralized configuration
- **`LanguageDetector`**: Main orchestrator for language detection
- **`model_config.py`**: Centralized configuration for all models and language mappings

### 🔧 Adding New Models

To add a new model combination, simply:

1. Create a new file in `backend/models/` (e.g., `model_c_dataset_a.py`)
2. Inherit from `BaseLanguageModel`
3. Implement the required methods
4. Add configuration to `model_config.py`
5. Register it in `ModelRegistry`

Example:
```python
# backend/models/model_c_dataset_a.py
from .base_model import BaseLanguageModel
from .model_config import get_model_config

class ModelCDatasetA(BaseLanguageModel):
    def __init__(self):
        self.model_key = "model-c-dataset-a"
        self.config = get_model_config(self.model_key)
        # Initialize your model
    
    def predict(self, text: str) -> Dict[str, Any]:
        # Implement prediction logic
        pass
    
    def get_supported_languages(self) -> List[str]:
        # Return supported language codes
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        # Return model metadata from config
        pass
```

Then add configuration in `model_config.py` and register in `language_detector.py`.

## 🧪 Testing

The project includes comprehensive test suites:

- **`test_app.py`**: General app functionality tests
- **`test_model_a_dataset_a.py`**: Tests for XLM-RoBERTa + standard dataset
- **`test_model_b_dataset_b.py`**: Tests for BERT + enhanced dataset (highest accuracy)
- **Model comparison tests**: Automated testing across all model combinations
- **Model switching tests**: Verify seamless model switching

## 🌐 Supported Languages

The models support different language sets based on their training:

- **Model A/B + Dataset A**: 60-100+ languages including major European, Asian, African, and other world languages
- **Model A/B + Dataset B**: 20 carefully selected high-performance languages (Arabic, Bulgarian, German, Greek, English, Spanish, French, Hindi, Italian, Japanese, Dutch, Polish, Portuguese, Russian, Swahili, Thai, Turkish, Urdu, Vietnamese, Chinese)

## 📊 Model Comparison

| Feature | Model A Dataset A | Model B Dataset A | Model A Dataset B | Model B Dataset B |
|---------|-------------------|-------------------|-------------------|-------------------|
| **Architecture** | XLM-RoBERTa | BERT | XLM-RoBERTa | BERT |
| **Dataset** | Standard | Standard | Enhanced | Enhanced |
| **Accuracy** | 97.9% | 96.17% | 99.72% | **99.85%** 🏆 |
| **Model Size** | 278M | 178M | 278M | 178M |
| **Languages** | 60+ | 100+ | 20 (curated) | 20 (curated) |
| **Training Loss** | N/A | N/A | 0.0176 | **0.0125** |
| **Speed** | Moderate | **Fast** | Moderate | **Fast** |
| **Memory Usage** | Higher | **Lower** | Higher | **Lower** |
| **Best For** | Balanced performance | Speed & broad coverage | Ultra-high accuracy | **Maximum precision** |

### 🎯 Model Selection Guide

- **🏆 Model B Dataset B**: Choose for maximum accuracy on 20 core languages (99.85%)
- **🔬 Model A Dataset B**: Choose for ultra-high accuracy on 20 core languages (99.72%)
- **⚖️ Model A Dataset A**: Choose for balanced performance and reliability (97.9%)
- **⚡ Model B Dataset A**: Choose for fast inference and broad language coverage (96.17%)

## 🔧 Configuration

You can configure models using the centralized configuration system:

```python
# Default model selection
detector = LanguageDetector(model_key="model-a-dataset-a")  # Balanced XLM-RoBERTa
detector = LanguageDetector(model_key="model-b-dataset-a")  # Fast BERT
detector = LanguageDetector(model_key="model-a-dataset-b")  # Ultra-high accuracy XLM-RoBERTa
detector = LanguageDetector(model_key="model-b-dataset-b")  # Maximum precision BERT

# All configurations are centralized in backend/models/model_config.py
```

## 📁 Project Structure

```
language-detection/
├── backend/
│   ├── models/
│   │   ├── model_config.py          # Centralized configuration
│   │   ├── base_model.py            # Abstract base class
│   │   ├── model_a_dataset_a.py     # XLM-RoBERTa + Standard
│   │   ├── model_b_dataset_a.py     # BERT + Standard
│   │   ├── model_a_dataset_b.py     # XLM-RoBERTa + Enhanced
│   │   ├── model_b_dataset_b.py     # BERT + Enhanced
│   │   └── __init__.py
│   └── language_detector.py         # Main orchestrator
├── tests/
├── app.py                           # Gradio interface
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-model-combination`)
3. Implement your model following the `BaseLanguageModel` interface
4. Add configuration to `model_config.py`
5. Add tests for your implementation
6. Commit your changes (`git commit -m 'Add new model combination'`)
7. Push to the branch (`git push origin feature/new-model-combination`)
8. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and model hosting platform
- **Model providers** for the fine-tuned language detection models used in this project
- **Gradio** for the excellent web interface framework
- **Open source community** for the foundational technologies that make this project possible