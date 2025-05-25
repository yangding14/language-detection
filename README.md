# 🌍 Language Detection App

A simple and elegant language detection application built with Gradio frontend and a modular backend designed for easy integration of ML/NLP models.

## ✨ Features

- **Clean Gradio Interface**: Simple, intuitive web interface for language detection
- **Modular Backend**: Easy-to-extend architecture for plugging in your own ML models
- **Real-time Detection**: Instant language detection with confidence scores
- **Multiple Predictions**: Shows top 5 language predictions with confidence levels
- **Example Texts**: Pre-loaded examples in various languages for testing
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
   - **Model**: `ZheYu03/xlm-r-langdetect-model`
   - **Accuracy**: 97.9%
   - **Size**: 278M parameters
   - **Status**: Available and working

2. **Language Model 2** 🚧
   - **Status**: Template ready for implementation
   - **Suggested**: `papluca/xlm-roberta-base-language-detection`

3. **Language Model 3** 🚧
   - **Status**: Template ready for implementation  
   - **Suggested**: Custom domain-specific models

4. **Language Model 4** 🚧
   - **Status**: Template ready for implementation
   - **Suggested**: Lightweight/fast models

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