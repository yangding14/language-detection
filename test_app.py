#!/usr/bin/env python3
"""
Test script for the Language Detection App

Run this script to test the backend functionality before launching the full app.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.language_detector import LanguageDetector, PlaceholderModel
from config import EXAMPLE_TEXTS

def test_language_detector():
    """Test the language detector with sample texts."""
    
    print("🧪 Testing Language Detection Backend")
    print("=" * 50)
    
    # Initialize detector
    detector = LanguageDetector()
    
    # Test with example texts
    test_texts = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?", 
        "Guten Tag, wie geht es Ihnen?",
        "こんにちは、元気ですか？",
        "Привет, как дела?",
        "This is a longer text to test the language detection capabilities of our system."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        print("-" * 40)
        
        try:
            result = detector.detect_language(text)
            
            print(f"🎯 Detected Language: {result['language']}")
            print(f"🏷️  Language Code: {result['language_code']}")
            print(f"📊 Confidence: {result['confidence']:.4f}")
            print(f"📈 Top Predictions:")
            
            for j, pred in enumerate(result['top_predictions'][:3], 1):
                print(f"   {j}. {pred['language']} ({pred['language_code']}) - {pred['confidence']:.4f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test supported languages
    print(f"\n🌍 Supported Languages: {len(detector.get_supported_languages())}")
    
    # Test edge cases
    print(f"\n🔍 Testing Edge Cases")
    print("-" * 40)
    
    edge_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        "123456",  # Numbers only
        "!@#$%^&*()",  # Special characters only
        "a",  # Single character
    ]
    
    for case in edge_cases:
        try:
            result = detector.detect_language(case)
            print(f"✅ '{case}' -> {result['language']} ({result['confidence']:.4f})")
        except Exception as e:
            print(f"⚠️  '{case}' -> Error: {e}")

def test_custom_model_interface():
    """Test the custom model interface."""
    
    print(f"\n🔧 Testing Custom Model Interface")
    print("=" * 50)
    
    # Test placeholder model directly
    model = PlaceholderModel()
    
    print(f"📋 Supported languages: {len(model.get_supported_languages())}")
    
    # Test prediction
    test_text = "Hello world"
    result = model.predict(test_text)
    
    print(f"📝 Test text: '{test_text}'")
    print(f"🔍 Model predictions: {len(result['predictions'])}")
    print(f"📊 Top prediction: {result['predictions'][0]}")

def main():
    """Run all tests."""
    
    print("🚀 Language Detection App - Test Suite")
    print("=" * 60)
    
    try:
        test_language_detector()
        test_custom_model_interface()
        
        print(f"\n✅ All tests completed successfully!")
        print(f"🎉 You can now run the app with: python app.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 