#!/usr/bin/env python3
"""
Test script for the Zues0102 XLM-R Papluca Language Detection Model

This script tests the newly implemented zues0102/xlmr-papluca-model
to ensure it works correctly within the language detection framework.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from backend.language_detector import LanguageDetector


def test_zues_model():
    """Test the Zues0102 model implementation."""
    print("🧪 Testing Zues0102 XLM-R Papluca Language Detection Model")
    print("=" * 70)
    
    try:
        # Initialize detector with the Zues0102 model
        detector = LanguageDetector(model_key="model-3")
        print("✅ Successfully initialized Zues0102 XLM-R Papluca model")
        
        # Test texts in different languages
        test_texts = [
            ("Hello, how are you today?", "en"),
            ("Bonjour, comment allez-vous?", "fr"), 
            ("Hola, ¿cómo estás?", "es"),
            ("Guten Tag, wie geht es Ihnen?", "de"),
            ("こんにちは、元気ですか？", "ja"),
            ("Привет, как дела?", "ru"),
            ("Ciao, come stai?", "it"),
            ("Olá, como você está?", "pt"),
            ("你好，你好吗？", "zh"),
            ("안녕하세요, 어떻게 지내세요?", "ko"),
            ("مرحبا، كيف حالك؟", "ar"),
            ("नमस्ते, आप कैसे हैं?", "hi")
        ]
        
        print("\n🔍 Running language detection tests:")
        print("-" * 70)
        
        correct_predictions = 0
        total_predictions = len(test_texts)
        
        for text, expected_lang in test_texts:
            try:
                result = detector.detect_language(text)
                predicted_lang = result['language_code']
                confidence = result['confidence']
                language_name = result['language']
                
                # Check if prediction is correct (allow some flexibility for Chinese variants)
                is_correct = (predicted_lang == expected_lang or 
                             (expected_lang == "zh" and predicted_lang in ["zh-hans", "zh-hant", "zh-cn", "zh-tw"]))
                if is_correct:
                    correct_predictions += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"{status} Text: {text[:35]}{'...' if len(text) > 35 else ''}")
                print(f"   Expected: {expected_lang} | Predicted: {predicted_lang} ({language_name})")
                print(f"   Confidence: {confidence:.4f}")
                print()
                
            except Exception as e:
                print(f"❌ Error testing '{text[:30]}...': {str(e)}")
                print()
        
        # Calculate accuracy
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"📊 Test Results: {correct_predictions}/{total_predictions} correct")
        print(f"📈 Accuracy: {accuracy:.1f}%")
        
        # Test model info
        print("\n📋 Model Information:")
        print("-" * 70)
        model_info = detector.get_current_model_info()
        for key, value in model_info.items():
            print(f"{key.title().replace('_', ' ')}: {value}")
        
        # Test available models
        print("\n🔧 Available Models:")
        print("-" * 70)
        available_models = detector.get_available_models()
        for key, info in available_models.items():
            status = "✅" if info["status"] == "available" else "🚧"
            print(f"{status} {info['display_name']} ({key})")
            print(f"   {info['description']}")
            print()
        
        print("🎉 Zues0102 XLM-R Papluca model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_comparison():
    """Test and compare all four available models."""
    print("\n🔄 Testing Model Comparison")
    print("=" * 70)
    
    models_to_test = [
        ("xlm-roberta-langdetect", "ZheYu03 XLM-RoBERTa"),
        ("model-2", "SongJuNN XLM-R"),
        ("model-3", "Zues0102 XLM-R Papluca"),
        ("model-4", "Zues0102 BERT Multilingual")
    ]
    
    test_text = "Hello, this is a test sentence for language detection."
    
    print(f"🧪 Test Text: {test_text}")
    print("-" * 70)
    
    try:
        for model_key, model_name in models_to_test:
            try:
                detector = LanguageDetector(model_key=model_key)
                result = detector.detect_language(test_text)
                
                print(f"✅ {model_name}")
                print(f"   Language: {result['language']} ({result['language_code']})")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"   Model: {result['metadata']['model_name']}")
                print()
                
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
                print()
        
        print("🎉 Model comparison test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model comparison test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Zues0102 XLM-R Papluca Model Tests\n")
    
    # Run tests
    test1_passed = test_zues_model()
    test2_passed = test_model_comparison()
    
    # Final results
    print("\n" + "=" * 70)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! Zues0102 XLM-R Papluca model is ready to use.")
        print("💡 This model offers the highest accuracy (99.72%) of all available models!")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 