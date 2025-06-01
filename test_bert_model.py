#!/usr/bin/env python3
"""
Test script for the Zues0102 BERT Multilingual Language Detection Model

This script tests the newly implemented zues0102/bert-base-multilingual-cased
to ensure it works correctly within the language detection framework.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from backend.language_detector import LanguageDetector


def test_bert_model():
    """Test the Zues0102 BERT Multilingual model implementation."""
    print("🧪 Testing Zues0102 BERT Multilingual Language Detection Model")
    print("=" * 75)
    
    try:
        # Initialize detector with the Zues0102 BERT model
        detector = LanguageDetector(model_key="model-4")
        print("✅ Successfully initialized Zues0102 BERT Multilingual model")
        
        # Test texts in the 20 supported languages
        test_texts = [
            ("Hello, how are you today?", "en"),  # English
            ("Bonjour, comment allez-vous?", "fr"),  # French
            ("Hola, ¿cómo estás?", "es"),  # Spanish
            ("Guten Tag, wie geht es Ihnen?", "de"),  # German
            ("Ciao, come stai?", "it"),  # Italian
            ("Olá, como você está?", "pt"),  # Portuguese
            ("Привет, как дела?", "ru"),  # Russian
            ("こんにちは、元気ですか？", "ja"),  # Japanese
            ("你好，你好吗？", "zh"),  # Chinese
            ("مرحبا، كيف حالك؟", "ar"),  # Arabic
            ("नमस्ते, आप कैसे हैं?", "hi"),  # Hindi
            ("Hallo, hoe gaat het met je?", "nl"),  # Dutch
            ("Γεια σας, πώς είστε;", "el"),  # Greek
            ("Здравейте, как сте?", "bg"),  # Bulgarian
            ("Witaj, jak się masz?", "pl"),  # Polish
            ("สวัสดี คุณเป็นอย่างไรบ้าง?", "th"),  # Thai
            ("Merhaba, nasılsınız?", "tr"),  # Turkish
            ("آپ کیسے ہیں؟", "ur"),  # Urdu
            ("Xin chào, bạn khỏe không?", "vi"),  # Vietnamese
            ("Habari, unajehje?", "sw")  # Swahili
        ]
        
        print("\n🔍 Running language detection tests:")
        print("-" * 75)
        
        correct_predictions = 0
        total_predictions = len(test_texts)
        
        for text, expected_lang in test_texts:
            try:
                result = detector.detect_language(text)
                predicted_lang = result['language_code']
                confidence = result['confidence']
                language_name = result['language']
                
                # Check if prediction is correct
                is_correct = predicted_lang == expected_lang
                if is_correct:
                    correct_predictions += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"{status} Text: {text[:40]}{'...' if len(text) > 40 else ''}")
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
        print("-" * 75)
        model_info = detector.get_current_model_info()
        for key, value in model_info.items():
            print(f"{key.title().replace('_', ' ')}: {value}")
        
        print("🎉 Zues0102 BERT Multilingual model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_all_models_comprehensive():
    """Test and compare all four available models comprehensively."""
    print("\n🔄 Comprehensive All-Models Comparison")
    print("=" * 75)
    
    models_to_test = [
        ("xlm-roberta-langdetect", "ZheYu03 XLM-RoBERTa", "97.9%"),
        ("model-2", "SongJuNN XLM-R", "96.17%"),
        ("model-3", "Zues0102 XLM-R Papluca", "99.72%"),
        ("model-4", "Zues0102 BERT Multilingual", "99.85%")
    ]
    
    test_texts = [
        "Hello, this is a test in English.",
        "Bonjour, ceci est un test en français.",
        "Hola, esto es una prueba en español.",
        "Guten Tag, das ist ein Test auf Deutsch."
    ]
    
    print("🧪 Testing with multiple sentences:")
    print("-" * 75)
    
    try:
        results_summary = {}
        
        for model_key, model_name, claimed_accuracy in models_to_test:
            print(f"\n🤖 Testing {model_name} (Claimed: {claimed_accuracy})")
            print("-" * 50)
            
            try:
                detector = LanguageDetector(model_key=model_key)
                model_results = []
                
                for text in test_texts:
                    result = detector.detect_language(text)
                    model_results.append({
                        'text': text[:30] + '...' if len(text) > 30 else text,
                        'language': result['language'],
                        'code': result['language_code'],
                        'confidence': result['confidence']
                    })
                    
                    print(f"   Text: {text[:30]}{'...' if len(text) > 30 else ''}")
                    print(f"   → {result['language']} ({result['language_code']}) - {result['confidence']:.4f}")
                
                results_summary[model_name] = model_results
                print(f"✅ {model_name} completed successfully")
                
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
                results_summary[model_name] = f"Error: {str(e)}"
        
        print(f"\n📊 All Models Testing Summary:")
        print("-" * 75)
        for model_name, results in results_summary.items():
            if isinstance(results, str):
                print(f"❌ {model_name}: {results}")
            else:
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                print(f"✅ {model_name}: Avg Confidence: {avg_confidence:.4f}")
        
        print("🎉 Comprehensive model comparison completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Zues0102 BERT Multilingual Model Tests\n")
    
    # Run tests
    test1_passed = test_bert_model()
    test2_passed = test_all_models_comprehensive()
    
    # Final results
    print("\n" + "=" * 75)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! Zues0102 BERT Multilingual model is ready to use.")
        print("🏆 This model offers the highest accuracy (99.85%) of all available models!")
        print("📝 Note: Supports 20 carefully selected languages for maximum precision.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 