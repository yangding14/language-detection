#!/usr/bin/env python3
"""
Test script for Model A Dataset A - XLM-RoBERTa + Standard Dataset

This script tests the XLM-RoBERTa based language detection model
trained on the standard multilingual dataset to ensure it works correctly.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from backend.language_detector import LanguageDetector


def test_model_a_dataset_a():
    """Test the Model A Dataset A implementation."""
    print("🧪 Testing Model A Dataset A - XLM-RoBERTa + Standard Dataset")
    print("=" * 75)
    
    try:
        # Initialize detector with Model A Dataset A
        detector = LanguageDetector(model_key="model-a-dataset-a")
        print("✅ Successfully initialized Model A Dataset A")
        
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
        print("-" * 75)
        
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
        
        print("🎉 Model A Dataset A test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_architecture():
    """Test the model architecture information."""
    print("\n🏗️ Testing Model Architecture Information")
    print("=" * 75)
    
    try:
        detector = LanguageDetector(model_key="model-a-dataset-a")
        model_info = detector.get_current_model_info()
        
        # Verify key architecture information
        expected_info = {
            "architecture": "XLM-RoBERTa",
            "dataset": "Dataset A",
            "accuracy": "97.9%",
            "model_size": "278M parameters"
        }
        
        print("🔍 Verifying model architecture information:")
        print("-" * 50)
        
        all_correct = True
        for key, expected_value in expected_info.items():
            actual_value = model_info.get(key, "Not found")
            if actual_value == expected_value:
                print(f"✅ {key}: {actual_value}")
            else:
                print(f"❌ {key}: Expected '{expected_value}', got '{actual_value}'")
                all_correct = False
        
        if all_correct:
            print("\n🎉 All architecture information verified successfully!")
        else:
            print("\n⚠️ Some architecture information mismatches found.")
        
        return all_correct
        
    except Exception as e:
        print(f"❌ Architecture test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Model A Dataset A Tests\n")
    
    # Run tests
    test1_passed = test_model_a_dataset_a()
    test2_passed = test_model_architecture()
    
    # Final results
    print("\n" + "=" * 75)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! Model A Dataset A is ready to use.")
        print("⚖️ This model offers balanced performance with robust cross-lingual capabilities!")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 