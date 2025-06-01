#!/usr/bin/env python3
"""
Test script for the demo functionality
"""

from demo_page import initialize_models, detect_with_all_models, create_results_dataframe, run_demo_tests, DEMO_SAMPLES

def test_model_initialization():
    """Test that all models can be initialized."""
    print("🔄 Testing model initialization...")
    models = initialize_models()
    
    print(f"✅ Initialized {len(models)} models:")
    for model_key, model_info in models.items():
        status_icon = "✅" if model_info["status"] == "Ready" else "❌"
        print(f"  {status_icon} {model_info['name']}: {model_info['status']}")
    
    return models

def test_single_detection():
    """Test detection with a single text across all models."""
    print("\n🔄 Testing single text detection...")
    
    models = initialize_models()
    test_text = "Hello, how are you today?"
    
    results = detect_with_all_models(test_text, models)
    
    print(f"Text: '{test_text}'")
    print("Results:")
    for model_key, result in results.items():
        print(f"  {model_key}: {result['language_code']} ({result['confidence']:.3f}) - {result['status']}")
    
    return results

def test_category_samples():
    """Test a few samples from each category."""
    print("\n🔄 Testing category samples...")
    
    models = initialize_models()
    
    for category, samples in DEMO_SAMPLES.items():
        print(f"\n📊 Category: {category}")
        # Test first sample from each category
        text, expected, description = samples[0]
        results = detect_with_all_models(text, models)
        
        print(f"  Text: '{text}' (Expected: {expected})")
        print(f"  Description: {description}")
        for model_key, result in results.items():
            match_icon = "✅" if result['language_code'] == expected or expected in ['ambiguous', 'mix', 'transliteration'] else "❌"
            print(f"    {model_key}: {result['language_code']} ({result['confidence']:.3f}) {match_icon}")

def test_dataframe_creation():
    """Test DataFrame creation with sample data."""
    print("\n🔄 Testing DataFrame creation...")
    
    models = initialize_models()
    
    # Test with a few samples
    test_texts = [
        "Hello world",
        "Bonjour le monde", 
        "Hola mundo"
    ]
    expected_langs = ["en", "fr", "es"]
    categories = ["Custom", "Custom", "Custom"]
    
    all_results = []
    for text in test_texts:
        results = detect_with_all_models(text, models)
        all_results.append(results)
    
    df = create_results_dataframe(test_texts, all_results, expected_langs, categories)
    
    print("DataFrame shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def test_demo_workflow():
    """Test the complete demo workflow."""
    print("\n🔄 Testing complete demo workflow...")
    
    models = initialize_models()
    
    # Test with selected categories and custom text
    selected_categories = ["Easy/Obvious", "Short Text"]
    custom_texts = "Hello world\nBonjour\n你好"
    
    summary, df = run_demo_tests(selected_categories, custom_texts, models)
    
    print(f"Summary: {summary}")
    if df is not None:
        print(f"Results DataFrame shape: {df.shape}")
        print("Sample results:")
        print(df.head())
    else:
        print("❌ No DataFrame returned")
    
    return summary, df

def main():
    """Run all tests."""
    print("🚀 Starting demo functionality tests...\n")
    
    try:
        # Test 1: Model initialization
        models = test_model_initialization()
        
        # Test 2: Single detection
        single_results = test_single_detection()
        
        # Test 3: Category samples
        test_category_samples()
        
        # Test 4: DataFrame creation
        df = test_dataframe_creation()
        
        # Test 5: Complete workflow
        summary, demo_df = test_demo_workflow()
        
        print("\n✅ All tests completed successfully!")
        print(f"📊 Total categories available: {len(DEMO_SAMPLES)}")
        print(f"📝 Total sample texts: {sum(len(samples) for samples in DEMO_SAMPLES.values())}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 