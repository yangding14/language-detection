import gradio as gr
import pandas as pd
from backend.language_detector import LanguageDetector
from typing import List, Dict, Any
import time

# Sample text database organized by difficulty categories
DEMO_SAMPLES = {
    "Easy/Obvious": [
        ("Hello, how are you doing today?", "en", "Clear English sentence"),
        ("Bonjour, comment allez-vous aujourd'hui?", "fr", "Clear French sentence"), 
        ("Hola, ¿cómo estás hoy?", "es", "Clear Spanish sentence"),
        ("Guten Tag, wie geht es Ihnen heute?", "de", "Clear German sentence"),
        ("こんにちは、今日はどうですか？", "ja", "Clear Japanese sentence"),
        ("Привет, как дела сегодня?", "ru", "Clear Russian sentence"),
        ("Ciao, come stai oggi?", "it", "Clear Italian sentence"),
        ("Olá, como você está hoje?", "pt", "Clear Portuguese sentence"),
        ("你好，你今天怎么样？", "zh", "Clear Chinese sentence"),
        ("안녕하세요, 오늘 어떻게 지내세요?", "ko", "Clear Korean sentence"),
    ],
    
    "Short Text": [
        ("Hi", "en", "Very short greeting"),
        ("Oui", "fr", "Single word French"),
        ("Sí", "es", "Single word Spanish"),
        ("Ja", "de", "Single word German"),
        ("はい", "ja", "Single word Japanese"),
        ("Да", "ru", "Single word Russian"),
        ("Sì", "it", "Single word Italian"),
        ("Sim", "pt", "Single word Portuguese"),
        ("是", "zh", "Single character Chinese"),
        ("네", "ko", "Single word Korean"),
    ],
    
    "False Friends": [
        ("actual", "en", "English word, but means 'current' in Spanish"),
        ("embarazada", "es", "Spanish for pregnant, not embarrassed"),
        ("gift", "en", "English word, but means 'poison' in German"),
        ("preservativo", "es", "Spanish for condom, not preservative"),
        ("sensible", "en", "English word, but means 'sensitive' in Spanish"),
        ("sympathique", "fr", "French for nice, not sympathetic"),
        ("biblioteca", "es", "Spanish for library, not Bible place"),
        ("realizzare", "it", "Italian for to achieve, not realize"),
        ("parents", "en", "English word, but means 'relatives' in French"),
        ("attualmente", "it", "Italian for currently, not actually"),
    ],
    
    "Mixed Scripts": [
        ("Hello123世界", "mix", "Mixed English, numbers, Chinese"),
        ("Café #1 في العالم", "mix", "Mixed French, numbers, Arabic"),
        ("2023年は良い年です", "ja", "Japanese with numbers"),
        ("Prix: €50,000", "fr", "French with currency and numbers"),
        ("iPhone 15 Pro Max", "en", "Product name with numbers"),
        ("COVID-19 パンデミック", "mix", "Mixed English acronym and Japanese"),
        ("Wi-Fi пароль: 123456", "mix", "Mixed English tech term and Russian"),
        ("GPS координаты", "mix", "Mixed English acronym and Russian"),
        ("URL: https://example.com", "en", "Web address"),
        ("HTML <div>content</div>", "en", "Code with markup"),
    ],
    
    "Proper Nouns": [
        ("Paris", "ambiguous", "City name - French or English context?"),
        ("Berlin", "ambiguous", "City name - German or English context?"),
        ("Madrid", "ambiguous", "City name - Spanish or English context?"),
        ("Tokyo", "ambiguous", "City name - Japanese or English context?"),
        ("Maria", "ambiguous", "Common name in many languages"),
        ("Alexander", "ambiguous", "Name used in many languages"),
        ("David", "ambiguous", "Biblical name used worldwide"),
        ("Anna", "ambiguous", "Name common across languages"),
        ("Michael", "ambiguous", "International name"),
        ("Sofia", "ambiguous", "Name and city, multiple languages"),
    ],
    
    "Common Words": [
        ("hotel", "ambiguous", "Same spelling in many languages"),
        ("restaurant", "ambiguous", "French origin, used worldwide"),
        ("taxi", "ambiguous", "International word"),
        ("pizza", "ambiguous", "Italian origin, used worldwide"),
        ("chocolate", "ambiguous", "Similar in many languages"),
        ("hospital", "ambiguous", "Medical term used internationally"),
        ("radio", "ambiguous", "Technology term used worldwide"),
        ("metro", "ambiguous", "Transportation term"),
        ("cafe", "ambiguous", "French origin, international use"),
        ("photo", "ambiguous", "Greek origin, used worldwide"),
    ],
    
    "Technical Terms": [
        ("algorithm", "en", "Technical English term"),
        ("algorithme", "fr", "Technical French term"),
        ("algoritmo", "es", "Technical Spanish term"),
        ("Algorithmus", "de", "Technical German term"),
        ("アルゴリズム", "ja", "Technical Japanese term"),
        ("алгоритм", "ru", "Technical Russian term"),
        ("algoritmo", "it", "Technical Italian term"),
        ("algoritmo", "pt", "Technical Portuguese term"),
        ("算法", "zh", "Technical Chinese term"),
        ("알고리즘", "ko", "Technical Korean term"),
    ],
    
    "Code-switching": [
        ("I love sushi とても美味しい", "mix", "English-Japanese code switching"),
        ("C'est très nice aujourd'hui", "mix", "French-English code switching"),
        ("Me gusta this song mucho", "mix", "Spanish-English code switching"),
        ("Das ist very interessant", "mix", "German-English code switching"),
        ("Это really хорошо", "mix", "Russian-English code switching"),
        ("È molto beautiful oggi", "mix", "Italian-English code switching"),
        ("Está muito good today", "mix", "Portuguese-English code switching"),
        ("这个 is very 好", "mix", "Chinese-English code switching"),
        ("이것은 really 좋다", "mix", "Korean-English code switching"),
        ("Merci beaucoup for everything", "mix", "French-English code switching"),
    ],
    
    "Transliterated Text": [
        ("Konnichiwa", "transliteration", "Japanese こんにちは in Latin script"),
        ("Spasibo", "transliteration", "Russian спасибо in Latin script"),
        ("Arigato", "transliteration", "Japanese ありがとう in Latin script"),
        ("Privyet", "transliteration", "Russian привет in Latin script"),
        ("Sayonara", "transliteration", "Japanese さようなら in Latin script"),
        ("Dosvedanya", "transliteration", "Russian до свидания in Latin script"),
        ("Nihao", "transliteration", "Chinese 你好 in Latin script"),
        ("Annyeonghaseyo", "transliteration", "Korean 안녕하세요 in Latin script"),
        ("Zdravstvuyte", "transliteration", "Russian здравствуйте in Latin script"),
        ("Ohayo gozaimasu", "transliteration", "Japanese おはようございます in Latin script"),
    ],
    
    "Ambiguous Script": [
        ("casa", "ambiguous", "House in Spanish/Italian/Portuguese"),
        ("rose", "ambiguous", "Flower in English or pink in French"),
        ("more", "ambiguous", "English word or Italian 'deaths'"),
        ("come", "ambiguous", "English verb or Italian 'how/like'"),
        ("no", "ambiguous", "English word or Spanish 'no'"),
        ("si", "ambiguous", "Spanish 'if' or Italian 'yes'"),
        ("la", "ambiguous", "English 'la' or French/Spanish/Italian article"),
        ("me", "ambiguous", "English pronoun or Spanish 'me'"),
        ("le", "ambiguous", "French article or Italian article"),
        ("son", "ambiguous", "English word or Spanish 'they are'"),
    ]
}

def initialize_models():
    """Initialize all four models for comparison."""
    models = {}
    model_configs = [
        ("model-a-dataset-a", "Model A Dataset A"),
        ("model-b-dataset-a", "Model B Dataset A"), 
        ("model-a-dataset-b", "Model A Dataset B"),
        ("model-b-dataset-b", "Model B Dataset B")
    ]
    
    for model_key, model_name in model_configs:
        try:
            models[model_key] = {
                "detector": LanguageDetector(model_key=model_key),
                "name": model_name,
                "status": "Ready"
            }
        except Exception as e:
            models[model_key] = {
                "detector": None,
                "name": model_name,
                "status": f"Error: {str(e)}"
            }
    
    return models

def detect_with_all_models(text: str, models: Dict) -> Dict[str, Any]:
    """Run language detection with all models and return results."""
    results = {}
    
    for model_key, model_info in models.items():
        if model_info["detector"] is None:
            results[model_key] = {
                "language": "Error",
                "confidence": 0.0,
                "language_code": "error",
                "status": model_info["status"]
            }
        else:
            try:
                result = model_info["detector"].detect_language(text)
                results[model_key] = {
                    "language": result["language"],
                    "confidence": result["confidence"],
                    "language_code": result["language_code"],
                    "status": "Success"
                }
            except Exception as e:
                results[model_key] = {
                    "language": "Error",
                    "confidence": 0.0,
                    "language_code": "error",
                    "status": f"Error: {str(e)}"
                }
    
    return results

def create_results_dataframe(texts: List[str], all_results: List[Dict], expected_langs: List[str] = None, categories: List[str] = None) -> pd.DataFrame:
    """Create a pandas DataFrame for results display."""
    data = []
    
    for i, (text, results) in enumerate(zip(texts, all_results)):
        row = {
            "Text": text[:40] + "..." if len(text) > 40 else text,  # Shortened text display
            "Expected": expected_langs[i] if expected_langs else "N/A",
            "Category": categories[i] if categories else "Custom"
        }
        
        expected_lang = expected_langs[i] if expected_langs else None
        
        # Add results from each model - combine language and confidence
        for model_key, result in results.items():
            # Shortened model names
            if model_key == "model-a-dataset-a":
                col_name = "A-A"
            elif model_key == "model-b-dataset-a":
                col_name = "B-A"  
            elif model_key == "model-a-dataset-b":
                col_name = "A-B"
            elif model_key == "model-b-dataset-b":
                col_name = "B-B"
            else:
                col_name = model_key[:6]
            
            # Determine if prediction is correct
            predicted_lang = result['language_code']
            is_correct = False
            
            if expected_lang and expected_lang not in ['ambiguous', 'mix', 'transliteration', 'unknown', 'N/A']:
                # For specific expected languages, check exact match
                is_correct = predicted_lang == expected_lang
                emoji = "✅" if is_correct else "🚫"
            else:
                # For ambiguous/mixed/transliterated/unknown cases, don't show emoji
                emoji = ""
            
            # Combine emoji, language code and confidence in one column
            if emoji:
                row[col_name] = f"{emoji} {predicted_lang} ({result['confidence']:.3f})"
            else:
                row[col_name] = f"{predicted_lang} ({result['confidence']:.3f})"
        
        data.append(row)
    
    return pd.DataFrame(data)

def run_demo_tests(selected_categories: List[str], custom_texts: str, models: Dict):
    """Run tests on selected categories and custom texts."""
    if not selected_categories and not custom_texts.strip():
        return "Please select at least one category or enter custom text.", None
    
    all_texts = []
    expected_langs = []
    categories = []
    
    # Add selected category samples
    for category in selected_categories:
        if category in DEMO_SAMPLES:
            for text, expected, description in DEMO_SAMPLES[category]:
                all_texts.append(text)
                expected_langs.append(expected)
                categories.append(category)
    
    # Add custom texts
    if custom_texts.strip():
        custom_lines = [line.strip() for line in custom_texts.strip().split('\n') if line.strip()]
        for text in custom_lines:
            all_texts.append(text)
            expected_langs.append("unknown")
            categories.append("Custom")
    
    if not all_texts:
        return "No texts to analyze.", None
    
    # Run detection on all texts
    all_results = []
    for text in all_texts:
        results = detect_with_all_models(text, models)
        all_results.append(results)
    
    # Create results DataFrame
    df = create_results_dataframe(all_texts, all_results, expected_langs, categories)
    
    summary = f"Analyzed {len(all_texts)} texts across {len(set(categories))} categories."
    
    return summary, df

def create_demo_interface():
    """Create the demo interface."""
    
    # Initialize models
    models = initialize_models()
    
    with gr.Blocks(title="Language Detection Demo - Model Comparison", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Language Detection Demo - Model Comparison")
        gr.Markdown("Compare all four language detection models simultaneously across various difficulty categories.")
        
        # Model Status Section
        with gr.Group():
            gr.Markdown("## 🤖 Model Status")
            model_status_text = ""
            for model_key, model_info in models.items():
                status_icon = "✅" if model_info["status"] == "Ready" else "❌"
                model_status_text += f"{status_icon} **{model_info['name']}**: {model_info['status']}\n\n"
            gr.Markdown(model_status_text)
        
        # Category Selection Section
        with gr.Group():
            gr.Markdown("## 📊 Test Categories")
            gr.Markdown("Select categories to test different aspects of language detection difficulty:")
            
            category_checkboxes = gr.CheckboxGroup(
                choices=list(DEMO_SAMPLES.keys()),
                label="Select Test Categories",
                value=["Easy/Obvious", "Short Text"],  # Default selection
                interactive=True
            )
        
        # Custom Text Input Section
        with gr.Group():
            gr.Markdown("## ✏️ Custom Text Input")
            gr.Markdown("Enter your own texts to test (one per line):")
            
            custom_text_input = gr.Textbox(
                label="Custom Texts",
                placeholder="Enter custom texts here, one per line...\nExample:\nHello world\nBonjour le monde\n你好世界",
                lines=5,
                max_lines=10
            )
        
        # Control Buttons
        with gr.Row():
            run_demo_btn = gr.Button("🔍 Run Demo Tests", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear Results", variant="secondary")
        
        # Sample Preview Section (moved up, condensed)
        with gr.Group():
            gr.Markdown("## 📚 Category Explanations")
            gr.Markdown("Understanding what each test category evaluates:")
            
            category_explanations = """
**Easy/Obvious:** Clear, unambiguous sentences in their native language. Tests basic language detection capability.

**Short Text:** Single words or very short phrases. Tests model performance with minimal context.

**False Friends:** Words that look similar across languages but have different meanings. Tests ability to distinguish between closely related languages.

**Mixed Scripts:** Text containing multiple languages, numbers, symbols, or scripts. Tests handling of multilingual content.

**Proper Nouns:** Names of people, places, or entities that exist across multiple languages. Tests context-dependent detection.

**Common Words:** International words with similar spelling across languages (hotel, taxi, etc.). Tests disambiguation of universal terms.

**Technical Terms:** Specialized vocabulary that may be borrowed or translated across languages. Tests domain-specific detection.

**Code-switching:** Text that switches between languages mid-sentence. Tests handling of bilingual communication patterns.

**Transliterated Text:** Non-Latin scripts written in Latin characters. Tests recognition of transliteration vs. native language.

**Ambiguous Script:** Words that could belong to multiple languages with identical spelling. Tests the model's decision-making under uncertainty.
"""
            
            gr.Markdown(category_explanations)
        
        # Results Section (moved to bottom)
        with gr.Group():
            gr.Markdown("## 📈 Results")
            
            summary_output = gr.Textbox(
                label="Summary",
                interactive=False,
                visible=False
            )
            
            results_dataframe = gr.Dataframe(
                label="Model Comparison Results (A-A: Model A Dataset A, B-A: Model B Dataset A, A-B: Model A Dataset B, B-B: Model B Dataset B)",
                wrap=True,
                interactive=False,
                visible=False
            )
        
        # Event Handlers
        def run_tests(selected_cats, custom_texts):
            summary, df = run_demo_tests(selected_cats, custom_texts, models)
            
            if df is not None:
                return (
                    gr.update(value=summary, visible=True),
                    gr.update(value=df, visible=True)
                )
            else:
                return (
                    gr.update(value=summary, visible=True),
                    gr.update(visible=False)
                )
        
        def clear_results():
            return (
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False)
            )
        
        # Connect event handlers
        run_demo_btn.click(
            fn=run_tests,
            inputs=[category_checkboxes, custom_text_input],
            outputs=[summary_output, results_dataframe]
        )
        
        clear_btn.click(
            fn=clear_results,
            outputs=[summary_output, results_dataframe]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        debug=True
    ) 