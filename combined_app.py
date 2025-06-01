import gradio as gr
import pandas as pd
from backend.language_detector import LanguageDetector
from typing import List, Dict, Any
import time

# Import demo samples from demo_page
from demo_page import DEMO_SAMPLES, initialize_models, detect_with_all_models, create_results_dataframe, run_demo_tests

def create_single_model_interface():
    """Create the original single model interface."""
    # Initialize the language detector with default model (Model A Dataset A)
    detector = LanguageDetector()
    
    with gr.Column() as single_interface:
        gr.Markdown("# 🌍 Language Detection App")
        gr.Markdown("Select a model and enter text below to detect its language with confidence scores.")
        
        # Model Selection Section with visual styling
        with gr.Group():
            gr.Markdown(
                "<div style='text-align: center; padding: 16px 0 8px 0; margin-bottom: 16px; font-size: 18px; font-weight: 600; border-bottom: 2px solid; background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent); border-radius: 8px 8px 0 0;'>🤖 Model Selection</div>"
            )
            
            # Get available models
            available_models = detector.get_available_models()
            model_choices = []
            model_info_map = {}
            
            for key, info in available_models.items():
                if info["status"] == "available":
                    model_choices.append((info["display_name"], key))
                else:
                    model_choices.append((f"{info['display_name']} (Coming Soon)", key))
                model_info_map[key] = info
            
            model_selector = gr.Dropdown(
                choices=model_choices,
                value="model-a-dataset-a",  # Default to Model A Dataset A
                label="Choose Language Detection Model",
                interactive=True
            )
            
            # Model Information Display
            model_info_display = gr.Markdown(
                value=_format_model_info(detector.get_current_model_info()),
                label="Model Information"
            )
        
        # Add visual separator
        gr.Markdown(
            "<div style='margin: 24px 0; border-top: 3px solid rgba(99, 102, 241, 0.2); background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.05), transparent); height: 2px;'></div>"
        )
        
        # Analysis Section
        with gr.Group():
            gr.Markdown(
                "<div style='text-align: center; padding: 16px 0 8px 0; margin-bottom: 16px; font-size: 18px; font-weight: 600; border-bottom: 2px solid; background: linear-gradient(90deg, transparent, rgba(34, 197, 94, 0.1), transparent); border-radius: 8px 8px 0 0;'>🔍 Language Analysis</div>"
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    text_input = gr.Textbox(
                        label="Text to Analyze",
                        placeholder="Enter text here to detect its language...",
                        lines=5,
                        max_lines=10
                    )
                    
                    detect_btn = gr.Button("🔍 Detect Language", variant="primary", size="lg")
                    
                    # Example texts
                    gr.Examples(
                        examples=[
                            ["Hello, how are you today?"],
                            ["Bonjour, comment allez-vous?"],
                            ["Hola, ¿cómo estás?"],
                            ["Guten Tag, wie geht es Ihnen?"],
                            ["こんにちは、元気ですか？"],
                            ["Привет, как дела?"],
                            ["Ciao, come stai?"],
                            ["Olá, como você está?"],
                            ["你好，你好吗？"],
                            ["안녕하세요, 어떻게 지내세요?"]
                        ],
                        inputs=text_input,
                        label="Try these examples:"
                    )
                
                with gr.Column(scale=2):
                    # Output section
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 16px 0 8px 0; margin-bottom: 12px; font-size: 18px; font-weight: 600; border-bottom: 2px solid; background: linear-gradient(90deg, transparent, rgba(168, 85, 247, 0.1), transparent); border-radius: 8px 8px 0 0;'>📊 Detection Results</div>"
                        )
                        
                        detected_language = gr.Textbox(
                            label="Detected Language",
                            interactive=False
                        )
                        
                        confidence_score = gr.Number(
                            label="Confidence Score",
                            interactive=False,
                            precision=4
                        )
                        
                        language_code = gr.Textbox(
                            label="Language Code (ISO 639-1)",
                            interactive=False
                        )
                        
                        # Top predictions table
                        top_predictions = gr.Dataframe(
                            headers=["Language", "Code", "Confidence"],
                            label="Top 5 Predictions",
                            interactive=False,
                            wrap=True
                        )
        
        # Status/Info section
        with gr.Row():
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
        
        # Event handlers
        def detect_language_wrapper(text, selected_model):
            if not text.strip():
                return (
                    "No text provided",
                    0.0,
                    "",
                    [],
                    gr.update(value="Please enter some text to analyze.", visible=True)
                )
            
            try:
                # Switch model if needed
                if detector.current_model_key != selected_model:
                    try:
                        detector.switch_model(selected_model)
                    except NotImplementedError:
                        return (
                            "Model unavailable",
                            0.0,
                            "",
                            [],
                            gr.update(value="This model is not yet implemented. Please select an available model.", visible=True)
                        )
                    except Exception as e:
                        return (
                            "Model error",
                            0.0,
                            "",
                            [],
                            gr.update(value=f"Error loading model: {str(e)}", visible=True)
                        )
                
                result = detector.detect_language(text)
                
                # Extract main prediction
                main_lang = result['language']
                main_confidence = result['confidence']
                main_code = result['language_code']
                
                # Format top predictions for table
                predictions_table = [
                    [pred['language'], pred['language_code'], f"{pred['confidence']:.4f}"]
                    for pred in result['top_predictions']
                ]
                
                model_info = result.get('metadata', {}).get('model_info', {})
                model_name = model_info.get('name', 'Unknown Model')
                
                return (
                    main_lang,
                    main_confidence,
                    main_code,
                    predictions_table,
                    gr.update(value=f"✅ Analysis Complete\n\nInput Text: {text[:100]}{'...' if len(text) > 100 else ''}\n\nDetected Language: {main_lang} ({main_code})\nConfidence: {main_confidence:.2%}\n\nModel: {model_name}", visible=True)
                )
                
            except Exception as e:
                return (
                    "Error occurred",
                    0.0,
                    "",
                    [],
                    gr.update(value=f"Error: {str(e)}", visible=True)
                )
        
        def update_model_info(selected_model):
            """Update model information display when model selection changes."""
            try:
                if detector.current_model_key != selected_model:
                    detector.switch_model(selected_model)
                model_info = detector.get_current_model_info()
                return _format_model_info(model_info)
            except NotImplementedError:
                return "**This model is not yet implemented.** Please select an available model."
            except Exception as e:
                return f"**Error loading model information:** {str(e)}"
        
        # Connect the button to the detection function
        detect_btn.click(
            fn=detect_language_wrapper,
            inputs=[text_input, model_selector],
            outputs=[detected_language, confidence_score, language_code, top_predictions, status_text]
        )
        
        # Also trigger on Enter key in text input
        text_input.submit(
            fn=detect_language_wrapper,
            inputs=[text_input, model_selector],
            outputs=[detected_language, confidence_score, language_code, top_predictions, status_text]
        )
        
        # Update model info when selection changes
        model_selector.change(
            fn=update_model_info,
            inputs=[model_selector],
            outputs=[model_info_display]
        )
    
    return single_interface

def create_demo_comparison_interface():
    """Create the demo comparison interface."""
    
    # Initialize models
    models = initialize_models()
    
    with gr.Column() as demo_interface:
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
    
    return demo_interface

def _format_model_info(model_info):
    """Format model information for display."""
    if not model_info:
        return "No model information available."
    
    formatted_info = f"""
**{model_info.get('name', 'Unknown Model')}**

{model_info.get('description', 'No description available.')}

**📊 Performance:**
- Accuracy: {model_info.get('accuracy', 'N/A')}
- Model Size: {model_info.get('model_size', 'N/A')}

**🏗️ Architecture:**
- Model Architecture: {model_info.get('architecture', 'N/A')}
- Base Model: {model_info.get('base_model', 'N/A')}
- Training Dataset: {model_info.get('dataset', 'N/A')}

**🌐 Languages:** {model_info.get('languages_supported', 'N/A')}

**⚙️ Training Details:** {model_info.get('training_details', 'N/A')}

**💡 Use Cases:** {model_info.get('use_cases', 'N/A')}

**✅ Strengths:** {model_info.get('strengths', 'N/A')}

**⚠️ Limitations:** {model_info.get('limitations', 'N/A')}
"""
    return formatted_info

def main():
    """Create the main application with tabbed interface."""
    
    with gr.Blocks(title="Language Detection App Suite", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🌍 Language Detection App Suite")
        gr.Markdown("Choose between single model testing or comprehensive model comparison.")
        
        with gr.Tabs():
            with gr.TabItem("🔍 Single Model Detection"):
                single_model_interface = create_single_model_interface()
            
            with gr.TabItem("🚀 Model Comparison Demo"):
                demo_comparison_interface = create_demo_comparison_interface()
    
    return app

if __name__ == "__main__":
    app = main()
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        debug=True
    ) 