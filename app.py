import gradio as gr
from backend.language_detector import LanguageDetector

def main():
    # Initialize the language detector with default model
    detector = LanguageDetector()
    
    # Create Gradio interface
    with gr.Blocks(title="Language Detection App", theme=gr.themes.Soft()) as app:
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
                value="xlm-roberta-langdetect",
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
    
    return app


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

**🌐 Languages:** {model_info.get('languages_supported', 'N/A')}

**⚙️ Training Details:** {model_info.get('training_details', 'N/A')}

**💡 Use Cases:** {model_info.get('use_cases', 'N/A')}

**✅ Strengths:** {model_info.get('strengths', 'N/A')}

**⚠️ Limitations:** {model_info.get('limitations', 'N/A')}
"""
    return formatted_info


if __name__ == "__main__":
    app = main()
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        debug=True
    ) 