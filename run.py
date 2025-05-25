#!/usr/bin/env python3
"""
Simple startup script for the Language Detection App

This script provides an easy way to run the app with different configurations.
"""

import sys
import os
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import gradio
        print("✅ Gradio is available")
    except ImportError:
        print("❌ Gradio not found. Install with: pip install -r requirements.txt")
        return False
    
    return True

def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    os.system("python test_app.py")

def run_app(model_type="placeholder", host="0.0.0.0", port=7860, share=False):
    """Run the main application."""
    
    if not check_dependencies():
        return 1
    
    # Set environment variables for configuration
    os.environ["MODEL_TYPE"] = model_type
    os.environ["HOST"] = host
    os.environ["PORT"] = str(port)
    os.environ["SHARE"] = str(share).lower()
    
    print(f"🚀 Starting Language Detection App...")
    print(f"📊 Model: {model_type}")
    print(f"🌐 Host: {host}:{port}")
    print(f"🔗 Share: {share}")
    print("-" * 50)
    
    # Import and run the app
    try:
        from app import main
        app = main()
        app.launch(
            server_name=host,
            server_port=port,
            share=share,
            debug=True
        )
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        return 1
    
    return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Language Detection App Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # Run with default settings
  python run.py --test                   # Run tests only
  python run.py --model huggingface      # Use Hugging Face model (if available)
  python run.py --port 8080              # Run on port 8080
  python run.py --share                  # Create public link
        """
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run tests instead of starting the app"
    )
    
    parser.add_argument(
        "--model",
        choices=["placeholder", "huggingface", "custom"],
        default="placeholder",
        help="Model type to use (default: placeholder)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link via Gradio"
    )
    
    args = parser.parse_args()
    
    print("🌍 Language Detection App Runner")
    print("=" * 40)
    
    if args.test:
        run_tests()
        return 0
    
    # Validate model choice
    if args.model == "huggingface":
        try:
            import transformers
            print("✅ Transformers available for Hugging Face model")
        except ImportError:
            print("⚠️  Transformers not available. Install with:")
            print("   pip install transformers torch")
            print("   Falling back to placeholder model...")
            args.model = "placeholder"
    
    return run_app(
        model_type=args.model,
        host=args.host,
        port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    sys.exit(main()) 