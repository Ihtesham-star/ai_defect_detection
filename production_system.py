# Production-Ready AI Model API
# Complete deployment system for construction and textile defect detection

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiDomainDefectCNN(nn.Module):
    """Production model architecture - exactly matching your trained model"""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(MultiDomainDefectCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Dropout2d(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ProductionDefectDetector:
    """Production-ready defect detection system"""
    
    def __init__(self, model_path='best_multi_domain_model_70k.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.load_model(model_path)
        self.setup_transforms()
        self.stats = {
            'total_predictions': 0,
            'defects_detected': 0,
            'start_time': datetime.now()
        }
        
        logger.info(f"Production AI system initialized on {self.device}")
    
    def load_model(self, model_path):
        """Load the trained model for production use"""
        try:
            self.model = MultiDomainDefectCNN(num_classes=2)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def setup_transforms(self):
        """Setup image preprocessing pipeline"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_input):
        """Preprocess image for model input"""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if image_input.startswith('data:image'):
                    # Base64 encoded image
                    image_data = image_input.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                else:
                    # File path
                    image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, bytes):
                # Raw bytes
                image = Image.open(io.BytesIO(image_input)).convert('RGB')
            else:
                # PIL Image
                image = image_input.convert('RGB')
            
            # Store original size for metadata
            original_size = image.size
            
            # Apply transforms
            processed_image = self.transform(image).unsqueeze(0).to(self.device)
            
            return processed_image, original_size, image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Invalid image input: {e}")
    
    def predict(self, image_input, threshold_high=0.95, threshold_medium=0.80):
        """Make production prediction with business logic"""
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image, original_size, pil_image = self.preprocess_image(image_input)
            
            # Model inference
            with torch.no_grad():
                output = self.model(processed_image)
                probabilities = torch.nn.functional.softmax(output, dim=1).cpu()
                confidence, predicted = torch.max(probabilities, 1)
            
            # Extract results
            defect_probability = probabilities[0][0].item()
            good_probability = probabilities[0][1].item()
            confidence_score = confidence.item()
            predicted_class = 'DEFECTIVE' if predicted.item() == 0 else 'GOOD'
            
            # Business decision logic
            if confidence_score >= threshold_high:
                decision = 'AUTOMATED'
                action = 'REJECT' if predicted_class == 'DEFECTIVE' else 'ACCEPT'
                review_required = False
            elif confidence_score >= threshold_medium:
                decision = 'REVIEW_RECOMMENDED'
                action = 'MANUAL_REVIEW'
                review_required = True
            else:
                decision = 'MANUAL_INSPECTION'
                action = 'MANUAL_INSPECTION'
                review_required = True
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.stats['total_predictions'] += 1
            if predicted_class == 'DEFECTIVE':
                self.stats['defects_detected'] += 1
            
            # Prepare response
            result = {
                'prediction': predicted_class,
                'confidence': confidence_score,
                'defect_probability': defect_probability,
                'good_probability': good_probability,
                'decision': decision,
                'action': action,
                'review_required': review_required,
                'processing_time_ms': processing_time,
                'timestamp': datetime.now().isoformat(),
                'image_size': original_size,
                'model_version': '1.0.0',
                'success': True
            }
            
            logger.info(f"Prediction: {predicted_class} ({confidence_score:.3f}) - {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_predict(self, image_list, **kwargs):
        """Process multiple images in batch"""
        results = []
        
        logger.info(f"Processing batch of {len(image_list)} images")
        
        for i, image in enumerate(image_list):
            try:
                result = self.predict(image, **kwargs)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'batch_index': i,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def get_system_stats(self):
        """Get system performance statistics"""
        uptime = datetime.now() - self.stats['start_time']
        
        return {
            'total_predictions': self.stats['total_predictions'],
            'defects_detected': self.stats['defects_detected'],
            'defect_rate': self.stats['defects_detected'] / max(1, self.stats['total_predictions']),
            'uptime_seconds': uptime.total_seconds(),
            'predictions_per_hour': self.stats['total_predictions'] / max(1, uptime.total_seconds() / 3600),
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'timestamp': datetime.now().isoformat()
        }

# Flask Web API (Optional - for web deployment)
def create_flask_app():
    """Create Flask web API for the defect detection system"""
    
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("Flask not installed. Install with: pip install flask flask-cors")
        return None
    
    app = Flask(__name__)
    CORS(app)  # Enable CORS for web browsers
    
    # Initialize the detector
    detector = ProductionDefectDetector()
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'stats': detector.get_system_stats()
        })
    
    @app.route('/predict', methods=['POST'])
    def predict_endpoint():
        """Single image prediction endpoint"""
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided'}), 400
            
            image_file = request.files['image']
            
            # Get optional parameters
            threshold_high = float(request.form.get('threshold_high', 0.95))
            threshold_medium = float(request.form.get('threshold_medium', 0.80))
            
            # Make prediction
            result = detector.predict(
                image_file.read(),
                threshold_high=threshold_high,
                threshold_medium=threshold_medium
            )
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/batch_predict', methods=['POST'])
    def batch_predict_endpoint():
        """Batch prediction endpoint"""
        try:
            files = request.files.getlist('images')
            if not files:
                return jsonify({'error': 'No images provided'}), 400
            
            # Process images
            image_data = [f.read() for f in files]
            results = detector.batch_predict(image_data)
            
            return jsonify({
                'results': results,
                'total_processed': len(results)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/stats', methods=['GET'])
    def stats_endpoint():
        """System statistics endpoint"""
        return jsonify(detector.get_system_stats())
    
    return app

# Streamlit Web Interface (Alternative to Flask)
def create_streamlit_app():
    """Create Streamlit web interface"""
    
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Install with: pip install streamlit")
        return None
    
    st.set_page_config(
        page_title="AI Defect Detection System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🏭 Production AI Defect Detection System")
    st.markdown("**Industries:** Construction (Concrete) • Textiles (Fabric)")
    
    # Initialize detector with proper caching (compatible with older Streamlit versions)
    @st.cache(allow_output_mutation=True, show_spinner=False)
    def load_detector():
        return ProductionDefectDetector()
    
    # Alternative for newer Streamlit versions - try this first
    try:
        @st.cache_resource
        def load_detector_new():
            return ProductionDefectDetector()
        detector = load_detector_new()
    except AttributeError:
        # Fallback for older Streamlit versions
        try:
            detector = load_detector()
        except:
            # If all caching fails, just load normally
            detector = ProductionDefectDetector()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    threshold_high = st.sidebar.slider("High Confidence Threshold", 0.8, 1.0, 0.95, 0.01)
    threshold_medium = st.sidebar.slider("Medium Confidence Threshold", 0.5, 0.9, 0.80, 0.01)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🔍 Single Image", "📊 Batch Processing", "📈 Statistics"])
    
    with tab1:
        st.header("Single Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload an image for defect detection",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file:
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if st.button("🔍 Analyze Image"):
                    with st.spinner("Analyzing..."):
                        result = detector.predict(
                            uploaded_file.getvalue(),
                            threshold_high=threshold_high,
                            threshold_medium=threshold_medium
                        )
                        
                        if result['success']:
                            # Display results
                            prediction = result['prediction']
                            confidence = result['confidence']
                            
                            # Color-coded result
                            if prediction == 'DEFECTIVE':
                                st.error(f"🚨 **DEFECTIVE DETECTED** ({confidence:.1%} confidence)")
                            else:
                                st.success(f"✅ **GOOD QUALITY** ({confidence:.1%} confidence)")
                            
                            # Detailed metrics
                            st.subheader("📊 Detailed Analysis")
                            
                            st.metric("Defect Probability", f"{result['defect_probability']:.1%}")
                            st.metric("Good Probability", f"{result['good_probability']:.1%}")
                            st.metric("Processing Time", f"{result['processing_time_ms']:.1f} ms")
                            st.metric("Business Action", result['action'])
                            
                            
                            # Business recommendation
                            st.subheader("💼 Business Recommendation")
                            if result['decision'] == 'AUTOMATED':
                                st.info(f"✅ **Automated Decision**: {result['action']}")
                            elif result['decision'] == 'REVIEW_RECOMMENDED':
                                st.warning("⚠️ **Manual Review Recommended**")
                            else:
                                st.error("🔍 **Manual Inspection Required**")
                        
                        else:
                            st.error(f"Analysis failed: {result['error']}")
    
    with tab2:
        st.header("Batch Processing")
        st.markdown("Upload multiple images for batch analysis")
        
        uploaded_files = st.file_uploader(
            "Select multiple images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} images")
            
            if st.button("🔍 Process Batch"):
                with st.spinner(f"Processing {len(uploaded_files)} images..."):
                    image_data = [f.getvalue() for f in uploaded_files]
                    results = detector.batch_predict(
                        image_data,
                        threshold_high=threshold_high,
                        threshold_medium=threshold_medium
                    )
                    
                    # Summary statistics
                    successful = [r for r in results if r.get('success', False)]
                    defective_count = len([r for r in successful if r['prediction'] == 'DEFECTIVE'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Processed", len(successful))
                    with col2:
                        st.metric("Defects Found", defective_count)
                    with col3:
                        st.metric("Success Rate", f"{len(successful)/len(results):.1%}")
                    
                    # Detailed results
                    for i, result in enumerate(results):
                        if result.get('success', False):
                            with st.expander(f"Image {i+1}: {result['prediction']} ({result['confidence']:.1%})"):
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(uploaded_files[i], width=200)
                                with col2:
                                    st.write(f"**Prediction**: {result['prediction']}")
                                    st.write(f"**Confidence**: {result['confidence']:.1%}")
                                    st.write(f"**Action**: {result['action']}")
                                    st.write(f"**Processing Time**: {result['processing_time_ms']:.1f} ms")
    
    with tab3:
        st.header("System Statistics")
        
        stats = detector.get_system_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", stats['total_predictions'])
        with col2:
            st.metric("Defects Detected", stats['defects_detected'])
        with col3:
            st.metric("Defect Rate", f"{stats['defect_rate']:.1%}")
        with col4:
            st.metric("Predictions/Hour", f"{stats['predictions_per_hour']:.0f}")
        
        st.subheader("System Information")
        st.write(f"**Device**: {stats['device']}")
        st.write(f"**Uptime**: {stats['uptime_seconds']:.0f} seconds")
        st.write(f"**Model Status**: {'✅ Loaded' if stats['model_loaded'] else '❌ Not Loaded'}")

# Command Line Interface
def create_cli():
    """Command line interface for the detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Defect Detection System')
    parser.add_argument('--mode', choices=['single', 'batch', 'api', 'streamlit'], 
                       default='single', help='Operation mode')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to folder with images')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--model', type=str, default='best_multi_domain_model_70k.pth',
                       help='Path to model file')
    parser.add_argument('--port', type=int, default=5000, help='Port for API server')
    
    return parser

# Streamlit detection and auto-execution
def is_streamlit():
    """Detect if running under Streamlit"""
    try:
        import streamlit as st
        # Check if we're in a Streamlit environment
        return hasattr(st, 'session_state')
    except ImportError:
        return False

# Main execution
if __name__ == "__main__":
    import sys
    
    # Check if running under Streamlit
    if is_streamlit() or 'streamlit' in sys.modules:
        # Automatically run Streamlit interface
        create_streamlit_app()
    else:
        # Command line interface
        parser = create_cli()
        args = parser.parse_args()
        
        if args.mode == 'single' and args.image:
            # Single image prediction
            detector = ProductionDefectDetector(args.model)
            result = detector.predict(args.image)
            print(json.dumps(result, indent=2))
        
        elif args.mode == 'batch' and args.batch:
            # Batch processing
            detector = ProductionDefectDetector(args.model)
            image_folder = Path(args.batch)
            
            images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                images.extend(image_folder.glob(f'*{ext}'))
            
            results = detector.batch_predict([str(img) for img in images])
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                print(json.dumps(results, indent=2))
        
        elif args.mode == 'api':
            # Flask API server
            app = create_flask_app()
            if app:
                print(f"🚀 Starting API server on port {args.port}")
                app.run(host='0.0.0.0', port=args.port, debug=False)
            else:
                print("Flask not available")
        
        elif args.mode == 'streamlit':
            # Streamlit interface
            print("🚀 Starting Streamlit interface...")
            print("Run: streamlit run production_system.py")
        
        else:
            # Interactive mode
            detector = ProductionDefectDetector(args.model)
            print("🔍 AI Defect Detection System - Interactive Mode")
            print("Enter image path (or 'quit' to exit):")
            
            while True:
                image_path = input("> ").strip()
                if image_path.lower() in ['quit', 'exit', 'q']:
                    break
                
                if os.path.exists(image_path):
                    result = detector.predict(image_path)
                    print(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")
                    print(f"Action: {result['action']}")
                else:
                    print("Image not found")
                    # Add this at the very end of your file to auto-run Streamlit
if __name__ == "__main__" and "streamlit" in globals():
    create_streamlit_app()