import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from streamlit_inference import YellowRustSegmentation

# Page configuration
st.set_page_config(
    page_title="Yellow Rust Segmentation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #4682B4;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stProgress .st-bo {
    background-color: #2E8B57;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        model_path = Path("models/checkpoints/best.pth")
        config_path = Path("configs/config.yaml")
        
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return None
            
        if not config_path.exists():
            st.error(f"Config file not found: {config_path}")
            return None
            
        segmentation_model = YellowRustSegmentation(
            model_path=str(model_path),
            config_path=str(config_path)
        )
        return segmentation_model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model, confidence_threshold=0.5, enhance_detection=False):
    """Process uploaded image and return results"""
    try:
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Run inference
        results = model.predict(
            image_np,
            confidence_threshold=confidence_threshold,
            enhance_detection=enhance_detection,
            return_visualization=True
        )
        
        return results
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def create_results_display(results, original_image):
    """Create a comprehensive results display"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(original_image, use_container_width=True)
        
    with col2:
        st.subheader("🎯 Segmentation Result")
        if 'visualization' in results:
            st.image(results['visualization'], use_container_width=True)
        else:
            st.warning("Visualization not available")
    
    # Metrics display
    st.subheader("📊 Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rust_percentage = results.get('rust_percentage', 0)
        st.metric(
            label="Rust Coverage",
            value=f"{rust_percentage:.2f}%",
            delta=f"{'High' if rust_percentage > 10 else 'Low'} Risk"
        )
    
    with col2:
        confidence = results.get('confidence_score', 0)
        st.metric(
            label="Confidence",
            value=f"{confidence:.3f}",
            delta=f"{'Good' if confidence > 0.8 else 'Fair'}"
        )
    
    with col3:
        total_pixels = results.get('total_pixels', 0)
        st.metric(
            label="Total Pixels",
            value=f"{total_pixels:,}"
        )
    
    with col4:
        rust_pixels = results.get('rust_pixels', 0)
        st.metric(
            label="Rust Pixels",
            value=f"{rust_pixels:,}"
        )
    
    # Additional visualizations
    if 'binary_mask' in results and 'probability_map' in results:
        st.subheader("🔍 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Binary Mask**")
            mask_display = (results['binary_mask'] * 255).astype(np.uint8)
            st.image(mask_display, use_container_width=True, caption="White areas indicate detected rust")
        
        with col2:
            st.write("**Probability Heatmap**")
            # Create heatmap visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(results['probability_map'], cmap='hot', vmin=0, vmax=1)
            ax.set_title('Rust Probability Heatmap')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close()

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Yellow Rust Segmentation</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-powered crop disease detection and analysis</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Model loading
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load the segmentation model. Please check if the model files exist.")
        st.info("📝 Make sure you have trained the model and the checkpoint files are available in `models/checkpoints/`")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Inference settings
    st.sidebar.subheader("🎛️ Inference Parameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Higher values = more conservative detection. Try 0.2-0.4 for better rust detection."
    )
    
    enhance_detection = st.sidebar.checkbox(
        "Enhanced Detection",
        value=False,
        help="Apply additional image processing for better detection"
    )
    
    # File upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a crop image to analyze for yellow rust disease"
    )
    
    if uploaded_file is not None:
        # Display upload info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        with st.expander("📋 File Details"):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
        
        # Load and display image
        try:
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            st.subheader("🔄 Processing")
            
            # Process button
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image for yellow rust..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    
                    # Simulate processing steps
                    progress_bar.progress(25)
                    st.write("📊 Preprocessing image...")
                    
                    progress_bar.progress(50)
                    st.write("🧠 Running AI inference...")
                    
                    # Run actual inference
                    results = process_image(
                        image, 
                        model, 
                        confidence_threshold=confidence_threshold,
                        enhance_detection=enhance_detection
                    )
                    
                    progress_bar.progress(75)
                    st.write("🎨 Generating visualizations...")
                    
                    progress_bar.progress(100)
                    st.write("✅ Analysis complete!")
                    
                    # Clear progress bar
                    progress_bar.empty()
                    
                    if results is not None:
                        # Display results
                        create_results_display(results, image)
                        
                        # Download options
                        st.subheader("💾 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'visualization' in results:
                                # Convert visualization to bytes for download
                                vis_pil = Image.fromarray(results['visualization'])
                                buf = io.BytesIO()
                                vis_pil.save(buf, format='PNG')
                                
                                st.download_button(
                                    label="📥 Download Visualization",
                                    data=buf.getvalue(),
                                    file_name=f"rust_analysis_{uploaded_file.name.split('.')[0]}.png",
                                    mime="image/png"
                                )
                        
                        with col2:
                            if 'binary_mask' in results:
                                # Convert mask to bytes for download
                                mask_pil = Image.fromarray((results['binary_mask'] * 255).astype(np.uint8))
                                buf = io.BytesIO()
                                mask_pil.save(buf, format='PNG')
                                
                                st.download_button(
                                    label="📥 Download Mask",
                                    data=buf.getvalue(),
                                    file_name=f"rust_mask_{uploaded_file.name.split('.')[0]}.png",
                                    mime="image/png"
                                )
                    
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    
    # Information section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### Yellow Rust Segmentation AI
        
        This application uses a deep learning model (U-Net with ResNet34 encoder) to detect and segment yellow rust disease in crop images.
        
        **Features:**
        - 🎯 Precise pixel-level segmentation
        - 📊 Quantitative analysis (rust percentage, confidence scores)
        - 🎨 Multiple visualization options
        - ⚙️ Adjustable detection parameters
        - 💾 Downloadable results
        
        **How to use:**
        1. Upload a crop image using the file uploader
        2. Adjust detection parameters in the sidebar if needed
        3. Click "Analyze Image" to run the AI analysis
        4. View results and download visualizations
        
        **Model Performance:**
        - Validation IoU: 94.21%
        - Architecture: U-Net with ResNet34 encoder
        - Training Dataset: YELLOW-RUST-19
        """)

if __name__ == "__main__":
    main()