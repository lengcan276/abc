# streamlit_ui/ModelingPage.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import base64
from PIL import Image
from io import BytesIO
import zipfile

def render_modeling_page(model_agent=None):
    """Render modeling analysis page"""
    st.title("Predictive Modeling")
    
    st.markdown("""
    ## S1-T1 Gap Predictive Modeling
    
    This page focuses on building and evaluating predictive models for S1-T1 gap properties.
    
    Two main models are built:
    
    1. **Classification Model** - Predicts whether a molecule will have a negative or positive S1-T1 gap
    2. **Regression Model** - Predicts the actual value of the S1-T1 gap
    
    The analysis includes:
    
    - Feature selection and importance ranking
    - Model performance evaluation
    - Feature engineering insights
    - Prediction visualization
    
    You can run the modeling pipeline on previously generated feature data or upload a feature CSV file.
    """)
    
    # Option to use existing data or upload new
    feature_file = None
    
    # Look for previously processed data
<<<<<<< HEAD
    extracted_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted'
=======
    extracted_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted'
>>>>>>> 0181d62 (update excited)
    if os.path.exists(extracted_dir):
        # Look for processed features file
        feature_files = [f for f in os.listdir(extracted_dir) if ('feature' in f.lower() or 'processed' in f.lower()) and f.endswith('.csv')]
        
        if feature_files:
            st.info("Found existing feature files.")
            selected_file = st.selectbox("Select feature file for modeling", feature_files)
            feature_file = os.path.join(extracted_dir, selected_file)
        else:
            st.warning("No feature files found. Please run feature engineering first.")
    else:
        st.warning("No extracted data directory found. Please extract data and run feature engineering first.")
        
    # Upload file option
    if not feature_file:
        st.subheader("Upload Feature Data")
        
        feature_upload = st.file_uploader("Upload processed features CSV", type="csv")
        
        if feature_upload:
            # Save uploaded CSV to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                temp_file.write(feature_upload.getvalue())
                feature_file = temp_file.name
                
    # Execute modeling
    if feature_file:
        # Check if pre-computed results exist
<<<<<<< HEAD
        results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
        models_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/models'
=======
        results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
        models_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/models'
>>>>>>> 0181d62 (update excited)
        
        if os.path.exists(results_dir) and os.path.exists(models_dir) and \
           len(os.listdir(results_dir)) > 0 and len(os.listdir(models_dir)) > 0:
            st.info("Found existing modeling results.")
            
            if st.button("Show Modeling Results"):
                display_modeling_results(results_dir)
                
            if st.button("Re-run Modeling"):
                run_modeling_analysis(feature_file, model_agent)
        else:
            if st.button("Run Modeling Analysis"):
                run_modeling_analysis(feature_file, model_agent)
                
    return None

def run_modeling_analysis(feature_file, model_agent):
    """Run modeling analysis and display results"""
    with st.spinner("Running modeling analysis..."):
        try:
            # Execute modeling analysis
            if model_agent:
                model_agent.feature_file = feature_file
                result = model_agent.run_modeling_pipeline()
                
                if result and ('classification' in result or 'regression' in result):
                    st.success("Modeling analysis completed.")
                    
                    # Display results
<<<<<<< HEAD
                    display_modeling_results('/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling')
=======
                    display_modeling_results('/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling')
>>>>>>> 0181d62 (update excited)
                    
                    # Return modeling results
                    return result
                else:
                    st.error("Modeling analysis failed")
            else:
                st.error("Modeling component not initialized")
        except Exception as e:
            st.error(f"Error during modeling analysis: {str(e)}")
            
    return None

def display_modeling_results(results_dir):
    """Display modeling analysis results"""
    st.subheader("Modeling Analysis Results")
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        st.error(f"Results directory {results_dir} not found.")
        return
        
    # Find all image files in the results directory
    image_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
    
    if not image_files:
        st.warning("No result images found.")
        return
        
    # Group images by type
    classification_images = [f for f in image_files if 'classification' in f or 'confusion_matrix' in f]
    regression_images = [f for f in image_files if 'regression' in f]
    feature_rank_images = [f for f in image_files if 'feature_ranks' in f]
    
    # Create tabs for classification and regression
    tabs = st.tabs(["Classification Model", "Regression Model", "Feature Selection"])
    
    # Classification tab
    with tabs[0]:
        st.markdown("### Classification Model Results")
        
        if classification_images:
            for file in classification_images:
                img = Image.open(os.path.join(results_dir, file))
                caption = file.replace('.png', '').replace('_', ' ').title()
                st.image(img, caption=caption, use_container_width=True)
        else:
            st.warning("No classification model results found.")
            
    # Regression tab
    with tabs[1]:
        st.markdown("### Regression Model Results")
        
        if regression_images:
            for file in regression_images:
                img = Image.open(os.path.join(results_dir, file))
                caption = file.replace('.png', '').replace('_', ' ').title()
                st.image(img, caption=caption, use_container_width=True)
        else:
            st.warning("No regression model results found.")
            
    # Feature selection tab
    with tabs[2]:
        st.markdown("### Feature Selection Results")
        
        if feature_rank_images:
            for file in feature_rank_images:
                img = Image.open(os.path.join(results_dir, file))
                target = file.replace('feature_ranks_', '').replace('.png', '')
                st.markdown(f"#### Feature Importance for {target}")
                st.image(img, use_container_width=True)
        else:
            st.warning("No feature selection results found.")
            
    # Check for model files
<<<<<<< HEAD
    models_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/models'
=======
    models_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/models'
>>>>>>> 0181d62 (update excited)
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') or f.endswith('.pkl')]
        
        if model_files:
            st.subheader("Trained Models")
            
            for file in model_files:
                model_path = os.path.join(models_dir, file)
                create_download_link(model_path, f"Download {file}")
                
    # Create download link for all results
    create_download_zip(results_dir, "Download all modeling results")

def create_download_link(file_path, text):
    """Create a download link for a file"""
    with open(file_path, 'rb') as f:
        data = f.read()
        
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def create_download_zip(directory, text):
    """Create a ZIP download link for all files in a directory"""
    # Create a BytesIO object
    zip_buffer = BytesIO()
    
    # Create a ZIP file in the BytesIO object
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.basename(file_path))
                
    # Reset the buffer position to the beginning
    zip_buffer.seek(0)
    
    # Encode as base64
    b64 = base64.b64encode(zip_buffer.read()).decode()
    
    # Get the directory name for the ZIP filename
    zip_filename = os.path.basename(directory) + "_results.zip"
    
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">{text}</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def load_modeling_page(model_agent=None):
    """Load the modeling page"""
    return render_modeling_page(model_agent)

if __name__ == "__main__":
    # For direct testing
    load_modeling_page()