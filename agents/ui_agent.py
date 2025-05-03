# agents/ui_agent.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import time
import base64
from io import BytesIO
import zipfile
import tempfile
import shutil
from PIL import Image

class UIAgent:
    """
    Agent responsible for managing Streamlit UI and coordinating 
    interactions between other agents and the user interface.
    """
    
    def __init__(self):
        """Initialize the UIAgent."""
        self.setup_logging()
        self.data_agent = None
        self.feature_agent = None
        self.exploration_agent = None
        self.model_agent = None
        self.insight_agent = None
        
    def setup_logging(self):
        """Configure logging for the UI agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/logs/ui_agent.log')
        self.logger = logging.getLogger('UIAgent')
        
    def initialize_agents(self):
        """Initialize all other agents."""
        from agents.data_agent import DataAgent
        from agents.feature_agent import FeatureAgent
        from agents.exploration_agent import ExplorationAgent
        from agents.model_agent import ModelAgent
        from agents.insight_agent import InsightAgent
        
        self.data_agent = DataAgent()
        self.feature_agent = FeatureAgent()
        self.exploration_agent = ExplorationAgent()
        self.model_agent = ModelAgent()
        self.insight_agent = InsightAgent()
        
        return True
        
    def display_home_page(self):
        """Display home page content."""
        st.title("Reverse TADF Analysis System")
        
        st.markdown("""
        ## Welcome to the Reverse TADF Molecular Analysis System
        
        This application helps you analyze molecular properties for reverse Thermally Activated Delayed Fluorescence (TADF) candidates.
        
        ### What is Reverse TADF?
        
        In typical molecules, the first triplet excited state (T1) has lower energy than the first singlet excited state (S1),
        following Hund's rule. However, in some special molecules, this ordering is reversed (S1 < T1),
        creating unique photophysical properties with applications in advanced optoelectronic devices.
        
        ### System Capabilities
        
        - **Data Extraction**: Process Gaussian and CREST calculation outputs
        - **Feature Engineering**: Generate molecular descriptors and alternative 3D features
        - **Exploration**: Analyze S1-T1 gap properties and identify reverse TADF candidates
        - **Modeling**: Build predictive models for S1-T1 gap classification and regression
        - **Insights**: Generate quantum chemistry explanations and design principles
        
        ### Getting Started
        
        Navigate through the sidebar menu to explore different functionalities:
        
        1. Start with the **Data Extraction** page to process molecular calculations
        2. Move to **Feature Engineering** to create and visualize molecular descriptors
        3. Use the **Exploration** page to identify reverse TADF candidates
        4. Explore the **Modeling** page to understand predictive model results
        5. Review the **Insights Report** for comprehensive analysis and design principles
        """)
        
        # Add system architecture diagram
        st.markdown("### System Architecture")
        
        architecture_md = """
        ```
        User Interaction (Streamlit)
             ↓
        Task Chain (LangChain)
         ├─> Data Agent
         │   └─Extract Gaussian and CREST feature data
         ├─> Feature Agent
         │   └─Generate combined features, polarity/conjugation/electronic effects
         ├─> Exploration Agent
         │   └─Filter S1-T1 < 0 samples, structure difference analysis
         ├─> Model Agent
         │   └─Build positive/negative S1-T1 classification or regression model
         ├─> Insight Agent
         │   └─Generate explanations based on feature importance
         └─> UI Agent (Streamlit)
             └─Display charts + Markdown explanations + Download results
        ```
        """
        
        st.markdown(architecture_md)
        
    def display_extraction_page(self):
        """Display data extraction page."""
        st.title("Data Extraction")
        
        st.markdown("""
        ## Gaussian & CREST Data Extraction
        
        Upload Gaussian log files and CREST results to extract molecular properties.
        
        ### Expected Data Structure
        
        The system expects a specific directory structure for molecular calculations:
        
        ```
        parent_directory/
        ├── molecule_name/
        │   ├── neutral/
        │   │   └── gaussian/
        │   │       └── conf_1/
        │   │           ├── ground.log
        │   │           └── excited.log
        │   ├── cation/
        │   │   └── gaussian/...
        │   ├── triplet/
        │   │   └── gaussian/...
        │   └── results/
        │       ├── neutral_results.txt
        │       ├── cation_results.txt
        │       └── triplet_results.txt
        ```
        
        You can upload a ZIP file containing this structure or provide a directory path.
        """)
        
        # Option to upload ZIP file
        uploaded_file = st.file_uploader("Upload ZIP containing molecular data", type="zip")
        
        # Option to provide directory path
        directory_path = st.text_input("Or provide directory path on server")
        
        # Execute extraction
        if st.button("Extract Data"):
            with st.spinner("Extracting data..."):
                if uploaded_file:
                    # Save uploaded ZIP to temp location and extract
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
                        temp_zip.write(uploaded_file.getvalue())
                        temp_zip_path = temp_zip.name
                        
                    # Create temp directory for extraction
                    temp_dir = tempfile.mkdtemp()
                    
                    # Extract zip to temp directory
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                        
                    # Initialize data agent with extracted path
                    if not self.data_agent:
                        from agents.data_agent import DataAgent
                        self.data_agent = DataAgent(base_dir=temp_dir)
                    else:
                        self.data_agent.base_dir = temp_dir
                        
                    # Process molecules
                    result_file = self.data_agent.process_molecules()
                    
                    # Clean up temp files
                    os.unlink(temp_zip_path)
                    shutil.rmtree(temp_dir)
                    
                elif directory_path:
                    # Use provided directory path
                    if not self.data_agent:
                        from agents.data_agent import DataAgent
                        self.data_agent = DataAgent(base_dir=directory_path)
                    else:
                        self.data_agent.base_dir = directory_path
                        
                    # Process molecules
                    result_file = self.data_agent.process_molecules()
                    
                else:
                    st.error("Please upload a ZIP file or provide a directory path.")
                    return
                
                if result_file:
                    st.success(f"Data extraction completed. Results saved to {result_file}")
                    
                    # Display results summary
                    df = pd.read_csv(result_file)
                    st.write(f"Extracted data for {df['Molecule'].nunique()} molecules")
                    st.write(f"Total conformers: {len(df)}")
                    
                    # Show sample data
                    st.subheader("Sample Data")
                    st.dataframe(df.head())
                    
                    # Create download link for results
                    self.create_download_link(result_file, "Download extracted data CSV")
                else:
                    st.error("Data extraction failed.")
                    
    def display_feature_page(self):
        """Display feature engineering page."""
        st.title("Feature Engineering")
        
        st.markdown("""
        ## Feature Engineering & Alternative 3D Descriptors
        
        This page helps you generate and explore various molecular descriptors derived from the extracted data.
        
        Key feature categories:
        
        1. **Electronic properties** - HOMO, LUMO, electron-donating/withdrawing effects
        2. **Structural features** - Rings, substituents, planarity, conjugation
        3. **Physical properties** - Polarity, hydrophobicity, size estimates
        4. **Quantum properties** - Energy levels, gaps, dipole moments
        
        You can run the feature engineering pipeline on previously extracted data or upload a new CSV file.
        """)
        
        # Option to use existing data or upload new
        data_source = st.radio("Data source", ["Use extracted data", "Upload CSV"])
        
        data_file = None
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload molecular data CSV", type="csv")
            if uploaded_file:
                # Save uploaded CSV to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                    temp_csv.write(uploaded_file.getvalue())
                    data_file = temp_csv.name
        else:
            # Look for previously extracted data
            extracted_dir = '../data/extracted'
            if os.path.exists(extracted_dir):
                csv_files = [f for f in os.listdir(extracted_dir) if f.endswith('.csv')]
                if csv_files:
                    selected_file = st.selectbox("Select extracted data file", csv_files)
                    data_file = os.path.join(extracted_dir, selected_file)
                else:
                    st.warning("No extracted data files found. Please extract data first.")
            else:
                st.warning("No extracted data directory found. Please extract data first.")
                
        # Execute feature engineering
        if data_file and st.button("Generate Features"):
            with st.spinner("Generating features..."):
                # Initialize feature agent
                if not self.feature_agent:
                    from agents.feature_agent import FeatureAgent
                    self.feature_agent = FeatureAgent(data_file=data_file)
                else:
                    self.feature_agent.data_file = data_file
                    
                # Run feature pipeline
                result = self.feature_agent.run_feature_pipeline()
                
                if result and 'feature_file' in result:
                    st.success(f"Feature engineering completed. Results saved to {result['feature_file']}")
                    
                    # Load and display feature data
                    feature_df = pd.read_csv(result['feature_file'])
                    
                    # Display basic stats
                    st.subheader("Feature Statistics")
                    st.write(f"Total features: {len(feature_df.columns)}")
                    
                    # 显示S1-T1能隙统计（如果有）
                    # 显示S1-T1能隙统计（如果有）
                    if 's1_t1_gap_ev' in feature_df.columns:
                        gap_data = feature_df[feature_df['s1_t1_gap_ev'].notna()]
                        neg_count = (gap_data['s1_t1_gap_ev'] < 0).sum()
                        pos_count = (gap_data['s1_t1_gap_ev'] >= 0).sum()
                        
                        st.write(f"含有S1-T1能隙数据的分子: {len(gap_data['Molecule'].unique())}")
                        st.write(f"具有负S1-T1能隙的分子(逆向TADF候选物): {neg_count}")
                        
                        # 创建S1-T1能隙分布图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=gap_data, x='s1_t1_gap_ev', bins=20, kde=True)
                        plt.axvline(x=0, color='red', linestyle='--')
                        plt.title('S1-T1能隙分布')
                        plt.xlabel('S1-T1能隙 (eV)')
                        st.pyplot(fig)
                        # 保存图表
                        save_path = "data/reports/s1_t1_gap_distribution.png"
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        st.success(f"图表已保存至: {save_path}")
                        
                    # 替代3D特征
                    st.subheader("替代3D特征示例")

                    # 选择一些有趣的3D特征
                    d3_features = [
                        'estimated_conjugation', 'estimated_polarity', 'electron_withdrawing_effect',
                        'electron_donating_effect', 'planarity_index', 'estimated_hydrophobicity'
                    ]

                    # 筛选数据框中存在的特征
                    valid_d3 = [f for f in d3_features if f in feature_df.columns]

                    if valid_d3:
                        # 创建3D特征之间的相关性热图
                        d3_corr = feature_df[valid_d3].corr()
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(d3_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                        plt.title('3D特征之间的相关性')
                        st.pyplot(fig)
                        # 保存图表
                        save_path = "data/reports/3d_features_correlation.png"
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        st.success(f"相关性热图已保存至: {save_path}")
                        
                        # 显示几个关键特征的分布
                        st.subheader("特征分布")
                        
                        for i, feature in enumerate(valid_d3[:3]):  # 显示前3个特征
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.histplot(data=feature_df, x=feature, kde=True)
                            plt.title(f'{feature}的分布')
                            st.pyplot(fig)
                            # 保存图表
                            save_path = f"data/reports/{feature}_distribution.png"
                            plt.savefig(save_path, dpi=300, bbox_inches='tight')
                            st.success(f"{feature}分布图已保存至: {save_path}")
                        
                        # 创建特征下载链接
                        self.create_download_link(result['feature_file'], "下载处理后的特征CSV")
                        
                        # 如果有S1-T1能隙数据，提供导航到探索页的选项
                        if 's1_t1_gap_ev' in feature_df.columns and neg_count > 0:
                            st.info("检测到负S1-T1能隙分子。转到'探索'页面分析这些逆向TADF候选物。")
                    else:
                        st.error("特征工程失败。")
                    
    def display_exploration_page(self):
        """Display exploration analysis page."""
        st.title("Reverse TADF Exploration")
        
        st.markdown("""
        ## Reverse TADF Candidate Exploration
        
        This page focuses on analyzing molecules with negative S1-T1 energy gaps, which are potential reverse TADF candidates.
        
        The analysis includes:
        
        1. **Structural pattern identification** - Common features in reverse TADF molecules
        2. **Electronic property analysis** - Unique electronic characteristics
        3. **Comparative visualization** - Differences between positive and negative gap molecules
        4. **Feature clustering** - Multidimensional analysis of molecular properties
        
        You can run the exploration on previously generated feature data or upload feature CSV files.
        """)
        
        # Option to use existing data or upload new
        neg_file = None
        pos_file = None
        
        # Look for previously processed data
        extracted_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted'
        if os.path.exists(extracted_dir):
            neg_path = os.path.join(extracted_dir, 'negative_s1t1_samples.csv')
            pos_path = os.path.join(extracted_dir, 'positive_s1t1_samples.csv')
            
            if os.path.exists(neg_path) and os.path.exists(pos_path):
                st.info("Found existing negative and positive S1-T1 gap data.")
                neg_file = neg_path
                pos_file = pos_path
            else:
                # Look for processed features file to generate gap data
                feature_files = [f for f in os.listdir(extracted_dir) if 'feature' in f.lower() and f.endswith('.csv')]
                
                if feature_files:
                    st.info("No pre-processed gap data found, but feature files are available.")
                    selected_file = st.selectbox("Select feature file to process", feature_files)
                    
                    if st.button("Process Gap Data"):
                        with st.spinner("Processing S1-T1 gap data..."):
                            # Initialize feature agent
                            if not self.feature_agent:
                                from agents.feature_agent import FeatureAgent
                                self.feature_agent = FeatureAgent(data_file=os.path.join(extracted_dir, selected_file))
                            else:
                                self.feature_agent.data_file = os.path.join(extracted_dir, selected_file)
                                
                            # Load data and extract gap samples
                            self.feature_agent.load_data()
                            gap_results = self.feature_agent.get_negative_s1t1_samples()
                            
                            if gap_results:
                                neg_file = gap_results['negative_file']
                                pos_file = gap_results['positive_file']
                                st.success(f"Found {gap_results['negative_count']} negative and {gap_results['positive_count']} positive S1-T1 gap samples.")
                else:
                    st.warning("No feature files found. Please run feature engineering first.")
                    
        else:
            st.warning("No extracted data directory found. Please extract data and run feature engineering first.")
            
        # Option to upload files
        if not neg_file or not pos_file:
            st.subheader("Upload Gap Data")
            
            neg_upload = st.file_uploader("Upload negative S1-T1 gap samples CSV", type="csv")
            pos_upload = st.file_uploader("Upload positive S1-T1 gap samples CSV", type="csv")
            
            if neg_upload and pos_upload:
                # Save uploaded CSVs to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_neg:
                    temp_neg.write(neg_upload.getvalue())
                    neg_file = temp_neg.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_pos:
                    temp_pos.write(pos_upload.getvalue())
                    pos_file = temp_pos.name
                    
        # Execute exploration
        if neg_file and pos_file:
            # Check if pre-computed results exist
            results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/exploration'
            
            if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
                st.info("Found existing exploration results.")
                
                if st.button("Show Exploration Results"):
                    self.display_exploration_results(results_dir)
                    
                if st.button("Re-run Exploration"):
                    self.run_exploration_analysis(neg_file, pos_file)
            else:
                if st.button("Run Exploration Analysis"):
                    self.run_exploration_analysis(neg_file, pos_file)
                    
    def run_exploration_analysis(self, neg_file, pos_file):
        """Run exploration analysis and display results."""
        with st.spinner("Running exploration analysis..."):
            # Print file paths to verify they're correct
            st.write(f"Negative file path: {neg_file}")
            st.write(f"Positive file path: {pos_file}")
            
            # Check if files exist
            if not os.path.exists(neg_file):
                st.error(f"Negative file does not exist: {neg_file}")
                return
            if not os.path.exists(pos_file):
                st.error(f"Positive file does not exist: {pos_file}")
                return
                
            # Check file contents
            try:
                neg_df = pd.read_csv(neg_file)
                pos_df = pd.read_csv(pos_file)
                
                st.write(f"Negative file contains {len(neg_df)} rows and {len(neg_df.columns)} columns")
                st.write(f"Positive file contains {len(pos_df)} rows and {len(pos_df.columns)} columns")
                
                # Show sample data if available
                if not neg_df.empty:
                    st.write("Negative data sample:")
                    st.write(neg_df.head(2))
                else:
                    st.warning("Negative data file is empty!")
                    
                if not pos_df.empty:
                    st.write("Positive data sample:")
                    st.write(pos_df.head(2))
                else:
                    st.warning("Positive data file is empty!")
                    
            except Exception as e:
                st.error(f"Error reading data files: {str(e)}")
                return
            
            # Initialize exploration agent
            if not self.exploration_agent:
                from agents.exploration_agent import ExplorationAgent
                self.exploration_agent = ExplorationAgent(neg_file=neg_file, pos_file=pos_file)
            else:
                self.exploration_agent.load_data(neg_file, pos_file)
                
            # Run exploration pipeline
            result = self.exploration_agent.run_exploration_pipeline()
            
            if result and 'analysis_results' in result:
                st.success("Exploration analysis completed.")
                
                # Display results
                self.display_exploration_results('/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/exploration')
                
                # Show report link
                if 'report' in result:
                    st.subheader("Exploration Report")
                    
                    with open(result['report'], 'r') as f:
                        report_text = f.read()
                        
                    st.markdown(report_text)
                    
                    # Create download link for report
                    self.create_download_link(result['report'], "Download exploration report")
            else:
                st.error("Exploration analysis failed.")
                
    def display_exploration_results(self, results_dir):
        """Display exploration analysis results."""
        st.subheader("Exploration Analysis Results")
        
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
        gap_dist = [f for f in image_files if 'gap_distribution' in f]
        structure_comparison = [f for f in image_files if 'structural_feature' in f]
        feature_comparisons = [f for f in image_files if '_comparison.png' in f]
        pca_analysis = [f for f in image_files if 'pca_analysis' in f]
        radar_comparison = [f for f in image_files if 'radar_feature' in f]
        
        # Display gap distribution
        if gap_dist:
            st.markdown("### S1-T1 Gap Distribution")
            img = Image.open(os.path.join(results_dir, gap_dist[0]))
            st.image(img, caption="Distribution of S1-T1 Energy Gaps", use_column_width=True)
            
        # Display structural comparison
        if structure_comparison:
            st.markdown("### Structural Feature Comparison")
            img = Image.open(os.path.join(results_dir, structure_comparison[0]))
            st.image(img, caption="Top Structural Features: Negative vs Positive S1-T1 Gap", use_column_width=True)
            
        # Display radar comparison
        if radar_comparison:
            st.markdown("### Feature Radar Comparison")
            img = Image.open(os.path.join(results_dir, radar_comparison[0]))
            st.image(img, caption="Feature Comparison: Negative vs Positive S1-T1 Gap", use_column_width=True)
            
        # Display PCA analysis
        if pca_analysis:
            st.markdown("### PCA Analysis")
            img = Image.open(os.path.join(results_dir, pca_analysis[0]))
            st.image(img, caption="PCA of Molecular Properties: Negative vs Positive S1-T1 Gap", use_column_width=True)
            
        # Display feature comparisons
        if feature_comparisons:
            st.markdown("### Feature Comparisons")
            
            # Create columns for displaying multiple images
            cols = st.columns(2)
            
            for i, file in enumerate(feature_comparisons[:6]):  # Limit to 6 comparison plots
                with cols[i % 2]:
                    img = Image.open(os.path.join(results_dir, file))
                    feature_name = file.replace('_comparison.png', '').replace('_', ' ').title()
                    st.image(img, caption=feature_name, use_column_width=True)
                    
        # Create download link for all results
        self.create_download_zip(results_dir, "Download all exploration results")
                    
    def display_modeling_page(self):
        """Display modeling analysis page."""
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
        extracted_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted'
        if os.path.exists(extracted_dir):
            # Look for processed features file
            feature_files = [f for f in os.listdir(extracted_dir) if 'feature' in f.lower() or 'processed' in f.lower() and f.endswith('.csv')]
            
            if feature_files:
                st.info("Found existing feature files.")
                selected_file = st.selectbox("Select feature file for modeling", feature_files)
                feature_file = os.path.join(extracted_dir, selected_file)
            else:
                st.warning("No feature files found. Please run feature engineering first.")
        else:
            st.warning("No extracted data directory found. Please extract data and run feature engineering first.")
            
        # Option to upload file
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
            results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/modeling'
            models_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/models'
            
            if os.path.exists(results_dir) and os.path.exists(models_dir) and \
               len(os.listdir(results_dir)) > 0 and len(os.listdir(models_dir)) > 0:
                st.info("Found existing modeling results.")
                
                if st.button("Show Modeling Results"):
                    self.display_modeling_results(results_dir)
                    
                if st.button("Re-run Modeling"):
                    self.run_modeling_analysis(feature_file)
            else:
                if st.button("Run Modeling Analysis"):
                    self.run_modeling_analysis(feature_file)
                    
    def run_modeling_analysis(self, feature_file):
        """Run modeling analysis and display results."""
        with st.spinner("Running modeling analysis..."):
            # Initialize model agent
            if not self.model_agent:
                from agents.model_agent import ModelAgent
                self.model_agent = ModelAgent(feature_file=feature_file)
            else:
                self.model_agent.feature_file = feature_file
                
            # Run modeling pipeline
            result = self.model_agent.run_modeling_pipeline()
            
            if result and ('classification' in result or 'regression' in result):
                st.success("Modeling analysis completed.")
                
                # Display results
                self.display_modeling_results('/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/modeling')
                
                # Store modeling results for insight agent
                self.modeling_results = result
            else:
                st.error("Modeling analysis failed.")
                
    def display_modeling_results(self, results_dir):
        """Display modeling analysis results."""
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
                    st.image(img, caption=caption, use_column_width=True)
            else:
                st.warning("No classification model results found.")
                
        # Regression tab
        with tabs[1]:
            st.markdown("### Regression Model Results")
            
            if regression_images:
                for file in regression_images:
                    img = Image.open(os.path.join(results_dir, file))
                    caption = file.replace('.png', '').replace('_', ' ').title()
                    st.image(img, caption=caption, use_column_width=True)
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
                    st.image(img, use_column_width=True)
            else:
                st.warning("No feature selection results found.")
                
        # Check for model files
        models_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') or f.endswith('.pkl')]
            
            if model_files:
                st.subheader("Trained Models")
                
                for file in model_files:
                    model_path = os.path.join(models_dir, file)
                    self.create_download_link(model_path, f"Download {file}")
                    
        # Create download link for all results
        self.create_download_zip(results_dir, "Download all modeling results")
                    
    def display_report_page(self):
        """Display insight report page."""
        st.title("Reverse TADF Insights Report")
        
        st.markdown("""
        ## Comprehensive Insights & Design Principles
        
        This page presents a comprehensive analysis of reverse TADF molecular design principles,
        combining results from exploration analysis and predictive modeling.
        
        The report includes:
        
        1. **Quantum chemistry explanations** - Why certain features influence S1-T1 gap direction
        2. **Design principles** - Strategies for developing reverse TADF materials
        3. **Feature importance analysis** - Understanding key molecular descriptors
        4. **Structure-property relationships** - Connections between molecular structure and photophysical properties
        
        You need to run both the exploration and modeling analyses before generating this report.
        """)
        
        # Check if we have modeling and exploration results
        has_modeling = os.path.exists('/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/modeling') and len(os.listdir('../data/reports/modeling')) > 0
        has_exploration = os.path.exists('/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/exploration') and len(os.listdir('../data/reports/exploration')) > 0
        
        # Check if report already exists
        report_path = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/reverse_tadf_insights_report.md'
        has_report = os.path.exists(report_path)
        
        if has_report:
            st.info("Found existing insights report.")
            
            # Display report
            with open(report_path, 'r') as f:
                report_text = f.read()
                
            st.markdown(report_text)
            
            # Create download link for report
            self.create_download_link(report_path, "Download insights report")
            
            if st.button("Regenerate Report"):
                self.generate_insight_report()
                
        elif has_modeling and has_exploration:
            if st.button("Generate Insights Report"):
                self.generate_insight_report()
        else:
            missing = []
            if not has_modeling:
                missing.append("modeling analysis")
            if not has_exploration:
                missing.append("exploration analysis")
                
            st.warning(f"Please run {' and '.join(missing)} first.")
            
    def generate_insight_report(self):
        """Generate comprehensive insight report."""
        with st.spinner("Generating insights report..."):
            # Load modeling results
            model_results = None
            if hasattr(self, 'modeling_results'):
                model_results = self.modeling_results
                
            # Load exploration results
            exploration_results = None
            if hasattr(self, 'exploration_results'):
                exploration_results = self.exploration_results
                
            # Initialize insight agent if needed
            if not self.insight_agent:
                from agents.insight_agent import InsightAgent
                self.insight_agent = InsightAgent(
                    modeling_results=model_results, 
                    exploration_results=exploration_results
                )
            else:
                self.insight_agent.load_results(model_results, exploration_results)
                
            # Run insight pipeline
            result = self.insight_agent.run_insight_pipeline()
            
            if result and 'report' in result:
                st.success("Insights report generated successfully.")
                
                # Display report
                with open(result['report'], 'r') as f:
                    report_text = f.read()
                    
                st.markdown(report_text)
                
                # Create download link for report
                self.create_download_link(result['report'], "Download insights report")
            else:
                st.error("Failed to generate insights report.")
                
    def create_download_link(self, file_path, text):
        """Create a download link for a file."""
        with open(file_path, 'rb') as f:
            data = f.read()
            
        b64 = base64.b64encode(data).decode()
        filename = os.path.basename(file_path)
        
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    def create_download_zip(self, directory, text):
        """Create a download link for a ZIP of all files in a directory."""
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
        
    def run_app(self):
        """Run the Streamlit application."""
        st.sidebar.title("Navigation")
        
        pages = {
            "Home": self.display_home_page,
            "Data Extraction": self.display_extraction_page,
            "Feature Engineering": self.display_feature_page,
            "Exploration Analysis": self.display_exploration_page,
            "Predictive Modeling": self.display_modeling_page,
            "Insights Report": self.display_report_page
        }
        
        # Initialize agents if not done already
        if not self.data_agent:
            self.initialize_agents()
            
        # Display navigation
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        
        # Display selected page
        pages[selection]()
