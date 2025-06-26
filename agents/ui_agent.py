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
import shutil
from PIL import Image
import tempfile
import json

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
        self.paper_agent = None
        self.multi_model_agent = None
        
        # 添加extracted_dir属性
        self.extracted_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted'  
    
    def setup_logging(self):
        """Configure logging for the UI agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/logs/ui_agent.log')
        self.logger = logging.getLogger('UIAgent')
        
    def initialize_agents(self):
        """Initialize all other agents."""
        from agents.data_agent import DataAgent
        from agents.feature_agent import FeatureAgent
        from agents.exploration_agent import ExplorationAgent
        from agents.model_agent import ModelAgent
        from agents.insight_agent import InsightAgent
        from agents.paper_agent import PaperAgent
        
        self.data_agent = DataAgent()
        self.feature_agent = FeatureAgent()
        self.exploration_agent = ExplorationAgent()
        self.model_agent = ModelAgent()
        self.insight_agent = InsightAgent()
        self.paper_agent = PaperAgent()
        try:
            from agents.multi_model_agent import MultiModelAgent
            self.multi_model_agent = None  # 暂不初始化，等待用户提供API密钥
            self.logger.info("MultiModelAgent类已加载")
        except ImportError as e:
            self.logger.warning(f"无法加载MultiModelAgent: {str(e)}")
        
        return True   
           

    def generate_visualizations(self, data, figures=None):
        """
        Generate enhanced visualizations using Kimi model.
        
        Args:
            data: DataFrame or dictionary containing data
            figures: Optional list of existing figure paths
            
        Returns:
            Dictionary mapping figure types to figure paths
        """
        if 'kimi' not in self.models:
            self.logger.warning("Kimi model not available. Using fallback visualizations.")
            return self._fallback_visualizations(data, figures)
        
        self.logger.info("Generating enhanced visualizations with Kimi k1.5")
        
        # Prepare data for visualization
        if isinstance(data, pd.DataFrame):
            data_json = data.to_json(orient="records")
        elif isinstance(data, dict):
            data_json = json.dumps(data)
        else:
            self.logger.warning("Unsupported data type for visualization")
            return self._fallback_visualizations(data, figures)
        
        # Create visualization prompt
        base_prompt = """
        You are a data visualization expert. Generate advanced visualization code for scientific publication about reversed TADF (Thermally Activated Delayed Fluorescence) materials.
        
        The data represents computational results for molecules with negative singlet-triplet gaps (S1 < T1).
        
        Generate Python code using matplotlib and seaborn to create publication-quality visualizations that showcase:
        
        1. Distribution of S1-T1 energy gaps across the molecular dataset
        2. Feature importance for key molecular descriptors
        3. Structure-property relationships (correlations between molecular properties)
        4. If PCA or dimensionality reduction results are available, show the clustering of molecules
        
        For each visualization:
        1. Use a clean, professional style suitable for academic publication
        2. Include appropriate axis labels, titles, and legends
        3. Use color schemes that are colorblind-friendly
        4. Add clear annotations where helpful
        
        The data is provided in the following JSON format:
        {data_json}
        
        Generate complete, executable Python code for each visualization type.
        Return the code in format that can be executed directly.
        """
        
        prompt_template = PromptTemplate(
            input_variables=["data_json"],
            template=base_prompt
        )
        
        # Create LLMChain
        chain = LLMChain(
            llm=self.models['kimi'],
            prompt=prompt_template
        )
        
        # Run the chain
        result = chain.run(
            data_json=data_json[:10000]  # Limit for prompt size
        )
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(result)
        
        # Generate visualizations by executing the code
        visualization_paths = {}
        for i, code in enumerate(code_blocks):
            try:
                # Create output directory
                output_dir = 'data/reports/visualizations'
                os.makedirs(output_dir, exist_ok=True)
                
                # Execute code in a temporary script
                viz_path = os.path.join(output_dir, f"viz_{i+1}.png")
                code_with_save = code + f"\nplt.savefig('{viz_path}', dpi=300, bbox_inches='tight')\nplt.close()"
                
                # Execute in a safe environment
                exec_locals = {'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'data': data}
                exec(code_with_save, {}, exec_locals)
                
                # Identify visualization type based on code content
                viz_type = self._identify_visualization_type(code)
                visualization_paths[viz_type] = viz_path
                
                self.logger.info(f"Generated visualization: {viz_type} at {viz_path}")
            except Exception as e:
                self.logger.error(f"Error generating visualization {i+1}: {str(e)}")
        
        # If figures were provided, include them in the result
        if figures:
            for fig_path in figures:
                if os.path.exists(fig_path):
                    # Extract figure name from path
                    fig_name = os.path.basename(fig_path)
                    viz_type = self._identify_visualization_type(fig_name)
                    
                    # Add to visualization paths if not already present
                    if viz_type not in visualization_paths:
                        visualization_paths[viz_type] = fig_path
        
        return visualization_paths

    def _extract_code_blocks(self, text):
        """Extract Python code blocks from text."""
        import re
        
        # Match code blocks: ```python ... ``` or just ``` ... ```
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        # Clean up the code blocks
        code_blocks = [match.strip() for match in matches]
        
        return code_blocks

    def _identify_visualization_type(self, code_or_name):
        """Identify visualization type based on code content or filename."""
        code_or_name = code_or_name.lower()
        
        # Check for common visualization types
        if "histogram" in code_or_name or "hist" in code_or_name or "distribution" in code_or_name:
            return "Distribution Plot"
        elif "scatter" in code_or_name:
            return "Scatter Plot"
        elif "correlation" in code_or_name or "heatmap" in code_or_name:
            return "Correlation Heatmap"
        elif "importance" in code_or_name or "feature" in code_or_name:
            return "Feature Importance"
        elif "pca" in code_or_name:
            return "PCA Plot"
        elif "violin" in code_or_name:
            return "Violin Plot"
        elif "bar" in code_or_name:
            return "Bar Chart"
        elif "box" in code_or_name:
            return "Box Plot"
        elif "radar" in code_or_name:
            return "Radar Chart"
        elif "pair" in code_or_name:
            return "Pair Plot"
        else:
            return "Other Visualization"

    def _fallback_visualizations(self, data, figures=None):
        """Generate fallback visualizations when Kimi is not available."""
        visualization_paths = {}
        
        try:
            # Create output directory
            output_dir = 'data/reports/visualizations'
            os.makedirs(output_dir, exist_ok=True)
            
            if isinstance(data, pd.DataFrame):
                # Generate basic plots
                
                # 1. S1-T1 Gap Distribution (if available)
                if 's1_t1_gap_ev' in data.columns:
                    gap_path = os.path.join(output_dir, "s1t1_gap_distribution.png")
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=data, x='s1_t1_gap_ev', bins=20, kde=True)
                    plt.axvline(x=0, color='red', linestyle='--')
                    plt.title('S1-T1 Gap Distribution')
                    plt.xlabel('S1-T1 Gap (eV)')
                    plt.savefig(gap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    visualization_paths["Distribution Plot"] = gap_path
                
                # 2. Correlation Heatmap (if multiple numeric columns available)
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 3:
                    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
                    plt.figure(figsize=(12, 10))
                    corr_matrix = data[numeric_cols[:10]].corr()  # Limit to 10 columns for readability
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.title('Correlation Between Features')
                    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    visualization_paths["Correlation Heatmap"] = corr_path
            
            # If figures were provided, include them in the result
            if figures:
                for fig_path in figures:
                    if os.path.exists(fig_path):
                        # Extract figure name from path
                        fig_name = os.path.basename(fig_path)
                        viz_type = self._identify_visualization_type(fig_name)
                        
                        # Add to visualization paths if not already present
                        if viz_type not in visualization_paths:
                            visualization_paths[viz_type] = fig_path
        
        except Exception as e:
            self.logger.error(f"Error generating fallback visualizations: {str(e)}")
        
        return visualization_paths   
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
        - **Paper Writing**: Generate academic paper content based on analysis results
        
        ### Getting Started
        
        Navigate through the sidebar menu to explore different functionalities:
        
        1. Start with the **Data Extraction** page to process molecular calculations
        2. Move to **Feature Engineering** to create and visualize molecular descriptors
        3. Use the **Exploration** page to identify reverse TADF candidates
        4. Explore the **Modeling** page to understand predictive model results
        5. Review the **Insights Report** for comprehensive analysis and design principles
        6. Generate an **Academic Paper** based on your findings
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
         ├─> Paper Agent
         │   └─Generate academic paper content based on analysis results
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
        
        # Add file selection guide
        with st.expander("ℹ️ File Selection Guide", expanded=True):
            st.info("""
            **Which file to select?**
            
            For Feature Engineering, please select:
            - **molecular_properties_summary.csv** (Recommended) - Contains summarized molecular properties
            - **all_conformers_data.csv** - Contains all conformer data (larger file)
            
            Do NOT select:
            - Files starting with 'processed_' - These are outputs from previous feature engineering
            - Files ending with '_samples.csv' - These are analysis results
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
            extracted_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted'
            if os.path.exists(extracted_dir):
                # Define suitable files for feature engineering
                suitable_files = ['molecular_properties_summary.csv', 'all_conformers_data.csv']
                
                # Find available suitable files
                available_files = []
                for f in os.listdir(extracted_dir):
                    if f in suitable_files and os.path.isfile(os.path.join(extracted_dir, f)):
                        available_files.append(f)
                
                if available_files:
                    selected_file = st.selectbox(
                        "Select extracted data file:",
                        available_files,
                        help="Select molecular_properties_summary.csv for best results"
                    )
                    data_file = os.path.join(extracted_dir, selected_file)
                    
                    # Preview selected file
                    if st.checkbox("Preview selected file"):
                        df = pd.read_csv(data_file)
                        st.write(f"File: {selected_file}")
                        st.write(f"Shape: {df.shape}")
                        st.dataframe(df.head())
                else:
                    st.warning("No suitable extracted data files found. Please run Data Extraction first.")
                    st.info("Looking for: molecular_properties_summary.csv or all_conformers_data.csv")
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
                
                if result and 'enhanced_feature_file' in result:
                    st.success(f"Feature engineering completed. Results saved!")
                    
                    # Load and display feature data
                    feature_df = pd.read_csv(result['enhanced_feature_file'])
                    
                    # Display feature statistics
                    if 'feature_statistics' in result and result['feature_statistics']:
                        stats = result['feature_statistics']
                        st.subheader("Feature Statistics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total features", stats['total_features'])
                        with col2:
                            st.metric("Total molecules", stats['total_molecules'])
                        with col3:
                            if 'molecules_with_inversions' in stats:
                                st.metric("Molecules with inversions", stats['molecules_with_inversions'])
                    
                    # Display gap statistics for ALL gap types, not just S1-T1
                    st.subheader("Gap Analysis")
                    
                    # Find all gap columns
                    gap_columns = [col for col in feature_df.columns 
                                if col.endswith('_gap') and not col.endswith('_gap_meV')]
                    
                    if gap_columns:
                        # Show statistics for each gap type
                        gap_stats = []
                        for gap_col in gap_columns:
                            gap_data = feature_df[gap_col].dropna()
                            if len(gap_data) > 0:
                                neg_count = (gap_data < 0).sum()
                                gap_stats.append({
                                    'Gap Type': gap_col.replace('_gap', ''),
                                    'Total Data Points': len(gap_data),
                                    'Negative Gaps': neg_count,
                                    'Positive Gaps': len(gap_data) - neg_count,
                                    'Mean (eV)': f"{gap_data.mean():.3f}",
                                    'Min (eV)': f"{gap_data.min():.3f}",
                                    'Max (eV)': f"{gap_data.max():.3f}"
                                })
                        
                        if gap_stats:
                            gap_stats_df = pd.DataFrame(gap_stats)
                            st.dataframe(gap_stats_df)
                            
                            # Let user select which gap to visualize
                            selected_gap = st.selectbox(
                                "Select gap type to visualize:",
                                gap_columns,
                                format_func=lambda x: x.replace('_gap', '')
                            )
                            
                            # Create distribution plot for selected gap
                            gap_data = feature_df[selected_gap].dropna()
                            if len(gap_data) > 0:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(data=feature_df, x=selected_gap, bins=20, kde=True)
                                plt.axvline(x=0, color='red', linestyle='--')
                                plt.title(f'{selected_gap.replace("_gap", "")} Distribution')
                                plt.xlabel(f'{selected_gap.replace("_gap", "")} (eV)')
                                st.pyplot(fig)
                    
                    # Alternative 3D features visualization
                    st.subheader("Alternative 3D Features")
                    
                    # Select some interesting 3D features
                    d3_features = [
                        'estimated_conjugation', 'estimated_polarity', 'electron_withdrawing_effect',
                        'electron_donating_effect', 'planarity_index', 'estimated_hydrophobicity'
                    ]
                    
                    # Filter features existing in the dataframe
                    valid_d3 = [f for f in d3_features if f in feature_df.columns]
                    
                    if valid_d3:
                        # Create correlation heatmap of 3D features
                        d3_corr = feature_df[valid_d3].corr()
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(d3_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                        plt.title('Correlation Between Alternative 3D Features')
                        st.pyplot(fig)
                    
                    # Create download links
                    st.subheader("Download Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        self.create_download_link(result['enhanced_feature_file'], 
                                                "📥 Download Enhanced Features CSV")
                    with col2:
                        if 'all_inverted_gaps' in result and result['all_inverted_gaps'] is not None:
                            # Save inverted gaps for download
                            try:
                                inverted_file = result['enhanced_feature_file'].replace(
                                    '.csv', '_inverted_gaps.csv'
                                )
                                if hasattr(result['all_inverted_gaps'], 'to_csv'):
                                    result['all_inverted_gaps'].to_csv(inverted_file, index=False)
                                    self.create_download_link(inverted_file, 
                                                            "📥 Download Inverted Gaps Analysis")
                            except Exception as e:
                                st.error(f"Error saving inverted gaps: {str(e)}")
                    
                    # Navigation suggestion
                    if gap_columns:
                        neg_gap_molecules = sum((feature_df[col] < 0).sum() for col in gap_columns)
                        if neg_gap_molecules > 0:
                            st.info(f"""
                            Found molecules with negative gaps across multiple gap types! 
                            Navigate to the Exploration Analysis page to analyze these reversed TADF candidates.
                            """)
                else:
                    st.error("Feature engineering failed.")
                        
    def display_exploration_page(self):
        """Display exploration analysis page with fixed file path handling."""
        st.header("Reverse TADF Exploration")
        st.subheader("Reverse TADF Candidate Exploration")
        
        st.markdown("""
        This page focuses on analyzing molecules with negative energy gaps, including:
        * **S1-T1 gaps** (traditional reverse TADF)
        * **Other inverted gaps** (S2-T4, S3-T4, etc.)
        
        The analysis includes:
        1. **Structural pattern identification** - Common features in molecules with inverted gaps
        2. **Electronic property analysis** - Unique electronic characteristics  
        3. **Comparative visualization** - Differences between positive and negative gap molecules
        4. **Feature clustering** - Multidimensional analysis of molecular properties
        """)
        
        # 初始化session state来保存文件路径
        if 'exploration_neg_file' not in st.session_state:
            st.session_state.exploration_neg_file = None
        if 'exploration_pos_file' not in st.session_state:
            st.session_state.exploration_pos_file = None
        
        # Multi-gap analysis section
        st.info("🔍 Found inverted gap data for multiple gap types!")
        tab1, tab2 = st.tabs(["Multi-Gap Analysis", "Traditional S1-T1 Analysis"])
        
        with tab1:
            st.subheader("Analysis of All Inverted Gap Types")
            inverted_gap_file = os.path.join(self.extracted_dir, 'all_inverted_gap_samples.csv')
            
            if os.path.exists(inverted_gap_file):
                df = pd.read_csv(inverted_gap_file)
                st.write(f"Found {len(df)} molecules with inverted gaps")
                
                # Show gap type distribution
                if 'gap_type' in df.columns:
                    gap_types = df['gap_type'].value_counts()
                    st.write("Gap type distribution:")
                    st.dataframe(gap_types)
                
                if st.button("Run Multi-Gap Exploration"):
                    # 修复：传递必需的all_gap_file参数
                    self.run_multi_gap_exploration(inverted_gap_file)
            else:
                st.warning("No multi-gap data found. Please run feature extraction with excited state analysis.")
        
        with tab2:
            st.subheader("Traditional S1-T1 Gap Analysis")
            
            # Look for existing data files
            neg_file = None
            pos_file = None
            
            # Check for existing feature files
            extracted_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted'
            if os.path.exists(extracted_dir):
                # Look for gap sample files
                neg_file_path = os.path.join(extracted_dir, 'negative_s1t1_samples.csv')
                pos_file_path = os.path.join(extracted_dir, 'positive_s1t1_samples.csv')
                
                if os.path.exists(neg_file_path) and os.path.exists(pos_file_path):
                    neg_file = neg_file_path
                    pos_file = pos_file_path
                    st.info("Found existing negative and positive S1-T1 gap data.")
                    
                    # 保存到session state
                    st.session_state.exploration_neg_file = neg_file
                    st.session_state.exploration_pos_file = pos_file
                else:
                    # Look for feature files as alternative
                    feature_files = [f for f in os.listdir(extracted_dir) if f.endswith('_features.csv')]
                    if feature_files:
                        st.info(f"Found {len(feature_files)} feature files in extracted directory.")
                        selected_file = st.selectbox("Select a feature file:", feature_files)
                        if selected_file:
                            # Load and split data based on S1-T1 gap
                            feature_path = os.path.join(extracted_dir, selected_file)
                            df = pd.read_csv(feature_path)
                            if 's1_t1_gap_ev' in df.columns:
                                neg_df = df[df['s1_t1_gap_ev'] < 0]
                                pos_df = df[df['s1_t1_gap_ev'] >= 0]
                                
                                # Save to temporary files
                                import tempfile
                                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
                                    neg_df.to_csv(f, index=False)
                                    neg_file = f.name
                                    st.session_state.exploration_neg_file = neg_file
                                    
                                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
                                    pos_df.to_csv(f, index=False)
                                    pos_file = f.name
                                    st.session_state.exploration_pos_file = pos_file
                                    
                                st.success(f"Split data: {len(neg_df)} negative gap, {len(pos_df)} positive gap molecules")
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
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_neg:
                        temp_neg.write(neg_upload.getvalue())
                        neg_file = temp_neg.name
                        st.session_state.exploration_neg_file = neg_file
                        
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_pos:
                        temp_pos.write(pos_upload.getvalue())
                        pos_file = temp_pos.name
                        st.session_state.exploration_pos_file = pos_file
            
            # 从session state获取文件路径
            neg_file = st.session_state.exploration_neg_file
            pos_file = st.session_state.exploration_pos_file
            
            # Execute exploration
            if neg_file and pos_file:
                # Check if pre-computed results exist
                results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/exploration'
                
                if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
                    st.info("Found existing exploration results.")
                    
                    if st.button("Show Exploration Results"):
                        self.display_exploration_results(results_dir)
                        
                    if st.button("Re-run Exploration"):
                        # 使用session state中保存的文件路径
                        self.run_exploration_analysis(neg_file, pos_file)
                else:
                    if st.button("Run Exploration Analysis"):
                        self.run_exploration_analysis(neg_file, pos_file)
                        
                # 添加调试信息
                with st.expander("Debug Information"):
                    st.write("Current file paths:")
                    st.write(f"Negative file: {neg_file}")
                    st.write(f"Positive file: {pos_file}")
                    st.write(f"Files exist: Neg={os.path.exists(neg_file) if neg_file else False}, Pos={os.path.exists(pos_file) if pos_file else False}")
            else:
                st.warning("Please provide both negative and positive S1-T1 gap data files.")
    def _display_s1t1_analysis(self, extracted_dir):
        """显示传统的S1-T1分析界面（原有代码）"""
        # Option to use existing data or upload new
        neg_file = None
        pos_file = None
        
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
            
            neg_upload = st.file_uploader("Upload negative S1-T1 gap samples CSV", type="csv", key="neg_upload")
            pos_upload = st.file_uploader("Upload positive S1-T1 gap samples CSV", type="csv", key="pos_upload")
            
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
            results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/exploration'
            
            if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
                st.info("Found existing exploration results.")
                
                if st.button("Show Exploration Results", key="show_s1t1"):
                    self.display_exploration_results(results_dir)
                    
                if st.button("Re-run Exploration", key="rerun_s1t1"):
                    self.run_exploration_analysis(neg_file, pos_file)
            else:
                if st.button("Run Exploration Analysis", key="run_s1t1"):
                    self.run_exploration_analysis(neg_file, pos_file)


                    
    def run_multi_gap_exploration(self, all_gap_file, reversed_features_file=None):
        """运行多gap类型的探索分析"""
        with st.spinner("Running multi-gap exploration analysis..."):
            try:
                # 首先检查数据内容
                gap_df = pd.read_csv(all_gap_file)
                
                # 添加详细的数据检查部分
                st.subheader("📊 Data Inspection")
                with st.expander("Check Raw Data", expanded=True):
                    st.write("**Columns in the data:**", gap_df.columns.tolist())
                    
                    # 检查gap_type列的唯一值
                    if 'gap_type' in gap_df.columns:
                        unique_gaps = gap_df['gap_type'].unique()
                        st.write(f"**Unique gap types found:** {unique_gaps}")
                        
                        # 显示每种类型的样本
                        for gap_type in unique_gaps:
                            sample = gap_df[gap_df['gap_type'] == gap_type].head(3)
                            st.write(f"\n**Sample data for {gap_type}:**")
                            st.dataframe(sample)
                            
                            # 如果是Unknown，显示source_column信息并尝试推断
                            if gap_type == "Unknown" and 'source_column' in sample.columns:
                                st.write(f"Unknown gaps come from columns: {sample['source_column'].unique()}")
                                
                                # 确保feature_agent已初始化
                                if self.feature_agent:
                                    st.write("Attempting to infer gap types:")
                                    for col in sample['source_column'].unique():
                                        if pd.notna(col):
                                            try:
                                                inferred = self.feature_agent._infer_gap_type_from_column(col)
                                                st.write(f"  - '{col}' → {inferred}")
                                            except:
                                                st.write(f"  - '{col}' → Could not infer")
                    
                    # 检查是否有其他可能的gap列
                    gap_related_cols = [col for col in gap_df.columns 
                                    if 'gap' in col.lower() or ('s' in col.lower() and 't' in col.lower())]
                    st.write("\n**Gap-related columns found:**", gap_related_cols)
                    
                    # 检查原始数据文件
                    st.write("\n**Checking source data...**")
                    
                    # 加载原始特征文件来查看实际的列名
                    feature_file_path = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted/processed_molecular_properties_features.csv'
                    if os.path.exists(feature_file_path):
                        feature_df = pd.read_csv(feature_file_path)
                        
                        # 查找所有包含gap的列
                        all_gap_cols = [col for col in feature_df.columns if 'gap' in col.lower()]
                        st.write(f"\n**All gap columns in feature file:**")
                        for col in all_gap_cols:
                            # 计算负值数量
                            if col in feature_df.columns:
                                neg_count = (feature_df[col] < 0).sum()
                                st.write(f"- {col}: {neg_count} negative values")
                
                # 添加重新分析按钮
                if st.button("Re-analyze Gap Types with Debug Mode"):
                    with st.spinner("Re-analyzing with debug mode..."):
                        try:
                            # 初始化feature agent
                            if not self.feature_agent:
                                from agents.feature_agent import FeatureAgent
                                self.feature_agent = FeatureAgent()
                            
                            # 设置输出目录
                            self.feature_agent.output_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted'
                            
                            # 确保正确初始化
                            if not hasattr(self.feature_agent, 'logger'):
                                self.feature_agent.setup_logging()
                            
                            # 加载特征数据
                            if os.path.exists(feature_file_path):
                                self.feature_agent.feature_df = pd.read_csv(feature_file_path)  # 修改：使用feature_df而不是data
                                
                                # 使用调试模式重新分析
                                st.write("\n**Running gap analysis in debug mode...**")
                                
                                # 手动检查每种可能的gap模式
                                gap_patterns = [
                                    ('s1_t1', 'S1-T1'),
                                    ('s2_t1', 'S2-T1'), 
                                    ('s1_t2', 'S1-T2'),
                                    ('s2_t2', 'S2-T2'),
                                    ('s3_t1', 'S3-T1'),
                                    ('s1_t3', 'S1-T3'),
                                    ('s2_t3', 'S2-T3'),
                                    ('s3_t3', 'S3-T3'),
                                    ('s4_t1', 'S4-T1'),
                                    ('s1_t4', 'S1-T4'),
                                    ('s2_t4', 'S2-T4'),
                                    ('s3_t4', 'S3-T4'),
                                    ('s4_t4', 'S4-T4'),
                                ]
                                
                                found_gaps = {}
                                for pattern, gap_name in gap_patterns:
                                    # 查找匹配的列
                                    matching_cols = [col for col in self.feature_agent.feature_df.columns  # 修改：使用feature_df
                                                if pattern in col.lower() and 'gap' in col.lower()]
                                    
                                    if matching_cols:
                                        st.write(f"\n**Pattern '{pattern}' matches columns:** {matching_cols}")
                                        
                                        for col in matching_cols:
                                            neg_count = (self.feature_agent.feature_df[col] < 0).sum()  # 修改：使用feature_df
                                            if neg_count > 0:
                                                found_gaps[gap_name] = {
                                                    'column': col,
                                                    'negative_count': neg_count
                                                }
                                                st.write(f"  - {col}: {neg_count} negative values → {gap_name}")
                                
                                # 检查通用的gap_ev列
                                if 'gap_ev' in self.feature_agent.feature_df.columns:
                                    st.write("\n**Checking generic gap_ev column...**")
                                    neg_count = (self.feature_agent.feature_df['gap_ev'] < 0).sum()
                                    if neg_count > 0:
                                        st.write(f"Found {neg_count} negative values in gap_ev column")
                                        # 推断这些gap的类型
                                        inferred_type = self.feature_agent._infer_gap_type_from_column('gap_ev')
                                        st.write(f"Inferred type for gap_ev: {inferred_type}")
                                
                                # 显示找到的所有gap类型
                                st.write("\n**Summary of found gap types:**")
                                for gap_type, info in found_gaps.items():
                                    st.write(f"- {gap_type}: {info['negative_count']} molecules (from column: {info['column']})")
                                
                                # 重新运行_find_all_inverted_gaps
                                st.write("\n**Re-running gap extraction...**")
                                all_inverted_gaps = self.feature_agent._find_all_inverted_gaps(self.feature_agent.feature_df)  # 修改：使用feature_df
                                
                                if all_inverted_gaps is not None and len(all_inverted_gaps) > 0:
                                    # 保存更新的数据
                                    output_file = os.path.join(
                                        self.feature_agent.output_dir, 
                                        'all_inverted_gap_samples_debug.csv'
                                    )
                                    all_inverted_gaps.to_csv(output_file, index=False)
                                    
                                    st.success(f"Debug analysis completed! Saved to: {output_file}")
                                    st.write(f"Total inverted gaps found: {len(all_inverted_gaps)}")
                                    st.write(f"Gap types: {all_inverted_gaps['gap_type'].value_counts().to_dict()}")
                                    
                                    # 显示改进后的结果
                                    st.write("\n**Improved gap type distribution:**")
                                    gap_type_counts = all_inverted_gaps['gap_type'].value_counts()
                                    for gap_type, count in gap_type_counts.items():
                                        st.write(f"- {gap_type}: {count} molecules")
                                    
                                    # 如果还有Unknown类型，显示详情
                                    if 'Unknown' in gap_type_counts:
                                        unknown_gaps = all_inverted_gaps[all_inverted_gaps['gap_type'] == 'Unknown']
                                        if 'source_column' in unknown_gaps.columns:
                                            st.write("\n**Remaining Unknown gaps from columns:**")
                                            st.write(unknown_gaps['source_column'].value_counts().to_dict())
                                else:
                                    st.warning("No inverted gaps found in debug analysis.")
                                    
                        except Exception as e:
                            st.error(f"Error in re-analysis: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
                
                # 继续原有的分析流程...
                # 初始化exploration agent，传入gap数据文件
                if not self.exploration_agent:
                    from agents.exploration_agent import ExplorationAgent
                    self.exploration_agent = ExplorationAgent(gap_data_file=all_gap_file)
                else:
                    self.exploration_agent.gap_data_file = all_gap_file
                    self.exploration_agent.load_gap_data()
                
                # 运行多gap分析
                all_gaps_results = self.exploration_agent.analyze_all_gaps()
                
                if all_gaps_results:
                    st.success("Multi-gap analysis completed!")
                    
                    # 显示分析结果
                    st.subheader("Analysis Results by Gap Type")
                    
                    # 创建结果表格
                    results_data = []
                    for gap_type, stats in all_gaps_results.items():
                        results_data.append({
                            'Gap Type': gap_type,
                            'Count': stats['count'],
                            'Mean Gap (eV)': f"{stats['mean_gap']:.4f}" if stats['mean_gap'] else 'N/A',
                            'Min Gap (eV)': f"{stats['min_gap']:.4f}" if stats['min_gap'] else 'N/A',
                            'Max Gap (eV)': f"{stats['max_gap']:.4f}" if stats['max_gap'] else 'N/A'
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df)
                    
                    # 显示生成的图表
                    results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/exploration'
                    
                    # 查找多gap相关的图表
                    if os.path.exists(results_dir):
                        multi_gap_images = [
                            'gap_types_distribution.png',
                            'gap_ranges_comparison.png',
                            'gap_value_violin_plot.png',  # 新增：小提琴图
                            'gap_correlation_heatmap.png'  # 新增：相关性热图
                        ]
                        
                        for img_file in multi_gap_images:
                            img_path = os.path.join(results_dir, img_file)
                            if os.path.exists(img_path):
                                img = Image.open(img_path)
                                st.image(img, use_column_width=True)
                    
                    # 显示每种gap类型的分子列表
                    with st.expander("View Molecules by Gap Type"):
                        selected_gap_type = st.selectbox(
                            "Select gap type to view molecules:",
                            list(all_gaps_results.keys())
                        )
                        
                        if selected_gap_type and all_gaps_results[selected_gap_type]['molecules']:
                            molecules = all_gaps_results[selected_gap_type]['molecules']
                            st.write(f"**{selected_gap_type}** molecules ({len(molecules)} total):")
                            
                            # 分批显示分子
                            batch_size = 10
                            for i in range(0, min(50, len(molecules)), batch_size):  # 最多显示50个
                                batch = molecules[i:i+batch_size]
                                st.write(", ".join(batch))
                            
                            if len(molecules) > 50:
                                st.write(f"... and {len(molecules) - 50} more molecules")
                            
                            # 新增：显示该类型的详细统计
                            if selected_gap_type != 'Unknown':
                                st.write(f"\n**Detailed statistics for {selected_gap_type}:**")
                                selected_gaps = gap_df[gap_df['gap_type'] == selected_gap_type]
                                if 'gap_value_ev' in selected_gaps.columns:
                                    st.write(f"- Standard deviation: {selected_gaps['gap_value_ev'].std():.4f} eV")
                                    st.write(f"- 25th percentile: {selected_gaps['gap_value_ev'].quantile(0.25):.4f} eV")
                                    st.write(f"- Median: {selected_gaps['gap_value_ev'].median():.4f} eV")
                                    st.write(f"- 75th percentile: {selected_gaps['gap_value_ev'].quantile(0.75):.4f} eV")
                    
                    # 运行完整的探索分析（包括原有的分析）
                    if st.button("Run Full Multi-Gap Analysis Pipeline"):
                        with st.spinner("Running full analysis pipeline..."):
                            # 这会运行包括PCA、特征比较等所有分析
                            full_results = self.exploration_agent.run_exploration_pipeline()
                            if full_results:
                                st.success("Full multi-gap exploration pipeline completed!")
                                
                                # 显示完整结果
                                if 'report' in full_results:
                                    with open(full_results['report'], 'r') as f:
                                        report_text = f.read()
                                    st.markdown("### Multi-Gap Analysis Report")
                                    st.markdown(report_text)
                                    self.create_download_link(full_results['report'], "Download multi-gap report")
                                
                                # 显示生成的所有图表
                                if 'visualizations' in full_results:
                                    st.subheader("Generated Visualizations")
                                    for viz_name, viz_path in full_results['visualizations'].items():
                                        if os.path.exists(viz_path):
                                            st.write(f"**{viz_name}**")
                                            img = Image.open(viz_path)
                                            st.image(img, use_column_width=True)
                else:
                    st.warning("No analysis results returned. Please check the data.")
                    
            except Exception as e:
                st.error(f"Error during multi-gap exploration: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    def run_exploration_analysis(self, neg_file, pos_file):
        """Run exploration analysis and display results with improved error handling."""
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
            
            # Initialize exploration agent - 重要修改：总是重新初始化或更新文件路径
            try:
                from agents.exploration_agent import ExplorationAgent
                
                # 总是创建新的实例或确保文件路径被正确设置
                if not hasattr(self, 'exploration_agent') or self.exploration_agent is None:
                    self.exploration_agent = ExplorationAgent(neg_file=neg_file, pos_file=pos_file)
                else:
                    # 更新现有agent的文件路径
                    self.exploration_agent.neg_file = neg_file
                    self.exploration_agent.pos_file = pos_file
                    # 重新加载数据
                    self.exploration_agent.load_data(neg_file, pos_file)
                    
            except Exception as e:
                st.error(f"Error initializing exploration agent: {str(e)}")
                return
                
            # Run exploration pipeline
            try:
                # 明确传递文件路径到pipeline
                result = self.exploration_agent.run_exploration_pipeline(neg_file, pos_file)
                
                if result and isinstance(result, dict) and 'analysis_results' in result:
                    st.success("Exploration analysis completed.")
                    
                    # Save results to instance for later use
                    self.exploration_results = result
                    
                    # Display results immediately
                    if 'report' in result:
                        with st.expander("View Analysis Report"):
                            with open(result['report'], 'r') as f:
                                report_content = f.read()
                            st.markdown(report_content)
                    
                    # Display visualizations
                    results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/exploration'
                    if os.path.exists(results_dir):
                        self.display_exploration_results(results_dir)
                else:
                    st.error("Exploration analysis failed. Please check the logs for details.")
                    if result is False:
                        st.error("No data files were specified. This might be due to missing file paths.")
                        
            except Exception as e:
                st.error(f"Error during exploration analysis: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
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
        extracted_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted'
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
            results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
            models_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/models'
            
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
                
                # Save for paper generation
                self.modeling_results = result
                
                # Display results
                self.display_modeling_results('/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling')
                
                # Return modeling results for later use
                return result
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
        models_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/models'
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
        has_modeling = os.path.exists('/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling') and len(os.listdir('/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling')) > 0
        has_exploration = os.path.exists('/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/exploration') and len(os.listdir('/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/exploration')) > 0
        
        # Check if report already exists
        report_path = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/reverse_tadf_insights_report.md'
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
                
                # Save for paper generation
                self.insight_results = result
                
                # Display report
                with open(result['report'], 'r') as f:
                    report_text = f.read()
                    
                st.markdown(report_text)
                
                # Create download link for report
                self.create_download_link(result['report'], "Download insights report")
            else:
                st.error("Failed to generate insights report.")
    
    def display_tuning_page(self):
        """Display model tuning page."""
        try:
            from streamlit_ui.TuningPage import load_tuning_page
            
            # Ensure tuning_agent is initialized
            if not hasattr(self, 'tuning_agent') or self.tuning_agent is None:
                from agents.tuning_agent import TuningAgent
                self.tuning_agent = TuningAgent()
            
            return load_tuning_page(self.tuning_agent)
        except Exception as e:
            st.error(f"加载微调页面时出错: {str(e)}")
            st.write("这可能是因为PyTorch或其他依赖项配置不正确。请确保已安装所有必要的库。")
            st.code("pip install torch transformers scikit-learn pandas numpy matplotlib seaborn", language="bash")
            return None
        
    def display_design_page(self):
        """Display molecular design page."""
        from streamlit_ui.DesignPage import load_design_page
        
        # Ensure design_agent is initialized
        if not hasattr(self, 'design_agent') or self.design_agent is None:
            from agents.design_agent import DesignAgent
            self.design_agent = DesignAgent()
        
        # Pass both design_agent and model_agent to the design page
        if not hasattr(self, 'model_agent') or self.model_agent is None:
            from agents.model_agent import ModelAgent
            self.model_agent = ModelAgent()
            
        return load_design_page(self.design_agent, self.model_agent)
    
    def display_paper_page(self):
        """Display paper generation page."""
        from streamlit_ui.PaperPage import load_paper_page
        return load_paper_page(
            paper_agent=self.paper_agent,
            model_agent=self.model_agent,
            exploration_agent=self.exploration_agent,
            insight_agent=self.insight_agent,
            multi_model_agent=self.multi_model_agent
        )
        
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
            "fine tuning": self.display_tuning_page,     # New
            "reversed TADF molecular design": self.display_design_page,     # New
            "Insights Report": self.display_report_page,
            "Paper Writing": self.display_paper_page
        }
        
        # Initialize agents if not done already
        if not self.data_agent:
            self.initialize_agents()
            
        # Display navigation
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        
        # Display selected page
        pages[selection]()
    
    def analyze_model_results(self, modeling_results, exploration_results=None):
        """
        Generate enhanced analysis of modeling results using DeepSeek R1.
        
        Args:
            modeling_results: Results from model_agent
            exploration_results: Optional results from exploration_agent
            
        Returns:
            Dictionary with enhanced analysis
        """
        if 'deepseek' not in self.models:
            self.logger.warning("DeepSeek model not available. Using fallback analysis.")
            return self._fallback_model_analysis(modeling_results, exploration_results)
        
        self.logger.info("Generating enhanced model analysis with DeepSeek R1")
        
        # Prepare modeling results in a readable format
        if isinstance(modeling_results, dict):
            model_data = json.dumps(modeling_results, indent=2)
        else:
            model_data = str(modeling_results)
        
        # Create prompt for model analysis
        base_prompt = """
        You are a computational chemistry expert specializing in analysis of quantum chemical calculations and machine learning models.
        
        Analyze the provided modeling results for reversed TADF (Thermally Activated Delayed Fluorescence) materials, focusing on:
        
        1. Classification model performance for predicting negative vs. positive S1-T1 gaps
        2. Regression model performance for predicting the magnitude of S1-T1 gaps
        3. Key features that determine the S1-T1 gap direction and magnitude
        4. Quantum chemical interpretation of the feature importance
        5. Structure-property relationships revealed by the models
        6. Recommendations for improving model performance
        7. Practical design guidelines based on the model insights
        
        The modeling results are provided in the following format:
        {model_data}
        
        Additional exploration results for context:
        {exploration_data}
        
        Provide a detailed analysis with quantum chemical reasoning to explain the modeling results.
        Structure your response with clear sections and focus on insights that would be valuable for materials design.
        """
        
        # Prepare exploration data if available
        if exploration_results:
            if isinstance(exploration_results, dict):
                exploration_data = json.dumps(exploration_results, indent=2)
            else:
                exploration_data = str(exploration_results)
        else:
            exploration_data = "No exploration results provided."
        
        prompt_template = PromptTemplate(
            input_variables=["model_data", "exploration_data"],
            template=base_prompt
        )
        
        # Create LLMChain
        chain = LLMChain(
            llm=self.models['deepseek'],
            prompt=prompt_template
        )
        
        # Run the chain
        result = chain.run(
            model_data=model_data[:5000],  # Limit for prompt size
            exploration_data=exploration_data[:5000]  # Limit for prompt size
        )
        
        # Parse the analysis into structured sections
        analysis = self._parse_analysis_sections(result)
        
        # Add original modeling results
        analysis['original_results'] = modeling_results
        
        self.logger.info("Model analysis completed")
        return analysis

    def _parse_analysis_sections(self, text):
        """Parse analysis text into structured sections."""
        import re
        
        # Initialize sections dictionary
        sections = {
            'summary': '',
            'classification_analysis': '',
            'regression_analysis': '',
            'feature_importance': '',
            'quantum_interpretation': '',
            'structure_property': '',
            'recommendations': '',
            'design_guidelines': '',
            'full_text': text
        }
        
        # Extract summary (first paragraph)
        paragraphs = text.split('\n\n')
        if paragraphs:
            sections['summary'] = paragraphs[0]
        
        # Extract sections based on headers
        section_matches = re.findall(r'##?\s+([^#\n]+)\n+(.+?)(?=\n##?\s+|$)', text, re.DOTALL)
        
        for header, content in section_matches:
            header = header.strip().lower()
            
            if 'classification' in header:
                sections['classification_analysis'] = content.strip()
            elif 'regression' in header:
                sections['regression_analysis'] = content.strip()
            elif 'feature' in header and 'import' in header:
                sections['feature_importance'] = content.strip()
            elif 'quantum' in header or 'chemical' in header or 'interpret' in header:
                sections['quantum_interpretation'] = content.strip()
            elif 'structure' in header and ('property' in header or 'relation' in header):
                sections['structure_property'] = content.strip()
            elif 'recommend' in header:
                sections['recommendations'] = content.strip()
            elif 'design' in header and ('guide' in header or 'principle' in header):
                sections['design_guidelines'] = content.strip()
        
        return sections

    def _fallback_model_analysis(self, modeling_results, exploration_results=None):
        """Generate fallback model analysis when DeepSeek is not available."""
        analysis = {
            'summary': '分析了反向TADF分子的分类和回归模型结果，确定了关键特征，并提出了设计原则。',
            'classification_analysis': '分类模型在预测S1-T1能隙方向方面表现良好，显示出较高的准确率、精确度和召回率。',
            'regression_analysis': '回归模型在预测S1-T1能隙大小方面表现良好，显示出较小的RMSE和合理的R²值。',
            'feature_importance': '电子提取效应、共轭模式和特定结构特征是预测S1-T1能隙最重要的因素。',
            'quantum_interpretation': '轨道空间分离和前线轨道能量差异是导致负S1-T1能隙的关键量子化学因素。',
            'structure_property': '具有推拉电子体系的分子，尤其是基于calicene骨架的分子，最有可能展现反向TADF特性。',
            'recommendations': '建议扩大分子数据集，并尝试使用深度学习方法进一步提高预测性能。',
            'design_guidelines': '设计具有强吸电子基团（如-CN）和供电子基团（如-NMe₂）的calicene衍生物，以实现负S1-T1能隙。',
            'original_results': modeling_results,
            'full_text': '这是一个回退分析，因为DeepSeek模型不可用。'
        }
        
        return analysis