# streamlit_ui/FeaturePage.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from agents.feature_agent import FeatureAgent

def display_gap_statistics(feature_df):
    """Display statistics for all gap types, not just S1-T1"""
    
    st.header("Gap Statistics")
    
    # Find all gap columns
    gap_columns = [col for col in feature_df.columns 
                   if col.endswith('_gap') and not col.endswith('_gap_meV')]
    
    if not gap_columns:
        st.warning("No gap data found in the features.")
        return
    
    # Display statistics for each gap type
    col1, col2, col3 = st.columns(3)
    
    total_molecules = len(feature_df)
    
    # Collect statistics for each gap
    gap_stats = {}
    for gap_col in gap_columns:
        gap_type = gap_col.replace('_gap', '')
        non_null_count = feature_df[gap_col].notna().sum()
        negative_count = (feature_df[gap_col] < 0).sum() if non_null_count > 0 else 0
        
        gap_stats[gap_type] = {
            'total': non_null_count,
            'negative': negative_count,
            'column': gap_col
        }
    
    # Display overall statistics
    with col1:
        st.metric("Total features", len(feature_df.columns))
    
    with col2:
        st.metric("Total molecules", total_molecules)
    
    with col3:
        st.metric("Gap types found", len(gap_columns))
    
    # Display detailed statistics for each gap type
    st.subheader("Gap Type Breakdown")
    
    # Create a summary table for all gap types
    gap_summary = []
    for gap_type, stats in gap_stats.items():
        gap_summary.append({
            'Gap Type': gap_type,
            'Molecules with data': stats['total'],
            'Negative gaps (inverted)': stats['negative'],
            'Percentage inverted': f"{(stats['negative']/stats['total']*100) if stats['total'] > 0 else 0:.1f}%"
        })
    
    gap_summary_df = pd.DataFrame(gap_summary)
    gap_summary_df = gap_summary_df.sort_values('Negative gaps (inverted)', ascending=False)
    
    st.dataframe(gap_summary_df, use_container_width=True)
    
    # Interactive gap selection
    st.subheader("Gap Distribution Analysis")
    
    selected_gap_type = st.selectbox(
        "Select gap type to visualize:",
        options=list(gap_stats.keys()),
        format_func=lambda x: f"{x} ({gap_stats[x]['negative']} inverted)"
    )
    
    # Display distribution for selected gap
    selected_gap_col = gap_stats[selected_gap_type]['column']
    gap_data = feature_df[selected_gap_col].dropna()
    
    if len(gap_data) > 0:
        # Create distribution plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(gap_data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero gap')
        ax1.set_xlabel(f'{selected_gap_type} (eV)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'{selected_gap_type} Distribution')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(gap_data)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_ylabel(f'{selected_gap_type} (eV)')
        ax2.set_title(f'{selected_gap_type} Box Plot')
        ax2.set_xticklabels([selected_gap_type])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display detailed statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{gap_data.mean():.3f} eV")
        with col2:
            st.metric("Std Dev", f"{gap_data.std():.3f} eV")
        with col3:
            st.metric("Min", f"{gap_data.min():.3f} eV")
        with col4:
            st.metric("Max", f"{gap_data.max():.3f} eV")
    
    # Multi-gap comparison
    if len(gap_columns) > 1:
        st.subheader("Multi-Gap Comparison")
        
        # Create comparison data
        comparison_data = []
        for gap_col in gap_columns:
            gap_type = gap_col.replace('_gap', '')
            gap_values = feature_df[gap_col].dropna()
            if len(gap_values) > 0:
                comparison_data.append({
                    'Gap Type': gap_type,
                    'Mean': gap_values.mean(),
                    'Negative Count': (gap_values < 0).sum(),
                    'Min': gap_values.min(),
                    'Max': gap_values.max()
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(comparison_df))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, comparison_df['Negative Count'], 
                           width, label='Negative Gaps')
            bars2 = ax.bar(x + width/2, comparison_df['Mean'], 
                           width, label='Mean Gap Value')
            
            ax.set_xlabel('Gap Type')
            ax.set_ylabel('Value')
            ax.set_title('Gap Types Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Gap Type'], rotation=45)
            ax.legend()
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)

def display_inversion_mechanisms(feature_df):
    """Display different inversion mechanisms"""
    
    st.subheader("Inversion Mechanisms")
    
    # Check for mechanism-related columns
    mechanism_cols = ['has_hRISC', 'has_inverted_ST', 
                     'has_high_order_inversion', 'has_multi_channel']
    
    available_mechanisms = [col for col in mechanism_cols if col in feature_df.columns]
    
    if available_mechanisms:
        mechanism_counts = {}
        for mech in available_mechanisms:
            count = feature_df[mech].sum() if mech in feature_df.columns else 0
            mechanism_counts[mech.replace('has_', '')] = count
        
        # Pie chart
        if any(mechanism_counts.values()):
            fig, ax = plt.subplots(figsize=(8, 8))
            labels = list(mechanism_counts.keys())
            sizes = list(mechanism_counts.values())
            
            # Show only non-zero mechanisms
            non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
            if non_zero_indices:
                labels = [labels[i] for i in non_zero_indices]
                sizes = [sizes[i] for i in non_zero_indices]
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title('Distribution of Inversion Mechanisms')
                st.pyplot(fig)
    else:
        # If no mechanism columns, try to infer from gap types
        if 'most_negative_gap_type' in feature_df.columns:
            gap_type_counts = feature_df['most_negative_gap_type'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            gap_type_counts.plot(kind='bar', ax=ax)
            ax.set_title('Primary Inversion Gap Types')
            ax.set_xlabel('Gap Type')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

def explore_gap_relationships(feature_df):
    """Explore relationships between different gap types"""
    
    st.subheader("Gap Relationships Explorer")
    
    gap_columns = [col for col in feature_df.columns 
                   if col.endswith('_gap') and not col.endswith('_gap_meV')]
    
    if len(gap_columns) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            gap_x = st.selectbox("X-axis gap:", gap_columns, index=0)
        with col2:
            gap_y = st.selectbox("Y-axis gap:", gap_columns, 
                               index=1 if len(gap_columns) > 1 else 0)
        
        # Create scatter plot
        valid_data = feature_df[[gap_x, gap_y]].dropna()
        
        if len(valid_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color by quadrant
            colors = []
            for _, row in valid_data.iterrows():
                if row[gap_x] < 0 and row[gap_y] < 0:
                    colors.append('red')  # Both inverted
                elif row[gap_x] < 0 or row[gap_y] < 0:
                    colors.append('orange')  # One inverted
                else:
                    colors.append('blue')  # None inverted
            
            scatter = ax.scatter(valid_data[gap_x], valid_data[gap_y], 
                               c=colors, alpha=0.6, s=50)
            
            # Add reference lines
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            ax.set_xlabel(f'{gap_x} (eV)')
            ax.set_ylabel(f'{gap_y} (eV)')
            ax.set_title(f'Correlation between {gap_x} and {gap_y}')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='Both inverted'),
                Patch(facecolor='orange', label='One inverted'),
                Patch(facecolor='blue', label='None inverted')
            ]
            ax.legend(handles=legend_elements)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show correlation coefficient
            correlation = valid_data[gap_x].corr(valid_data[gap_y])
            st.metric("Correlation coefficient", f"{correlation:.3f}")

def show_feature_visualizations(feature_df):
    """Display various feature visualizations"""
    
    st.subheader("Feature Visualizations")
    
    # Create multiple tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Correlation Heatmap", 
        "Gap Distribution", 
        "Feature Importance",
        "3D Scatter",
        "Statistical Summary"
    ])
    
    with tab1:
        # Correlation heatmap
        st.write("### Correlation Between Features")
        
        # Let users select feature categories
        feature_categories = {
            "3D Features": ['estimated_conjugation', 'estimated_size', 'planarity_index',
                           'estimated_polarity', 'estimated_hydrophobicity'],
            "Electronic Features": ['homo', 'lumo', 'homo_lumo_gap', 'dipole',
                                   'electron_donating_effect', 'electron_withdrawing_effect'],
            "Gap Features": [col for col in feature_df.columns if col.endswith('_gap')],
            "Substituent Features": [col for col in feature_df.columns if col.startswith('has_')],
            "All Numeric": feature_df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:20]
        }
        
        # Filter out empty categories
        feature_categories = {k: v for k, v in feature_categories.items() 
                             if any(col in feature_df.columns for col in v)}
        
        if feature_categories:
            category = st.selectbox("Select feature category:", list(feature_categories.keys()))
            selected_features = [f for f in feature_categories[category] if f in feature_df.columns]
            
            if len(selected_features) > 1:
                corr_matrix = feature_df[selected_features].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax, fmt='.2f')
                st.pyplot(fig)
    
    with tab2:
        # Gap distribution
        display_gap_statistics(feature_df)
    
    with tab3:
        # Feature importance
        st.write("### Feature Importance Analysis")
        
        # Calculate feature importance using variance
        numeric_features = feature_df.select_dtypes(include=['float64', 'int64']).columns
        
        importance_data = []
        for col in numeric_features[:30]:  # Show top 30
            variance = feature_df[col].var()
            non_zero_ratio = (feature_df[col] != 0).sum() / len(feature_df)
            importance_data.append({
                'Feature': col,
                'Variance': variance,
                'Non-zero Ratio': non_zero_ratio
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('Variance', ascending=False)
        
        # Bar plot
        fig = px.bar(importance_df.head(20), 
                     x='Variance', 
                     y='Feature',
                     orientation='h',
                     title='Top 20 Features by Variance')
        st.plotly_chart(fig)
    
    with tab4:
        # 3D scatter plot
        st.write("### 3D Feature Relationships")
        
        # Select 3 features for 3D visualization
        numeric_cols = feature_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_cols) >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_feat = st.selectbox("X-axis:", numeric_cols, index=0)
            with col2:
                y_feat = st.selectbox("Y-axis:", numeric_cols, 
                                     index=min(1, len(numeric_cols)-1))
            with col3:
                z_feat = st.selectbox("Z-axis:", numeric_cols, 
                                     index=min(2, len(numeric_cols)-1))
            
            # Color encoding
            color_feat = st.selectbox("Color by:", ['None'] + numeric_cols)
            
            # Create 3D scatter plot
            if color_feat != 'None' and color_feat in feature_df.columns:
                fig = px.scatter_3d(feature_df, 
                                   x=x_feat, 
                                   y=y_feat, 
                                   z=z_feat,
                                   color=color_feat,
                                   title=f"3D Scatter: {x_feat} vs {y_feat} vs {z_feat}")
            else:
                fig = px.scatter_3d(feature_df, 
                                   x=x_feat, 
                                   y=y_feat, 
                                   z=z_feat,
                                   title=f"3D Scatter: {x_feat} vs {y_feat} vs {z_feat}")
            
            st.plotly_chart(fig)
    
    with tab5:
        # Statistical summary
        st.write("### Statistical Summary")
        
        # Select features to analyze
        selected_features = st.multiselect(
            "Select features to analyze:",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
        
        if selected_features:
            # Display descriptive statistics
            st.write("#### Descriptive Statistics")
            st.dataframe(feature_df[selected_features].describe())
            
            # Box plots
            if len(selected_features) <= 6:
                st.write("#### Box Plots")
                fig, axes = plt.subplots(1, len(selected_features), 
                                       figsize=(4*len(selected_features), 6))
                
                if len(selected_features) == 1:
                    axes = [axes]
                
                for i, feat in enumerate(selected_features):
                    feature_df.boxplot(column=feat, ax=axes[i])
                    axes[i].set_title(feat)
                
                plt.tight_layout()
                st.pyplot(fig)

def preview_data(file_path):
    """Preview selected data file"""
    try:
        df = pd.read_csv(file_path)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'Molecule' in df.columns:
                st.metric("Total Molecules", len(df['Molecule'].unique()))
            else:
                st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Data Points", len(df))
        
        # Display data preview
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Display column information
        with st.expander("Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-null Count': df.count(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info)
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_download_link(file_path, text):
    """Create file download link"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            
        b64 = base64.b64encode(data).decode()
        filename = os.path.basename(file_path)
        
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")

def render_feature_page(feature_agent=None):
    """Render feature engineering page"""
    st.title("Feature Engineering & Alternative 3D Descriptors")
    
    st.markdown("""
    This page helps you generate and explore various molecular descriptors derived from the extracted data.
    
    Key feature categories:
    1. **Electronic properties** - HOMO, LUMO, electron-donating/withdrawing effects
    2. **Structural features** - Rings, substituents, planarity, conjugation
    3. **Physical properties** - Polarity, hydrophobicity, size estimates
    4. **Quantum properties** - Energy levels, gaps, dipole moments
    
    You can run the feature engineering pipeline on previously extracted data or upload a new CSV file.
    """)
    
    # File selection guide
    with st.expander("ℹ️ File Selection Guide", expanded=True):
        st.info("""
        **Which file to select?**
        
        For Feature Engineering, please select:
        - **molecular_properties_summary.csv** (Recommended) - Contains summarized molecular properties
        - **all_conformers_data.csv** - Contains all conformer data (larger file)
        
        Do NOT select:
        - Files starting with 'processed_' - These are outputs from previous feature engineering
        - Files ending with '_samples.csv' - These are analysis results
        - Files in subdirectories like 'reversed_tadf_analysis/' - These are analysis outputs
        """)
    
    # Data source selection
    data_source = st.radio("Data source", ["Use extracted data", "Upload CSV"])
    
    data_file = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            # Save uploaded CSV to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                temp_csv.write(uploaded_file.getvalue())
                data_file = temp_csv.name
    else:
        # Find previously extracted data
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        extracted_dir = os.path.join(base_dir, 'data', 'extracted')
        
        if os.path.exists(extracted_dir):
            # Define suitable files for feature engineering
            suitable_files = ['molecular_properties_summary.csv', 'all_conformers_data.csv']
            
            # Find available suitable files
            available_files = []
            for f in os.listdir(extracted_dir):
                if f in suitable_files and f.endswith('.csv'):
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
                    preview_data(data_file)
            else:
                st.warning("No suitable extracted data files found. Please run Data Extraction first.")
        else:
            st.warning(f"Extracted data directory not found: {extracted_dir}. Please run Data Extraction first.")
    
    # Generate features button
    if st.button("Generate Features"):
        if data_file:
            with st.spinner("Generating features..."):
                try:
                    # Initialize feature agent if not provided
                    if feature_agent is None:
                        feature_agent = FeatureAgent()
                    
                    # Run feature pipeline
                    feature_agent.data_file = data_file
                    result = feature_agent.run_feature_pipeline()
                    
                    if result and 'enhanced_feature_file' in result:
                        st.success(f"Feature engineering completed! Results saved to {result['enhanced_feature_file']}")
                        
                        # Load and display feature data
                        feature_df = pd.read_csv(result['enhanced_feature_file'])
                        
                        # Display feature statistics
                        if 'feature_statistics' in result and result['feature_statistics']:
                            stats = result['feature_statistics']
                            st.subheader("Feature Engineering Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Features", stats['total_features'])
                            with col2:
                                st.metric("Total Molecules", stats['total_molecules'])
                            with col3:
                                if 'molecules_with_inversions' in stats:
                                    st.metric("Molecules with Inversions", stats['molecules_with_inversions'])
                        
                        # Display visualizations
                        show_feature_visualizations(feature_df)
                        
                        # Display inversion mechanisms
                        display_inversion_mechanisms(feature_df)
                        
                        # Explore gap relationships
                        explore_gap_relationships(feature_df)
                        
                        # Create download links
                        st.subheader("Download Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            create_download_link(result['enhanced_feature_file'], 
                                               "📥 Download Enhanced Features CSV")
                        with col2:
                            if 'all_inverted_gaps' in result and result['all_inverted_gaps'] is not None:
                                # Save inverted gaps to file for download
                                inverted_file = result['enhanced_feature_file'].replace(
                                    'enhanced.csv', 'inverted_gaps.csv'
                                )
                                result['all_inverted_gaps'].to_csv(inverted_file, index=False)
                                create_download_link(inverted_file, 
                                                   "📥 Download Inverted Gaps Analysis")
                        
                        # Navigation suggestion
                        if 'molecules_with_inversions' in stats and stats['molecules_with_inversions'] > 0:
                            st.info(f"""
                            Found {stats['molecules_with_inversions']} molecules with inverted gaps! 
                            Navigate to the Exploration Analysis page to analyze these reversed TADF candidates.
                            """)
                        
                        return result
                    else:
                        st.error("Feature engineering failed. Please check the logs.")
                        
                except Exception as e:
                    st.error(f"Error in feature engineering: {str(e)}")
                    st.exception(e)
        else:
            st.warning("Please select a data file first.")
    
    return None

def load_feature_page(feature_agent=None):
    """Load feature engineering page"""
    return render_feature_page(feature_agent)

if __name__ == "__main__":
    # For direct testing
    load_feature_page()