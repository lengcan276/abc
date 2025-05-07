# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend

class VisualizationUtils:
    """
    Utility class for creating visualizations
    """
    
    @staticmethod
    def set_plot_style():
        """Set plotting style"""
        # Set seaborn style
        sns.set(style="whitegrid")
        
        # Set matplotlib parameters
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 100
        
        # Handle font issues
        try:
            # Try to use system fonts
            import matplotlib.font_manager as fm
            
            # Use default fonts - no need to search for Chinese fonts
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
            
            # For proper display of minus sign
            plt.rcParams['axes.unicode_minus'] = True
        
        except Exception as e:
            # If there's an error, fall back to defaults
            import logging
            logging.warning(f"Error setting font: {e}")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    @staticmethod
    def create_s1t1_gap_distribution(df, save_path=None):
        """Create S1-T1 gap distribution plot"""
        # Set plotting style
        VisualizationUtils.set_plot_style()
        
        # Ensure s1_t1_gap_ev column exists
        if 's1_t1_gap_ev' not in df.columns:
            logging.error("No s1_t1_gap_ev column in dataframe")
            return None
            
        # Filter rows with gap data
        gap_data = df[df['s1_t1_gap_ev'].notna()].copy()
        
        if len(gap_data) == 0:
            logging.error("No valid S1-T1 gap data found")
            return None
            
        # Add category column (negative vs positive)
        gap_data['gap_type'] = gap_data['s1_t1_gap_ev'].apply(
            lambda x: 'Negative Gap' if x < 0 else 'Positive Gap'
        )
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data=gap_data, x='s1_t1_gap_ev', hue='gap_type', bins=20, kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('S1-T1 Gap Distribution')
        plt.xlabel('S1-T1 Gap (eV)')
        plt.tight_layout()
        
        # Save image (if path provided)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"S1-T1 gap distribution saved to {save_path}")
            
        return plt.gcf()
        
    @staticmethod
    def create_feature_comparison(negative_df, positive_df, feature, save_path=None):
        """Create feature comparison plot (negative vs positive gap)"""
        # Set plotting style
        VisualizationUtils.set_plot_style()
        
        # Ensure both dataframes have the specified feature
        if feature not in negative_df.columns or feature not in positive_df.columns:
            logging.error(f"No {feature} column in dataframe")
            return None
            
        # Get feature data
        neg_data = negative_df[feature].dropna()
        pos_data = positive_df[feature].dropna()
        
        if len(neg_data) == 0 or len(pos_data) == 0:
            logging.error(f"No valid {feature} data found")
            return None
            
        # Create box plot
        plt.figure(figsize=(8, 6))
        
        box_data = [neg_data.values, pos_data.values]
        
        plt.boxplot(box_data, labels=['Negative Gap', 'Positive Gap'])
        plt.title(f'{feature.replace("_", " ").title()}: Negative vs Positive Gap')
        plt.ylabel(feature)
        
        # Add data points
        x_pos = [1, 2]
        for i, data in enumerate([neg_data, pos_data]):
            # Add jitter to x positions
            x = np.random.normal(x_pos[i], 0.05, size=len(data))
            plt.scatter(x, data, alpha=0.3, s=10)
            
        # Add means as stars
        means = [neg_data.mean(), pos_data.mean()]
        plt.plot(x_pos, means, 'r*', markersize=10)
        
        plt.tight_layout()
        
        # Save image (if path provided)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"{feature} comparison chart saved to {save_path}")
            
        return plt.gcf()
        
    @staticmethod
    def create_pca_plot(negative_df, positive_df, features, save_path=None):
        """Create PCA analysis plot"""
        # Set plotting style
        VisualizationUtils.set_plot_style()
        
        # Ensure both dataframes have all specified features
        for feature in features:
            if feature not in negative_df.columns or feature not in positive_df.columns:
                logging.error(f"No {feature} column in dataframe")
                return None
                
        # Prepare data
        neg_subset = negative_df[features].dropna().copy()
        neg_subset['group'] = 'Negative Gap'
        
        pos_subset = positive_df[features].dropna().copy()
        pos_subset['group'] = 'Positive Gap'
        
        # Merge data
        combined = pd.concat([neg_subset, pos_subset], ignore_index=True)
        
        if len(combined) < 5:
            logging.error("Too few data points for PCA analysis")
            return None
            
        # Standardize features
        X = combined[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        # Create PCA plot
        plt.figure(figsize=(10, 8))
        for group, color in zip(['Negative Gap', 'Positive Gap'], ['red', 'blue']):
            mask = combined['group'] == group
            plt.scatter(
                pca_result[mask, 0], pca_result[mask, 1],
                label=group, color=color, alpha=0.7
            )
            
        # Add feature vectors
        feature_vectors = pca.components_.T
        feature_names = features
        
        # Scale vectors for visibility
        scale = 5
        for i, feature in enumerate(feature_names):
            plt.arrow(
                0, 0, 
                feature_vectors[i, 0] * scale, 
                feature_vectors[i, 1] * scale,
                head_width=0.1, head_length=0.1, 
                fc='k', ec='k'
            )
            plt.text(
                feature_vectors[i, 0] * scale * 1.15, 
                feature_vectors[i, 1] * scale * 1.15, 
                feature.replace('_', ' ')
            )
            
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} Variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} Variance)')
        plt.title('PCA of Molecular Properties: Negative vs Positive S1-T1 Gap')
        plt.legend()
        plt.tight_layout()
        
        # Save image (if path provided)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"PCA analysis plot saved to {save_path}")
            
        return plt.gcf()
        
    @staticmethod
    def create_radar_chart(negative_df, positive_df, features, save_path=None):
        """Create radar chart comparing features"""
        # Set plotting style
        VisualizationUtils.set_plot_style()
        
        # Ensure enough features
        if len(features) < 3:
            logging.error("Radar chart requires at least 3 features")
            return None
            
        # Ensure both dataframes have all specified features
        valid_features = []
        for feature in features:
            if feature in negative_df.columns and feature in positive_df.columns:
                valid_features.append(feature)
            else:
                logging.warning(f"Feature {feature} missing in dataframe, skipped")
                
        if len(valid_features) < 3:
            logging.error(f"Not enough valid features ({len(valid_features)}), radar chart requires at least 3 features")
            return None
                
        # Calculate means for both groups, handling NaN and inf values
        neg_means = []
        pos_means = []
        for f in valid_features:
            neg_val = negative_df[f].replace([np.inf, -np.inf], np.nan).mean()
            pos_val = positive_df[f].replace([np.inf, -np.inf], np.nan).mean()
            
            # If still NaN, use 0 as placeholder
            neg_means.append(0.0 if np.isnan(neg_val) else neg_val)
            pos_means.append(0.0 if np.isnan(pos_val) else pos_val)
            
        # Print debug info
        logging.info(f"Negative gap feature means: {neg_means}")
        logging.info(f"Positive gap feature means: {pos_means}")
        
        # Normalize values to [0,1] for radar chart
        all_values = np.array(neg_means + pos_means)
        if len(all_values) == 0 or np.all(np.isnan(all_values)):
            logging.error("Cannot create radar chart: all values are NaN")
            return None
            
        min_vals = np.nanmin(all_values)
        max_vals = np.nanmax(all_values)
        
        # Handle edge cases
        if np.isnan(min_vals) or np.isnan(max_vals) or np.isclose(max_vals, min_vals):
            logging.warning("All values approximately equal or NaN present, using default normalization values")
            normalized_neg = [0.5 for _ in neg_means]
            normalized_pos = [0.5 for _ in pos_means]
        else:
            # Ensure no division by zero
            range_vals = max_vals - min_vals
            if np.isclose(range_vals, 0):
                range_vals = 1.0
                
            normalized_neg = [(x - min_vals) / range_vals for x in neg_means]
            normalized_pos = [(x - min_vals) / range_vals for x in pos_means]
        
        # Create radar chart
        labels = [f.replace('_', ' ').title() for f in valid_features]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        
        # Close polygon
        normalized_neg.append(normalized_neg[0])
        normalized_pos.append(normalized_pos[0])
        angles.append(angles[0])
        labels.append(labels[0])
        
        # Create polar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot negative gap polygon
        ax.plot(angles, normalized_neg, 'r-', linewidth=2, label='Negative Gap')
        ax.fill(angles, normalized_neg, 'r', alpha=0.1)
        
        # Plot positive gap polygon
        ax.plot(angles, normalized_pos, 'b-', linewidth=2, label='Positive Gap')
        ax.fill(angles, normalized_pos, 'b', alpha=0.1)
        
        # Set ticks and labels
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        ax.set_title('Feature Comparison: Negative vs Positive S1-T1 Gap', size=15)
        ax.legend(loc='upper right')
        
        # Save image (if path provided)
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Radar chart saved to {save_path}")
            except Exception as e:
                logging.error(f"Error saving radar chart: {e}")
        
        return fig
        
    @staticmethod
    def create_feature_importance_plot(importance_df, target_name, save_path=None, top_n=15):
        """Create feature importance plot"""
        # Set plotting style
        VisualizationUtils.set_plot_style()
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
        plt.title(f'Top Features for Predicting {target_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save image (if path provided)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"Feature importance plot saved to {save_path}")
            
        return plt.gcf()
        
    @staticmethod
    def create_correlation_heatmap(df, features, save_path=None):
        """Create correlation heatmap"""
        # Set plotting style
        VisualizationUtils.set_plot_style()
        
        # Ensure dataframe has specified features
        valid_features = [f for f in features if f in df.columns]
        
        if not valid_features:
            logging.error("No valid features in dataframe")
            return None
            
        # Calculate correlation matrix
        corr = df[valid_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Save image (if path provided)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"Correlation heatmap saved to {save_path}")
            
        return plt.gcf()
    
    @staticmethod
    def create_structure_feature_radar(negative_df, positive_df, save_path=None):
        """Create radar chart comparing structural features"""
        # Set plotting style
        VisualizationUtils.set_plot_style()
        
        # Select structural features to display in radar chart
        features = [
            'max_conjugation_length', 'twist_ratio', 
            'max_h_bond_strength', 'planarity',
            'aromatic_rings_count'
        ]
        
        # Check if both dataframes have these features
        valid_features = [f for f in features 
                        if f in negative_df.columns and f in positive_df.columns]
        
        if len(valid_features) < 3:  # Need at least 3 features
            logging.warning("Not enough structural features for radar chart comparison")
            return None
        
        # Calculate means
        neg_means = [negative_df[f].mean() for f in valid_features]
        pos_means = [positive_df[f].mean() for f in valid_features]
        
        # Normalize to [0,1] range
        all_values = neg_means + pos_means
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        normalized_neg = [(v - min_val) / range_val for v in neg_means]
        normalized_pos = [(v - min_val) / range_val for v in pos_means]
        
        # Friendly name mapping
        feature_names = {
            'max_conjugation_length': 'π-Conjugation Length',
            'twist_ratio': 'Twist Ratio', 
            'max_h_bond_strength': 'H-Bond Strength',
            'planarity': 'Planarity',
            'aromatic_rings_count': 'Aromatic Rings Count',
            'conjugation_path_count': 'Conjugation Paths Count',
            'dihedral_angles_count': 'Key Dihedral Angles Count',
            'hydrogen_bonds_count': 'Hydrogen Bonds Count'
        }
        
        # Create labels
        labels = [feature_names.get(f, f.replace('_', ' ').title()) for f in valid_features]
        
        # Prepare radar chart
        angles = np.linspace(0, 2*np.pi, len(valid_features), endpoint=False).tolist()
        
        # Close the figure
        normalized_neg.append(normalized_neg[0])
        normalized_pos.append(normalized_pos[0])
        angles.append(angles[0])
        labels.append(labels[0])
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
        
        ax.plot(angles, normalized_neg, 'r-', linewidth=2, label='Negative S1-T1 Gap')
        ax.fill(angles, normalized_neg, 'r', alpha=0.1)
        
        ax.plot(angles, normalized_pos, 'b-', linewidth=2, label='Positive S1-T1 Gap')
        ax.fill(angles, normalized_pos, 'b', alpha=0.1)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        ax.set_title('Structural Feature Comparison: Negative vs Positive S1-T1 Gap')
        ax.legend(loc='upper right')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"Structural feature radar chart saved to: {save_path}")
        
        return fig

    @staticmethod
    def create_dihedral_vs_s1t1_plot(df, save_path=None):
        """Create dihedral angle vs S1-T1 gap relationship plot"""
        if 'max_dihedral_angle' not in df.columns or 's1_t1_gap_ev' not in df.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(
            df['max_dihedral_angle'], 
            df['s1_t1_gap_ev'],
            alpha=0.7, c=df['s1_t1_gap_ev'], cmap='coolwarm'
        )
        
        # Add trend line
        x = df['max_dihedral_angle'].values.reshape(-1, 1)
        y = df['s1_t1_gap_ev'].values
        
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x, y)
            
            x_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
            y_pred = model.predict(x_pred)
            
            plt.plot(x_pred, y_pred, 'k--', linewidth=1)
        except:
            # Skip trend line if it can't be calculated
            pass
        
        # Add zero line
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Add 40° reference line
        plt.axvline(x=40, color='g', linestyle='--', alpha=0.5)
        
        plt.xlabel('Maximum Dihedral Angle (degrees)')
        plt.ylabel('S1-T1 Gap (eV)')
        plt.title('Dihedral Angle vs S1-T1 Gap Relationship')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"Dihedral angle relationship plot saved to: {save_path}")
        
        return plt.gcf()

    @staticmethod
    def create_conjugation_vs_s1t1_plot(df, save_path=None):
        """Create conjugation length vs S1-T1 gap relationship plot"""
        if 'max_conjugation_length' not in df.columns or 's1_t1_gap_ev' not in df.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Group by maximum conjugation length
        grouped = df.groupby('max_conjugation_length')['s1_t1_gap_ev'].agg(['mean', 'std', 'count'])
        grouped = grouped[grouped['count'] >= 2]  # At least 2 samples
        
        x = grouped.index
        y = grouped['mean']
        yerr = grouped['std']
        
        # Draw bar chart
        plt.bar(x, y, yerr=yerr, alpha=0.7, color='skyblue')
        
        # Add zero line
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Maximum π-Conjugation Path Length')
        plt.ylabel('Average S1-T1 Gap (eV)')
        plt.title('Conjugation Length vs S1-T1 Gap Relationship')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"Conjugation length relationship plot saved to: {save_path}")
        
        return plt.gcf()

    @staticmethod
    def create_hydrogen_bonds_effect_plot(df, save_path=None):
        """Create hydrogen bonding effect on S1-T1 gap plot"""
        if 'hydrogen_bonds_count' not in df.columns or 's1_t1_gap_ev' not in df.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Categorize based on hydrogen bond count
        df['h_bond_category'] = df['hydrogen_bonds_count'].apply(
            lambda x: 'No H-Bonds' if x == 0 else ('1-2 H-Bonds' if x <= 2 else 'Multiple H-Bonds')
        )
        
        # Group and calculate means
        grouped = df.groupby('h_bond_category')['s1_t1_gap_ev'].agg(['mean', 'std', 'count']).reset_index()
        
        # Ensure order
        order = ['No H-Bonds', '1-2 H-Bonds', 'Multiple H-Bonds']
        grouped['order'] = grouped['h_bond_category'].map({cat: i for i, cat in enumerate(order)})
        grouped = grouped.sort_values('order')
        
        # Plot
        plt.bar(
            grouped['h_bond_category'], 
            grouped['mean'],
            yerr=grouped['std'],
            alpha=0.7,
            color=['#ff9999', '#66b3ff', '#99ff99']
        )
        
        # Add zero line
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Number of Hydrogen Bonds')
        plt.ylabel('Average S1-T1 Gap (eV)')
        plt.title('Hydrogen Bonding vs S1-T1 Gap Relationship')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"Hydrogen bond effect plot saved to: {save_path}")
        
        return plt.gcf()