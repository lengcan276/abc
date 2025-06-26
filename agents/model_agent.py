# agents/model_agent.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  # 添加这一行导入streamlit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import os
import logging
import pickle
import joblib


class ModelAgent:
    """
    Agent responsible for building predictive models for S1-T1 gap properties,
    with focus on classifying potential reverse TADF molecules.
    """
    
    def __init__(self, feature_file=None, feature_agent=None):
        """Initialize the ModelAgent with feature data file."""
        self.feature_file = feature_file
        self.feature_df = None
        self.selected_features = {}
        self.models = {}
        self.feature_importance = {}
        self.feature_agent = feature_agent
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the model agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/logs/model_agent.log')
        self.logger = logging.getLogger('ModelAgent')
        
    def load_data(self, file_path=None):
        """Load feature data from CSV file."""
        if file_path:
            self.feature_file = file_path
            
        if not self.feature_file or not os.path.exists(self.feature_file):
            self.logger.error(f"Feature file not found: {self.feature_file}")
            return False
            
        print(f"Loading feature data from {self.feature_file}...")
        self.feature_df = pd.read_csv(self.feature_file)

        # Print basic data info
        print(f"Feature dataset shape: {self.feature_df.shape}")
        
        # Check S1-T1 gap data availability
        if 's1_t1_gap_ev' in self.feature_df.columns:
            s1t1_count = self.feature_df['s1_t1_gap_ev'].notna().sum()
            print(f"Number of entries with S1-T1 gap data: {s1t1_count}")
            
            # Count negative and positive gap molecules
            if s1t1_count > 0:
                neg_count = (self.feature_df['s1_t1_gap_ev'] < 0).sum()
                print(f"Number of entries with negative S1-T1 gap: {neg_count}")
                print(f"Number of entries with positive S1-T1 gap: {s1t1_count - neg_count}")
        else:
            self.logger.warning("No S1-T1 gap data found in the feature file.")
            
        return True
    
    @staticmethod
    def enhance_dataset_with_crest(feature_df):
        """
        Enhance dataset with CREST data
        
        Args:
            feature_df: Original feature dataframe
        
        Returns:
            Enhanced feature dataframe
        """
        print("Starting to enhance dataset with CREST data...")
        original_rows = len(feature_df)
        
        # 1. Create expanded dataframe based on original molecules
        enhanced_df = feature_df.copy()
        
        # 2. Identify molecules with CREST data
        crest_columns = [col for col in feature_df.columns if 'crest' in col.lower()]
        molecules_with_crest = feature_df[feature_df[crest_columns].notna().any(axis=1)]['Molecule'].unique()
        
        print(f"Found {len(molecules_with_crest)} molecules with CREST data")
        
        # 3. For each molecule with CREST data, create additional "virtual samples"
        synthetic_samples = []
        
        for molecule in molecules_with_crest:
            # Get data for this molecule
            mol_data = feature_df[feature_df['Molecule'] == molecule].copy()
            
            # Check if there is conformer count information
            conformer_cols = [col for col in crest_columns if 'num_conformers' in col]
            
            for _, row in mol_data.iterrows():
                # For each state (neutral, cation, triplet), try to create synthetic samples
                for state in ['neutral', 'cation', 'triplet']:
                    conformer_col = f"{state}_crest_num_conformers"
                    energy_range_col = f"{state}_crest_energy_range"
                    
                    # Check if there is CREST data for this state
                    if conformer_col in row and not pd.isna(row[conformer_col]) and row[conformer_col] > 1:
                        # Get the number of conformers
                        num_conformers = int(row[conformer_col])
                        
                        # Create synthetic samples based on this molecule
                        for i in range(min(num_conformers-1, 2)):  # Create at most 2 extra samples to avoid too many
                            # Copy original data
                            synthetic_row = row.copy()
                            
                            # Modify molecule name to identify this as a synthetic sample
                            synthetic_row['Molecule'] = f"{molecule}_crest_synth_{i+1}"
                            
                            # Make slight perturbations to CREST features
                            if energy_range_col in row and not pd.isna(row[energy_range_col]):
                                # Create reasonable perturbation based on energy range
                                energy_range = row[energy_range_col]
                                # Add small random fluctuation to original energy range (-15% to +15%)
                                perturbation = np.random.uniform(-0.15, 0.15) * energy_range
                                synthetic_row[energy_range_col] = energy_range + perturbation
                                
                                # If there is S1-T1 gap data, also make slight perturbation
                                if 's1_t1_gap_ev' in row and not pd.isna(row['s1_t1_gap_ev']):
                                    gap = row['s1_t1_gap_ev']
                                    # Keep gap sign the same, but add small change (max ±0.1eV)
                                    gap_perturbation = np.random.uniform(-0.1, 0.1)
                                    synthetic_row['s1_t1_gap_ev'] = gap + gap_perturbation
                                    # Ensure sign doesn't change
                                    if (gap < 0 and synthetic_row['s1_t1_gap_ev'] > 0) or \
                                    (gap > 0 and synthetic_row['s1_t1_gap_ev'] < 0):
                                        synthetic_row['s1_t1_gap_ev'] = gap
                                
                                # Add this synthetic sample
                                synthetic_samples.append(synthetic_row)
                                print(f"Created synthetic sample based on {state} state CREST data for molecule {molecule}")
        
        # 4. Add synthetic samples to dataframe
        if synthetic_samples:
            synthetic_df = pd.DataFrame(synthetic_samples)
            enhanced_df = pd.concat([enhanced_df, synthetic_df], ignore_index=True)
            print(f"Added {len(synthetic_samples)} synthetic samples based on CREST data")
            print(f"Dataset expanded from {original_rows} rows to {len(enhanced_df)} rows")
        else:
            print("No synthetic samples created")
        
        # 5. Ensure all CREST features are retained
        for col in crest_columns:
            if col not in enhanced_df.columns:
                print(f"Warning: CREST feature {col} not in enhanced dataset")
        
        return enhanced_df

    def select_features(self, target_col, n_features=15):
        """Select the most relevant features for a target variable."""
        if self.feature_df is None:
            self.logger.error("No data loaded. Call load_data() first.")
            return None
            
        print(f"Selecting features for target: {target_col}...")
        
        # Keep only data with target values
        df_target = self.feature_df[self.feature_df[target_col].notna()].copy()
        print(f"Number of samples with {target_col} data: {len(df_target)}")
        
        if len(df_target) < 10:
            print(f"Warning: Too few samples with {target_col} data. Need at least 10 samples.")
            return None
            
        # Determine features usable for training
        # Exclude non-numeric features, target column, and obviously unrelated columns
        exclude_cols = ['Molecule', 'conformer', 'State', 'is_primary',
                    target_col, 'excited_energy', 'excited_opt_success',
                    'excited_no_imaginary', 'excited_homo', 'excited_lumo',
                    'excited_homo_lumo_gap', 'excited_dipole']
        
        # Select numeric features
        numeric_cols = df_target.select_dtypes(include=['float64', 'int64']).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Check if there are CREST features
        crest_features = [col for col in feature_cols if 'crest' in col.lower()]
        print(f"Found {len(crest_features)} CREST features: {crest_features}")
        
        # Remove features with too many NaN values
        valid_features = []
        for col in feature_cols:
            nan_ratio = df_target[col].isna().mean()
            # Use more lenient missing value standard for CREST features
            if 'crest' in col.lower():
                if nan_ratio < 0.5:  # If CREST feature has less than 50% missing values, keep it
                    valid_features.append(col)
                    print(f"Keeping CREST feature '{col}' with {nan_ratio*100:.1f}% missing values")
            else:
                if nan_ratio < 0.3:  # Non-CREST features keep original 30% threshold
                    valid_features.append(col)
                    
        print(f"Number of valid features: {len(valid_features)}")
        print(f"Number of valid CREST features: {len([f for f in valid_features if 'crest' in f.lower()])}")
        
        if len(valid_features) == 0:
            print("No valid features found.")
            return None
            
        # Prepare feature matrix and target vector
        X = df_target[valid_features].copy()
        y = df_target[target_col].copy()
        
        # Replace infinity values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Special handling for CREST features
        for col in X.columns:
            if 'crest' in col.lower():
                # For CREST features with missing values, fill with 0 instead of median
                missing_count = X[col].isna().sum()
                if missing_count > 0:
                    X[col] = X[col].fillna(0)
                    print(f"Filled {missing_count} missing values in CREST feature '{col}' with 0")
        
        # Check for extreme values
        for col in X.columns:
            # Calculate column mean and std
            col_mean = X[col].mean()
            col_std = X[col].std()
            
            # Skip if std is NaN or infinity
            if pd.isna(col_std) or np.isinf(col_std):
                print(f"Warning: Column {col} has problematic standard deviation. Skipping extreme value check.")
                continue
                
            # For CREST features, use a more lenient threshold for extreme values
            z_threshold = 5 if 'crest' in col.lower() else 3
            
            # Use Z-score to identify extreme values
            extreme_mask = abs(X[col] - col_mean) > z_threshold * col_std
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                # Treat extreme values as missing rather than removing them
                extreme_indices = X.index[extreme_mask]
                X.loc[extreme_indices, col] = np.nan
                print(f"Identified {extreme_count} extreme values in '{col}', replaced with NaN")
        
        # Fill remaining NaN values
        # For non-CREST features, use median to fill missing values
        non_crest_cols = [col for col in X.columns if 'crest' not in col.lower()]
        if non_crest_cols:
            X[non_crest_cols] = X[non_crest_cols].fillna(X[non_crest_cols].median())
        
        # Feature normalization - use robust scaling for handling outliers
        scaler = RobustScaler()
        try:
            X_scaled = scaler.fit_transform(X)
            print("Successfully applied robust scaling to features")
        except Exception as e:
            print(f"Error during scaling: {e}")
            print("Falling back to simple standardization")
            # Re-check infinity and NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN differently for CREST and non-CREST features
            crest_cols = [col for col in X.columns if 'crest' in col.lower()]
            non_crest_cols = [col for col in X.columns if 'crest' not in col.lower()]
            
            if crest_cols:
                X[crest_cols] = X[crest_cols].fillna(0)
            if non_crest_cols:
                X[non_crest_cols] = X[non_crest_cols].fillna(X[non_crest_cols].median())
            
            # Use simple Z-score standardization
            X_mean = X.mean()
            X_std = X.std()
            X_std = X_std.replace(0, 1)  # Avoid division by zero
            X_scaled = (X - X_mean) / X_std
            
        X_scaled_df = pd.DataFrame(X_scaled, columns=valid_features)
        
        # 1. Use mutual information for feature selection
        print("Using mutual information for feature selection...")
        mi_selector = SelectKBest(mutual_info_regression, k=min(n_features, len(valid_features)))
        mi_selector.fit(X_scaled, y)
        mi_scores = pd.DataFrame({
            'Feature': valid_features,
            'MI_Score': mi_selector.scores_
        }).sort_values('MI_Score', ascending=False)
        
        # 2. Use F-statistics for feature selection
        print("Using F-statistics for feature selection...")
        f_selector = SelectKBest(f_regression, k=min(n_features, len(valid_features)))
        f_selector.fit(X_scaled, y)
        f_scores = pd.DataFrame({
            'Feature': valid_features,
            'F_Score': f_selector.scores_
        }).sort_values('F_Score', ascending=False)
        
        # 3. Use Random Forest feature importance
        print("Using Random Forest feature importance...")
        if target_col == 'is_negative_gap':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        rf_scores = pd.DataFrame({
            'Feature': valid_features,
            'RF_Importance': rf.feature_importances_
        }).sort_values('RF_Importance', ascending=False)
        
        # 4. Combined scoring: combine rankings from the three methods
        print("Calculating combined feature importance scores...")
        combined_scores = pd.DataFrame({'Feature': valid_features})
        
        # Add rankings from three methods
        combined_scores['MI_Rank'] = combined_scores['Feature'].map(
            {row['Feature']: i+1 for i, (_, row) in enumerate(mi_scores.iterrows())}
        )
        combined_scores['F_Rank'] = combined_scores['Feature'].map(
            {row['Feature']: i+1 for i, (_, row) in enumerate(f_scores.iterrows())}
        )
        combined_scores['RF_Rank'] = combined_scores['Feature'].map(
            {row['Feature']: i+1 for i, (_, row) in enumerate(rf_scores.iterrows())}
        )
        
        # Calculate average ranking
        combined_scores['Avg_Rank'] = (
            combined_scores['MI_Rank'] +
            combined_scores['F_Rank'] +
            combined_scores['RF_Rank']
        ) / 3
        
        # Ensure CREST features are retained
        # Get top n_features*0.7 features
        n_auto_select = int(n_features * 0.7)
        auto_features = combined_scores.sort_values('Avg_Rank').head(n_auto_select)['Feature'].tolist()
        
        # Ensure at least top 3 CREST features are included
        crest_in_scores = combined_scores[combined_scores['Feature'].str.contains('crest', case=False)]
        top_crest = crest_in_scores.sort_values('Avg_Rank').head(3)['Feature'].tolist()
        
        # Merge automatically selected features and top CREST features
        final_features = list(set(auto_features + top_crest))
        
        # If total feature count is still less than n_features, add more features
        remaining_features = [f for f in combined_scores.sort_values('Avg_Rank')['Feature'].tolist() 
                            if f not in final_features]
        
        while len(final_features) < n_features and remaining_features:
            final_features.append(remaining_features.pop(0))
        
        print(f"Selected top {len(final_features)} features:")
        # Display regular features and CREST features separately
        regular_features = [f for f in final_features if 'crest' not in f.lower()]
        crest_features_selected = [f for f in final_features if 'crest' in f.lower()]
        
        print("Regular features:")
        for feature in regular_features:
            print(f"  - {feature}")
        
        print("\nCREST features:")
        for feature in crest_features_selected:
            print(f"  - {feature}")
        
        # Store results for later use
        self.selected_features[target_col] = final_features
        
        # Create results directory
        results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
        os.makedirs(results_dir, exist_ok=True)
        
        # Visualize feature importance with CREST features highlighted
        plt.figure(figsize=(12, 8))
        top_features = combined_scores.sort_values('Avg_Rank').head(15)
        
        # Use different colors for CREST features
        colors = ['#1f77b4' if 'crest' not in feat.lower() else '#ff7f0e' 
                for feat in top_features['Feature']]
        
        sns.barplot(x='Avg_Rank', y='Feature', data=top_features, palette=colors)
        plt.title(f'Top 15 Features for Predicting {target_col} (Lower Rank is Better)')
        plt.xlabel('Average Rank')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Regular Features'),
            Patch(facecolor='#ff7f0e', label='CREST Features')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'feature_ranks_{target_col}.png'))
        plt.close()
        
        # Create additional chart specifically for CREST feature importance
        plt.figure(figsize=(10, 6))
        crest_importance = combined_scores[combined_scores['Feature'].str.contains('crest', case=False)]
        if not crest_importance.empty:
            crest_importance = crest_importance.sort_values('Avg_Rank')
            sns.barplot(x='Avg_Rank', y='Feature', data=crest_importance, color='#ff7f0e')
            plt.title(f'CREST Feature Importance for {target_col}')
            plt.xlabel('Average Rank')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'crest_feature_importance_{target_col}.png'))
        plt.close()
        
        return {
            'features': final_features,
            'scores': combined_scores,
            'mi_scores': mi_scores,
            'f_scores': f_scores,
            'rf_scores': rf_scores
        }
    
    def build_classification_model(self):
        """Build a classification model to predict negative vs positive S1-T1 gap."""
        if self.feature_df is None:
            self.logger.error("No data loaded. Call load_data() first.")
            return None
            
        print("Building classification model for S1-T1 gap direction...")
        
        # First, create target variable for classification
        # 1 = negative gap (reverse TADF), 0 = positive gap
        if 's1_t1_gap_ev' not in self.feature_df.columns:
            self.logger.error("S1-T1 gap data not found in features.")
            return None
            
        # Create classification target
        self.feature_df['is_negative_gap'] = (self.feature_df['s1_t1_gap_ev'] < 0).astype(int)
        
        # Filter to samples with S1-T1 gap data
        df_class = self.feature_df[self.feature_df['s1_t1_gap_ev'].notna()].copy()
        
        if len(df_class) < 10:
            print("Not enough samples with S1-T1 gap data for classification.")
            return None
            
        # Select features for classification
        feature_results = self.select_features('is_negative_gap', n_features=15)
        
        if not feature_results:
            self.logger.error("Feature selection failed.")
            return None
            
        selected_features = feature_results['features']
        
        # Create results directory
        results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare data for modeling
        X = df_class[selected_features].copy()
        y = df_class['is_negative_gap'].copy()
        
        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        
        rf_classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_classifier.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Classification model performance:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision (Negative Gap): {class_report['1']['precision']:.4f}")
        print(f"  - Recall (Negative Gap): {class_report['1']['recall']:.4f}")
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Positive Gap', 'Negative Gap'],
                   yticklabels=['Positive Gap', 'Negative Gap'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for S1-T1 Gap Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Calculate feature importance
        perm_importance = permutation_importance(
            rf_classifier, X_test, y_test, n_repeats=10, random_state=42
        )
        
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance for S1-T1 Gap Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'classification_feature_importance.png'))
        plt.close()
        
        # Store model and resultsdel_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = os.path.join(model_dir, 's1t1_gap_classifier.joblib')
        joblib.dump(rf_classifier, model_file)
        
        scaler_file = os.path.join(model_dir, 's1t1_gap_classifier_scaler.joblib')
        joblib.dump(scaler, scaler_file)
        
        # Save selected features
        with open(os.path.join(model_dir, 's1t1_gap_classifier_features.txt'), 'w') as f:
            f.write('\n'.join(selected_features))
            
        self.models['s1t1_classifier'] = {
            'model': rf_classifier,
            'scaler': scaler,
            'features': selected_features,
            'accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'class_report': class_report,
            'importance': importance_df
        }
        
        self.feature_importance['is_negative_gap'] = importance_df
        
        return {
            'model_file': model_file,
            'scaler_file': scaler_file,
            'features': selected_features,
            'accuracy': accuracy,
            'importance': importance_df,
            'conf_matrix_plot': os.path.join(results_dir, 'confusion_matrix.png'),
            'importance_plot': os.path.join(results_dir, 'classification_feature_importance.png')
        }
    
    def build_regression_model(self):
        """Build a regression model to predict S1-T1 gap values."""
        if self.feature_df is None:
            self.logger.error("No data loaded. Call load_data() first.")
            return None
            
        print("Building regression model for S1-T1 gap prediction...")
        
        if 's1_t1_gap_ev' not in self.feature_df.columns:
            self.logger.error("S1-T1 gap data not found in features.")
            return None
            
        # Filter to samples with S1-T1 gap data
        df_reg = self.feature_df[self.feature_df['s1_t1_gap_ev'].notna()].copy()
        
        if len(df_reg) < 10:
            print("Not enough samples with S1-T1 gap data for regression.")
            return None
            
        # Select features for regression
        feature_results = self.select_features('s1_t1_gap_ev', n_features=15)
        
        if not feature_results:
            self.logger.error("Feature selection failed.")
            return None
            
        selected_features = feature_results['features']
        
        # Create results directory
        results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare data for modeling
        X = df_reg[selected_features].copy()
        y = df_reg['s1_t1_gap_ev'].copy()
        
        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Train Random Forest regressor
        rf_regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        
        rf_regressor.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_regressor.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Regression model performance:")
        print(f"  - Mean Squared Error: {mse:.4f}")
        print(f"  - Root Mean Squared Error: {rmse:.4f}")
        print(f"  - R² Score: {r2:.4f}")
        
        # Visualize predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual S1-T1 Gap (eV)')
        plt.ylabel('Predicted S1-T1 Gap (eV)')
        plt.title('Random Forest Regression: Predicted vs Actual S1-T1 Gap')
        
        # Add text for model performance
        plt.text(
            0.05, 0.95, 
            f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', alpha=0.1)
        )
        
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # Line at y=0
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)  # Line at x=0
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'regression_prediction.png'))
        plt.close()
        
        # Calculate feature importance
        perm_importance = permutation_importance(
            rf_regressor, X_test, y_test, n_repeats=10, random_state=42
        )
        
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance for S1-T1 Gap Regression')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'regression_feature_importance.png'))
        plt.close()
        
        # Store model and results
        model_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = os.path.join(model_dir, 's1t1_gap_regressor.joblib')
        joblib.dump(rf_regressor, model_file)
        
        scaler_file = os.path.join(model_dir, 's1t1_gap_regressor_scaler.joblib')
        joblib.dump(scaler, scaler_file)
        
        # Save selected features
        with open(os.path.join(model_dir, 's1t1_gap_regressor_features.txt'), 'w') as f:
            f.write('\n'.join(selected_features))
            
        self.models['s1t1_regressor'] = {
            'model': rf_regressor,
            'scaler': scaler,
            'features': selected_features,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'importance': importance_df
        }
        
        self.feature_importance['s1_t1_gap_ev'] = importance_df
        
        return {
            'model_file': model_file,
            'scaler_file': scaler_file,
            'features': selected_features,
            'rmse': rmse,
            'r2': r2,
            'importance': importance_df,
            'prediction_plot': os.path.join(results_dir, 'regression_prediction.png'),
            'importance_plot': os.path.join(results_dir, 'regression_feature_importance.png')
        }
    
    def run_modeling_pipeline(self, feature_file=None, feature_agent=None):
        """Run the complete modeling workflow"""
        try:
            st.write("Starting modeling analysis workflow...")
            print("Starting modeling analysis workflow...")
            
            # 如果传入了feature_agent，则保存它
            if feature_agent:
                self.feature_agent = feature_agent
            
            # 加载特征数据到ModelAgent
            if feature_file:
                self.feature_file = feature_file
                
            if not self.load_data():
                st.error("Failed to load data, exiting modeling workflow")
                print("Failed to load data, exiting modeling workflow")
                return None
                    
            # 确保在self.feature_df中创建is_negative_gap目标变量
            if 's1_t1_gap_ev' in self.feature_df.columns:
                self.feature_df['is_negative_gap'] = (self.feature_df['s1_t1_gap_ev'] < 0).astype(int)
                st.write(f"Created classification target variable: is_negative_gap")
                st.write(f"Negative samples (S1-T1 < 0): {self.feature_df['is_negative_gap'].sum()}")
                st.write(f"Positive samples (S1-T1 >= 0): {(self.feature_df['is_negative_gap'] == 0).sum()}")
                print(f"Created classification target variable: is_negative_gap")
                print(f"Negative samples (S1-T1 < 0): {self.feature_df['is_negative_gap'].sum()}")
                print(f"Positive samples (S1-T1 >= 0): {(self.feature_df['is_negative_gap'] == 0).sum()}")
                
                # 如果有feature_agent，确保它也有相同的数据
                if hasattr(self, 'feature_agent') and self.feature_agent:
                    # 复制目标变量和数据到feature_agent
                    self.feature_agent.feature_df = self.feature_df.copy()
                    print("已将数据复制到feature_agent")
            
            st.write("Enhancing dataset with CREST data...")
            print("Enhancing dataset with CREST data...")
            # 确保方法存在
            try:
                self.feature_df = self.enhance_dataset_with_crest(self.feature_df)
                
                # 更新feature_agent中的数据
                if hasattr(self, 'feature_agent') and self.feature_agent:
                    self.feature_agent.feature_df = self.feature_df.copy()
                    print("已更新feature_agent中的增强数据")
            except Exception as e:
                st.error(f"Error enhancing dataset: {str(e)}")
                print(f"Error enhancing dataset: {str(e)}")
                    
            # Create results directory
            results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
            models_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/models'

            # Ensure directories exist
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)
            
            st.write(f"Results will be saved to: {results_dir}")
            st.write(f"Models will be saved to: {models_dir}")
            print(f"Results will be saved to: {results_dir}")
            print(f"Models will be saved to: {models_dir}")
            
            # Check for previous run results
            existing_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
            if existing_files:
                st.write(f"Found {len(existing_files)} existing chart files, will be replaced with new results")
                print(f"Found {len(existing_files)} existing chart files, will be replaced with new results")
            
            # Prepare classification data
            st.write("Preparing S1-T1 Gap classification data...")
            print("Preparing S1-T1 Gap classification data...")
            
            # Create classification target
            if 's1_t1_gap_ev' in self.feature_df.columns:
                # Print basic statistics of s1_t1_gap_ev column to confirm data is reasonable
                st.write(f"Descriptive statistics for s1_t1_gap_ev column: {self.feature_df['s1_t1_gap_ev'].describe()}")
                print(f"Descriptive statistics for s1_t1_gap_ev column: {self.feature_df['s1_t1_gap_ev'].describe()}")
                
                # Check for invalid values (NaN or infinity)
                invalid_count = self.feature_df['s1_t1_gap_ev'].isna().sum()
                if invalid_count > 0:
                    st.write(f"Warning: {invalid_count} invalid values in s1_t1_gap_ev column")
                    print(f"Warning: {invalid_count} invalid values in s1_t1_gap_ev column")
                
                self.feature_df['is_negative_gap'] = (self.feature_df['s1_t1_gap_ev'] < 0).astype(int)
                st.write(f"Created classification target variable: is_negative_gap")
                st.write(f"Negative samples (S1-T1 < 0): {self.feature_df['is_negative_gap'].sum()}")
                st.write(f"Positive samples (S1-T1 >= 0): {(self.feature_df['is_negative_gap'] == 0).sum()}")
                print(f"Created classification target variable: is_negative_gap")
                print(f"Negative samples (S1-T1 < 0): {self.feature_df['is_negative_gap'].sum()}")
                print(f"Positive samples (S1-T1 >= 0): {(self.feature_df['is_negative_gap'] == 0).sum()}")
            else:
                st.error(f"Warning: Could not find 's1_t1_gap_ev' column, cannot create classification target")
                print(f"Warning: Could not find 's1_t1_gap_ev' column, cannot create classification target")
                # Check if there might be other columns with gap in the name
                potential_gap_cols = [col for col in self.feature_df.columns if 'gap' in col.lower()]
                if potential_gap_cols:
                    st.write(f"Found potential gap columns: {potential_gap_cols}")
                    print(f"Found potential gap columns: {potential_gap_cols}")
                return None
            
            # Select features for classification target
            st.write("Selecting features for classification target...")
            print("Selecting features for classification target...")
            try:
                # 检查feature_agent是否存在且有select_features方法
                if hasattr(self, 'feature_agent') and self.feature_agent and hasattr(self.feature_agent, 'select_features'):
                    class_selection = self.feature_agent.select_features('is_negative_gap')
                else:
                    st.write("Using internal select_features method for classification")
                    print("Using internal select_features method for classification")
                    class_selection = self.select_features('is_negative_gap')
                    
                if not class_selection:
                    st.error("Classification feature selection failed")
                    print("Classification feature selection failed")
                    return None
                    
                class_features = class_selection['features']
                st.write(f"Selected {len(class_features)} classification features")
                print(f"Selected {len(class_features)} classification features")
            except Exception as e:
                st.error(f"Error during feature selection process: {str(e)}")
                print(f"Error during feature selection process: {str(e)}")
                # 尝试使用自身方法作为回退
                try:
                    class_selection = self.select_features('is_negative_gap')
                    if class_selection:
                        class_features = class_selection['features']
                        st.write(f"Fallback feature selection successful: {len(class_features)} features")
                        print(f"Fallback feature selection successful: {len(class_features)} features")
                    else:
                        return None
                except Exception as e2:
                    st.error(f"Fallback feature selection also failed: {str(e2)}")
                    print(f"Fallback feature selection also failed: {str(e2)}")
                    return None
                
            # Check if CREST features are included
            crest_features = [f for f in class_features if 'crest' in f.lower()]
            st.write(f"Classification features include {len(crest_features)} CREST features: {crest_features}")
            print(f"Classification features include {len(crest_features)} CREST features: {crest_features}")
            
            # Prepare regression data
            st.write("Preparing S1-T1 Gap regression data...")
            print("Preparing S1-T1 Gap regression data...")
            try:
                # 检查feature_agent是否存在且有select_features方法
                if hasattr(self, 'feature_agent') and self.feature_agent and hasattr(self.feature_agent, 'select_features'):
                    reg_selection = self.feature_agent.select_features('s1_t1_gap_ev')
                else:
                    st.write("Using internal select_features method for regression")
                    print("Using internal select_features method for regression")
                    reg_selection = self.select_features('s1_t1_gap_ev')
                    
                if not reg_selection:
                    st.error("Regression feature selection failed")
                    print("Regression feature selection failed")
                    return None
                    
                reg_features = reg_selection['features']
                st.write(f"Selected {len(reg_features)} regression features")
                print(f"Selected {len(reg_features)} regression features")
            except Exception as e:
                st.error(f"Error during regression feature selection process: {str(e)}")
                print(f"Error during regression feature selection process: {str(e)}")
                # 尝试使用自身方法作为回退
                try:
                    reg_selection = self.select_features('s1_t1_gap_ev')
                    if reg_selection:
                        reg_features = reg_selection['features']
                        st.write(f"Fallback regression feature selection successful: {len(reg_features)} features")
                        print(f"Fallback regression feature selection successful: {len(reg_features)} features")
                    else:
                        return None
                except Exception as e2:
                    st.error(f"Fallback regression feature selection also failed: {str(e2)}")
                    print(f"Fallback regression feature selection also failed: {str(e2)}")
                    return None
                
            # Check CREST features in regression features
            crest_reg_features = [f for f in reg_features if 'crest' in f.lower()]
            st.write(f"Regression features include {len(crest_reg_features)} CREST features: {crest_reg_features}")
            print(f"Regression features include {len(crest_reg_features)} CREST features: {crest_reg_features}")
            
            # Prepare classification training data
            st.write("Preparing classification training data...")
            print("Preparing classification training data...")
            try:
                class_data = self.feature_df[self.feature_df['is_negative_gap'].notna()].copy()
                
                # Check if class_features exist in dataset
                missing_features = [f for f in class_features if f not in class_data.columns]
                if missing_features:
                    st.error(f"The following classification features are missing from data: {missing_features}")
                    print(f"The following classification features are missing from data: {missing_features}")
                    # Filter out missing features
                    class_features = [f for f in class_features if f in class_data.columns]
                    st.write(f"Continuing with {len(class_features)} available features")
                    print(f"Continuing with {len(class_features)} available features")
                    
                if not class_features:
                    st.error("No classification features available, cannot continue")
                    print("No classification features available, cannot continue")
                    return None
                    
                X_class = class_data[class_features].values
                y_class = class_data['is_negative_gap'].values
                st.write(f"Classification training data: {X_class.shape[0]} rows, {X_class.shape[1]} columns")
                print(f"Classification training data: {X_class.shape[0]} rows, {X_class.shape[1]} columns")
            except Exception as e:
                st.error(f"Error preparing classification data: {str(e)}")
                print(f"Error preparing classification data: {str(e)}")
                return None
                
            # Prepare regression training data
            st.write("Preparing regression training data...")
            print("Preparing regression training data...")
            try:
                reg_data = self.feature_df[self.feature_df['s1_t1_gap_ev'].notna()].copy()
                
                # Check if reg_features exist in dataset
                missing_features = [f for f in reg_features if f not in reg_data.columns]
                if missing_features:
                    st.error(f"The following regression features are missing from data: {missing_features}")
                    print(f"The following regression features are missing from data: {missing_features}")
                    # Filter out missing features
                    reg_features = [f for f in reg_features if f in reg_data.columns]
                    st.write(f"Continuing with {len(reg_features)} available features")
                    print(f"Continuing with {len(reg_features)} available features")
                    
                if not reg_features:
                    st.error("No regression features available, cannot continue")
                    print("No regression features available, cannot continue")
                    return None
                    
                X_reg = reg_data[reg_features].values
                y_reg = reg_data['s1_t1_gap_ev'].values
                st.write(f"Regression training data: {X_reg.shape[0]} rows, {X_reg.shape[1]} columns")
                print(f"Regression training data: {X_reg.shape[0]} rows, {X_reg.shape[1]} columns")
            except Exception as e:
                st.error(f"Error preparing regression data: {str(e)}")
                print(f"Error preparing regression data: {str(e)}")
                return None
                
            # Train classification model
            st.write("Training S1-T1 Gap classification model...")
            print("Training S1-T1 Gap classification model...")
            try:
                from sklearn.ensemble import RandomForestClassifier
                
                # Use simple decision tree if sample size is less than 10
                if len(y_class) < 10:
                    st.write("Warning: Sample size is very small, using simplified model")
                    print("Warning: Sample size is very small, using simplified model")
                    from sklearn.tree import DecisionTreeClassifier
                    clf_model = DecisionTreeClassifier(max_depth=2, random_state=42)
                else:
                    clf_model = RandomForestClassifier(
                        n_estimators=100, 
                        max_depth=5,
                        min_samples_split=2,
                        class_weight='balanced',
                        random_state=42
                    )
                
                # Training and evaluation
                st.write("Using cross-validation to train and evaluate classification model...")
                print("Using cross-validation to train and evaluate classification model...")
                from sklearn.model_selection import cross_val_score, LeaveOneOut
                
                # Use leave-one-out cross-validation if sample size less than 10
                if len(y_class) < 10:
                    cv = LeaveOneOut()
                    scores = cross_val_score(clf_model, X_class, y_class, cv=cv, scoring='accuracy')
                    class_result = {
                        'accuracy': scores.mean(),
                        'model': clf_model,
                        'features': class_features
                    }
                else:
                    scores = cross_val_score(clf_model, X_class, y_class, cv=5, scoring='accuracy')
                    class_result = {
                        'accuracy': scores.mean(),
                        'model': clf_model,
                        'features': class_features
                    }
                
                st.write(f"Classification model average accuracy: {scores.mean():.4f}")
                print(f"Classification model average accuracy: {scores.mean():.4f}")
                
                # Train final classification model
                clf_model.fit(X_class, y_class)
                
                # Generate classification model visualizations
                st.write("Generating classification model visualizations...")
                print("Generating classification model visualizations...")
                self.generate_visualizations(clf_model, X_class, y_class, class_features, 'is_negative_gap')
            except Exception as e:
                st.error(f"Error during classification model training: {str(e)}")
                print(f"Error during classification model training: {str(e)}")
                class_result = None
                
            # Train regression model
            st.write("Training S1-T1 Gap regression model...")
            print("Training S1-T1 Gap regression model...")
            try:
                from sklearn.ensemble import RandomForestRegressor
                
                # Use simple model if sample size is less than 10
                if len(y_reg) < 10:
                    st.write("Warning: Sample size is very small, using simplified regression model")
                    print("Warning: Sample size is very small, using simplified regression model")
                    from sklearn.tree import DecisionTreeRegressor
                    reg_model = DecisionTreeRegressor(max_depth=2, random_state=42)
                else:
                    reg_model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=5,
                        min_samples_split=2,
                        random_state=42
                    )
                    
                # Training and evaluation
                st.write("Using cross-validation to train and evaluate regression model...")
                print("Using cross-validation to train and evaluate regression model...")
                from sklearn.model_selection import cross_val_score
                from sklearn.metrics import mean_squared_error, r2_score
                
                # Use leave-one-out cross-validation if sample size less than 10
                if len(y_reg) < 10:
                    cv = LeaveOneOut()
                    neg_mse_scores = cross_val_score(reg_model, X_reg, y_reg, cv=cv, 
                                                scoring='neg_mean_squared_error')
                    r2_scores = cross_val_score(reg_model, X_reg, y_reg, cv=cv, scoring='r2')
                else:
                    neg_mse_scores = cross_val_score(reg_model, X_reg, y_reg, cv=5, 
                                                scoring='neg_mean_squared_error')
                    r2_scores = cross_val_score(reg_model, X_reg, y_reg, cv=5, scoring='r2')
                    
                mse = -neg_mse_scores.mean()
                rmse = np.sqrt(mse)
                r2 = r2_scores.mean()
                
                reg_result = {
                    'rmse': rmse,
                    'r2': r2,
                    'model': reg_model,
                    'features': reg_features
                }
                
                st.write(f"Regression model RMSE: {rmse:.4f}, R²: {r2:.4f}")
                print(f"Regression model RMSE: {rmse:.4f}, R²: {r2:.4f}")
                
                # Train final regression model
                reg_model.fit(X_reg, y_reg)
                
                # Generate regression model visualizations
                st.write("Generating regression model visualizations...")
                print("Generating regression model visualizations...")
                self.generate_visualizations(reg_model, X_reg, y_reg, reg_features, 's1_t1_gap_ev')
            except Exception as e:
                st.error(f"Error during regression model training: {str(e)}")
                print(f"Error during regression model training: {str(e)}")
                reg_result = None
                
            # Save models
            st.write("Saving models to disk...")
            print("Saving models to disk...")
            try:
                import joblib
                
                # Save classification model
                if class_result:
                    clf_path = os.path.join(models_dir, 's1t1_gap_classifier.joblib')
                    joblib.dump(clf_model, clf_path)
                    st.write(f"Classification model saved: {clf_path}")
                    print(f"Classification model saved: {clf_path}")
                    
                    # Save feature names
                    with open(os.path.join(models_dir, 'classification_features.txt'), 'w') as f:
                        for feature in class_features:
                            f.write(f"{feature}\n")
                
                # Save regression model
                if reg_result:
                    reg_path = os.path.join(models_dir, 's1t1_gap_regressor.joblib')
                    joblib.dump(reg_model, reg_path)
                    st.write(f"Regression model saved: {reg_path}")
                    print(f"Regression model saved: {reg_path}")
                    
                    # Save feature names
                    with open(os.path.join(models_dir, 'regression_features.txt'), 'w') as f:
                        for feature in reg_features:
                            f.write(f"{feature}\n")
            except Exception as e:
                st.error(f"Error saving models: {str(e)}")
                print(f"Error saving models: {str(e)}")
                
            # Finally, check generated chart files
            st.write("Checking generated chart files:")
            print("Checking generated chart files:")
            generated_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
            st.write(f"Number of generated chart files: {len(generated_files)}")
            print(f"Number of generated chart files: {len(generated_files)}")
            for file in generated_files:
                st.write(f" - {file}")
                print(f" - {file}")
            
            result = {
                'classification': class_result,
                'regression': reg_result
            }
            
            if class_result or reg_result:
                return result
            else:
                st.error("Both classification and regression models failed to train")
                print("Both classification and regression models failed to train")
                return None
                
        except Exception as e:
            st.error(f"Error in modeling workflow: {str(e)}")
            print(f"Error in modeling workflow: {str(e)}")
            import traceback
            tb = traceback.format_exc()
            st.error(f"Complete error information: {tb}")
            print(f"Complete error information: {tb}")
            return None
    
    def generate_visualizations(self, model, X, y, feature_names, target_col):
        """Generate model visualizations"""
        try:
            # Create results directory
            results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
            os.makedirs(results_dir, exist_ok=True)
            
            print(f"Generating visualizations for {target_col}, saving to: {results_dir}")
            
            # Print detailed file path
            feature_importance_path = os.path.join(results_dir, f'feature_ranks_{target_col}.png')
            print(f"Feature importance chart path: {feature_importance_path}")
            
            # Ensure matplotlib backend is set correctly
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Critical font configuration to avoid errors
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial'],
                'font.size': 10,
                'text.color': 'black',
                'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
                'figure.dpi': 300,
                'savefig.dpi': 300
            })
            
            # Feature importance visualization
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Only show top 15 features
                n_features = min(15, len(feature_names))
                plt.barh(range(n_features), importances[indices[:n_features]], align='center')
                plt.yticks(range(n_features), [feature_names[i] for i in indices[:n_features]])
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                
                # NO TITLE - avoid font rendering issues
                
                # Save without tight_layout
                plt.savefig(feature_importance_path)
                plt.close()
                
                print(f"Feature importance chart saved: {feature_importance_path}")
            
            # For classification model, create confusion matrix
            if target_col == 'is_negative_gap':
                from sklearn.metrics import confusion_matrix
                from sklearn.model_selection import cross_val_predict
                
                # Use cross-validation to get predictions
                y_pred = cross_val_predict(model, X, y, cv=5)
                cm = confusion_matrix(y, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Positive', 'Negative'],
                            yticklabels=['Positive', 'Negative'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # NO TITLE - avoid font rendering issues
                
                # Save confusion matrix
                cm_path = os.path.join(results_dir, 'classification_confusion_matrix.png')
                plt.savefig(cm_path)
                plt.close()
                
                print(f"Confusion matrix chart saved: {cm_path}")
            
            # For regression model, create prediction vs actual plot
            if target_col == 's1_t1_gap_ev':
                from sklearn.model_selection import cross_val_predict
                
                # Use cross-validation to get predictions
                y_pred = cross_val_predict(model, X, y, cv=5)
                
                plt.figure(figsize=(8, 8))
                plt.scatter(y, y_pred, alpha=0.5)
                
                # Add ideal line (y=x)
                min_val = min(min(y), min(y_pred))
                max_val = max(max(y), max(y_pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                
                # Add zero lines
                plt.axhline(y=0, color='green', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='green', linestyle='-', alpha=0.3)
                
                # NO TITLE - avoid font rendering issues
                
                # Save regression plot
                reg_path = os.path.join(results_dir, 'regression_prediction.png')
                plt.savefig(reg_path)
                plt.close()
                
                print(f"Regression prediction chart saved: {reg_path}")
            
            # Create feature importance visualization (simplified version)
            feature_imp_path = os.path.join(results_dir, f'classification_feature_importance.png' if target_col == 'is_negative_gap' 
                                        else 'regression_feature_importance.png')
            
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 8))
                
                # Create simple feature importance bars without using seaborn or fancy features
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                # Basic bar chart without text labels that might cause issues
                plt.barh(range(len(importance_df)), importance_df['Importance'].values)
                plt.yticks(range(len(importance_df)), importance_df['Feature'].values)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                
                # NO TITLE - avoid font rendering issues
                
                plt.savefig(feature_imp_path)
                plt.close()
                
                print(f"Feature importance chart saved: {feature_imp_path}")
            
            # Add additional chart: highlight CREST features
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                
                # Create DataFrame for easier processing
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                })
                
                # Mark CREST features
                is_crest = [1 if 'crest' in str(f).lower() else 0 for f in importance_df['Feature']]
                importance_df['Is_CREST'] = is_crest
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Take only top 15 features
                plot_df = importance_df.head(15)
                
                # Create simple bars with different colors but minimal text
                y_pos = range(len(plot_df))
                for i, (_, row) in enumerate(plot_df.iterrows()):
                    color = 'orange' if row['Is_CREST'] == 1 else 'blue'
                    plt.barh(i, row['Importance'], color=color)
                
                plt.yticks(y_pos, plot_df['Feature'].values)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                
                # NO TITLE - avoid font rendering issues
                
                # Save chart
                crest_path = os.path.join(results_dir, f'crest_feature_importance_{target_col}.png')
                plt.savefig(crest_path)
                plt.close()
                
                print(f"CREST feature importance chart saved: {crest_path}")
            
            # Add text feature importance file for reference
            if hasattr(model, 'feature_importances_'):
                txt_path = os.path.join(results_dir, f'feature_importance_{target_col}.txt')
                with open(txt_path, 'w') as f:
                    f.write(f"Feature Importance Analysis - {target_col}\n")
                    f.write("="*50 + "\n\n")
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Mark CREST features
                    for i, row in importance_df.iterrows():
                        is_crest = 'CREST Feature' if 'crest' in str(row['Feature']).lower() else ''
                        f.write(f"{row['Feature']}: {row['Importance']:.6f} {is_crest}\n")
                
                print(f"Feature importance text file saved: {txt_path}")
            
            return True
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            return False