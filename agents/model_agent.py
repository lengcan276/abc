# agents/model_agent.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    def __init__(self, feature_file=None):
        """Initialize the ModelAgent with feature data file."""
        self.feature_file = feature_file
        self.feature_df = None
        self.selected_features = {}
        self.models = {}
        self.feature_importance = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the model agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/logs/model_agent.log')
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
    
    def enhance_dataset_with_crest(feature_df):
        """
        使用CREST数据增强数据集
        
        Args:
            feature_df: 原始特征数据框
        
        Returns:
            增强后的特征数据框
        """
        print("开始使用CREST数据增强数据集...")
        original_rows = len(feature_df)
        
        # 1. 创建基于原始分子的扩展数据框
        enhanced_df = feature_df.copy()
        
        # 2. 识别包含CREST数据的分子
        crest_columns = [col for col in feature_df.columns if 'crest' in col.lower()]
        molecules_with_crest = feature_df[feature_df[crest_columns].notna().any(axis=1)]['Molecule'].unique()
        
        print(f"找到{len(molecules_with_crest)}个带有CREST数据的分子")
        
        # 3. 对每个有CREST数据的分子，创建额外的"虚拟样本"
        synthetic_samples = []
        
        for molecule in molecules_with_crest:
            # 获取该分子的数据
            mol_data = feature_df[feature_df['Molecule'] == molecule].copy()
            
            # 检查是否有构象数量信息
            conformer_cols = [col for col in crest_columns if 'num_conformers' in col]
            
            for _, row in mol_data.iterrows():
                # 对于每个状态(neutral, cation, triplet)，尝试创建合成样本
                for state in ['neutral', 'cation', 'triplet']:
                    conformer_col = f"{state}_crest_num_conformers"
                    energy_range_col = f"{state}_crest_energy_range"
                    
                    # 检查是否有该状态的CREST数据
                    if conformer_col in row and not pd.isna(row[conformer_col]) and row[conformer_col] > 1:
                        # 获取构象数量
                        num_conformers = int(row[conformer_col])
                        
                        # 创建基于该分子的合成样本
                        for i in range(min(num_conformers-1, 2)):  # 最多创建2个额外样本，避免过多
                            # 复制原始数据
                            synthetic_row = row.copy()
                            
                            # 修改分子名称以标识这是合成样本
                            synthetic_row['Molecule'] = f"{molecule}_crest_synth_{i+1}"
                            
                            # 对CREST特征进行轻微扰动
                            if energy_range_col in row and not pd.isna(row[energy_range_col]):
                                # 根据能量范围创建合理的扰动
                                energy_range = row[energy_range_col]
                                # 在原始能量范围基础上添加小的随机波动 (-15% 到 +15%)
                                perturbation = np.random.uniform(-0.15, 0.15) * energy_range
                                synthetic_row[energy_range_col] = energy_range + perturbation
                                
                                # 如果有S1-T1能隙数据，也进行轻微扰动
                                if 's1_t1_gap_ev' in row and not pd.isna(row['s1_t1_gap_ev']):
                                    gap = row['s1_t1_gap_ev']
                                    # 保持间隙符号相同，但添加小的变化（最多±0.1eV）
                                    gap_perturbation = np.random.uniform(-0.1, 0.1)
                                    synthetic_row['s1_t1_gap_ev'] = gap + gap_perturbation
                                    # 确保符号不变
                                    if (gap < 0 and synthetic_row['s1_t1_gap_ev'] > 0) or \
                                    (gap > 0 and synthetic_row['s1_t1_gap_ev'] < 0):
                                        synthetic_row['s1_t1_gap_ev'] = gap
                                    
                                # 添加该合成样本
                                synthetic_samples.append(synthetic_row)
                                print(f"为分子 {molecule} 创建基于{state}状态CREST数据的合成样本")
        
        # 4. 将合成样本添加到数据框
        if synthetic_samples:
            synthetic_df = pd.DataFrame(synthetic_samples)
            enhanced_df = pd.concat([enhanced_df, synthetic_df], ignore_index=True)
            print(f"添加了{len(synthetic_samples)}个基于CREST数据的合成样本")
            print(f"数据集从{original_rows}行扩展到{len(enhanced_df)}行")
        else:
            print("没有创建任何合成样本")
        
        # 5. 确保all CREST特征被保留
        for col in crest_columns:
            if col not in enhanced_df.columns:
                print(f"警告: CREST特征 {col} 不在增强数据集中")
        
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
        
        # 检查是否有CREST特征
        crest_features = [col for col in feature_cols if 'crest' in col.lower()]
        print(f"Found {len(crest_features)} CREST features: {crest_features}")
        
        # Remove features with too many NaN values
        valid_features = []
        for col in feature_cols:
            nan_ratio = df_target[col].isna().mean()
            # 对CREST特征使用更宽松的缺失值标准
            if 'crest' in col.lower():
                if nan_ratio < 0.5:  # 如果CREST特征缺失值少于50%，仍保留它
                    valid_features.append(col)
                    print(f"Keeping CREST feature '{col}' with {nan_ratio*100:.1f}% missing values")
            else:
                if nan_ratio < 0.3:  # 非CREST特征保持原有30%的阈值
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
        
        # 针对CREST特征的特殊处理
        for col in X.columns:
            if 'crest' in col.lower():
                # 对于CREST特征中的缺失值，使用0填充而不是中位数
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
        # 对于非CREST特征，使用中位数填充缺失值
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
        
        # 确保CREST特征的保留
        # 获取前n_features*0.7个特征
        n_auto_select = int(n_features * 0.7)
        auto_features = combined_scores.sort_values('Avg_Rank').head(n_auto_select)['Feature'].tolist()
        
        # 确保至少包含top 3的CREST特征
        crest_in_scores = combined_scores[combined_scores['Feature'].str.contains('crest', case=False)]
        top_crest = crest_in_scores.sort_values('Avg_Rank').head(3)['Feature'].tolist()
        
        # 合并自动选择的特征和top CREST特征
        final_features = list(set(auto_features + top_crest))
        
        # 如果特征总数仍小于n_features，添加更多特征
        remaining_features = [f for f in combined_scores.sort_values('Avg_Rank')['Feature'].tolist() 
                            if f not in final_features]
        
        while len(final_features) < n_features and remaining_features:
            final_features.append(remaining_features.pop(0))
        
        print(f"Selected top {len(final_features)} features:")
        # 分别显示常规特征和CREST特征
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
        results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/modeling'
        os.makedirs(results_dir, exist_ok=True)
        
        # Visualize feature importance with CREST features highlighted
        plt.figure(figsize=(12, 8))
        top_features = combined_scores.sort_values('Avg_Rank').head(15)
        
        # 为CREST特征使用不同的颜色
        colors = ['#1f77b4' if 'crest' not in feat.lower() else '#ff7f0e' 
                for feat in top_features['Feature']]
        
        sns.barplot(x='Avg_Rank', y='Feature', data=top_features, palette=colors)
        plt.title(f'Top 15 Features for Predicting {target_col} (Lower Rank is Better)')
        plt.xlabel('Average Rank')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Regular Features'),
            Patch(facecolor='#ff7f0e', label='CREST Features')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'feature_ranks_{target_col}.png'))
        plt.close()
        
        # 额外创建CREST特征重要性专用图表
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
        results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/modeling'
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
        
        # Store model and results
        model_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/models'
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
        results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/modeling'
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
        model_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/models'
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
    
    def run_modeling_pipeline(self):
        """运行完整的建模流程"""
        try:
            print("开始建模分析流程...")
            
            # 加载特征数据
            if not self.load_data():
                print("加载数据失败，退出建模流程")
                return None
            print("使用CREST数据增强数据集...")
            self.feature_df = self.enhance_dataset_with_crest(self.feature_df)
            # 创建结果目录
            results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/modeling'
            models_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/models'
            
            # 确保目录存在
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)
            
            print(f"结果将保存到: {results_dir}")
            print(f"模型将保存到: {models_dir}")
            
            # 检查之前运行的结果
            existing_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
            if existing_files:
                print(f"发现{len(existing_files)}个现有图表文件，将被新结果替换")
            
            # 准备分类数据
            print("准备S1-T1 Gap分类数据...")
            
            # 创建分类目标
            if 's1_t1_gap_ev' in self.feature_df.columns:
                self.feature_df['is_negative_gap'] = (self.feature_df['s1_t1_gap_ev'] < 0).astype(int)
                print(f"创建了分类目标变量: is_negative_gap")
                print(f"阴性样本(S1-T1 < 0): {self.feature_df['is_negative_gap'].sum()}")
                print(f"阳性样本(S1-T1 >= 0): {(self.feature_df['is_negative_gap'] == 0).sum()}")
            else:
                print(f"警告: 找不到's1_t1_gap_ev'列，无法创建分类目标")
                return None
            
            # 对分类目标进行特征选择
            print("对分类目标进行特征选择...")
            class_selection = self.feature_agent.select_features('is_negative_gap')
            if not class_selection:
                print("分类特征选择失败")
                return None
                
            class_features = class_selection['features']
            print(f"选择了{len(class_features)}个分类特征")
            
            # 检查是否包含CREST特征
            crest_features = [f for f in class_features if 'crest' in f.lower()]
            print(f"分类特征中包含{len(crest_features)}个CREST特征: {crest_features}")
            
            # 准备回归数据
            print("准备S1-T1 Gap回归数据...")
            reg_selection = self.feature_agent.select_features('s1_t1_gap_ev')
            if not reg_selection:
                print("回归特征选择失败")
                return None
                
            reg_features = reg_selection['features']
            print(f"选择了{len(reg_features)}个回归特征")
            
            # 检查回归特征中的CREST特征
            crest_reg_features = [f for f in reg_features if 'crest' in f.lower()]
            print(f"回归特征中包含{len(crest_reg_features)}个CREST特征: {crest_reg_features}")
            
            # 准备分类训练数据
            print("准备分类训练数据...")
            class_data = self.feature_df[self.feature_df['is_negative_gap'].notna()].copy()
            X_class = class_data[class_features].values
            y_class = class_data['is_negative_gap'].values
            print(f"分类训练数据: {X_class.shape[0]}行, {X_class.shape[1]}列")
            
            # 准备回归训练数据
            print("准备回归训练数据...")
            reg_data = self.feature_df[self.feature_df['s1_t1_gap_ev'].notna()].copy()
            X_reg = reg_data[reg_features].values
            y_reg = reg_data['s1_t1_gap_ev'].values
            print(f"回归训练数据: {X_reg.shape[0]}行, {X_reg.shape[1]}列")
            
            # 训练分类模型
            print("训练S1-T1 Gap分类模型...")
            from sklearn.ensemble import RandomForestClassifier
            
            # 如果样本量小于10，使用简单决策树
            if len(y_class) < 10:
                print("警告: 样本量很小，使用简化模型")
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
            
            # 训练和评估
            print("使用交叉验证训练和评估分类模型...")
            from sklearn.model_selection import cross_val_score, LeaveOneOut
            
            # 样本量小于10使用留一交叉验证
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
            
            print(f"分类模型平均准确率: {scores.mean():.4f}")
            
            # 训练最终分类模型
            clf_model.fit(X_class, y_class)
            
            # 生成分类模型可视化
            print("生成分类模型可视化...")
            self.generate_visualizations(clf_model, X_class, y_class, class_features, 'is_negative_gap')
            
            # 训练回归模型
            print("训练S1-T1 Gap回归模型...")
            from sklearn.ensemble import RandomForestRegressor
            
            # 如果样本量小于10，使用简单模型
            if len(y_reg) < 10:
                print("警告: 样本量很小，使用简化回归模型")
                from sklearn.tree import DecisionTreeRegressor
                reg_model = DecisionTreeRegressor(max_depth=2, random_state=42)
            else:
                reg_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=2,
                    random_state=42
                )
                
            # 训练和评估
            print("使用交叉验证训练和评估回归模型...")
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import mean_squared_error, r2_score
            
            # 样本量小于10使用留一交叉验证
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
            
            print(f"回归模型RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # 训练最终回归模型
            reg_model.fit(X_reg, y_reg)
            
            # 生成回归模型可视化
            print("生成回归模型可视化...")
            self.generate_visualizations(reg_model, X_reg, y_reg, reg_features, 's1_t1_gap_ev')
            
            # 保存模型
            print("保存模型到磁盘...")
            import joblib
            
            # 保存分类模型
            clf_path = os.path.join(models_dir, 's1t1_gap_classifier.joblib')
            joblib.dump(clf_model, clf_path)
            print(f"分类模型已保存: {clf_path}")
            
            # 保存回归模型
            reg_path = os.path.join(models_dir, 's1t1_gap_regressor.joblib')
            joblib.dump(reg_model, reg_path)
            print(f"回归模型已保存: {reg_path}")
            
            # 保存特征名称
            with open(os.path.join(models_dir, 'classification_features.txt'), 'w') as f:
                for feature in class_features:
                    f.write(f"{feature}\n")
            
            with open(os.path.join(models_dir, 'regression_features.txt'), 'w') as f:
                for feature in reg_features:
                    f.write(f"{feature}\n")
            
            # 最后，检查生成的图表文件
            print("检查生成的图表文件:")
            generated_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
            print(f"生成的图表文件数量: {len(generated_files)}")
            for file in generated_files:
                print(f" - {file}")
            
            return {
                'classification': class_result,
                'regression': reg_result
            }
        except Exception as e:
            print(f"建模流程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_visualizations(self, model, X, y, feature_names, target_col):
        """生成模型可视化图表"""
        try:
            # 创建结果目录
            results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/modeling'
            os.makedirs(results_dir, exist_ok=True)
            
            print(f"正在生成{target_col}的可视化图表，保存到: {results_dir}")
            
            # 添加详细的文件路径打印
            feature_importance_path = os.path.join(results_dir, f'feature_ranks_{target_col}.png')
            print(f"特征重要性图表路径: {feature_importance_path}")
            
            # 确保matplotlib后端正确设置（避免GUI问题）
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            # 特征重要性可视化
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # 只展示前15个特征
                n_features = min(15, len(feature_names))
                plt.barh(range(n_features), importances[indices[:n_features]], align='center')
                plt.yticks(range(n_features), [feature_names[i] for i in indices[:n_features]])
                plt.xlabel('特征重要性')
                plt.ylabel('特征')
                plt.title(f'{target_col}特征重要性')
                
                # 使用高DPI和紧凑布局保存
                plt.tight_layout()
                plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"特征重要性图表已保存: {feature_importance_path}")
                
                # 检查文件是否存在
                if os.path.exists(feature_importance_path):
                    print(f"确认文件已成功创建: {feature_importance_path}")
                    print(f"文件大小: {os.path.getsize(feature_importance_path)} 字节")
                else:
                    print(f"警告: 文件未能创建: {feature_importance_path}")
            
            # 为分类模型创建混淆矩阵
            if target_col == 'is_negative_gap':
                from sklearn.metrics import confusion_matrix
                from sklearn.model_selection import cross_val_predict
                
                # 使用交叉验证获取预测
                y_pred = cross_val_predict(model, X, y, cv=5)
                cm = confusion_matrix(y, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['正S1-T1', '负S1-T1'],
                            yticklabels=['正S1-T1', '负S1-T1'])
                plt.xlabel('预测')
                plt.ylabel('实际')
                plt.title('分类模型混淆矩阵')
                
                cm_path = os.path.join(results_dir, 'classification_confusion_matrix.png')
                plt.tight_layout()
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"混淆矩阵图表已保存: {cm_path}")
            
            # 为回归模型创建预测vs实际值图
            if target_col == 's1_t1_gap_ev':
                from sklearn.model_selection import cross_val_predict
                
                # 使用交叉验证获取预测
                y_pred = cross_val_predict(model, X, y, cv=5)
                
                plt.figure(figsize=(8, 8))
                plt.scatter(y, y_pred, alpha=0.5)
                
                # 添加理想线(y=x)
                min_val = min(min(y), min(y_pred))
                max_val = max(max(y), max(y_pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                plt.xlabel('实际值')
                plt.ylabel('预测值')
                plt.title('回归模型: 预测 vs 实际')
                
                # 添加零线
                plt.axhline(y=0, color='green', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='green', linestyle='-', alpha=0.3)
                
                reg_path = os.path.join(results_dir, 'regression_prediction.png')
                plt.tight_layout()
                plt.savefig(reg_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"回归预测图表已保存: {reg_path}")
                
            # 添加额外的图表：突出显示CREST特征
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                
                # 创建数据框以便于处理
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                })
                
                # 标记CREST特征
                importance_df['Is_CREST'] = importance_df['Feature'].apply(
                    lambda x: 'CREST特征' if 'crest' in str(x).lower() else '其他特征'
                )
                
                # 按重要性排序
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # 只取前15个特征
                plot_df = importance_df.head(15)
                
                # 使用seaborn创建彩色条形图
                ax = sns.barplot(x='Importance', y='Feature', hue='Is_CREST', data=plot_df)
                plt.title(f'{target_col}特征重要性 (带CREST特征标记)')
                plt.tight_layout()
                
                crest_path = os.path.join(results_dir, f'crest_feature_importance_{target_col}.png')
                plt.savefig(crest_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"CREST特征重要性图表已保存: {crest_path}")
                
            # 添加文本特征重要性文件以防图像加载失败
            if hasattr(model, 'feature_importances_'):
                txt_path = os.path.join(results_dir, f'feature_importance_{target_col}.txt')
                with open(txt_path, 'w') as f:
                    f.write(f"特征重要性分析 - {target_col}\n")
                    f.write("="*50 + "\n\n")
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # 标记CREST特征
                    for i, row in importance_df.iterrows():
                        is_crest = 'CREST特征' if 'crest' in str(row['Feature']).lower() else ''
                        f.write(f"{row['Feature']}: {row['Importance']:.6f} {is_crest}\n")
                
                print(f"特征重要性文本文件已保存: {txt_path}")
            
            return True
        except Exception as e:
            print(f"生成可视化时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False