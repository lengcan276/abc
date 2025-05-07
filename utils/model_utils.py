# utils/model_utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import logging
import os
import joblib

class ModelUtils:
    """
    General utility class for modeling and evaluation
    """
    
    @staticmethod
    def prepare_features_targets(df, target_col, feature_cols=None, exclude_cols=None):
        """Prepare features and target variables"""
        # Filter data with target values
        df_target = df[df[target_col].notna()].copy()
        
        if len(df_target) < 10:
            logging.warning(f"{target_col} has too few samples ({len(df_target)}), at least 10 samples needed")
            return None, None
            
        # Determine features to use
        if feature_cols is None:
            # Exclude columns that shouldn't be used as features
            if exclude_cols is None:
                exclude_cols = ['Molecule', 'conformer', 'State', 'is_primary', target_col]
                
            # Select numeric features
            numeric_cols = df_target.select_dtypes(include=['float64', 'int64']).columns
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
        # Remove features with too many NaN values
        valid_features = []
        for col in feature_cols:
            if col in df_target.columns:
                nan_ratio = df_target[col].isna().mean()
                if nan_ratio < 0.3:  # If missing values less than 30%
                    valid_features.append(col)
                    
        if len(valid_features) == 0:
            logging.error("No valid features found")
            return None, None
            
        # Prepare feature matrix and target vector
        X = df_target[valid_features].copy()
        y = df_target[target_col].copy()
        
        # Replace infinity values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values (using median)
        X = X.fillna(X.median())
        
        return X, y, valid_features
        
    @staticmethod
    def scale_features(X, scaler_type='robust'):
        """Standardize features"""
        if scaler_type == 'robust':
            # Use robust scaler (less sensitive to outliers)
            scaler = RobustScaler()
        else:
            # Standard standardization
            scaler = StandardScaler()
            
        try:
            X_scaled = scaler.fit_transform(X)
            return X_scaled, scaler
        except Exception as e:
            logging.error(f"Error standardizing features: {e}")
            
            # Check for infinity and NaN again
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Use simple Z-score standardization
            X_mean = X.mean()
            X_std = X.std()
            X_std = X_std.replace(0, 1)  # Avoid division by zero
            X_scaled = (X - X_mean) / X_std
            
            # Create a simple scaler object to maintain interface consistency
            class SimpleScaler:
                def transform(self, X):
                    X = X.copy()
                    X = X.replace([np.inf, -np.inf], np.nan)
                    X = X.fillna(X_mean)
                    return (X - X_mean) / X_std
                    
            return X_scaled, SimpleScaler()
            
    @staticmethod
    def train_evaluate_classification(X, y, feature_names, results_dir=None, model_dir=None, class_names=None):
        """Train and evaluate classification model"""
        # Create results directory
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            
        # If no class names provided, use defaults
        if class_names is None:
            class_names = ['Positive', 'Negative']
            
        # Feature standardization
        X_scaled, scaler = ModelUtils.scale_features(X)
        
        # Split data
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
        
        # Predict
        y_pred = rf_classifier.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Output evaluation results
        logging.info(f"Classification model performance:")
        logging.info(f"  - Accuracy: {accuracy:.4f}")
        logging.info(f"  - Precision (Negative): {class_report['1']['precision']:.4f}")
        logging.info(f"  - Recall (Negative): {class_report['1']['recall']:.4f}")
        
        # Visualize confusion matrix
        if results_dir is not None:
            # Set matplotlib parameters to use English fonts
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('S1-T1 Gap Classification Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
            plt.close()
            
        # Calculate feature importance
        perm_importance = permutation_importance(
            rf_classifier, X_test, y_test, n_repeats=10, random_state=42
        )
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # Visualize feature importance
        if results_dir is not None:
            # Set matplotlib parameters to use English fonts
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Feature Importance for S1-T1 Gap Classification')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'classification_feature_importance.png'))
            plt.close()
            
        # Save model and results
        if model_dir is not None:
            model_file = os.path.join(model_dir, 's1t1_gap_classifier.joblib')
            joblib.dump(rf_classifier, model_file)
            
            scaler_file = os.path.join(model_dir, 's1t1_gap_classifier_scaler.joblib')
            joblib.dump(scaler, scaler_file)
            
            # Save selected features
            with open(os.path.join(model_dir, 's1t1_gap_classifier_features.txt'), 'w') as f:
                f.write('\n'.join(feature_names))
                
        return {
            'model': rf_classifier,
            'scaler': scaler,
            'features': feature_names,
            'accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'class_report': class_report,
            'importance': importance_df
        }
        
    @staticmethod
    def train_evaluate_regression(X, y, feature_names, results_dir=None, model_dir=None):
        """Train and evaluate regression model"""
        # Create results directory
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            
        # Feature standardization
        X_scaled, scaler = ModelUtils.scale_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Train Random Forest regressor
        rf_regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        
        rf_regressor.fit(X_train, y_train)
        
        # Predict
        y_pred = rf_regressor.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Output evaluation results
        logging.info(f"Regression model performance:")
        logging.info(f"  - Mean Squared Error: {mse:.4f}")
        logging.info(f"  - Root Mean Squared Error: {rmse:.4f}")
        logging.info(f"  - R² Score: {r2:.4f}")
        
        # Visualize predictions vs actual
        if results_dir is not None:
            # Set matplotlib parameters to use English fonts
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.7)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            plt.xlabel('Actual S1-T1 Gap (eV)')
            plt.ylabel('Predicted S1-T1 Gap (eV)')
            plt.title('Random Forest Regression: Predicted vs Actual S1-T1 Gap')
            
            # Add model performance text
            plt.text(
                0.05, 0.95, 
                f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', alpha=0.1)
            )
            
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # y=0 line
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)  # x=0 line
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'regression_prediction.png'))
            plt.close()
            
        # Calculate feature importance
        perm_importance = permutation_importance(
            rf_regressor, X_test, y_test, n_repeats=10, random_state=42
        )
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # Visualize feature importance
        if results_dir is not None:
            # Set matplotlib parameters to use English fonts
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Feature Importance for S1-T1 Gap Regression')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'regression_feature_importance.png'))
            plt.close()
            
        # Save model and results
        if model_dir is not None:
            model_file = os.path.join(model_dir, 's1t1_gap_regressor.joblib')
            joblib.dump(rf_regressor, model_file)
            
            scaler_file = os.path.join(model_dir, 's1t1_gap_regressor_scaler.joblib')
            joblib.dump(scaler, scaler_file)
            
            # Save selected features
            with open(os.path.join(model_dir, 's1t1_gap_regressor_features.txt'), 'w') as f:
                f.write('\n'.join(feature_names))
                
        return {
            'model': rf_regressor,
            'scaler': scaler,
            'features': feature_names,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'importance': importance_df,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
    @staticmethod
    def feature_selection(X, y, feature_names, classification=False):
        """Feature selection using various methods"""
        # Feature standardization
        X_scaled, _ = ModelUtils.scale_features(X)
        
        # Use Random Forest feature importance
        if classification:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
        rf.fit(X_scaled, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Use K-fold cross-validation to evaluate each feature's importance
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if classification else KFold(n_splits=5, shuffle=True, random_state=42)
        
        feature_scores = []
        for i, feature in enumerate(feature_names):
            feature_X = X_scaled[:, i].reshape(-1, 1)
            if classification:
                scores = cross_val_score(RandomForestClassifier(n_estimators=10, random_state=42), 
                                       feature_X, y, cv=cv, scoring='accuracy')
            else:
                scores = cross_val_score(RandomForestRegressor(n_estimators=10, random_state=42), 
                                       feature_X, y, cv=cv, scoring='r2')
                
            feature_scores.append({
                'Feature': feature,
                'CV_Score': scores.mean()
            })
            
        cv_importance = pd.DataFrame(feature_scores).sort_values('CV_Score', ascending=False)
        
        # Merge two importance scores
        merged_importance = importance.merge(cv_importance, on='Feature')
        merged_importance['Combined_Score'] = merged_importance[['Importance', 'CV_Score']].mean(axis=1)
        
        # Sort by combined score
        final_importance = merged_importance.sort_values('Combined_Score', ascending=False)
        
        return final_importance