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
    用于建模和评估的通用工具类
    """
    
    @staticmethod
    def prepare_features_targets(df, target_col, feature_cols=None, exclude_cols=None):
        """准备特征和目标变量"""
        # 筛选有目标值的数据
        df_target = df[df[target_col].notna()].copy()
        
        if len(df_target) < 10:
            logging.warning(f"{target_col} 数据样本太少（{len(df_target)}），需要至少 10 个样本")
            return None, None
            
        # 确定要使用的特征
        if feature_cols is None:
            # 排除不应该用作特征的列
            if exclude_cols is None:
                exclude_cols = ['Molecule', 'conformer', 'State', 'is_primary', target_col]
                
            # 选择数值特征
            numeric_cols = df_target.select_dtypes(include=['float64', 'int64']).columns
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
        # 移除含有大量 NaN 的特征
        valid_features = []
        for col in feature_cols:
            if col in df_target.columns:
                nan_ratio = df_target[col].isna().mean()
                if nan_ratio < 0.3:  # 如果缺失值少于 30%
                    valid_features.append(col)
                    
        if len(valid_features) == 0:
            logging.error("没有找到有效特征")
            return None, None
            
        # 准备特征矩阵和目标向量
        X = df_target[valid_features].copy()
        y = df_target[target_col].copy()
        
        # 替换无穷大值为 NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # 填充 NaN 值（使用中位数）
        X = X.fillna(X.median())
        
        return X, y, valid_features
        
    @staticmethod
    def scale_features(X, scaler_type='robust'):
        """标准化特征"""
        if scaler_type == 'robust':
            # 使用稳健缩放器（对异常值不敏感）
            scaler = RobustScaler()
        else:
            # 标准标准化
            scaler = StandardScaler()
            
        try:
            X_scaled = scaler.fit_transform(X)
            return X_scaled, scaler
        except Exception as e:
            logging.error(f"标准化特征时出错: {e}")
            
            # 再次检查无穷大和 NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # 使用简单的 Z 分数标准化
            X_mean = X.mean()
            X_std = X.std()
            X_std = X_std.replace(0, 1)  # 避免除以零
            X_scaled = (X - X_mean) / X_std
            
            # 创建一个简单的缩放器对象以保持接口一致
            class SimpleScaler:
                def transform(self, X):
                    X = X.copy()
                    X = X.replace([np.inf, -np.inf], np.nan)
                    X = X.fillna(X_mean)
                    return (X - X_mean) / X_std
                    
            return X_scaled, SimpleScaler()
            
    @staticmethod
    def train_evaluate_classification(X, y, feature_names, results_dir=None, model_dir=None, class_names=None):
        """训练和评估分类模型"""
        # 创建结果目录
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            
        # 如果没有提供类名，使用默认值
        if class_names is None:
            class_names = ['阳性', '阴性']
            
        # 特征标准化
        X_scaled, scaler = ModelUtils.scale_features(X)
        
        # 拆分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 训练随机森林分类器
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        
        rf_classifier.fit(X_train, y_train)
        
        # 预测
        y_pred = rf_classifier.predict(X_test)
        
        # 评估模型
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # 输出评估结果
        logging.info(f"分类模型性能:")
        logging.info(f"  - 准确率: {accuracy:.4f}")
        logging.info(f"  - 精确率 (阴性): {class_report['1']['precision']:.4f}")
        logging.info(f"  - 召回率 (阴性): {class_report['1']['recall']:.4f}")
        
        # 可视化混淆矩阵
        if results_dir is not None:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                      xticklabels=class_names,
                      yticklabels=class_names)
            plt.xlabel('预测')
            plt.ylabel('实际')
            plt.title('S1-T1 能隙分类的混淆矩阵')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
            plt.close()
            
        # 计算特征重要性
        perm_importance = permutation_importance(
            rf_classifier, X_test, y_test, n_repeats=10, random_state=42
        )
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # 可视化特征重要性
        if results_dir is not None:
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('S1-T1 能隙分类的特征重要性')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'classification_feature_importance.png'))
            plt.close()
            
        # 保存模型和结果
        if model_dir is not None:
            model_file = os.path.join(model_dir, 's1t1_gap_classifier.joblib')
            joblib.dump(rf_classifier, model_file)
            
            scaler_file = os.path.join(model_dir, 's1t1_gap_classifier_scaler.joblib')
            joblib.dump(scaler, scaler_file)
            
            # 保存所选特征
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
        """训练和评估回归模型"""
        # 创建结果目录
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            
        # 特征标准化
        X_scaled, scaler = ModelUtils.scale_features(X)
        
        # 拆分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # 训练随机森林回归器
        rf_regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        
        rf_regressor.fit(X_train, y_train)
        
        # 预测
        y_pred = rf_regressor.predict(X_test)
        
        # 评估模型
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # 输出评估结果
        logging.info(f"回归模型性能:")
        logging.info(f"  - 均方误差: {mse:.4f}")
        logging.info(f"  - 均方根误差: {rmse:.4f}")
        logging.info(f"  - R² 分数: {r2:.4f}")
        
        # 可视化预测 vs 实际
        if results_dir is not None:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.7)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            plt.xlabel('实际 S1-T1 能隙 (eV)')
            plt.ylabel('预测 S1-T1 能隙 (eV)')
            plt.title('随机森林回归: 预测 vs 实际 S1-T1 能隙')
            
            # 添加模型性能文本
            plt.text(
                0.05, 0.95, 
                f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', alpha=0.1)
            )
            
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # y=0 线
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)  # x=0 线
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'regression_prediction.png'))
            plt.close()
            
        # 计算特征重要性
        perm_importance = permutation_importance(
            rf_regressor, X_test, y_test, n_repeats=10, random_state=42
        )
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # 可视化特征重要性
        if results_dir is not None:
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('S1-T1 能隙回归的特征重要性')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'regression_feature_importance.png'))
            plt.close()
            
        # 保存模型和结果
        if model_dir is not None:
            model_file = os.path.join(model_dir, 's1t1_gap_regressor.joblib')
            joblib.dump(rf_regressor, model_file)
            
            scaler_file = os.path.join(model_dir, 's1t1_gap_regressor_scaler.joblib')
            joblib.dump(scaler, scaler_file)
            
            # 保存所选特征
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
        """使用多种方法进行特征选择"""
        # 特征标准化
        X_scaled, _ = ModelUtils.scale_features(X)
        
        # 使用随机森林特征重要性
        if classification:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
        rf.fit(X_scaled, y)
        
        # 获取特征重要性
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # 使用 K 折交叉验证评估每个特征的重要性
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
        
        # 合并两种重要性分数
        merged_importance = importance.merge(cv_importance, on='Feature')
        merged_importance['Combined_Score'] = merged_importance[['Importance', 'CV_Score']].mean(axis=1)
        
        # 按组合分数排序
        final_importance = merged_importance.sort_values('Combined_Score', ascending=False)
        
        return final_importance
