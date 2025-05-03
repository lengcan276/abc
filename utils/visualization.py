# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

class VisualizationUtils:
    """
    用于创建可视化图表的工具类
    """
    
    @staticmethod
    def set_plot_style():
        """设置绘图样式"""
        # 设置 seaborn 样式
        sns.set(style="whitegrid")
        
        # 设置 matplotlib 参数
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 100
        
    @staticmethod
    def create_s1t1_gap_distribution(df, save_path=None):
        """创建 S1-T1 能隙分布图"""
        # 设置绘图样式
        VisualizationUtils.set_plot_style()
        
        # 确保有 s1_t1_gap_ev 列
        if 's1_t1_gap_ev' not in df.columns:
            logging.error("数据框中没有 s1_t1_gap_ev 列")
            return None
            
        # 筛选有能隙数据的行
        gap_data = df[df['s1_t1_gap_ev'].notna()].copy()
        
        if len(gap_data) == 0:
            logging.error("没有找到有效的 S1-T1 能隙数据")
            return None
            
        # 添加类别列（负值 vs 正值）
        gap_data['gap_type'] = gap_data['s1_t1_gap_ev'].apply(
            lambda x: '负值能隙' if x < 0 else '正值能隙'
        )
        
        # 创建分布图
        plt.figure(figsize=(10, 6))
        sns.histplot(data=gap_data, x='s1_t1_gap_ev', hue='gap_type', bins=20, kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('S1-T1 能隙分布')
        plt.xlabel('S1-T1 能隙 (eV)')
        plt.tight_layout()
        
        # 保存图像（如果提供了路径）
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"S1-T1 能隙分布图已保存至 {save_path}")
            
        return plt.gcf()
        
    @staticmethod
    def create_feature_comparison(negative_df, positive_df, feature, save_path=None):
        """创建特征比较图（负值 vs 正值能隙）"""
        # 设置绘图样式
        VisualizationUtils.set_plot_style()
        
        # 确保两个数据框都有指定特征
        if feature not in negative_df.columns or feature not in positive_df.columns:
            logging.error(f"数据框中没有 {feature} 列")
            return None
            
        # 获取特征数据
        neg_data = negative_df[feature].dropna()
        pos_data = positive_df[feature].dropna()
        
        if len(neg_data) == 0 or len(pos_data) == 0:
            logging.error(f"没有找到有效的 {feature} 数据")
            return None
            
        # 创建箱线图
        plt.figure(figsize=(8, 6))
        
        box_data = [neg_data.values, pos_data.values]
        
        plt.boxplot(box_data, labels=['负值能隙', '正值能隙'])
        plt.title(f'{feature.replace("_", " ").title()}: 负值 vs 正值能隙')
        plt.ylabel(feature)
        
        # 添加数据点
        x_pos = [1, 2]
        for i, data in enumerate([neg_data, pos_data]):
            # 添加抖动到 x 位置
            x = np.random.normal(x_pos[i], 0.05, size=len(data))
            plt.scatter(x, data, alpha=0.3, s=10)
            
        # 添加均值作为星号
        means = [neg_data.mean(), pos_data.mean()]
        plt.plot(x_pos, means, 'r*', markersize=10)
        
        plt.tight_layout()
        
        # 保存图像（如果提供了路径）
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"{feature} 比较图已保存至 {save_path}")
            
        return plt.gcf()
        
    @staticmethod
    def create_pca_plot(negative_df, positive_df, features, save_path=None):
        """创建 PCA 分析图"""
        # 设置绘图样式
        VisualizationUtils.set_plot_style()
        
        # 确保两个数据框都有指定特征
        for feature in features:
            if feature not in negative_df.columns or feature not in positive_df.columns:
                logging.error(f"数据框中没有 {feature} 列")
                return None
                
        # 准备数据
        neg_subset = negative_df[features].dropna().copy()
        neg_subset['group'] = '负值能隙'
        
        pos_subset = positive_df[features].dropna().copy()
        pos_subset['group'] = '正值能隙'
        
        # 合并数据
        combined = pd.concat([neg_subset, pos_subset], ignore_index=True)
        
        if len(combined) < 5:
            logging.error("PCA 分析的数据太少")
            return None
            
        # 标准化特征
        X = combined[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 应用 PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        # 创建 PCA 图
        plt.figure(figsize=(10, 8))
        for group, color in zip(['负值能隙', '正值能隙'], ['red', 'blue']):
            mask = combined['group'] == group
            plt.scatter(
                pca_result[mask, 0], pca_result[mask, 1],
                label=group, color=color, alpha=0.7
            )
            
        # 添加特征向量
        feature_vectors = pca.components_.T
        feature_names = features
        
        # 缩放向量以增加可见性
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
            
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
        plt.title('分子属性 PCA: 负值 vs 正值 S1-T1 能隙')
        plt.legend()
        plt.tight_layout()
        
        # 保存图像（如果提供了路径）
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"PCA 分析图已保存至 {save_path}")
            
        return plt.gcf()
        
    @staticmethod
    def create_radar_chart(negative_df, positive_df, features, save_path=None):
        """创建雷达图比较特征"""
        # 设置绘图样式
        VisualizationUtils.set_plot_style()
        
        # 确保有足够的特征
        if len(features) < 3:
            logging.error("雷达图需要至少 3 个特征")
            return None
            
        # 确保两个数据框都有指定特征
        for feature in features:
            if feature not in negative_df.columns or feature not in positive_df.columns:
                logging.error(f"数据框中没有 {feature} 列")
                return None
                
        # 计算两组的均值
        neg_means = [negative_df[f].mean() for f in features]
        pos_means = [positive_df[f].mean() for f in features]
        
        # 归一化值到 [0,1] 用于雷达图
        all_values = np.concatenate([neg_means, pos_means])
        min_vals = np.min(all_values)
        max_vals = np.max(all_values)
        
        # 处理所有值相同的情况
        if max_vals == min_vals:
            normalized_neg = [0.5 for _ in neg_means]
            normalized_pos = [0.5 for _ in pos_means]
        else:
            normalized_neg = [(x - min_vals) / (max_vals - min_vals) for x in neg_means]
            normalized_pos = [(x - min_vals) / (max_vals - min_vals) for x in pos_means]
        
        # 创建雷达图
        labels = [f.replace('_', ' ').title() for f in features]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        
        # 闭合多边形
        normalized_neg.append(normalized_neg[0])
        normalized_pos.append(normalized_pos[0])
        angles.append(angles[0])
        labels.append(labels[0])
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        ax.plot(angles, normalized_neg, 'r-', linewidth=2, label='负值能隙')
        ax.fill(angles, normalized_neg, 'r', alpha=0.1)
        
        ax.plot(angles, normalized_pos, 'b-', linewidth=2, label='正值能隙')
        ax.fill(angles, normalized_pos, 'b', alpha=0.1)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        ax.set_title('特征比较: 负值 vs 正值 S1-T1 能隙', size=15)
        ax.legend(loc='upper right')
        
        # 保存图像（如果提供了路径）
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"雷达图已保存至 {save_path}")
            
        return fig
        
    @staticmethod
    def create_feature_importance_plot(importance_df, target_name, save_path=None, top_n=15):
        """创建特征重要性图"""
        # 设置绘图样式
        VisualizationUtils.set_plot_style()
        
        # 获取前 N 个特征
        top_features = importance_df.head(top_n)
        
        # 创建特征重要性图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
        plt.title(f'预测 {target_name} 的顶级特征')
        plt.xlabel('重要性')
        plt.tight_layout()
        
        # 保存图像（如果提供了路径）
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"特征重要性图已保存至 {save_path}")
            
        return plt.gcf()
        
    @staticmethod
    def create_correlation_heatmap(df, features, save_path=None):
        """创建相关性热图"""
        # 设置绘图样式
        VisualizationUtils.set_plot_style()
        
        # 确保数据框有指定特征
        valid_features = [f for f in features if f in df.columns]
        
        if not valid_features:
            logging.error("数据框中没有有效特征")
            return None
            
        # 计算相关性矩阵
        corr = df[valid_features].corr()
        
        # 创建热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('特征相关性矩阵')
        plt.tight_layout()
        
        # 保存图像（如果提供了路径）
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"相关性热图已保存至 {save_path}")
            
        return plt.gcf()
