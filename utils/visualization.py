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
        valid_features = []
        for feature in features:
            if feature in negative_df.columns and feature in positive_df.columns:
                valid_features.append(feature)
            else:
                logging.warning(f"数据框中缺少特征 {feature}，已跳过")
                
        if len(valid_features) < 3:
            logging.error(f"有效特征不足（只有{len(valid_features)}个），雷达图需要至少3个特征")
            return None
                
        # 计算两组的均值，同时处理NaN和无穷大值
        neg_means = []
        pos_means = []
        for f in valid_features:
            neg_val = negative_df[f].replace([np.inf, -np.inf], np.nan).mean()
            pos_val = positive_df[f].replace([np.inf, -np.inf], np.nan).mean()
            
            # 如果依然是NaN，使用0代替
            neg_means.append(0.0 if np.isnan(neg_val) else neg_val)
            pos_means.append(0.0 if np.isnan(pos_val) else pos_val)
            
        # 打印调试信息
        logging.info(f"负值能隙特征均值: {neg_means}")
        logging.info(f"正值能隙特征均值: {pos_means}")
        
        # 归一化值到 [0,1] 用于雷达图
        all_values = np.array(neg_means + pos_means)
        if len(all_values) == 0 or np.all(np.isnan(all_values)):
            logging.error("无法创建雷达图：所有值都是NaN")
            return None
            
        min_vals = np.nanmin(all_values)
        max_vals = np.nanmax(all_values)
        
        # 处理极端情况
        if np.isnan(min_vals) or np.isnan(max_vals) or np.isclose(max_vals, min_vals):
            logging.warning("所有值近似相等或存在NaN，使用默认归一化值")
            normalized_neg = [0.5 for _ in neg_means]
            normalized_pos = [0.5 for _ in pos_means]
        else:
            # 确保不会除以零
            range_vals = max_vals - min_vals
            if np.isclose(range_vals, 0):
                range_vals = 1.0
                
            normalized_neg = [(x - min_vals) / range_vals for x in neg_means]
            normalized_pos = [(x - min_vals) / range_vals for x in pos_means]
        
        # 创建雷达图
        labels = [f.replace('_', ' ').title() for f in valid_features]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        
        # 闭合多边形
        normalized_neg.append(normalized_neg[0])
        normalized_pos.append(normalized_pos[0])
        angles.append(angles[0])
        labels.append(labels[0])
        
        # 创建极坐标图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 绘制负值能隙多边形
        ax.plot(angles, normalized_neg, 'r-', linewidth=2, label='负值能隙')
        ax.fill(angles, normalized_neg, 'r', alpha=0.1)
        
        # 绘制正值能隙多边形
        ax.plot(angles, normalized_pos, 'b-', linewidth=2, label='正值能隙')
        ax.fill(angles, normalized_pos, 'b', alpha=0.1)
        
        # 设置刻度和标签
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        ax.set_title('特征比较: 负值 vs 正值 S1-T1 能隙', size=15)
        ax.legend(loc='upper right')
        
        # 保存图像（如果提供了路径）
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"雷达图已保存至 {save_path}")
            except Exception as e:
                logging.error(f"保存雷达图时出错: {e}")
        
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
    
    # 添加到visualization.py
    @staticmethod
    def create_structure_feature_radar(negative_df, positive_df, save_path=None):
        """创建结构特征的雷达图比较"""
        # 设置绘图样式
        VisualizationUtils.set_plot_style()
        
        # 选择要在雷达图中显示的结构特征
        features = [
            'max_conjugation_length', 'twist_ratio', 
            'max_h_bond_strength', 'planarity',
            'aromatic_rings_count'
        ]
        
        # 检查两个数据框是否都有这些特征
        valid_features = [f for f in features 
                        if f in negative_df.columns and f in positive_df.columns]
        
        if len(valid_features) < 3:  # 需要至少3个特征
            print("警告：没有足够的结构特征进行雷达图比较")
            return None
        
        # 计算平均值
        neg_means = [negative_df[f].mean() for f in valid_features]
        pos_means = [positive_df[f].mean() for f in valid_features]
        
        # 归一化至[0,1]范围
        all_values = neg_means + pos_means
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        normalized_neg = [(v - min_val) / range_val for v in neg_means]
        normalized_pos = [(v - min_val) / range_val for v in pos_means]
        
        # 友好名称映射
        feature_names = {
            'max_conjugation_length': 'π共轭长度',
            'twist_ratio': '扭曲程度', 
            'max_h_bond_strength': '氢键强度',
            'planarity': '平面性',
            'aromatic_rings_count': '芳香环数',
            'conjugation_path_count': '共轭路径数',
            'dihedral_angles_count': '关键二面角数',
            'hydrogen_bonds_count': '氢键数'
        }
        
        # 创建标签
        labels = [feature_names.get(f, f.replace('_', ' ').title()) for f in valid_features]
        
        # 准备雷达图
        angles = np.linspace(0, 2*np.pi, len(valid_features), endpoint=False).tolist()
        
        # 闭合图形
        normalized_neg.append(normalized_neg[0])
        normalized_pos.append(normalized_pos[0])
        angles.append(angles[0])
        labels.append(labels[0])
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
        
        ax.plot(angles, normalized_neg, 'r-', linewidth=2, label='负值S1-T1能隙')
        ax.fill(angles, normalized_neg, 'r', alpha=0.1)
        
        ax.plot(angles, normalized_pos, 'b-', linewidth=2, label='正值S1-T1能隙')
        ax.fill(angles, normalized_pos, 'b', alpha=0.1)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        ax.set_title('结构特征比较: 负值 vs 正值 S1-T1能隙')
        ax.legend(loc='upper right')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"结构特征雷达图已保存至: {save_path}")
        
        return fig

    @staticmethod
    def create_dihedral_vs_s1t1_plot(df, save_path=None):
        """创建二面角与S1-T1能隙关系图"""
        if 'max_dihedral_angle' not in df.columns or 's1_t1_gap_ev' not in df.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # 散点图
        plt.scatter(
            df['max_dihedral_angle'], 
            df['s1_t1_gap_ev'],
            alpha=0.7, c=df['s1_t1_gap_ev'], cmap='coolwarm'
        )
        
        # 添加趋势线
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
            # 如果无法计算趋势线则跳过
            pass
        
        # 添加零线
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 添加40°参考线
        plt.axvline(x=40, color='g', linestyle='--', alpha=0.5)
        
        plt.xlabel('最大二面角 (度)')
        plt.ylabel('S1-T1 能隙 (eV)')
        plt.title('二面角扭曲与S1-T1能隙关系')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"二面角关系图已保存至: {save_path}")
        
        return plt.gcf()

    @staticmethod
    def create_conjugation_vs_s1t1_plot(df, save_path=None):
        """创建共轭长度与S1-T1能隙关系图"""
        if 'max_conjugation_length' not in df.columns or 's1_t1_gap_ev' not in df.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # 根据最大共轭长度分组
        grouped = df.groupby('max_conjugation_length')['s1_t1_gap_ev'].agg(['mean', 'std', 'count'])
        grouped = grouped[grouped['count'] >= 2]  # 至少2个样本
        
        x = grouped.index
        y = grouped['mean']
        yerr = grouped['std']
        
        # 绘制条形图
        plt.bar(x, y, yerr=yerr, alpha=0.7, color='skyblue')
        
        # 添加零线
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('最大π共轭路径长度')
        plt.ylabel('平均S1-T1能隙 (eV)')
        plt.title('共轭长度与S1-T1能隙关系')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"共轭长度关系图已保存至: {save_path}")
        
        return plt.gcf()

    @staticmethod
    def create_hydrogen_bonds_effect_plot(df, save_path=None):
        """创建氢键对S1-T1能隙的影响图"""
        if 'hydrogen_bonds_count' not in df.columns or 's1_t1_gap_ev' not in df.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # 基于氢键数分类
        df['h_bond_category'] = df['hydrogen_bonds_count'].apply(
            lambda x: '无氢键' if x == 0 else ('1-2个氢键' if x <= 2 else '多个氢键')
        )
        
        # 分组并计算平均值
        grouped = df.groupby('h_bond_category')['s1_t1_gap_ev'].agg(['mean', 'std', 'count']).reset_index()
        
        # 确保顺序
        order = ['无氢键', '1-2个氢键', '多个氢键']
        grouped['order'] = grouped['h_bond_category'].map({cat: i for i, cat in enumerate(order)})
        grouped = grouped.sort_values('order')
        
        # 绘图
        plt.bar(
            grouped['h_bond_category'], 
            grouped['mean'],
            yerr=grouped['std'],
            alpha=0.7,
            color=['#ff9999', '#66b3ff', '#99ff99']
        )
        
        # 添加零线
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('氢键数量')
        plt.ylabel('平均S1-T1能隙 (eV)')
        plt.title('氢键与S1-T1能隙关系')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"氢键影响图已保存至: {save_path}")
        
        return plt.gcf()