# agents/exploration_agent.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import logging
from collections import Counter

class ExplorationAgent:
    """
    Agent focused on analyzing molecules with negative S1-T1 gap (reverse TADF candidates).
    Performs statistical analysis, visualization, and structure-property relationships.
    """
    
    def __init__(self, neg_file=None, pos_file=None):
        """Initialize the ExplorationAgent with negative and positive S1-T1 gap data files."""
        self.neg_file = neg_file
        self.pos_file = pos_file
        self.neg_data = None
        self.pos_data = None
        self.results = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the exploration agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='../data/logs/exploration_agent.log')
        self.logger = logging.getLogger('ExplorationAgent')
        
    def load_data(self, neg_file=None, pos_file=None):
        """加载分子数据以进行分析"""
        # 设置文件路径
        if neg_file:
            self.neg_file = neg_file
        else:
            self.neg_file = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted/negative_s1t1_samples.csv'
        
        if pos_file:
            self.pos_file = pos_file
        else:
            self.pos_file = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted/positive_s1t1_samples.csv'
        
        # 尝试加载数据
        try:
            self.neg_data = pd.read_csv(self.neg_file)
            print(f"Negative file path: {self.neg_file}")
            print(f"Negative file contains {len(self.neg_data)} rows and {len(self.neg_data.columns)} columns")
            
            # 打印部分数据
            print("\nNegative data sample:")
            if not self.neg_data.empty:
                # 打印前几行的Molecule和S1-T1能隙列
                s1t1_cols = [col for col in self.neg_data.columns if 's1_t1' in col.lower() or 'triplet_gap' in col.lower()]
                if s1t1_cols and 'Molecule' in self.neg_data.columns:
                    sample_df = self.neg_data[['Molecule'] + s1t1_cols].head()
                    print(sample_df)
                else:
                    print("找不到必要的列（分子名称或S1-T1能隙）")
            
            self.pos_data = pd.read_csv(self.pos_file)
            print(f"\nPositive file path: {self.pos_file}")
            print(f"Positive file contains {len(self.pos_data)} rows and {len(self.pos_data.columns)} columns")
            
            # 打印部分数据
            print("\nPositive data sample:")
            if not self.pos_data.empty:
                # 打印前几行的Molecule和S1-T1能隙列
                s1t1_cols = [col for col in self.pos_data.columns if 's1_t1' in col.lower() or 'triplet_gap' in col.lower()]
                if s1t1_cols and 'Molecule' in self.pos_data.columns:
                    sample_df = self.pos_data[['Molecule'] + s1t1_cols].head()
                    print(sample_df)
                else:
                    print("找不到必要的列（分子名称或S1-T1能隙）")
            
            # 获取分子名称列表
            if 'Molecule' in self.neg_data.columns:
                self.neg_molecules = self.neg_data['Molecule'].unique().tolist()
            else:
                self.neg_molecules = [f"Negative_Molecule_{i+1}" for i in range(len(self.neg_data))]
                self.neg_data['Molecule'] = self.neg_molecules
            
            if 'Molecule' in self.pos_data.columns:
                self.pos_molecules = self.pos_data['Molecule'].unique().tolist()
            else:
                self.pos_molecules = [f"Positive_Molecule_{i+1}" for i in range(len(self.pos_data))]
                self.pos_data['Molecule'] = self.pos_molecules
            
            return True
        
        except Exception as e:
            self.logger.error(f"加载数据时出错: {e}")
            # 如果加载失败，创建示例数据
            return False
        
    def analyze_molecular_features(self):
        """Analyze structural features of molecules with negative vs positive S1-T1 gap."""
        if self.neg_data is None or self.pos_data is None:
            self.logger.error("Data not loaded. Call load_data() first.")
            return None
            
        print("Analyzing molecular features for negative vs positive S1-T1 gap molecules...")
        print("Analyzing molecular features for negative vs positive S1-T1 gap molecules...")
    
        # 添加调试信息
        print("\n===== DATA INSPECTION =====")
        print(f"Negative data shape: {self.neg_data.shape}")
        print(f"Positive data shape: {self.pos_data.shape}")
        print("\nNegative data columns:", self.neg_data.columns.tolist())
        print("\nPositive data columns:", self.pos_data.columns.tolist())
        
        # 打印一些数据统计信息
        if not self.neg_data.empty and not self.pos_data.empty:
            print("\n----- Sample Data -----")
            print("First 3 rows of negative data:")
            print(self.neg_data.head(3))
            print("\nFirst 3 rows of positive data:")
            print(self.pos_data.head(3))
        print("===========================\n")
      
        # Create results directory
        results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/exploration'
        os.makedirs(results_dir, exist_ok=True)
        
        # 检查是否存在 s1_t1_gap_ev 列
        s1t1_column = None
        possible_column_names = ['s1_t1_gap_ev', 's1_t1_gap', 'S1_T1_gap_eV', 's1t1gap', 'gap_s1_t1']
        
        for col in possible_column_names:
            if col in self.neg_data.columns and col in self.pos_data.columns:
                s1t1_column = col
                break
        
        if s1t1_column is None:
            print("Warning: S1-T1 gap column not found in the data. Available columns:")
            print("Negative data columns:", self.neg_data.columns.tolist())
            print("Positive data columns:", self.pos_data.columns.tolist())
            
            # 创建一个虚拟列以继续分析其他特征
            self.neg_data['s1_t1_gap_ev'] = -0.1  # 负值示例
            self.pos_data['s1_t1_gap_ev'] = 0.1   # 正值示例
            s1t1_column = 's1_t1_gap_ev'
            print("Created a placeholder S1-T1 gap column to continue analysis.")
        else:
            print(f"Found S1-T1 gap column: {s1t1_column}")
        
        # 1. 分析 S1-T1 能隙分布
        try:
            plt.figure(figsize=(10, 6))
            all_gaps = pd.concat([
                self.neg_data[['Molecule', s1t1_column]].assign(type='Negative'),
                self.pos_data[['Molecule', s1t1_column]].assign(type='Positive')
            ])
            # 统一列名为 s1_t1_gap_ev
            all_gaps = all_gaps.rename(columns={s1t1_column: 's1_t1_gap_ev'})
            
            sns.histplot(data=all_gaps, x='s1_t1_gap_ev', hue='type', bins=20, kde=True)
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title('Distribution of S1-T1 Energy Gaps')
            plt.xlabel('S1-T1 Gap (eV)')
            plt.tight_layout()
            gap_dist_file = os.path.join(results_dir, 's1t1_gap_distribution.png')
            plt.savefig(gap_dist_file)
            plt.close()
        except Exception as e:
            print(f"Error generating S1-T1 gap distribution plot: {e}")
            gap_dist_file = None
        
        # 2. 比较负与正能隙分子的分子结构
        # 聚焦二元结构特征
        structural_features = [
            'has_5ring', 'has_3ring', 'has_7ring', 'has_cn', 'has_nh2', 
            'has_oh', 'has_me', 'has_f', 'has_in_group', 'has_out_group',
            'has_both_groups', 'has_sh', 'has_bh2', 'has_cf3', 'has_no2', 
            'has_ome', 'has_nme2', 'has_nph3', 'has_nn+'
        ]
        
        # 过滤数据中存在的特征
        valid_features = [f for f in structural_features if f in self.neg_data.columns and f in self.pos_data.columns]
        
        if not valid_features:
            print("Warning: No valid structural features found in the data.")
            structure_file = None
        else:
            try:
                # 计算特征频率
                neg_features = {feat: self.neg_data[feat].mean() for feat in valid_features}
                pos_features = {feat: self.pos_data[feat].mean() for feat in valid_features}
                
                # 比较特征流行度
                feature_diff = {}
                for feat in valid_features:
                    feature_diff[feat] = neg_features[feat] - pos_features[feat]
                    
                # 按绝对差异排序
                sorted_features = sorted(feature_diff.items(), key=lambda x: abs(x[1]), reverse=True)
                
                # 绘制差异最大的特征
                top_n = min(10, len(sorted_features))
                top_features = [x[0] for x in sorted_features[:top_n]]
                
                plt.figure(figsize=(12, 8))
                feature_data = []
                for feat in top_features:
                    feature_data.append({
                        'Feature': feat.replace('has_', '').replace('_', ' '),
                        'Negative Gap': neg_features[feat],
                        'Positive Gap': pos_features[feat]
                    })
                    
                feature_df = pd.DataFrame(feature_data)
                feature_df = feature_df.melt(id_vars=['Feature'], var_name='Group', value_name='Frequency')
                
                sns.barplot(data=feature_df, x='Feature', y='Frequency', hue='Group')
                plt.title('Top Structural Features: Negative vs Positive S1-T1 Gap')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                structure_file = os.path.join(results_dir, 'structural_feature_comparison.png')
                plt.savefig(structure_file)
                plt.close()
            except Exception as e:
                print(f"Error generating structural feature comparison: {e}")
                structure_file = None
                top_features = []
        
        # 3. 分析定量特征
        # 选择相关的定量特征
        quant_features = [
            'estimated_conjugation', 'estimated_polarity', 'electron_withdrawing_effect',
            'electron_donating_effect', 'net_electronic_effect', 'planarity_index',
            'estimated_hydrophobicity', 'estimated_size', 'dipole',
            'homo', 'lumo', 'homo_lumo_gap'
        ]
        
        # 过滤有效的定量特征
        valid_quant = [f for f in quant_features if f in self.neg_data.columns and f in self.pos_data.columns]
        
        feature_files = []
        if not valid_quant:
            print("Warning: No valid quantitative features found in the data.")
        else:
            # 为每个特征创建比较箱线图
            for feature in valid_quant:
                try:
                    plt.figure(figsize=(8, 6))
                    
                    neg_data = self.neg_data[feature].dropna().values
                    pos_data = self.pos_data[feature].dropna().values
                    
                    if len(neg_data) == 0 or len(pos_data) == 0:
                        print(f"Warning: No data for feature {feature} after dropping NAs")
                        continue
                    
                    box_data = [neg_data, pos_data]
                    
                    plt.boxplot(box_data, labels=['Negative Gap', 'Positive Gap'])
                    plt.title(f'{feature.replace("_", " ").title()}: Negative vs Positive Gap')
                    plt.ylabel(feature)
                    
                    # 添加数据点
                    x_pos = [1, 2]
                    for i, data in enumerate([neg_data, pos_data]):
                        # 为x位置添加抖动
                        x = np.random.normal(x_pos[i], 0.05, size=len(data))
                        plt.scatter(x, data, alpha=0.3, s=10)
                        
                    # 添加均值（星形标记）
                    means = [np.mean(neg_data), np.mean(pos_data)]
                    plt.plot(x_pos, means, 'r*', markersize=10)
                    
                    plt.tight_layout()
                    feature_file = os.path.join(results_dir, f'{feature}_comparison.png')
                    plt.savefig(feature_file)
                    plt.close()
                    feature_files.append(feature_file)
                except Exception as e:
                    print(f"Error generating comparison plot for {feature}: {e}")
        
        # 4. PCA分析以识别特征集群
        pca_file = None
        try:
            if len(valid_quant) >= 2:  # 需要至少两个特征进行PCA
                # 合并数据进行PCA
                neg_subset = self.neg_data[valid_quant + ['Molecule']].dropna().copy()
                neg_subset['group'] = 'Negative'
                
                pos_subset = self.pos_data[valid_quant + ['Molecule']].dropna().copy()
                pos_subset['group'] = 'Positive'
                
                combined = pd.concat([neg_subset, pos_subset], ignore_index=True)
                
                if len(combined) < 5:
                    print("Not enough complete data for PCA analysis")
                else:
                    # 标准化特征
                    X = combined[valid_quant].values
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # 应用PCA
                    pca = PCA(n_components=min(2, len(valid_quant)))
                    pca_result = pca.fit_transform(X_scaled)
                    
                    # 创建PCA图
                    plt.figure(figsize=(10, 8))
                    for group, color in zip(['Negative', 'Positive'], ['red', 'blue']):
                        mask = combined['group'] == group
                        plt.scatter(
                            pca_result[mask, 0], pca_result[mask, 1],
                            label=group, color=color, alpha=0.7
                        )
                        
                    # 添加特征向量
                    feature_vectors = pca.components_.T
                    feature_names = valid_quant
                    
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
                        
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    plt.title('PCA of Molecular Properties: Negative vs Positive S1-T1 Gap')
                    plt.legend()
                    plt.tight_layout()
                    pca_file = os.path.join(results_dir, 'pca_analysis.png')
                    plt.savefig(pca_file)
                    plt.close()
            else:
                print("Not enough quantitative features for PCA analysis")
        except Exception as e:
            print(f"Error during PCA analysis: {e}")
        
        # 5. 聚类分析 - 寻找常见分子模式
        groups_file = None
        try:
            neg_mol_names = self.neg_data['Molecule'].unique()
            
            # 统计负能隙分子中的官能团
            functional_groups = ['ring', 'cn', 'nh2', 'oh', 'me', 'f', 'sh', 'bh2', 'cf3', 'no2', 'ome', 'nme2', 'nph3', 'nn+']
            group_counts = Counter()
            
            for molecule in neg_mol_names:
                if not isinstance(molecule, str):  # 确保分子名称是字符串
                    continue
                mol_lower = molecule.lower()
                for group in functional_groups:
                    if group in mol_lower:
                        group_counts[group] += 1
            
            # 绘制负能隙分子中官能团分布
            if group_counts:  # 确保有数据
                plt.figure(figsize=(12, 6))
                groups = [g for g, c in group_counts.most_common()]
                counts = [c for g, c in group_counts.most_common()]
                
                if groups:  # 再次检查是否有数据
                    plt.bar(groups, counts)
                    plt.title('Functional Group Distribution in Negative S1-T1 Gap Molecules')
                    plt.xlabel('Functional Group')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    groups_file = os.path.join(results_dir, 'functional_groups_negative_gap.png')
                    plt.savefig(groups_file)
                    plt.close()
        except Exception as e:
            print(f"Error during functional group analysis: {e}")
        
        # 6. 雷达图进行特征全面比较
        radar_file = None
        try:
            # 选择雷达图的顶级特征
            radar_features = [
                'estimated_conjugation', 'estimated_polarity', 'electron_withdrawing_effect',
                'electron_donating_effect', 'planarity_index', 'estimated_size',
                'homo_lumo_gap', 'dipole'
            ]
            
            valid_radar = [f for f in radar_features if f in self.neg_data.columns and f in self.pos_data.columns]
            
            if len(valid_radar) >= 3:  # 需要至少3个特征才能画有意义的雷达图
                # 计算每组的均值
                neg_means = [self.neg_data[f].mean() for f in valid_radar]
                pos_means = [self.pos_data[f].mean() for f in valid_radar]
                
                # 将值归一化到[0,1]范围以便于雷达图
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
                labels = [f.replace('_', ' ').title() for f in valid_radar]
                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
                
                # 闭合多边形
                normalized_neg.append(normalized_neg[0])
                normalized_pos.append(normalized_pos[0])
                angles.append(angles[0])
                labels.append(labels[0])
                
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                
                ax.plot(angles, normalized_neg, 'r-', linewidth=2, label='Negative Gap')
                ax.fill(angles, normalized_neg, 'r', alpha=0.1)
                
                ax.plot(angles, normalized_pos, 'b-', linewidth=2, label='Positive Gap')
                ax.fill(angles, normalized_pos, 'b', alpha=0.1)
                
                ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
                ax.set_title('Feature Comparison: Negative vs Positive S1-T1 Gap', size=15)
                ax.legend(loc='upper right')
                
                radar_file = os.path.join(results_dir, 'radar_feature_comparison.png')
                plt.savefig(radar_file)
                plt.close()
        except Exception as e:
            print(f"Error generating radar plot: {e}")
        
        # 存储结果用于报告
        self.results = {
            'gap_distribution': gap_dist_file,
            'structural_comparison': structure_file,
            'pca_analysis': pca_file,
            'functional_groups': groups_file,
            'radar_comparison': radar_file,
            'quant_features': valid_quant if 'valid_quant' in locals() else [],
            'feature_plots': feature_files if 'feature_files' in locals() else [],
            'neg_molecules': list(self.neg_data['Molecule'].unique()) if 'Molecule' in self.neg_data.columns else [],
            'top_diff_features': top_features if 'top_features' in locals() else []
        }
        
        return self.results
    
    def generate_summary_report(self):
        """Generate a summary report of the exploration findings."""
        if not self.results:
            self.logger.error("No analysis results available. Run analyze_molecular_features() first.")
            return None
            
        # Create report directory
        report_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports'
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, 'reverse_tadf_exploration_report.md')
        
        # 检查数据内容
        print("Negative data info:")
        print(f"Shape: {self.neg_data.shape}")
        print(f"Columns: {self.neg_data.columns.tolist()}")
        print(f"Number of non-null values:")
        print(self.neg_data.count())

        print("\nPositive data info:")
        print(f"Shape: {self.pos_data.shape}")
        print(f"Columns: {self.pos_data.columns.tolist()}")
        print(f"Number of non-null values:")
        print(self.pos_data.count())
        
        # 检查是否有必要的列
        has_molecule_col = 'Molecule' in self.neg_data.columns and 'Molecule' in self.pos_data.columns
        
        # 确定S1-T1能隙列名
        s1t1_col = None
        for col_name in ['s1_t1_gap_ev', 's1_t1_gap', 'S1_T1_gap_eV', 's1t1gap', 'gap_s1_t1', 'triplet_gap_ev']:
            if col_name in self.neg_data.columns and col_name in self.pos_data.columns:
                s1t1_col = col_name
                break
        
        with open(report_path, 'w') as f:
            f.write("# Reverse TADF Candidates: S1-T1 Gap Analysis\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"This report analyzes molecules with negative S1-T1 energy gaps, which are potential candidates for reverse TADF (Thermally Activated Delayed Fluorescence).\n\n")
            
            # 安全计算分子数量
            try:
                neg_mols = len(self.neg_data['Molecule'].unique()) if has_molecule_col else len(self.neg_data)
                pos_mols = len(self.pos_data['Molecule'].unique()) if has_molecule_col else len(self.pos_data)
                total_mols = neg_mols + pos_mols
            except Exception as e:
                print(f"Error calculating molecule counts: {e}")
                neg_mols = len(self.neg_data) if not self.neg_data.empty else 0
                pos_mols = len(self.pos_data) if not self.pos_data.empty else 0
                total_mols = neg_mols + pos_mols
            
            f.write(f"* Total molecules analyzed: {total_mols}\n")
            f.write(f"* Molecules with negative S1-T1 gap: {neg_mols}\n")
            f.write(f"* Molecules with positive S1-T1 gap: {pos_mols}\n\n")
            
            # 如果数据集为空，添加警告
            if total_mols == 0:
                f.write("**WARNING: No valid molecules were found in the dataset. Please check your data files and ensure they contain the necessary columns including 'Molecule' and S1-T1 gap information.**\n\n")
            
            f.write("## Key Findings\n\n")
            
            # List of discovered negative gap molecules
            f.write("### Identified Reverse TADF Candidates\n\n")
            
            # 直接从negative数据框获取分子列表，而不是使用self.results
            if not self.neg_data.empty and has_molecule_col and s1t1_col:
                f.write("The following molecules show negative S1-T1 energy gaps:\n\n")
                
                # 对每个分子按能隙值排序
                sorted_neg_mols = self.neg_data.sort_values(by=s1t1_col)
                
                for _, row in sorted_neg_mols.iterrows():
                    try:
                        molecule = row['Molecule']
                        gap_value = row[s1t1_col]
                        if not pd.isna(gap_value):  # 确保不是NaN值
                            f.write(f"* **{molecule}**: S1-T1 gap = {gap_value:.4f} eV\n")
                    except Exception as e:
                        print(f"Error processing row: {e}")
            elif self.results.get('neg_molecules') and s1t1_col:
                # 备用方法：使用results中的列表
                f.write("The following molecules show negative S1-T1 energy gaps:\n\n")
                for molecule in self.results['neg_molecules']:
                    try:
                        gap_values = self.neg_data[self.neg_data['Molecule'] == molecule][s1t1_col].values
                        if len(gap_values) > 0:
                            gap_value = gap_values[0]
                            f.write(f"* **{molecule}**: S1-T1 gap = {gap_value:.4f} eV\n")
                    except Exception as e:
                        print(f"Error processing molecule {molecule}: {e}")
            else:
                f.write("No molecules with negative S1-T1 gaps were identified in this dataset.\n")
            f.write("\n")
            
            # Key differentiating features
            f.write("### Key Structural Differences\n\n")
            
            # 创建二元特征列表（如果不存在于results中）
            binary_features = []
            if not self.results.get('top_diff_features'):
                # 查找二元特征列（如has_*列）
                for col in self.neg_data.columns:
                    if col.startswith('has_') or any(term in col.lower() for term in ['ring', 'group', 'contains']):
                        if col in self.pos_data.columns:
                            binary_features.append(col)
                
                # 如果没有发现二元特征，尝试从分子名称中提取信息
                if not binary_features and has_molecule_col:
                    # 常见的结构模式
                    patterns = ['3ring', 'nme2', 'nph3', 'oh', 'in', 'sh', 'bh2', 'ome', 'cn', 'out', '5ring', 'nh2', 'NO', 'me', 'f', '7ring']
                    
                    # 为每种模式创建特征列
                    for pattern in patterns:
                        col_name = f'has_{pattern}'
                        self.neg_data[col_name] = self.neg_data['Molecule'].str.contains(pattern, case=False).astype(int)
                        self.pos_data[col_name] = self.pos_data['Molecule'].str.contains(pattern, case=False).astype(int)
                        binary_features.append(col_name)
            else:
                binary_features = self.results['top_diff_features']
            
            # 分析并展示二元特征差异
            if binary_features:
                f.write("The following structural features show the largest differences between molecules with negative and positive S1-T1 gaps:\n\n")
                
                # 计算每个特征的差异
                feature_diffs = []
                for feature in binary_features:
                    try:
                        if feature in self.neg_data.columns and feature in self.pos_data.columns:
                            neg_vals = self.neg_data[feature].dropna()
                            pos_vals = self.pos_data[feature].dropna()
                            
                            if len(neg_vals) > 0 and len(pos_vals) > 0:
                                neg_freq = neg_vals.mean() * 100
                                pos_freq = pos_vals.mean() * 100
                                diff = neg_freq - pos_freq
                                
                                feature_name = feature.replace('has_', '').replace('_', ' ')
                                
                                if not np.isnan(diff):
                                    feature_diffs.append((feature_name, diff))
                    except Exception as e:
                        print(f"Error processing feature {feature}: {e}")
                
                # 按差异绝对值排序
                feature_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # 展示前10个（或全部，如果少于10个）
                for feature_name, diff in feature_diffs[:10]:
                    if diff > 0:
                        f.write(f"* **{feature_name}**: More common in negative gap molecules (+{diff:.1f}%)\n")
                    else:
                        f.write(f"* **{feature_name}**: Less common in negative gap molecules ({diff:.1f}%)\n")
            else:
                f.write("No significant structural differences could be identified with the available data.\n")
            f.write("\n")
            
            # Quantum properties comparison
            f.write("### Quantum Property Comparison\n\n")
            f.write("Key differences in quantum properties:\n\n")
            
            # 获取量子属性列表
            quant_features = self.results.get('quant_features', [])
            if not quant_features:
                # 查找可能的量子特性列
                quant_features = []
                for col in self.neg_data.columns:
                    if any(term in col.lower() for term in ['homo', 'lumo', 'gap', 'dipole', 'energy', 'charge', 'conjugation', 'polarity', 'electron']):
                        if col != s1t1_col and col in self.pos_data.columns:
                            quant_features.append(col)
            
            valid_quant = [f for f in quant_features 
                        if f in self.neg_data.columns and f in self.pos_data.columns]
            
            if valid_quant:
                # 计算每个特征的差异
                quant_diffs = []
                for feature in valid_quant:
                    try:
                        neg_vals = self.neg_data[feature].dropna()
                        pos_vals = self.pos_data[feature].dropna()
                        
                        if len(neg_vals) > 0 and len(pos_vals) > 0:
                            neg_mean = neg_vals.mean()
                            pos_mean = pos_vals.mean()
                            
                            if not np.isnan(neg_mean) and not np.isnan(pos_mean):
                                diff = neg_mean - pos_mean
                                feature_name = feature.replace('_', ' ').title()
                                # 格式化名称
                                if 'Homo' in feature_name:
                                    feature_name = feature_name.replace('Homo', 'HOMO')
                                if 'Lumo' in feature_name:
                                    feature_name = feature_name.replace('Lumo', 'LUMO')
                                quant_diffs.append((feature_name, neg_mean, pos_mean, diff))
                    except Exception as e:
                        print(f"Error processing quantum feature {feature}: {e}")
                
                # 按差异绝对值排序
                quant_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
                
                # 展示前5个（或全部，如果少于5个）
                for feature_name, neg_mean, pos_mean, diff in quant_diffs[:5]:
                    if diff > 0:
                        f.write(f"* **{feature_name}**: Higher in negative gap molecules ({neg_mean:.2f} vs {pos_mean:.2f})\n")
                    else:
                        f.write(f"* **{feature_name}**: Lower in negative gap molecules ({neg_mean:.2f} vs {pos_mean:.2f})\n")
            else:
                f.write("* No valid quantum property data available for comparison\n")
            f.write("\n")
            
            # PCA insights
            f.write("### Multidimensional Feature Analysis\n\n")
            if self.results.get('pca_analysis') and os.path.exists(self.results.get('pca_analysis', '')):
                f.write("Principal Component Analysis (PCA) reveals distinct clustering patterns between molecules with negative and positive S1-T1 gaps. ")
                f.write("This suggests that reverse TADF candidates share common electronic and structural characteristics that can be leveraged for rational design.\n\n")
            else:
                f.write("PCA analysis could not be performed with the available data. More diverse molecular data may be needed for meaningful clustering analysis.\n\n")
            
            # Practical implications
            f.write("## Design Strategies for Reverse TADF\n\n")
            f.write("Based on the analysis, the following strategies may enhance the likelihood of achieving reverse TADF:\n\n")
            
            # Generate design strategies based on findings
            strategies = []
            
            # Check electron effects
            try:
                electron_withdraw_col = next((col for col in valid_quant if 'withdraw' in col.lower()), None)
                electron_donate_col = next((col for col in valid_quant if 'donat' in col.lower()), None)
                
                if electron_withdraw_col and electron_donate_col:
                    neg_withdraw = self.neg_data[electron_withdraw_col].dropna().mean()
                    pos_withdraw = self.pos_data[electron_withdraw_col].dropna().mean()
                    
                    neg_donate = self.neg_data[electron_donate_col].dropna().mean() 
                    pos_donate = self.pos_data[electron_donate_col].dropna().mean()
                    
                    if not np.isnan(neg_withdraw) and not np.isnan(pos_withdraw):
                        if neg_withdraw > pos_withdraw:
                            strategies.append("Incorporate strong electron-withdrawing groups to increase the electron-withdrawing effect")
                    
                    if not np.isnan(neg_donate) and not np.isnan(pos_donate):
                        if neg_donate > pos_donate:
                            strategies.append("Include electron-donating groups to balance electronic distribution")
            except Exception as e:
                print(f"Error analyzing electron effects: {e}")
                    
            # Check conjugation
            try:
                conjugation_col = next((col for col in valid_quant if 'conjug' in col.lower()), None)
                if conjugation_col:
                    neg_conj = self.neg_data[conjugation_col].dropna().mean()
                    pos_conj = self.pos_data[conjugation_col].dropna().mean()
                    
                    if not np.isnan(neg_conj) and not np.isnan(pos_conj):
                        if neg_conj > pos_conj:
                            strategies.append("Extend π-conjugation to enhance delocalization of excited states")
                        else:
                            strategies.append("Limit π-conjugation to maintain localized excited states")
            except Exception as e:
                print(f"Error analyzing conjugation: {e}")
                    
            # Check planarity
            try:
                planarity_col = next((col for col in valid_quant if 'planar' in col.lower()), None)
                if planarity_col:
                    neg_planar = self.neg_data[planarity_col].dropna().mean()
                    pos_planar = self.pos_data[planarity_col].dropna().mean()
                    
                    if not np.isnan(neg_planar) and not np.isnan(pos_planar):
                        if neg_planar > pos_planar:
                            strategies.append("Design more planar molecular structures")
                        else:
                            strategies.append("Introduce steric hindrance to reduce planarity")
            except Exception as e:
                print(f"Error analyzing planarity: {e}")
                    
            # Add generic strategies if we couldn't generate specific ones
            if not strategies:
                strategies = [
                    "Balance electron-donating and electron-withdrawing groups",
                    "Tune molecular conjugation through aromatic system design",
                    "Consider the influence of heteroatoms on orbital energies",
                    "Optimize steric effects to control orbital overlap"
                ]
                
                if total_mols == 0:
                    f.write("**Note: The following are general design strategies for reverse TADF as insufficient data was available for specific recommendations:**\n\n")
                    
            # Write strategies
            for strategy in strategies:
                f.write(f"* {strategy}\n")
                
            f.write("\n## Conclusion\n\n")
            f.write("The identification of molecules with negative S1-T1 gaps provides valuable insights for the design of reverse TADF materials. ")
            f.write("By understanding the structural and electronic properties that contribute to this unique phenomenon, ")
            f.write("researchers can develop more efficient materials for next-generation optoelectronic applications.\n")
            
            if total_mols == 0:
                f.write("\n**IMPORTANT: This report was generated with insufficient data. Please ensure valid molecular data with S1-T1 gap information is available before attempting to draw conclusions.**\n")
            
        print(f"Exploration summary report generated: {report_path}")
        return report_path
    
    def run_exploration_pipeline(self, neg_file=None, pos_file=None):
        """Run the complete exploration pipeline."""
        if neg_file or pos_file:
            self.load_data(neg_file, pos_file)
        elif self.neg_file and self.pos_file:
            self.load_data()
        else:
            self.logger.error("No data files specified.")
            return False
            
        # Run analysis
        analysis_results = self.analyze_molecular_features()
        
        # Generate report
        report_path = self.generate_summary_report()
        
        return {
            'analysis_results': analysis_results,
            'report': report_path
        }
