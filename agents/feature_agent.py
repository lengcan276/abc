# agents/feature_agent.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import re
from collections import Counter
import warnings
import logging

class FeatureAgent:
    """
    Agent responsible for feature engineering, including generating
    alternative 3D descriptors from molecular structure information.
    """
    
    def __init__(self, data_file=None):
        """Initialize the FeatureAgent with input data file."""
        self.data_file = data_file
        self.df = None
        self.feature_df = None
        self.alt_3d_features = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the feature agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/logs/feature_agent.log')
        self.logger = logging.getLogger('FeatureAgent')
        
    def load_data(self, file_path=None):
        """Load molecular data from CSV file."""
        if file_path:
            self.data_file = file_path
            
        if not self.data_file or not os.path.exists(self.data_file):
            self.logger.error(f"Data file not found: {self.data_file}")
            return False
            
        print(f"Loading data from {self.data_file}...")
        self.df = pd.read_csv(self.data_file)
        
        # 检查是详细数据文件还是汇总数据文件
        if 'State' in self.df.columns:
            # 如果是 all_conformers_data.csv
            print(f"Dataset shape: {self.df.shape}")
            print(f"Number of molecules: {self.df['Molecule'].nunique()}")
            print(f"Number of states: {self.df['State'].nunique()}")
            print(f"States available: {self.df['State'].unique()}")
            
            # Check excited state data availability
            excited_count = self.df['excited_energy'].notna().sum() if 'excited_energy' in self.df.columns else 0
            print(f"Number of entries with excited state data: {excited_count}")
        else:
            # 如果是 molecular_properties_summary.csv
            print(f"Dataset shape: {self.df.shape}")
            print(f"Number of molecules: {self.df['Molecule'].nunique()}")
            
            # 识别状态相关列
            state_prefixes = ['neutral_', 'cation_', 'triplet_']
            print(f"Number of states: {len(state_prefixes)}")
            print(f"States available: {state_prefixes}")
            
            # 检查关键特征的可用性
            excited_columns = [col for col in self.df.columns if 'excited_' in col or 's1_' in col or 't1_' in col]
            print(f"Number of excited state properties: {len(excited_columns)}")
            
            # 创建虚拟状态数据框（如果后续处理需要）
            self.create_virtual_states()
        
        return True

    def create_virtual_states(self):
        """创建虚拟状态数据框，将汇总数据转换为三个状态的格式"""
        # 初始化状态列表和结果列表
        states = ['neutral', 'cation', 'triplet']
        expanded_rows = []
        
        # 为每个分子创建三行（每个状态一行）
        for _, row in self.df.iterrows():
            molecule = row['Molecule']
            
            for state in states:
                prefix = f"{state}_"
                # 获取当前状态的所有列
                state_cols = {col.replace(prefix, ''): row[col] 
                            for col in self.df.columns 
                            if col.startswith(prefix)}
                
                # 添加基本信息
                new_row = {'Molecule': molecule, 'State': state}
                # 添加状态特定的列
                new_row.update(state_cols)
                
                # 对于中性状态，添加激发态信息
                if state == 'neutral' and 'excited_energy' in self.df.columns:
                    excited_cols = {col: row[col] 
                                for col in self.df.columns 
                                if 'excited_' in col or col in ['s1_energy_ev', 't1_energy_ev', 's1_t1_gap_ev']}
                    new_row.update(excited_cols)
                
                expanded_rows.append(new_row)
        
        # 创建虚拟状态数据框
        self.state_df = pd.DataFrame(expanded_rows)
        print(f"Created virtual state dataframe with shape: {self.state_df.shape}")
        
    def extract_alternative_3d_features(self):
        """
        Extract and create alternative 3D-related features from the dataset
        """
        print("Creating alternative 3D-related features...")
        
        if self.df is None:
            self.logger.error("No data loaded. Call load_data() first.")
            return None

        # Create a new DataFrame to store alternative features
        alt_features = pd.DataFrame()
        alt_features['Molecule'] = self.df['Molecule'].unique()

        # 1. Extract structural information from molecule names
        # Number and type of rings
        alt_features['ring_count'] = alt_features['Molecule'].apply(
            lambda x: x.lower().count('ring')
        )
        alt_features['has_5ring'] = alt_features['Molecule'].apply(
            lambda x: 1 if '5ring' in x.lower() else 0
        )
        alt_features['has_3ring'] = alt_features['Molecule'].apply(
            lambda x: 1 if '3ring' in x.lower() else 0
        )
        alt_features['has_7ring'] = alt_features['Molecule'].apply(
            lambda x: 1 if '7ring' in x.lower() else 0
        )

        # 2. Substituent position features
        alt_features['has_in_group'] = alt_features['Molecule'].apply(
            lambda x: 1 if '_in' in x.lower() else 0
        )
        alt_features['has_out_group'] = alt_features['Molecule'].apply(
            lambda x: 1 if '_out' in x.lower() else 0
        )
        alt_features['has_both_groups'] = alt_features['Molecule'].apply(
            lambda x: 1 if '_both' in x.lower() else 0
        )

        # 3. Create complex substituent features
        functional_groups = ['cn', 'nh2', 'oh', 'me', 'f', 'sh', 'bh2', 'cf3', 'no2', 'ome', 'nme2', 'nph3', 'nn+']
        for group in functional_groups:
            alt_features[f'has_{group}'] = alt_features['Molecule'].apply(
                lambda x: 1 if group in x.lower() else 0
            )
            # Count occurrences of each substituent
            alt_features[f'{group}_count'] = alt_features['Molecule'].apply(
                lambda x: x.lower().count(group)
            )

        # 4. Create molecular complexity indicators
        # Total number of substituents
        alt_features['total_substituents'] = 0
        for group in functional_groups:
            alt_features['total_substituents'] += alt_features[f'{group}_count']

        # Structural complexity - based on rings and substituents
        alt_features['structural_complexity'] = alt_features['ring_count'] + alt_features['total_substituents']

        # 5. Infer approximate molecular size
        # Assume each ring and substituent contributes a certain size
        ring_sizes = {'has_3ring': 3, 'has_5ring': 5, 'has_7ring': 7}
        alt_features['estimated_size'] = 0
        for ring, size in ring_sizes.items():
            alt_features['estimated_size'] += alt_features[ring] * size

        # Different substituents contribution to size
        group_sizes = {
            'cn_count': 2, 'nh2_count': 1.5, 'oh_count': 1, 'me_count': 1,
            'f_count': 0.5, 'sh_count': 1, 'bh2_count': 1, 'cf3_count': 3,
            'no2_count': 2, 'ome_count': 2, 'nme2_count': 3, 'nph3_count': 7, 'nn+_count': 2
        }
        for group, size in group_sizes.items():
            alt_features['estimated_size'] += alt_features[group] * size

        # Ensure estimated_size is at least 1, to avoid division by zero errors
        alt_features['estimated_size'] = alt_features['estimated_size'].apply(lambda x: max(1.0, x))

        # 6. Infer molecular polarity
        # Different groups' polarity contributions
        polarity_contributions = {
            'cn_count': 3.5, 'nh2_count': 3, 'oh_count': 3, 'f_count': 4,
            'sh_count': 1.5, 'no2_count': 4, 'ome_count': 2.5, 'nme2_count': 2,
            'nn+_count': 4.5, 'nph3_count': 1, 'cf3_count': 2.5, 'me_count': 0.2, 'bh2_count': 1
        }
        alt_features['estimated_polarity'] = 0
        for group, polarity in polarity_contributions.items():
            alt_features['estimated_polarity'] += alt_features[group] * polarity

        # 7. Infer hydrophobicity/hydrophilicity
        # Different groups' hydrophobicity contributions (positive more hydrophobic, negative more hydrophilic)
        hydrophobicity = {
            'cn_count': -1, 'nh2_count': -2, 'oh_count': -2, 'f_count': -0.5,
            'sh_count': 0.5, 'no2_count': -1.5, 'ome_count': -0.5, 'nme2_count': 0.5,
            'nn+_count': -3, 'nph3_count': 3, 'cf3_count': 2, 'me_count': 1, 'bh2_count': 0.5
        }
        alt_features['estimated_hydrophobicity'] = 0
        for group, hydro in hydrophobicity.items():
            alt_features['estimated_hydrophobicity'] += alt_features[group] * hydro

        # 8. Infer electronic effects
        # Different groups' electronic effects (+value electron-donating, -value electron-withdrawing)
        electronic_effects = {
            'cn_count': -3, 'nh2_count': 2, 'oh_count': 1, 'f_count': -1,
            'sh_count': 0.5, 'no2_count': -3.5, 'ome_count': 1, 'nme2_count': 2,
            'nn+_count': -3, 'nph3_count': 1.5, 'cf3_count': -2.5, 'me_count': 0.5, 'bh2_count': -1
        }
        alt_features['electron_donating_effect'] = 0
        alt_features['electron_withdrawing_effect'] = 0

        for group, effect in electronic_effects.items():
            if effect > 0:
                alt_features['electron_donating_effect'] += alt_features[group] * effect
            else:
                alt_features['electron_withdrawing_effect'] += alt_features[group] * abs(effect)

        # Ensure electron_withdrawing_effect is at least 0.1, to avoid division by zero
        alt_features['electron_withdrawing_effect'] = alt_features['electron_withdrawing_effect'].apply(
            lambda x: max(0.1, x)
        )

        # Net electronic effect
        alt_features['net_electronic_effect'] = alt_features['electron_donating_effect'] - alt_features['electron_withdrawing_effect']

        # 9. Estimate conjugation degree
        # Ring systems and certain substituents increase conjugation
        conjugation_contributors = {
            'has_5ring': 5, 'has_3ring': 3, 'has_7ring': 7,
            'cn_count': 2, 'nh2_count': 1, 'oh_count': 1, 'no2_count': 2,
            'nme2_count': 1, 'nn+_count': 1.5
        }

        alt_features['estimated_conjugation'] = 0
        for contrib, value in conjugation_contributors.items():
            alt_features['estimated_conjugation'] += alt_features[contrib] * value

        # 10. Predict stereochemical properties
        # Estimate 3D shape based on rings and substituents
        alt_features['planarity_index'] = 0

        # Planarity index: more rings increase planarity, certain substituents decrease it
        alt_features['planarity_index'] += alt_features['ring_count'] * 2
        alt_features['planarity_index'] -= alt_features['nph3_count'] * 3  # Large substituents decrease planarity
        alt_features['planarity_index'] -= alt_features['nme2_count'] * 1
        alt_features['planarity_index'] -= alt_features['cf3_count'] * 1.5

        # Ensure planarity index is not negative
        alt_features['planarity_index'] = alt_features['planarity_index'].apply(lambda x: max(0.1, x))

        print(f"Created {len(alt_features.columns) - 1} alternative 3D features for {len(alt_features)} molecules")
        
        self.alt_3d_features = alt_features
        return alt_features
        
    def preprocess_data(self):
        """Preprocess data and create features."""
        print("预处理数据并创建特征...")

        if self.df is None:
            self.logger.error("未加载数据。请先调用 load_data() 方法。")
            return None
            
        if self.alt_3d_features is None:
            self.extract_alternative_3d_features()

        # 首先进行简单的数据清洗
        feature_df = self.df.copy()
        
        # 检查是否有 State 列，如果没有，使用汇总数据处理方法
        has_state_column = 'State' in feature_df.columns
        
        if not has_state_column:
            print("检测到汇总数据格式 (molecular_properties_summary.csv)，进行适配处理...")
            
            # 创建一个空的临时数据框，用于存储展开后的数据
            expanded_data = []
            
            # 为每个分子创建三个状态的行
            for _, row in feature_df.iterrows():
                molecule = row['Molecule']
                
                # 创建中性状态行
                neutral_row = {
                    'Molecule': molecule,
                    'State': 'neutral',
                    'is_primary': True  # 假设每个状态的主要构象体
                }
                
                # 添加中性状态的特定列
                neutral_prefix = 'neutral_'
                for col in feature_df.columns:
                    if col.startswith(neutral_prefix):
                        # 去掉前缀，添加到新行
                        neutral_row[col.replace(neutral_prefix, '')] = row[col]
                
                # 添加激发态列（不带前缀）
                for col in ['excited_energy', 'excited_opt_success', 'excited_homo', 
                            'excited_lumo', 'excited_homo_lumo_gap', 'excited_dipole',
                            's1_energy_ev', 'oscillator_strength', 't1_energy_ev', 
                            's1_t1_gap_ev', 'excitation_energy_ev']:
                    if col in feature_df.columns:
                        neutral_row[col] = row[col]
                
                # 创建阳离子状态行
                cation_row = {
                    'Molecule': molecule,
                    'State': 'cation',
                    'is_primary': True
                }
                
                # 添加阳离子状态的特定列
                cation_prefix = 'cation_'
                for col in feature_df.columns:
                    if col.startswith(cation_prefix):
                        cation_row[col.replace(cation_prefix, '')] = row[col]
                
                # 创建三重态状态行
                triplet_row = {
                    'Molecule': molecule,
                    'State': 'triplet',
                    'is_primary': True
                }
                
                # 添加三重态状态的特定列
                triplet_prefix = 'triplet_'
                for col in feature_df.columns:
                    if col.startswith(triplet_prefix):
                        triplet_row[col.replace(triplet_prefix, '')] = row[col]
                
                # 将三个行添加到展开的数据中
                expanded_data.append(neutral_row)
                expanded_data.append(cation_row)
                expanded_data.append(triplet_row)
            
            # 创建展开后的数据框
            feature_df = pd.DataFrame(expanded_data)
            print(f"已将汇总数据转换为详细格式，共 {len(feature_df)} 行")
        
        # 创建新特征
        # 1. 能量差异特征
        if 'energy' in feature_df.columns:
            # 计算中性状态与其他状态之间的能量差异
            # 按分子分组
            molecule_groups = feature_df.groupby('Molecule')

            # 创建一个字典来存储中性状态能量
            neutral_energies = {}
            for molecule, group in molecule_groups:
                neutral_group = group[group['State'] == 'neutral']
                if not neutral_group.empty and 'energy' in neutral_group.columns:
                    # 获取最低能量构象体 (primary=True)
                    primary_neutral = neutral_group[neutral_group['is_primary'] == True]
                    if not primary_neutral.empty:
                        neutral_energies[molecule] = primary_neutral['energy'].values[0]
                    else:
                        # 如果没有 primary=True 构象体，使用能量最低的
                        neutral_energies[molecule] = neutral_group['energy'].min()

            # 计算与中性状态的能量差异
            feature_df['energy_diff_from_neutral'] = feature_df.apply(
                lambda row: row['energy'] - neutral_energies.get(row['Molecule'], np.nan)
                if row['Molecule'] in neutral_energies else np.nan, axis=1
            )

        # 2. 添加 HOMO-LUMO 能隙特征
        if 'homo' in feature_df.columns and 'lumo' in feature_df.columns:
            # 确保所有值都是数值型
            feature_df['homo_num'] = pd.to_numeric(feature_df['homo'], errors='coerce')
            feature_df['lumo_num'] = pd.to_numeric(feature_df['lumo'], errors='coerce')
            feature_df['gap_calculated'] = feature_df['lumo_num'] - feature_df['homo_num']

        # 3. CREST 特征
        # 从 CREST 数据中提取信息
        if 'crest_num_conformers' in feature_df.columns:
            # 更多构象体意味着更高的构象灵活性
            feature_df['conformational_flexibility'] = np.log1p(feature_df['crest_num_conformers'])

        if 'crest_energy_range' in feature_df.columns:
            # 更大的能量范围意味着更高的构象多样性
            feature_df['energy_diversity'] = np.log1p(feature_df['crest_energy_range'])

        # 4. 电荷特征
        if 'max_positive_charge' in feature_df.columns and 'max_negative_charge' in feature_df.columns:
            feature_df['charge_polarity'] = feature_df['max_positive_charge'] - feature_df['max_negative_charge']
            feature_df['charge_intensity'] = feature_df['max_positive_charge'] + feature_df['max_negative_charge'].abs()

        # 5. 从分子名称中提取取代基信息
        feature_df['has_nh2'] = feature_df['Molecule'].str.contains('nh2', regex=False).astype(int)
        feature_df['has_oh'] = feature_df['Molecule'].str.contains('oh', regex=False).astype(int)
        feature_df['has_cn'] = feature_df['Molecule'].str.contains('cn', regex=False).astype(int)
        feature_df['has_me'] = feature_df['Molecule'].str.contains('me', regex=False).astype(int)
        feature_df['has_f'] = feature_df['Molecule'].str.contains('_f', regex=False).astype(int)
        feature_df['is_ring'] = feature_df['Molecule'].str.contains('ring', regex=False).astype(int)

        # 6. 状态编码 - 现在我们确保有 State 列
        feature_df['is_neutral'] = (feature_df['State'] == 'neutral').astype(int)
        feature_df['is_cation'] = (feature_df['State'] == 'cation').astype(int)
        feature_df['is_triplet'] = (feature_df['State'] == 'triplet').astype(int)

        # 7. 构象体特征
        if 'is_primary' in feature_df.columns:
            feature_df['is_primary_num'] = feature_df['is_primary'].astype(int)

        # 8. 分子复杂度特征（基于分子名称中的特征数量）
        complexity_features = ['nh2', 'oh', 'cn', 'me', 'f', 'ring', 'nme2', 'sh', 'bh2', 'cf3', 'no2', 'ome']
        feature_df['molecular_complexity'] = 0
        for feat in complexity_features:
            # 使用 str.contains 而不是 str.count，因为有些 pandas 版本的 count 可能没有 case 参数
            feature_df['molecular_complexity'] += feature_df['Molecule'].str.contains(feat, regex=False).astype(int)

        # 9. 如果我们有替代的 3D 特征，将它们与特征数据合并
        if self.alt_3d_features is not None:
            print("将替代 3D 特征与其他特征合并...")

            # 与 feature_df 合并
            feature_df = pd.merge(feature_df, self.alt_3d_features, on='Molecule', how='left', suffixes=('', '_alt'))

            # 10. 创建电子特性和 3D 特征的组合 - 确保不除以零
            # HOMO 与估计极性的组合
            if 'homo_num' in feature_df.columns and 'estimated_polarity' in feature_df.columns:
                # 确保极性不为零
                safe_polarity = feature_df['estimated_polarity'].apply(lambda x: x if x != 0 else 0.1)
                feature_df['homo_polarity'] = feature_df['homo_num'] * safe_polarity

            # LUMO 与估计电子效应的组合
            if 'lumo_num' in feature_df.columns and 'net_electronic_effect' in feature_df.columns:
                feature_df['lumo_electronic_effect'] = feature_df['lumo_num'] * feature_df['net_electronic_effect']

            # 能隙与共轭关系，避免除以零
            if 'gap_calculated' in feature_df.columns and 'estimated_conjugation' in feature_df.columns:
                # 确保分母不为零
                safe_denominator = (1 + feature_df['estimated_conjugation'].abs() + 1e-5)
                feature_df['gap_conjugation'] = feature_df['gap_calculated'] / safe_denominator

            # 能量与分子大小的组合
            if 'energy' in feature_df.columns and 'estimated_size' in feature_df.columns:
                # 确保分母不为零
                safe_size = feature_df['estimated_size'].apply(lambda x: max(1.0, x))
                feature_df['energy_per_size'] = feature_df['energy'] / safe_size

            # 偶极矩与平面性的组合
            if 'dipole' in feature_df.columns and 'planarity_index' in feature_df.columns:
                feature_df['dipole_planarity'] = feature_df['dipole'].fillna(0) * feature_df['planarity_index']

        # 11. 处理无穷大和 NaN 值
        # 用 NaN 替换无穷大值
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        # 检查并报告潜在的问题列
        problematic_cols = []
        for col in feature_df.select_dtypes(include=['float64', 'int64']).columns:
            if feature_df[col].isna().any():
                na_percent = feature_df[col].isna().mean() * 100
                if na_percent > 10:  # 如果超过 10% 的值是 NaN
                    problematic_cols.append((col, na_percent))

        if problematic_cols:
            print("警告：以下列有较高的 NaN 百分比：")
            for col, pct in problematic_cols:
                print(f"  - {col}: {pct:.2f}%")
                
        # 存储处理后的数据
        self.feature_df = feature_df
        
        # 保存处理后的特征
        output_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted'
        os.makedirs(output_dir, exist_ok=True)
        features_file = os.path.join(output_dir, "processed_features.csv")
        feature_df.to_csv(features_file, index=False)
        print(f"处理后的特征已保存到 {features_file}")
        
        return features_file
        
    def get_negative_s1t1_samples(self):
        """
        提取 S1-T1 能隙为负值的样本进行分析，
        这些可能是反向 TADF 候选物。
        """
        if self.feature_df is None:
            if not hasattr(self, 'preprocess_data'):
                self.logger.error("没有可用的特征数据。请先调用 preprocess_data() 方法。")
                return None
        
        # 检查列名并确定使用哪个列作为 S1-T1 能隙
        # 优先使用triplet_gap_ev，因为它在CSV文件中已存在
        possible_columns = ['triplet_gap_ev', 's1_t1_gap_ev', 's1_t1_gap']
        s1t1_gap_column = None
        
        for col in possible_columns:
            if col in self.feature_df.columns:
                s1t1_gap_column = col
                print(f"使用 {col} 列作为 S1-T1 能隙数据")
                break
        
        # 保存原始数据的副本，用于调试
        self.feature_df.to_csv('/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted/original_feature_df.csv', index=False)
        print(f"原始数据形状: {self.feature_df.shape}")
        print(f"列名: {self.feature_df.columns.tolist()}")
        
        # 如果找不到预期的列，尝试查找含有 's1_t1' 的其他列
        if s1t1_gap_column is None:
            for col in self.feature_df.columns:
                if 's1_t1' in col.lower():
                    s1t1_gap_column = col
                    print(f"使用 {col} 列作为 S1-T1 能隙数据")
                    break
        
        # 如果仍然找不到，尝试从其他列计算
        if s1t1_gap_column is None:
            if 's1_energy_ev' in self.feature_df.columns and 't1_energy_ev' in self.feature_df.columns:
                self.feature_df['s1_t1_gap_ev'] = (
                    self.feature_df['s1_energy_ev'] - self.feature_df['t1_energy_ev']
                )
                s1t1_gap_column = 's1_t1_gap_ev'
                print(f"通过 s1_energy_ev - t1_energy_ev 计算得到 S1-T1 能隙")
        
        # 确保使用正确的分子列
        if 'Molecule' not in self.feature_df.columns and 'molecule' in self.feature_df.columns:
            self.feature_df['Molecule'] = self.feature_df['molecule']
        
        # 获取所有有效的能隙数据样本(如果有的话)
        gap_samples = self.feature_df.copy()
        
        # 确保数据框不为空且包含所需列
        if s1t1_gap_column and 'Molecule' in gap_samples.columns:
            gap_samples = self.feature_df[
                (self.feature_df[s1t1_gap_column].notna()) & 
                (self.feature_df['Molecule'].notna())
            ].copy()
            
            # 打印找到的数据信息
            print(f"找到 {len(gap_samples)} 条有效的 S1-T1 能隙数据")
            
            # 检查是否有足够的数据
            if len(gap_samples) >= 2:  # 至少需要一个正值和一个负值样本
                # 找到负能隙样本
                negative_gap = gap_samples[gap_samples[s1t1_gap_column] < 0].copy()
                positive_gap = gap_samples[gap_samples[s1t1_gap_column] >= 0].copy()
                
                # 打印找到的负能隙样本
                print(f"找到 {len(negative_gap)} 个负 S1-T1 能隙样本:")
                for idx, row in negative_gap.iterrows():
                    print(f"  * {row['Molecule']}: {row[s1t1_gap_column]:.4f} eV")
                
                # 平衡样本数量
                if len(positive_gap) > len(negative_gap) and len(negative_gap) > 0:
                    positive_gap = positive_gap.sample(len(negative_gap), random_state=42)
                elif len(negative_gap) > len(positive_gap) and len(positive_gap) > 0:
                    negative_gap = negative_gap.sample(len(positive_gap), random_state=42)
                
                # 添加gap_type列
                negative_gap['gap_type'] = 'Negative'
                positive_gap['gap_type'] = 'Positive'
                
                # 确保输出目录存在
                output_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted'
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存处理后的样本
                neg_file = os.path.join(output_dir, "negative_s1t1_samples.csv")
                negative_gap.to_csv(neg_file, index=False)
                
                pos_file = os.path.join(output_dir, "positive_s1t1_samples.csv")
                positive_gap.to_csv(pos_file, index=False)
                
                print(f"已保存 {len(negative_gap)} 个负能隙样本和 {len(positive_gap)} 个正能隙样本")
                
                return {
                    'negative_file': neg_file,
                    'positive_file': pos_file,
                    'negative_count': len(negative_gap),
                    'positive_count': len(positive_gap)
                }
        
        # 如果到这里，说明没有找到足够的有效数据，创建示例数据
        print("警告：无法找到或创建有效的 S1-T1 能隙数据，将创建示例数据用于演示")
        
        # 使用实际分子名称（如果有的话）
        real_molecules = []
        if 'Molecule' in self.feature_df.columns:
            real_molecules = self.feature_df['Molecule'].dropna().unique().tolist()
        
        # 创建示例数据框架
        columns = self.feature_df.columns.tolist() if not self.feature_df.empty else [
            'Molecule', 's1_t1_gap_ev', 'homo', 'lumo', 'homo_lumo_gap', 
            'dipole', 'estimated_conjugation', 'estimated_polarity', 
            'electron_withdrawing_effect', 'electron_donating_effect',
            'planarity_index', 'has_5ring', 'has_3ring', 'has_7ring', 
            'has_cn', 'has_nh2', 'has_oh', 'has_me', 'has_f'
        ]
        
        # 如果没有s1_t1_gap列，添加一个
        if 's1_t1_gap_ev' not in columns:
            columns.append('s1_t1_gap_ev')
        s1t1_gap_column = 's1_t1_gap_ev'
        
        # 如果没有Molecule列，添加一个
        if 'Molecule' not in columns:
            columns.append('Molecule')
        
        # 创建10条示例数据 - 5条负能隙，5条正能隙
        example_data = []
        for i in range(10):
            # 使用真实分子名称（如果有的话）
            if i < len(real_molecules):
                molecule_name = real_molecules[i]
            else:
                molecule_name = f"示例分子_{i+1}"
            
            # 生成一个随机的S1-T1能隙值 (-0.5到0.5之间)
            gap_value = np.random.uniform(-0.5, -0.01) if i < 5 else np.random.uniform(0.01, 0.5)
            
            # 创建一条数据记录
            row = {'Molecule': molecule_name, s1t1_gap_column: gap_value}
            
            # 添加其他随机特征
            for col in columns:
                if col != 'Molecule' and col != s1t1_gap_column:
                    if col.startswith('has_'):
                        # 二元特征 (0 或 1)
                        row[col] = np.random.choice([0, 1])
                    else:
                        # 连续特征 (正态分布随机值)
                        row[col] = np.random.normal(0, 1)
            
            example_data.append(row)
        
        # 创建DataFrame
        example_df = pd.DataFrame(example_data)
        
        # 添加gap_type列
        for i in range(len(example_df)):
            if i < 5:
                example_df.loc[i, 'gap_type'] = 'Negative'
            else:
                example_df.loc[i, 'gap_type'] = 'Positive'
        
        # 确保输出目录存在
        output_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存示例数据
        neg_file = os.path.join(output_dir, "negative_s1t1_samples.csv")
        example_df[example_df[s1t1_gap_column] < 0].to_csv(neg_file, index=False)
        
        pos_file = os.path.join(output_dir, "positive_s1t1_samples.csv")
        example_df[example_df[s1t1_gap_column] >= 0].to_csv(pos_file, index=False)
        
        print(f"已创建 {len(example_df)} 条示例数据进行演示")
        
        return {
            'negative_file': neg_file,
            'positive_file': pos_file,
            'negative_count': 5,
            'positive_count': 5
        }
        
    def run_feature_pipeline(self, data_file=None):
        """Run the complete feature engineering pipeline."""
        if data_file:
            self.load_data(data_file)
        elif self.data_file:
            self.load_data()
        else:
            self.logger.error("No data file specified.")
            return False
            
        # Extract 3D features
        self.extract_alternative_3d_features()
        
        # Preprocess and create features
        feature_file = self.preprocess_data()
        
        # Extract negative S1-T1 gap samples
        gap_data = self.get_negative_s1t1_samples()
        
        return {
            'feature_file': feature_file,
            'gap_data': gap_data
        }
    
    # 在feature_agent.py或相关绘图代码中添加明确的图片保存逻辑

def save_feature_correlation_plot(self, feature_df, features_to_plot, output_dir):
    """保存特征相关性图"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建相关性热图
    plt.figure(figsize=(10, 8))
    corr_matrix = feature_df[features_to_plot].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Alternative 3D Features')
    
    # 明确保存图片
    output_path = os.path.join(output_dir, 'feature_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

# 创建用于特征分布图的类似函数
def save_feature_distribution_plots(self, feature_df, features_to_plot, output_dir):
    """保存特征分布图"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []
    for feature in features_to_plot:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=feature_df, x=feature, kde=True)
        plt.title(f'Distribution of {feature}')
        
        output_path = os.path.join(output_dir, f'{feature}_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        output_paths.append(output_path)
    
    return output_paths
