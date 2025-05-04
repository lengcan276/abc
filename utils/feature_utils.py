# utils/feature_utils.py
import pandas as pd
import numpy as np
import logging

class FeatureUtils:
    """
    用于特征工程和处理的通用工具类
    """
    
    @staticmethod
    def estimate_molecular_conjugation(molecule_name):
        """估计分子的共轭程度基于名称中的特征"""
        molecule_name = molecule_name.lower()
        
        # 基础共轭分数
        conjugation_score = 0
        
        # 根据环系统贡献
        if '5ring' in molecule_name:
            conjugation_score += 5
        if '3ring' in molecule_name:
            conjugation_score += 3
        if '7ring' in molecule_name:
            conjugation_score += 7
        if 'ring' in molecule_name and not any(x in molecule_name for x in ['3ring', '5ring', '7ring']):
            conjugation_score += 6  # 默认为六元环
            
        # 特定共轭基团的贡献
        conjugation_contributors = {
            'cn': 2,    # 氰基延长共轭
            'nh2': 1,   # 胺基有助于共轭
            'oh': 1,    # 羟基有助于共轭
            'no2': 2,   # 硝基延长共轭
            'nme2': 1,  # 二甲胺基有助于共轭
            'nn+': 1.5  # 重氮基团
        }
        
        for group, value in conjugation_contributors.items():
            if group in molecule_name:
                count = molecule_name.count(group)
                conjugation_score += value * count
                
        return conjugation_score
    
    @staticmethod
    def estimate_electronic_effects(molecule_name):
        """估计电子效应（给电子和吸电子效应）"""
        molecule_name = molecule_name.lower()
        
        # 不同取代基的电子效应（+值为给电子，-值为吸电子）
        electronic_effects = {
            'cn': -3,    # 氰基：强吸电子
            'nh2': 2,    # 胺基：给电子
            'oh': 1,     # 羟基：轻微给电子
            'f': -1,     # 氟：吸电子
            'sh': 0.5,   # 硫醇基：轻微给电子
            'no2': -3.5, # 硝基：强吸电子
            'ome': 1,    # 甲氧基：给电子
            'nme2': 2,   # 二甲胺基：给电子
            'nn+': -3,   # 重氮基团：强吸电子
            'nph3': 1.5, # 三苯胺基：给电子
            'cf3': -2.5, # 三氟甲基：吸电子
            'me': 0.5,   # 甲基：轻微给电子
            'bh2': -1    # 硼基：轻微吸电子
        }
        
        donating_effect = 0
        withdrawing_effect = 0
        
        for group, effect in electronic_effects.items():
            count = molecule_name.count(group)
            if count > 0:
                if effect > 0:
                    donating_effect += effect * count
                else:
                    withdrawing_effect += abs(effect) * count
                    
        # 确保没有零值，以避免除零错误
        if withdrawing_effect == 0:
            withdrawing_effect = 0.1
            
        return donating_effect, withdrawing_effect, donating_effect - withdrawing_effect
    
    @staticmethod
    def estimate_polarity(molecule_name):
        """估计分子极性基于名称中的取代基"""
        molecule_name = molecule_name.lower()
        
        # 不同基团的极性贡献
        polarity_contributions = {
            'cn': 3.5,   # 氰基：高极性
            'nh2': 3,    # 胺基：高极性
            'oh': 3,     # 羟基：高极性
            'f': 4,      # 氟：高极性
            'sh': 1.5,   # 硫醇基：中等极性
            'no2': 4,    # 硝基：高极性
            'ome': 2.5,  # 甲氧基：中等极性
            'nme2': 2,   # 二甲胺基：中等极性
            'nn+': 4.5,  # 重氮基团：高极性
            'nph3': 1,   # 三苯胺基：低极性
            'cf3': 2.5,  # 三氟甲基：中等极性
            'me': 0.2,   # 甲基：低极性
            'bh2': 1     # 硼基：低极性
        }
        
        polarity_score = 0
        
        for group, polarity in polarity_contributions.items():
            count = molecule_name.count(group)
            polarity_score += polarity * count
            
        return polarity_score
    
    @staticmethod
    def estimate_molecular_polarity(molecule_name):
        """估计分子的极性/非极性特性基于名称中的取代基 - 关系到电荷分离能力和TADF性能"""
        molecule_name = molecule_name.lower()
        
        # 不同基团的极性贡献（正值表示非极性，负值表示极性）
        polarity_contributions = {
            'cn': -1,    # 氰基：极性，有助于电荷分离
            'nh2': -2,   # 胺基：极性，有助于电荷分离
            'oh': -2,    # 羟基：极性，有助于电荷分离
            'f': -0.5,   # 氟：轻微极性
            'sh': 0.5,   # 硫醇基：轻微非极性
            'no2': -1.5, # 硝基：极性，强电子吸引
            'ome': -0.5, # 甲氧基：轻微极性
            'nme2': 0.5, # 二甲胺基：轻微非极性
            'nn+': -3,   # 重氮基团：高极性，显著影响激发态
            'nph3': 3,   # 三苯胺基：非极性，影响分子扭曲
            'cf3': 2,    # 三氟甲基：非极性，但电负性高
            'me': 1,     # 甲基：非极性，可影响分子扭曲
            'bh2': 0.5   # 硼基：轻微非极性
        }
        
        polarity_score = 0
        
        for group, polarity in polarity_contributions.items():
            count = molecule_name.count(group)
            polarity_score += polarity * count
            
        return polarity_score
    
    @staticmethod
    def estimate_planarity(molecule_name):
        """估计分子平面性基于名称中的特征"""
        molecule_name = molecule_name.lower()
        
        # 基础平面性分数
        planarity_score = 0
        
        # 环越多越平面
        ring_count = molecule_name.count('ring')
        planarity_score += ring_count * 2
        
        # 某些取代基会降低平面性
        planar_reducers = {
            'nph3': 3,   # 大型取代基降低平面性
            'nme2': 1,   # 二甲胺基降低平面性
            'cf3': 1.5   # 三氟甲基降低平面性
        }
        
        for group, reduction in planar_reducers.items():
            count = molecule_name.count(group)
            planarity_score -= reduction * count
            
        # 确保平面性指数不为负
        planarity_score = max(0.1, planarity_score)
        
        return planarity_score
    
    @staticmethod
    def estimate_size(molecule_name):
        """估计分子大小基于名称中的环和取代基"""
        molecule_name = molecule_name.lower()
        
        # 基础大小
        size_score = 0
        
        # 环的贡献
        ring_sizes = {'3ring': 3, '5ring': 5, '7ring': 7}
        for ring, size in ring_sizes.items():
            if ring in molecule_name:
                size_score += size
                
        # 如果有环但不是特定大小的环，假设是六元环
        if 'ring' in molecule_name and not any(r in molecule_name for r in ring_sizes.keys()):
            size_score += 6
            
        # 不同取代基对大小的贡献
        group_sizes = {
            'cn': 2,     # 氰基
            'nh2': 1.5,  # 胺基
            'oh': 1,     # 羟基
            'me': 1,     # 甲基
            'f': 0.5,    # 氟
            'sh': 1,     # 硫醇基
            'bh2': 1,    # 硼基
            'cf3': 3,    # 三氟甲基
            'no2': 2,    # 硝基
            'ome': 2,    # 甲氧基
            'nme2': 3,   # 二甲胺基
            'nph3': 7,   # 三苯胺基
            'nn+': 2     # 重氮基团
        }
        
        for group, size in group_sizes.items():
            count = molecule_name.count(group)
            size_score += size * count
            
        # 确保大小至少为 1，避免除零错误
        size_score = max(1.0, size_score)
        
        return size_score
    
    @staticmethod
    def handle_outliers(df, columns, z_threshold=3.0):
        """处理极端值（通过 Z 分数）"""
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                # 计算 Z 分数
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                
                # 避免零标准差
                if std_val == 0 or pd.isna(std_val):
                    continue
                    
                # 识别极端值
                z_scores = abs((df_clean[col] - mean_val) / std_val)
                outliers = z_scores > z_threshold
                
                # 将极端值替换为 NaN
                if outliers.sum() > 0:
                    df_clean.loc[outliers, col] = np.nan
                    logging.info(f"在 '{col}' 列中识别出 {outliers.sum()} 个极端值，已替换为 NaN")
                    
        return df_clean
    
    @staticmethod
    def add_molecular_features(df):
        """为数据框添加分子特征"""
        # 确保有 Molecule 列
        if 'Molecule' not in df.columns:
            logging.error("数据框中没有 Molecule 列")
            return df
            
        # 创建特征
        df['ring_count'] = df['Molecule'].str.lower().str.count('ring')
        
        # 环类型
        df['has_5ring'] = df['Molecule'].str.lower().str.contains('5ring').astype(int)
        df['has_3ring'] = df['Molecule'].str.lower().str.contains('3ring').astype(int)
        df['has_7ring'] = df['Molecule'].str.lower().str.contains('7ring').astype(int)
        
        # 取代基位置
        df['has_in_group'] = df['Molecule'].str.lower().str.contains('_in').astype(int)
        df['has_out_group'] = df['Molecule'].str.lower().str.contains('_out').astype(int)
        df['has_both_groups'] = df['Molecule'].str.lower().str.contains('_both').astype(int)
        
        # 功能基团
        functional_groups = ['cn', 'nh2', 'oh', 'me', 'f', 'sh', 'bh2', 'cf3', 'no2', 'ome', 'nme2', 'nph3', 'nn+']
        for group in functional_groups:
            df[f'has_{group}'] = df['Molecule'].str.lower().str.contains(group).astype(int)
            df[f'{group}_count'] = df['Molecule'].str.lower().str.count(group)
            
        # 使用估计函数
        df['estimated_conjugation'] = df['Molecule'].apply(FeatureUtils.estimate_molecular_conjugation)
        
        # 电子效应
        electronic_effects = df['Molecule'].apply(lambda x: FeatureUtils.estimate_electronic_effects(x))
        df['electron_donating_effect'] = electronic_effects.apply(lambda x: x[0])
        df['electron_withdrawing_effect'] = electronic_effects.apply(lambda x: x[1])
        df['net_electronic_effect'] = electronic_effects.apply(lambda x: x[2])
        
        # 其他估计的属性
        df['estimated_polarity'] = df['Molecule'].apply(FeatureUtils.estimate_polarity)
        df['molecular_polarity'] = df['Molecule'].apply(FeatureUtils.estimate_molecular_polarity)
        df['planarity_index'] = df['Molecule'].apply(FeatureUtils.estimate_planarity)
        df['estimated_size'] = df['Molecule'].apply(FeatureUtils.estimate_size)
        
        # 计算总取代基数量
        df['total_substituents'] = df[[f'{group}_count' for group in functional_groups]].sum(axis=1)
        
        # 创建结构复杂度指标
        df['structural_complexity'] = df['ring_count'] + df['total_substituents']
        
        return df
        
    @staticmethod
    def create_combined_features(df):
        """创建组合特征"""
        # 确保需要的列存在
        required_cols = ['homo', 'lumo', 'estimated_polarity', 'molecular_polarity',
                        'net_electronic_effect', 'estimated_conjugation', 'energy',
                        'estimated_size', 'dipole', 'planarity_index']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logging.warning(f"创建组合特征时缺少列: {missing_cols}")
            
        # 创建组合特征（如果基础列存在）
        if 'homo' in df.columns and 'estimated_polarity' in df.columns:
            # 确保极性不为零
            safe_polarity = df['estimated_polarity'].apply(lambda x: x if x != 0 else 0.1)
            df['homo_polarity'] = df['homo'] * safe_polarity
            
        if 'homo' in df.columns and 'molecular_polarity' in df.columns:
            # 添加分子极性与HOMO的相互作用
            safe_mol_polarity = df['molecular_polarity'].apply(lambda x: x if x != 0 else 0.1)
            df['homo_molecular_polarity'] = df['homo'] * safe_mol_polarity
            
        if 'lumo' in df.columns and 'net_electronic_effect' in df.columns:
            df['lumo_electronic_effect'] = df['lumo'] * df['net_electronic_effect']
            
        if 'homo_lumo_gap' in df.columns and 'estimated_conjugation' in df.columns:
            # 确保分母不为零
            safe_denominator = (1 + df['estimated_conjugation'].abs() + 1e-5)
            df['gap_conjugation'] = df['homo_lumo_gap'] / safe_denominator
            
        if 'energy' in df.columns and 'estimated_size' in df.columns:
            # 确保尺寸不为零
            safe_size = df['estimated_size'].apply(lambda x: max(1.0, x))
            df['energy_per_size'] = df['energy'] / safe_size
            
        if 'dipole' in df.columns and 'planarity_index' in df.columns:
            df['dipole_planarity'] = df['dipole'].fillna(0) * df['planarity_index']
        
        # 添加结构相关的组合特征
        structure_cols = [
            'conjugation_path_count', 'max_conjugation_length', 
            'max_dihedral_angle', 'twist_ratio', 
            'hydrogen_bonds_count', 'max_h_bond_strength',
            'planarity'
        ]
        
        # 检查必要的列是否存在
        existing_struct_cols = [col for col in structure_cols if col in df.columns]
        
        if existing_struct_cols:
            logging.info(f"检测到结构特征列: {existing_struct_cols}")
            
            if 'homo_lumo_gap' in df.columns:
                # 共轭与能隙关系
                if 'max_conjugation_length' in df.columns:
                    df['gap_vs_conjugation'] = df['homo_lumo_gap'] * df['max_conjugation_length']
                    
                # 扭曲与能隙关系
                if 'twist_ratio' in df.columns:
                    # 确保分母不为零
                    safe_twist = df['twist_ratio'].apply(lambda x: max(x, 0.01))
                    df['gap_vs_twist'] = df['homo_lumo_gap'] / safe_twist
                    
                # 氢键与能隙的关系
                if 'max_h_bond_strength' in df.columns:
                    df['gap_vs_h_bond'] = df['homo_lumo_gap'] * (1 + df['max_h_bond_strength'])
                    
            # S1-T1能隙与结构的关系
            if 's1_t1_gap_ev' in df.columns:
                # 与扭曲的关系
                if 'twist_ratio' in df.columns:
                    df['s1t1_vs_twist'] = df['s1_t1_gap_ev'] * df['twist_ratio']
                    
                # 与平面性的关系
                if 'planarity' in df.columns:
                    # 平面性越高（接近1），分母越小，效应越强
                    safe_nonplanar = (1.01 - df['planarity'].clip(0, 0.99))
                    df['s1t1_vs_nonplanar'] = df['s1_t1_gap_ev'] / safe_nonplanar
                    
                # 与共轭的关系
                if 'max_conjugation_length' in df.columns:
                    # 确保分母不为零
                    safe_conj = df['max_conjugation_length'].apply(lambda x: max(x, 1))
                    df['s1t1_vs_conjugation'] = df['s1_t1_gap_ev'] / safe_conj
                    
                # 与氢键相关的特征
                if 'hydrogen_bonds_count' in df.columns:
                    h_bond_factor = df['hydrogen_bonds_count'] + 1  # 加1避免零值
                    df['s1t1_vs_hbonds'] = df['s1_t1_gap_ev'] * h_bond_factor
                    
                # 计算扭曲角对S1-T1能隙的影响
                if 'max_dihedral_angle' in df.columns:
                    df['s1t1_vs_dihedral'] = df['s1_t1_gap_ev'] * df['max_dihedral_angle'] / 90
            
        return df
   
        
    @staticmethod
    def clean_and_prepare_features(df):
        """清理和准备特征数据框"""
        # 替换无穷大值为 NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 检查缺失值比例高的列
        na_ratios = df.isna().mean()
        high_na_cols = na_ratios[na_ratios > 0.3].index.tolist()
        
        if high_na_cols:
            logging.warning(f"以下列缺失值比例高于 30%: {high_na_cols}")
            
        # 填充数值列中的 NaN（使用中位数）
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 进一步处理处理极端值
        numeric_cols_filtered = [col for col in numeric_cols if col not in ['Molecule', 'conformer', 'State']]
        df = FeatureUtils.handle_outliers(df, numeric_cols_filtered)
        
        # 再次填充可能产生的 NaN
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df