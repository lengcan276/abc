import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Import configuration
try:
    from config import EXCITATION_SETTINGS
except ImportError:
    # Fallback if config not available
    EXCITATION_SETTINGS = {}

class ExcitationAnalyzer:
    """激发态分析工具，基于wB97X-D/def2-TZVP基准测试"""
    
    def __init__(self, reference_data=None):
        """
        初始化分析器
        reference_data: 包含CCSD参考数据的字典
        """
        # Load from config or use defaults
        config_reference = EXCITATION_SETTINGS.get('reference_molecules', {})
        default_reference = {
            'calicene': {'gap': -0.101, 'type': 'S3-T4'},
            '3ring_cn': {'gap': -0.028, 'type': 'S2-T4'},
            '5ring_nme2': {'gap': -0.016, 'type': 'S2-T4'},
            '5ring_nme2_3ring_cn_in': {'gap': -0.011, 'type': 'S1-T1'},
            '5ring_npme3_3ring_CN': {'gap': -0.013, 'type': 'S1-T1'}
        }
        
        # Merge config with defaults, config takes precedence
        self.reference_data = reference_data or {**default_reference, **config_reference}
        
        # Load systematic corrections from config
        self.systematic_correction = EXCITATION_SETTINGS.get(
            'systematic_corrections',
            {
                'S1-T1': -0.037,
                'S2-T4': -0.040,
                'S3-T4': -0.045,
                'default': -0.040
            }
        )
        
        # Load screening criteria
        self.screening_criteria = EXCITATION_SETTINGS.get(
            'screening_criteria',
            {
                'min_transition_similarity': 0.7,
                'confidence_threshold': 0.7,
                'gap_threshold': -0.005,
                'max_state_number': 6
            }
        )
        
        # Method and basis from config
        self.method = EXCITATION_SETTINGS.get('method', 'wb97xd')
        self.basis = EXCITATION_SETTINGS.get('basis', 'def2-tzvp')
    
    def analyze_molecule_excitations(self, states_data, molecule_name):
        """分析单个分子的激发态"""
        analysis = {
            'molecule': molecule_name,
            'total_singlets': len(states_data['singlets']),
            'total_triplets': len(states_data['triplets']),
            'inverted_gaps': []
        }
        
        # 分析所有反转能隙
        for gap_info in states_data['inverted_gaps']:
            # 应用系统误差修正
            gap_type = gap_info['type']
            correction = self.systematic_correction.get(
                gap_type, 
                self.systematic_correction.get('default', -0.040)
            )
            corrected_gap = gap_info['gap'] + correction
            
            gap_analysis = {
                'type': gap_type,
                'raw_gap': gap_info['gap'],
                'corrected_gap': corrected_gap,
                'similarity': gap_info['transition_similarity'],
                'confidence': self.calculate_confidence(gap_info)
            }
            
            # 与参考数据比较
            if molecule_name in self.reference_data:
                ref = self.reference_data[molecule_name]
                if gap_type == ref['type']:
                    gap_analysis['reference_gap'] = ref['gap']
                    gap_analysis['error'] = corrected_gap - ref['gap']
                    gap_analysis['error_meV'] = gap_analysis['error'] * 1000
            
            analysis['inverted_gaps'].append(gap_analysis)
        
        # 确定主要反转类型
        if analysis['inverted_gaps']:
            # 选择置信度最高的反转
            analysis['primary_inversion'] = max(
                analysis['inverted_gaps'], 
                key=lambda x: x['confidence']
            )
        
        return analysis
    
    def calculate_confidence(self, gap_info):
        """
        计算反转能隙的置信度
        基于：跃迁相似度、能隙大小、振子强度等
        """
        confidence = 0.0
        
        # 跃迁相似度贡献 (权重40%)
        min_similarity = self.screening_criteria['min_transition_similarity']
        if gap_info['transition_similarity'] >= min_similarity:
            confidence += gap_info['transition_similarity'] * 0.4
        
        # 能隙大小贡献 (权重30%)
        gap_magnitude = abs(gap_info['gap'])
        gap_threshold = abs(self.screening_criteria['gap_threshold'])
        if gap_magnitude > gap_threshold:
            confidence += min(gap_magnitude / 0.1, 1.0) * 0.3
        
        # 低激发态优先 (权重20%)
        max_state = self.screening_criteria['max_state_number']
        state_penalty = (gap_info['singlet_state'] + gap_info['triplet_state']) / (2 * max_state)
        confidence += max(0, 1 - state_penalty) * 0.2
        
        # 振子强度贡献 (权重10%)
        if gap_info['singlet_osc_strength'] > 0.01:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def generate_benchmark_comparison(self, analyzed_molecules):
        """生成与基准测试的比较报告"""
        comparison_data = []
        
        for mol_analysis in analyzed_molecules:
            mol_name = mol_analysis['molecule']
            
            if 'primary_inversion' in mol_analysis:
                primary = mol_analysis['primary_inversion']
                
                row = {
                    'molecule': mol_name,
                    'gap_type': primary['type'],
                    'calculated_gap': primary['corrected_gap'],
                    'confidence': primary['confidence']
                }
                
                # 添加参考数据
                if 'reference_gap' in primary:
                    row['reference_gap'] = primary['reference_gap']
                    row['error'] = primary['error']
                    row['error_meV'] = primary['error_meV']
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_analysis_results(self, comparison_df, output_dir='./'):
        """绘制分析结果图表"""
        # 创建类似于基准测试的图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 能隙类型分布
        ax1 = axes[0, 0]
        gap_type_counts = comparison_df['gap_type'].value_counts()
        gap_type_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Inverted Gap Types')
        ax1.set_xlabel('Gap Type')
        ax1.set_ylabel('Count')
        
        # 2. 置信度分布
        ax2 = axes[0, 1]
        ax2.hist(comparison_df['confidence'], bins=20, color='lightgreen', 
                edgecolor='black', alpha=0.7)
        ax2.set_title('Confidence Score Distribution')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Count')
        ax2.axvline(x=0.7, color='red', linestyle='--', 
                   label='High Confidence Threshold')
        ax2.legend()
        
        # 3. 能隙值分布
        ax3 = axes[1, 0]
        for gap_type in comparison_df['gap_type'].unique():
            data = comparison_df[comparison_df['gap_type'] == gap_type]['calculated_gap']
            ax3.scatter([gap_type] * len(data), data * 1000, alpha=0.6, s=50)
        ax3.set_title('Calculated Gap Values by Type')
        ax3.set_xlabel('Gap Type')
        ax3.set_ylabel('Gap (meV)')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 4. 如果有参考数据，绘制相关性图
        ax4 = axes[1, 1]
        if 'reference_gap' in comparison_df.columns:
            has_ref = comparison_df.dropna(subset=['reference_gap'])
            ax4.scatter(has_ref['reference_gap'] * 1000, 
                       has_ref['calculated_gap'] * 1000,
                       s=100, alpha=0.7)
            
            # 添加理想线
            min_val = min(has_ref['reference_gap'].min(), has_ref['calculated_gap'].min()) * 1000
            max_val = max(has_ref['reference_gap'].max(), has_ref['calculated_gap'].max()) * 1000
            ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            ax4.set_xlabel('Reference Gap (meV)')
            ax4.set_ylabel('Calculated Gap (meV)')
            ax4.set_title('Calculated vs Reference Gaps')
            
            # 计算并显示统计
            if len(has_ref) > 1:
                mae = has_ref['error_meV'].abs().mean()
                rmse = np.sqrt((has_ref['error_meV']**2).mean())
                ax4.text(0.05, 0.95, f'MAE: {mae:.1f} meV\nRMSE: {rmse:.1f} meV',
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax4.text(0.5, 0.5, 'No reference data available',
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Reference Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inverted_gap_analysis.png'), dpi=300)
        plt.close()