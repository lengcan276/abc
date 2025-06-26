# main.py
import streamlit as st
from agents.ui_agent import UIAgent
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reverse_tadf_system.log'),
        logging.StreamHandler()
    ]
)

def debug_data_files():
    """检查数据文件是否正确生成，包括多种激发态能隙"""
    import pandas as pd
    
    # 检查分子特性摘要文件
    summary_file = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted/molecular_properties_summary.csv'
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        print(f"分子特性摘要文件包含 {len(df)} 行和 {len(df.columns)} 列")
        print(f"列名: {list(df.columns)[:20]}...")  # 显示前20个列名
        
        # 检查各种激发态能隙
        gap_patterns = {
            'S1-T1': ['s1_t1', 'singlet_triplet_gap', 'delta_s1_t1'],
            'S1-T2': ['s1_t2', 'singlet_t2_gap', 'delta_s1_t2'],
            'T1-T2': ['t1_t2', 'triplet_gap', 'delta_t1_t2'],
            'S2-S1': ['s2_s1', 'singlet_gap', 'delta_s2_s1'],
            'S2-T1': ['s2_t1', 's2_triplet_gap', 'delta_s2_t1'],
            'S2-T2': ['s2_t2', 's2_t2_gap', 'delta_s2_t2']
        }
        
        found_gaps = {}
        for gap_type, patterns in gap_patterns.items():
            gap_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in patterns):
                    gap_columns.append(col)
            
            if gap_columns:
                found_gaps[gap_type] = gap_columns
                print(f"\n找到 {gap_type} 能隙列: {gap_columns}")
                
                # 统计能隙分布
                for col in gap_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        neg_count = (df[col] < 0).sum()
                        zero_count = (df[col] == 0).sum()
                        pos_count = (df[col] > 0).sum()
                        
                        print(f"  列 {col}:")
                        print(f"    负值: {neg_count} 个")
                        print(f"    零值: {zero_count} 个")
                        print(f"    正值: {pos_count} 个")
                        
                        if neg_count > 0:
                            neg_mols = df[df[col] < 0]['Molecule'].tolist()[:5]  # 只显示前5个
                            print(f"    负能隙分子示例: {neg_mols}")
        
        if not found_gaps:
            print("\n警告：未找到任何激发态能隙列！")
            print("请检查数据提取过程是否正确计算了各种激发态能隙。")
        
        # 检查是否有REVERSED TADF候选分子（满足特定能隙条件）
        print("\n\n=== REVERSED TADF 候选分子筛选 ===")
        
        # 条件1：S1-T2 < 0 (S1低于T2)
        if 'S1-T2' in found_gaps:
            for col in found_gaps['S1-T2']:
                if pd.api.types.is_numeric_dtype(df[col]):
                    condition1 = df[col] < 0
                    print(f"\n条件1 (S1 < T2): {condition1.sum()} 个分子")
                    
        # 条件2：T1-T2能隙较小（便于三重态之间的转换）
        if 'T1-T2' in found_gaps:
            for col in found_gaps['T1-T2']:
                if pd.api.types.is_numeric_dtype(df[col]):
                    small_gap = df[col].abs() < 0.3  # eV
                    print(f"\n条件2 (|T1-T2| < 0.3 eV): {small_gap.sum()} 个分子")
    else:
        print(f"警告：摘要文件 {summary_file} 不存在")
    
    # 检查负能隙样本文件
    neg_file = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted/negative_s1t1_samples.csv'
    if os.path.exists(neg_file):
        df = pd.read_csv(neg_file)
        print(f"负能隙样本文件包含 {len(df)} 行和 {len(df.columns)} 列")
        
        # 检查是否包含分子名称
        if 'Molecule' in df.columns:
            print(f"负能隙分子: {df['Molecule'].tolist()}")
        else:
            print("警告：负能隙样本文件中缺少 'Molecule' 列")
    else:
        print(f"警告：负能隙样本文件 {neg_file} 不存在")
    
    # 检查正能隙样本文件
    pos_file = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted/positive_s1t1_samples.csv'
    if os.path.exists(pos_file):
        df = pd.read_csv(pos_file)
        print(f"正能隙样本文件包含 {len(df)} 行和 {len(df.columns)} 列")
        
        # 检查是否包含分子名称
        if 'Molecule' in df.columns:
            print(f"正能隙分子: {df['Molecule'].tolist()}")
        else:
            print("警告：正能隙样本文件中缺少 'Molecule' 列")
    else:
        print(f"警告：正能隙样本文件 {pos_file} 不存在")
    
    # 运行REVERSED TADF分析
    print("\n\n=== 运行REVERSED TADF深度分析 ===")
    try:
        from utils.reversed_tadf_analyzer import ReversedTADFAnalyzer
        analyzer = ReversedTADFAnalyzer(summary_file)
        if analyzer.load_data():
            output_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted/reversed_tadf_analysis'
            analyzer.export_results(output_dir)
    except ImportError:
        print("提示：将reversed_tadf_analyzer.py放入utils文件夹以启用高级分析功能")
    except Exception as e:
        print(f"REVERSED TADF分析出错: {e}")

def main():
    """Main entry point for the Reverse TADF Analysis System."""
    # Set page configuration
    st.set_page_config(
        page_title="Reverse TADF Analysis System",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create necessary directories
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/extracted', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/reports', exist_ok=True)
    
    # Initialize and run UI agent
    ui_agent = UIAgent()
    ui_agent.run_app()

if __name__ == "__main__":
    debug_data_files()
    main()