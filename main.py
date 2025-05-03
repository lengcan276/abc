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
    """检查数据文件是否正确生成"""
    import pandas as pd
    
    # 检查分子特性摘要文件
    summary_file = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted/molecular_properties_summary.csv'
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        print(f"分子特性摘要文件包含 {len(df)} 行和 {len(df.columns)} 列")
        
        # 检查S1-T1能隙列
        gap_columns = [col for col in df.columns if 's1_t1' in col.lower() or 'triplet_gap' in col.lower()]
        if gap_columns:
            print(f"找到的能隙列: {gap_columns}")
            
            # 统计负能隙分子
            for col in gap_columns:
                neg_count = (df[col] < 0).sum()
                print(f"列 {col} 中有 {neg_count} 个负值")
                
                if neg_count > 0:
                    neg_mols = df[df[col] < 0]['Molecule'].tolist()
                    print(f"负能隙分子: {neg_mols}")
        else:
            print("警告：未找到S1-T1能隙列")
    else:
        print(f"警告：摘要文件 {summary_file} 不存在")
    
    # 检查负能隙样本文件
    neg_file = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted/negative_s1t1_samples.csv'
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
    pos_file = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/extracted/positive_s1t1_samples.csv'
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

def main():
    """Main entry point for the Reverse TADF Analysis System."""
    # Set page configuration
    st.set_page_config(
        page_title="Reverse TADF Analysis System",
        page_icon="������",
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
