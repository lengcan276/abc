# streamlit_ui/FeaturePage.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import base64
from io import BytesIO

def render_feature_page(feature_agent=None):
    """渲染特征工程页面"""
    st.title("特征工程")
    
    st.markdown("""
    ## 特征工程与替代 3D 描述符
    
    本页面帮助您生成和探索从提取数据中派生的各种分子描述符。
    
    关键特征类别：
    
    1. **电子属性** - HOMO、LUMO、给/吸电子效应
    2. **结构特征** - 环、取代基、平面性、共轭性
    3. **物理属性** - 极性、疏水性、尺寸估计
    4. **量子属性** - 能级、能隙、偶极矩
    
    您可以在先前提取的数据上运行特征工程流程，或上传新的 CSV 文件。
    """)
    
    # 使用现有数据或上传新数据的选项
    data_source = st.radio("数据来源", ["使用提取的数据", "上传 CSV"])
    
    data_file = None
    
    if data_source == "上传 CSV":
        uploaded_file = st.file_uploader("上传分子数据 CSV", type="csv")
        if uploaded_file:
            # 将上传的 CSV 保存到临时位置
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                temp_csv.write(uploaded_file.getvalue())
                data_file = temp_csv.name
    else:
        # 查找先前提取的数据
        extracted_dir = '../data/extracted'
        if os.path.exists(extracted_dir):
            csv_files = [f for f in os.listdir(extracted_dir) if f.endswith('.csv')]
            if csv_files:
                selected_file = st.selectbox("选择提取的数据文件", csv_files)
                data_file = os.path.join(extracted_dir, selected_file)
            else:
                st.warning("未找到提取的数据文件。请先提取数据。")
        else:
            st.warning("未找到提取的数据目录。请先提取数据。")
    
    # 执行特征工程
    feature_button = st.button("生成特征")
    
    if feature_button and data_file:
        with st.spinner("正在生成特征..."):
            try:
                # 执行特征工程（假设 feature_agent 已经传入）
                if feature_agent:
                    feature_agent.data_file = data_file
                    result = feature_agent.run_feature_pipeline()
                    
                    if result and 'feature_file' in result:
                        st.success(f"特征工程完成，结果保存至 {result['feature_file']}")
                        
                        # 加载并显示特征数据
                        feature_df = pd.read_csv(result['feature_file'])
                        
                        # 显示基本统计信息
                        st.subheader("特征统计")
                        st.write(f"总特征数: {len(feature_df.columns)}")
                        
                        # 如果有 S1-T1 能隙数据，显示相应统计信息
                        if 's1_t1_gap_ev' in feature_df.columns:
                            gap_data = feature_df[feature_df['s1_t1_gap_ev'].notna()]
                            neg_count = (gap_data['s1_t1_gap_ev'] < 0).sum()
                            pos_count = (gap_data['s1_t1_gap_ev'] >= 0).sum()
                            
                            st.write(f"具有 S1-T1 能隙数据的分子: {len(gap_data['Molecule'].unique())}")
                            st.write(f"具有负 S1-T1 能隙的分子 (反向 TADF 候选体): {neg_count}")
                            
                            # 创建 S1-T1 能隙分布图
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(data=gap_data, x='s1_t1_gap_ev', bins=20, kde=True)
                            plt.axvline(x=0, color='red', linestyle='--')
                            plt.title('S1-T1 能隙分布')
                            plt.xlabel('S1-T1 能隙 (eV)')
                            st.pyplot(fig)
                        
                        # 替代 3D 特征
                        st.subheader("替代 3D 特征示例")
                        
                        # 选择一些有趣的 3D 特征
                        d3_features = [
                            'estimated_conjugation', 'estimated_polarity', 'electron_withdrawing_effect',
                            'electron_donating_effect', 'planarity_index', 'estimated_hydrophobicity'
                        ]
                        
                        # 筛选存在于数据框中的特征
                        valid_d3 = [f for f in d3_features if f in feature_df.columns]
                        
                        if valid_d3:
                            # 创建 3D 特征之间的相关性热图
                            d3_corr = feature_df[valid_d3].corr()
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(d3_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                            plt.title('替代 3D 特征之间的相关性')
                            st.pyplot(fig)
                            
                            # 显示几个关键特征的分布
                            st.subheader("特征分布")
                            
                            for feature in valid_d3[:3]:  # 显示前 3 个特征
                                fig, ax = plt.subplots(figsize=(8, 5))
                                sns.histplot(data=feature_df, x=feature, kde=True)
                                plt.title(f'{feature} 的分布')
                                st.pyplot(fig)
                        
                        # 创建特征下载链接
                        create_download_link(result['feature_file'], "下载处理后的特征 CSV")
                        
                        # 如果有 S1-T1 能隙数据，提供导航到探索页面的选项
                        if 's1_t1_gap_ev' in feature_df.columns and neg_count > 0:
                            st.info("检测到负 S1-T1 能隙分子。前往探索分析页面分析这些反向 TADF 候选体。")
                            
                        # 返回特征生成结果
                        return result
                    else:
                        st.error("特征工程失败")
                else:
                    st.error("特征工程组件未初始化")
            except Exception as e:
                st.error(f"特征工程过程中出错: {str(e)}")
                
    return None

def create_download_link(file_path, text):
    """创建文件下载链接"""
    with open(file_path, 'rb') as f:
        data = f.read()
        
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def load_feature_page(feature_agent=None):
    """加载特征工程页面"""
    return render_feature_page(feature_agent)

if __name__ == "__main__":
    # 用于直接运行测试
    load_feature_page()
