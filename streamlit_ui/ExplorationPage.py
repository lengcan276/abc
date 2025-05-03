# streamlit_ui/ExplorationPage.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import base64
from PIL import Image
from io import BytesIO
import zipfile

def render_exploration_page(exploration_agent=None, feature_agent=None):
    """渲染探索分析页面"""
    st.title("反向 TADF 探索分析")
    
    st.markdown("""
    ## 反向 TADF 候选体探索
    
    本页面专注于分析具有负 S1-T1 能隙的分子，这些分子是潜在的反向 TADF 候选体。
    
    分析包括：
    
    1. **结构模式识别** - 反向 TADF 分子中的共同特征
    2. **电子属性分析** - 独特的电子特性
    3. **比较可视化** - 正负能隙分子之间的差异
    4. **特征聚类** - 分子属性的多维分析
    
    您可以在先前生成的特征数据上运行探索分析，或上传特征 CSV 文件。
    """)
    
    # 查找数据选项
    neg_file = None
    pos_file = None
    
    # 查找先前处理的数据
    extracted_dir = '../data/extracted'
    if os.path.exists(extracted_dir):
        neg_path = os.path.join(extracted_dir, 'negative_s1t1_samples.csv')
        pos_path = os.path.join(extracted_dir, 'positive_s1t1_samples.csv')
        
        if os.path.exists(neg_path) and os.path.exists(pos_path):
            st.info("找到现有的负值和正值 S1-T1 能隙数据。")
            neg_file = neg_path
            pos_file = pos_path
        else:
            # 查找处理过的特征文件以生成能隙数据
            feature_files = [f for f in os.listdir(extracted_dir) if 'feature' in f.lower() and f.endswith('.csv')]
            
            if feature_files:
                st.info("未找到预处理的能隙数据，但可用特征文件。")
                selected_file = st.selectbox("选择要处理的特征文件", feature_files)
                
                if st.button("处理能隙数据"):
                    with st.spinner("正在处理 S1-T1 能隙数据..."):
                        try:
                            # 使用特征代理处理能隙数据
                            if feature_agent:
                                feature_agent.data_file = os.path.join(extracted_dir, selected_file)
                                feature_agent.load_data()
                                gap_results = feature_agent.get_negative_s1t1_samples()
                                
                                if gap_results:
                                    neg_file = gap_results['negative_file']
                                    pos_file = gap_results['positive_file']
                                    st.success(f"找到 {gap_results['negative_count']} 个负值和 {gap_results['positive_count']} 个正值 S1-T1 能隙样本。")
                                else:
                                    st.error("能隙数据处理失败")
                            else:
                                st.error("特征工程组件未初始化")
                        except Exception as e:
                            st.error(f"处理能隙数据时出错: {str(e)}")
            else:
                st.warning("未找到特征文件。请先运行特征工程。")
    else:
        st.warning("未找到提取的数据目录。请先提取数据并运行特征工程。")
        
    # 上传文件选项
    if not neg_file or not pos_file:
        st.subheader("上传能隙数据")
        
        neg_upload = st.file_uploader("上传负 S1-T1 能隙样本 CSV", type="csv")
        pos_upload = st.file_uploader("上传正 S1-T1 能隙样本 CSV", type="csv")
        
        if neg_upload and pos_upload:
            # 将上传的 CSV 保存到临时位置
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_neg:
                temp_neg.write(neg_upload.getvalue())
                neg_file = temp_neg.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_pos:
                temp_pos.write(pos_upload.getvalue())
                pos_file = temp_pos.name
                
    # 执行探索分析
    if neg_file and pos_file:
        # 检查是否存在预计算结果
        results_dir = '../data/reports/exploration'
        
        if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
            st.info("找到现有的探索分析结果。")
            
            if st.button("显示探索分析结果"):
                display_exploration_results(results_dir)
                
            if st.button("重新运行探索分析"):
                run_exploration_analysis(neg_file, pos_file, exploration_agent)
        else:
            if st.button("运行探索分析"):
                run_exploration_analysis(neg_file, pos_file, exploration_agent)
                
    return None

def run_exploration_analysis(neg_file, pos_file, exploration_agent):
    """运行探索分析并显示结果"""
    with st.spinner("正在运行探索分析..."):
        try:
            # 执行探索分析
            if exploration_agent:
                exploration_agent.load_data(neg_file, pos_file)
                result = exploration_agent.run_exploration_pipeline()
                
                if result and 'analysis_results' in result:
                    st.success("探索分析完成。")
                    
                    # 显示结果
                    display_exploration_results('../data/reports/exploration')
                    
                    # 显示报告链接
                    if 'report' in result:
                        st.subheader("探索分析报告")
                        
                        try:
                            with open(result['report'], 'r') as f:
                                report_text = f.read()
                                
                            st.markdown(report_text)
                            
                            # 创建报告下载链接
                            create_download_link(result['report'], "下载探索分析报告")
                        except Exception as e:
                            st.error(f"读取报告文件时出错: {str(e)}")
                            
                    # 返回探索分析结果
                    return result
                else:
                    st.error("探索分析失败")
            else:
                st.error("探索分析组件未初始化")
        except Exception as e:
            st.error(f"探索分析过程中出错: {str(e)}")
            
    return None

def display_exploration_results(results_dir):
    """显示探索分析结果"""
    st.subheader("探索分析结果")
    
    # 检查结果目录是否存在
    if not os.path.exists(results_dir):
        st.error(f"结果目录 {results_dir} 未找到。")
        return
        
    # 查找结果目录中的所有图像文件
    image_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
    
    if not image_files:
        st.warning("未找到结果图像。")
        return
        
    # 按类型分组图像
    gap_dist = [f for f in image_files if 'gap_distribution' in f]
    structure_comparison = [f for f in image_files if 'structural_feature' in f]
    feature_comparisons = [f for f in image_files if '_comparison.png' in f]
    pca_analysis = [f for f in image_files if 'pca_analysis' in f]
    radar_comparison = [f for f in image_files if 'radar_feature' in f]
    
    # 显示能隙分布
    if gap_dist:
        st.markdown("### S1-T1 能隙分布")
        img = Image.open(os.path.join(results_dir, gap_dist[0]))
        st.image(img, caption="S1-T1 能隙分布", use_column_width=True)
        
    # 显示结构比较
    if structure_comparison:
        st.markdown("### 结构特征比较")
        img = Image.open(os.path.join(results_dir, structure_comparison[0]))
        st.image(img, caption="顶级结构特征: 负值 vs 正值 S1-T1 能隙", use_column_width=True)
        
    # 显示雷达图比较
    if radar_comparison:
        st.markdown("### 特征雷达图比较")
        img = Image.open(os.path.join(results_dir, radar_comparison[0]))
        st.image(img, caption="特征比较: 负值 vs 正值 S1-T1 能隙", use_column_width=True)
        
    # 显示 PCA 分析
    if pca_analysis:
        st.markdown("### PCA 分析")
        img = Image.open(os.path.join(results_dir, pca_analysis[0]))
        st.image(img, caption="分子属性 PCA: 负值 vs 正值 S1-T1 能隙", use_column_width=True)
        
    # 显示特征比较
    if feature_comparisons:
        st.markdown("### 特征比较")
        
        # 创建用于显示多个图像的列
        cols = st.columns(2)
        
        for i, file in enumerate(feature_comparisons[:6]):  # 限制为 6 个比较图
            with cols[i % 2]:
                img = Image.open(os.path.join(results_dir, file))
                feature_name = file.replace('_comparison.png', '').replace('_', ' ').title()
                st.image(img, caption=feature_name, use_column_width=True)
                
    # 创建所有结果的下载链接
    create_download_zip(results_dir, "下载所有探索分析结果")

def create_download_link(file_path, text):
    """创建文件下载链接"""
    with open(file_path, 'rb') as f:
        data = f.read()
        
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def create_download_zip(directory, text):
    """创建目录中所有文件的 ZIP 下载链接"""
    # 创建 BytesIO 对象
    zip_buffer = BytesIO()
    
    # 在 BytesIO 对象中创建 ZIP 文件
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.basename(file_path))
                
    # 将缓冲区位置重置到开头
    zip_buffer.seek(0)
    
    # 编码为 base64
    b64 = base64.b64encode(zip_buffer.read()).decode()
    
    # 获取目录名作为 ZIP 文件名
    zip_filename = os.path.basename(directory) + "_results.zip"
    
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">{text}</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def load_exploration_page(exploration_agent=None, feature_agent=None):
    """加载探索分析页面"""
    return render_exploration_page(exploration_agent, feature_agent)

if __name__ == "__main__":
    # 用于直接运行测试
    load_exploration_page()
