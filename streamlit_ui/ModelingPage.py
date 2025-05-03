# streamlit_ui/ModelingPage.py
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

def render_modeling_page(model_agent=None):
    """渲染建模分析页面"""
    st.title("预测建模")
    
    st.markdown("""
    ## S1-T1 能隙预测建模
    
    本页面专注于构建和评估 S1-T1 能隙属性的预测模型。
    
    构建了两种主要模型：
    
    1. **分类模型** - 预测分子是否具有负值或正值 S1-T1 能隙
    2. **回归模型** - 预测 S1-T1 能隙的实际值
    
    分析包括：
    
    - 特征选择和重要性排名
    - 模型性能评估
    - 特征工程洞察
    - 预测可视化
    
    您可以在先前生成的特征数据上运行建模流程，或上传特征 CSV 文件。
    """)
    
    # 使用现有数据或上传新数据的选项
    feature_file = None
    
    # 查找先前处理的数据
    extracted_dir = '../data/extracted'
    if os.path.exists(extracted_dir):
        # 查找处理过的特征文件
        feature_files = [f for f in os.listdir(extracted_dir) if ('feature' in f.lower() or 'processed' in f.lower()) and f.endswith('.csv')]
        
        if feature_files:
            st.info("找到现有特征文件。")
            selected_file = st.selectbox("选择用于建模的特征文件", feature_files)
            feature_file = os.path.join(extracted_dir, selected_file)
        else:
            st.warning("未找到特征文件。请先运行特征工程。")
    else:
        st.warning("未找到提取的数据目录。请先提取数据并运行特征工程。")
        
    # 上传文件选项
    if not feature_file:
        st.subheader("上传特征数据")
        
        feature_upload = st.file_uploader("上传处理后的特征 CSV", type="csv")
        
        if feature_upload:
            # 将上传的 CSV 保存到临时位置
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                temp_file.write(feature_upload.getvalue())
                feature_file = temp_file.name
                
    # 执行建模
    if feature_file:
        # 检查是否存在预计算结果
        results_dir = '../data/reports/modeling'
        models_dir = '../data/models'
        
        if os.path.exists(results_dir) and os.path.exists(models_dir) and \
           len(os.listdir(results_dir)) > 0 and len(os.listdir(models_dir)) > 0:
            st.info("找到现有的建模结果。")
            
            if st.button("显示建模结果"):
                display_modeling_results(results_dir)
                
            if st.button("重新运行建模"):
                run_modeling_analysis(feature_file, model_agent)
        else:
            if st.button("运行建模分析"):
                run_modeling_analysis(feature_file, model_agent)
                
    return None

def run_modeling_analysis(feature_file, model_agent):
    """运行建模分析并显示结果"""
    with st.spinner("正在运行建模分析..."):
        try:
            # 执行建模分析
            if model_agent:
                model_agent.feature_file = feature_file
                result = model_agent.run_modeling_pipeline()
                
                if result and ('classification' in result or 'regression' in result):
                    st.success("建模分析完成。")
                    
                    # 显示结果
                    display_modeling_results('../data/reports/modeling')
                    
                    # 返回建模结果
                    return result
                else:
                    st.error("建模分析失败")
            else:
                st.error("建模组件未初始化")
        except Exception as e:
            st.error(f"建模分析过程中出错: {str(e)}")
            
    return None

def display_modeling_results(results_dir):
    """显示建模分析结果"""
    st.subheader("建模分析结果")
    
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
    classification_images = [f for f in image_files if 'classification' in f or 'confusion_matrix' in f]
    regression_images = [f for f in image_files if 'regression' in f]
    feature_rank_images = [f for f in image_files if 'feature_ranks' in f]
    
    # 创建分类和回归的选项卡
    tabs = st.tabs(["分类模型", "回归模型", "特征选择"])
    
    # 分类选项卡
    with tabs[0]:
        st.markdown("### 分类模型结果")
        
        if classification_images:
            for file in classification_images:
                img = Image.open(os.path.join(results_dir, file))
                caption = file.replace('.png', '').replace('_', ' ').title()
                st.image(img, caption=caption, use_column_width=True)
        else:
            st.warning("未找到分类模型结果。")
            
    # 回归选项卡
    with tabs[1]:
        st.markdown("### 回归模型结果")
        
        if regression_images:
            for file in regression_images:
                img = Image.open(os.path.join(results_dir, file))
                caption = file.replace('.png', '').replace('_', ' ').title()
                st.image(img, caption=caption, use_column_width=True)
        else:
            st.warning("未找到回归模型结果。")
            
    # 特征选择选项卡
    with tabs[2]:
        st.markdown("### 特征选择结果")
        
        if feature_rank_images:
            for file in feature_rank_images:
                img = Image.open(os.path.join(results_dir, file))
                target = file.replace('feature_ranks_', '').replace('.png', '')
                st.markdown(f"#### {target} 的特征重要性")
                st.image(img, use_column_width=True)
        else:
            st.warning("未找到特征选择结果。")
            
    # 检查模型文件
    models_dir = '../data/models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') or f.endswith('.pkl')]
        
        if model_files:
            st.subheader("训练好的模型")
            
            for file in model_files:
                model_path = os.path.join(models_dir, file)
                create_download_link(model_path, f"下载 {file}")
                
    # 创建所有结果的下载链接
    create_download_zip(results_dir, "下载所有建模结果")

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
    
def load_modeling_page(model_agent=None):
    """加载建模页面"""
    return render_modeling_page(model_agent)

if __name__ == "__main__":
    # 用于直接运行测试
    load_modeling_page()
