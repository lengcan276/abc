# streamlit_ui/ExtractorPage.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import zipfile
import shutil
import base64
from io import BytesIO

def render_extractor_page(data_agent=None):
    """渲染数据提取页面"""
    st.title("数据提取")
    
    st.markdown("""
    ## Gaussian 和 CREST 数据提取
    
    上传 Gaussian 日志文件和 CREST 结果以提取分子属性。
    
    ### 预期的数据结构
    
    系统期望分子计算具有特定的目录结构：
    
    ```
    父目录/
    ├── 分子名称/
    │   ├── neutral/（中性态）
    │   │   └── gaussian/
    │   │       └── conf_1/
    │   │           ├── ground.log
    │   │           └── excited.log
    │   ├── cation/（阳离子态）
    │   │   └── gaussian/...
    │   ├── triplet/（三重态）
    │   │   └── gaussian/...
    │   └── results/
    │       ├── neutral_results.txt
    │       ├── cation_results.txt
    │       └── triplet_results.txt
    ```
    
    您可以上传包含此结构的 ZIP 文件或提供服务器上的目录路径。
    """)
    
    # 提供示例数据结构信息
    with st.expander("查看更多关于数据格式的信息"):
        st.markdown("""
        ### 文件内容要求
        
        #### Gaussian 日志文件 (.log)
        - **ground.log**: 包含基态优化和能量计算结果
        - **excited.log**: 包含激发态计算结果（TD-DFT）
        
        #### CREST 结果文件
        - **neutral_results.txt**: 中性态构象搜索结果
        - **cation_results.txt**: 阳离子态构象搜索结果
        - **triplet_results.txt**: 三重态构象搜索结果
        
        系统将提取以下关键信息：
        - HF 能量值
        - HOMO-LUMO 能隙
        - 偶极矩
        - Mulliken 电荷分布
        - S1-T1 能隙
        - 振子强度
        - 构象灵活性指标
        """)
    
    # 上传 ZIP 文件选项
    uploaded_file = st.file_uploader("上传包含分子数据的 ZIP 文件", type="zip")
    
    # 提供目录路径选项
    directory_path = st.text_input("或提供服务器上的目录路径")
    
    # 执行提取
    extract_button = st.button("提取数据")
    
    if extract_button:
        if uploaded_file or directory_path:
            with st.spinner("正在提取数据..."):
                try:
                    if uploaded_file:
                        # 将上传的 ZIP 保存到临时位置并解压
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
                            temp_zip.write(uploaded_file.getvalue())
                            temp_zip_path = temp_zip.name
                            
                        # 创建临时目录用于解压
                        temp_dir = tempfile.mkdtemp()
                        
                        # 解压 ZIP 到临时目录
                        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                            
                        # 使用临时目录路径
                        effective_path = temp_dir
                        
                    else:
                        # 使用提供的目录路径
                        effective_path = directory_path
                    
                    # 执行数据提取（假设 data_agent 已经传入）
                    if data_agent:
                        data_agent.base_dir = effective_path
                        result_file = data_agent.process_molecules()
                        
                        if result_file:
                            # 显示结果
                            st.success(f"数据提取完成，结果保存至 {result_file}")
                            
                            # 加载并显示数据摘要
                            df = pd.read_csv(result_file)
                            st.write(f"已提取 {df['Molecule'].nunique()} 个分子的数据")
                            st.write(f"总构象数: {len(df)}")
                            
                            # 显示样本数据
                            st.subheader("样本数据")
                            st.dataframe(df.head())
                            
                            # 创建下载链接
                            create_download_link(result_file, "下载提取的数据 CSV")
                            
                            # 清理临时文件（如果使用上传的 ZIP）
                            if uploaded_file:
                                os.unlink(temp_zip_path)
                                shutil.rmtree(temp_dir)
                                
                            # 返回提取结果供后续使用
                            return result_file
                        else:
                            st.error("数据提取失败")
                    else:
                        st.error("数据提取组件未初始化")
                except Exception as e:
                    st.error(f"数据提取过程中出错: {str(e)}")
                    # 确保清理临时文件
                    if uploaded_file and 'temp_zip_path' in locals() and 'temp_dir' in locals():
                        try:
                            os.unlink(temp_zip_path)
                            shutil.rmtree(temp_dir)
                        except:
                            pass
        else:
            st.error("请上传 ZIP 文件或提供目录路径")
            
    return None

def create_download_link(file_path, text):
    """创建文件下载链接"""
    with open(file_path, 'rb') as f:
        data = f.read()
        
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def load_extractor_page(data_agent=None):
    """加载数据提取页面"""
    return render_extractor_page(data_agent)

if __name__ == "__main__":
    # 用于直接运行测试
    load_extractor_page()
