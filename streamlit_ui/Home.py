# streamlit_ui/Home.py
import streamlit as st
import os
import base64
from PIL import Image

def render_home_page():
    """渲染首页内容"""
    st.title("反向 TADF 分析系统")
    
    st.markdown("""
    ## 欢迎使用反向 TADF 分子分析系统
    
    本应用帮助您分析分子属性，特别关注具有反向热活化延迟荧光（Reverse TADF）特性的候选分子。
    
    ### 什么是反向 TADF？
    
    在常规分子中，第一三重态激发态（T1）的能量通常低于第一单重态激发态（S1），符合洪特规则。
    然而，在一些特殊分子中，这种排序被颠倒（S1 < T1），创造出具有独特光物理性质的材料，
    这类材料在先进的光电子器件中有重要应用。
    
    ### 系统功能
    
    - **数据提取**：处理 Gaussian 和 CREST 计算输出
    - **特征工程**：生成分子描述符和替代性 3D 特征
    - **探索分析**：分析 S1-T1 能隙特性，识别反向 TADF 候选分子
    - **建模预测**：构建 S1-T1 能隙分类和回归预测模型
    - **洞察生成**：产生量子化学解释和设计原则
    
    ### 开始使用
    
    通过侧边栏菜单导航不同功能：
    
    1. 从**数据提取**页面开始，处理分子计算结果
    2. 转到**特征工程**页面创建和可视化分子描述符
    3. 使用**探索分析**页面识别反向 TADF 候选分子
    4. 探索**建模预测**页面了解预测模型结果
    5. 查看**洞察报告**获取综合分析和设计原则
    """)
    
    # 添加系统架构图
    st.markdown("### 系统架构")
    
    architecture_md = """
    ```
    用户交互 (Streamlit)
         ↓
    任务链 (LangChain)
     ├─> 数据提取 Agent
     │   └─提取 Gaussian 和 CREST 特征数据
     ├─> 特征工程 Agent
     │   └─生成组合特征、极性/共轭/电子效应等
     ├─> 探索分析 Agent
     │   └─筛选 S1-T1 < 0 样本，结构差异分析
     ├─> 建模预测 Agent
     │   └─构建正负 S1-T1 分类或回归模型
     ├─> 洞察生成 Agent
     │   └─基于特征重要性生成解释报告
     └─> UI Agent (Streamlit)
         └─显示图表 + Markdown 解释 + 结果下载
    ```
    """
    
    st.markdown(architecture_md)
    
    # 添加反向 TADF 分子示意图（如果有的话）
    # 这里可以添加一个示意图，展示 S1 < T1 的能级排列
    
    st.markdown("""
    ### 系统工作流程
    
    1. **分子数据提取**：从量子化学计算结果中自动提取关键信息
    2. **特征工程**：基于分子结构和电子属性生成关键特征
    3. **数据分析**：探索性分析发现潜在的反向 TADF 分子
    4. **预测建模**：构建机器学习模型预测 S1-T1 能隙符号和大小
    5. **设计洞察**：生成具有可操作性的分子设计原则
    
    通过侧边栏开始您的分析旅程！
    """)
    
def load_home_page():
    """加载首页"""
    render_home_page()

if __name__ == "__main__":
    load_home_page()
