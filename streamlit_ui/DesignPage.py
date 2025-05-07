# streamlit_ui/DesignPage.py
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
from rdkit import Chem
from rdkit.Chem import Draw

def render_design_page(design_agent=None, model_agent=None):
    """渲染分子设计页面"""
    st.title("反向TADF分子设计")
    
    st.markdown("""
    ## 分子设计与优化
    
    本页面使用强化学习方法设计具有负S1-T1能隙的新型反向TADF分子。
    
    系统基于以下原理：
    
    1. **启发式搜索** - 使用已知有效结构（如Calicene）作为起点
    2. **目标导向探索** - 优化分子结构以获得理想的S1-T1能隙值
    3. **结构-性能关系** - 应用从探索和建模中学到的规律
    
    您可以选择使用传统机器学习模型或微调的深度学习模型进行分子设计。
    """)
    
    # 创建侧边栏设置
    st.sidebar.header("设计参数")
    
    # 检查模型是否已自动加载
    model_already_loaded = False
    if design_agent and design_agent.predictive_model is not None:
        model_already_loaded = True
        # 在日志中记录已加载的模型信息
        st.sidebar.success("系统已自动加载预测模型")
    
    # 选择模型类型
    model_type = st.sidebar.radio(
        "选择使用的预测模型",
        ["传统机器学习模型", "微调深度学习模型", "两者结合"]
    )
    
    # 设置目标S1-T1能隙范围
    target_gap = st.sidebar.slider(
        "目标S1-T1能隙 (eV)",
        min_value=-0.5,
        max_value=0.0,
        value=-0.1,
        step=0.01,
        help="选择期望的S1-T1能隙值，负值表示反向TADF特性"
    )
    
    # 选择起始骨架
    scaffold_options = {
        "Calicene": "C1=CC=CC=1C=C1C=C1",
        "Azulene": "c1ccc2cccc-2cc1",
        "Heptazine": "c1nc2nc3nc(nc3nc2n1)n1c2nc3nc(nc3nc2n1)n1c2nc3nc(nc3nc2n1)n1",
        "自定义": "custom"
    }
    
    scaffold_selection = st.sidebar.selectbox(
        "选择起始骨架",
        list(scaffold_options.keys())
    )
    
    scaffold = None
    if scaffold_selection == "自定义":
        scaffold = st.sidebar.text_input("输入自定义SMILES", "")
    else:
        scaffold = scaffold_options[scaffold_selection]
        
        # 显示所选骨架的结构
        if scaffold:
            mol = Chem.MolFromSmiles(scaffold)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 200))
                st.sidebar.image(img, caption=f"{scaffold_selection}结构")
    
    # 生成分子数量
    n_samples = st.sidebar.slider(
        "生成分子数量",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )
    
    # 模型路径
    models_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/models/'
    
    # 检查可用模型
    available_ml_models = []
    available_dl_models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.joblib'):
                available_ml_models.append(file)
            elif file.endswith('.pt') or file.endswith('.pth') or os.path.isdir(os.path.join(models_dir, file)):
                available_dl_models.append(file)
    
    # 选择模型
    ml_model_path = None
    dl_model_path = None
    
    # 如果已经自动加载了模型，显示已加载的模型信息
    if model_already_loaded:
        st.sidebar.info("系统已自动加载分类模型或回归模型。可以直接运行分子设计。")
        # 在此处不需要选择模型，因为已经加载了
        # 设置ml_model_path为一个特殊标志，表示已加载模型
        ml_model_path = "AUTO_LOADED"
    else:
        # 原始的模型选择逻辑
        if model_type in ["传统机器学习模型", "两者结合"]:
            if available_ml_models:
                ml_model = st.sidebar.selectbox(
                    "选择传统机器学习模型",
                    available_ml_models
                )
                ml_model_path = os.path.join(models_dir, ml_model)
            else:
                st.sidebar.warning("未找到可用的传统机器学习模型。请先运行建模页面训练模型。")
        
        if model_type in ["微调深度学习模型", "两者结合"]:
            if available_dl_models:
                dl_model = st.sidebar.selectbox(
                    "选择微调深度学习模型",
                    available_dl_models
                )
                dl_model_path = os.path.join(models_dir, dl_model)
            else:
                st.sidebar.warning("未找到可用的微调深度学习模型。请先运行微调页面训练模型。")
    
    # 执行设计
    run_button = st.sidebar.button("运行分子设计")
    
    # 在主页面区域显示模型状态
    if model_already_loaded:
        st.success("系统已自动加载传统机器学习模型，可以直接进行分子设计。")
    else:
        st.warning("未选择传统机器学习模型。")
    
    if run_button:
        if design_agent is None:
            st.error("设计代理未初始化。请检查系统配置。")
        # 允许在已自动加载模型的情况下运行
        elif not model_already_loaded and model_type == "传统机器学习模型" and ml_model_path is None:
            st.error("未选择传统机器学习模型。")
        elif not model_already_loaded and model_type == "微调深度学习模型" and dl_model_path is None:
            st.error("未选择微调深度学习模型。")
        elif not model_already_loaded and model_type == "两者结合" and (ml_model_path is None or dl_model_path is None):
            st.error("需要同时选择传统机器学习模型和微调深度学习模型。")
        else:
            # 执行分子设计
            with st.spinner("正在设计分子...这可能需要几分钟时间..."):
                # 只有在没有自动加载模型的情况下才执行加载操作
                if not model_already_loaded:
                    # 初始化设计代理
                    if model_type == "传统机器学习模型":
                        design_agent.load_predictive_model(ml_model_path)
                    elif model_type == "微调深度学习模型":
                        design_agent.load_fine_tuned_model(dl_model_path)
                    else:  # 两者结合
                        design_agent.load_predictive_model(ml_model_path)
                        design_agent.load_fine_tuned_model(dl_model_path)
                
                # 运行设计流程
                result = design_agent.run_design_pipeline(
                    target_gap=target_gap,
                    scaffold=scaffold,
                    n_samples=n_samples
                )
                
                if result:
                    st.success("分子设计完成！")
                    display_design_results(result)
                else:
                    st.error("分子设计失败。请检查日志以获取更多信息。")
    
    # 如果还没有运行设计，显示一些背景信息
    if not run_button:
        display_design_info()

def display_design_results(result):
    """显示分子设计结果"""
    molecules = result.get('molecules', [])
    results_df = result.get('results_df')
    
    if not molecules:
        st.warning("未生成任何分子。")
        return
    
    st.write(f"成功生成 {len(molecules)} 个分子")
    
    # 创建选项卡
    tabs = st.tabs(["分子结构", "性能分布", "数据表"])
    
    # 分子结构选项卡
    with tabs[0]:
        st.subheader("生成的分子结构")
        
        # 获取前10个分子并显示
        top_mols = molecules[:min(10, len(molecules))]
        
        # 每行3个分子
        cols = st.columns(3)
        for i, mol_data in enumerate(top_mols):
            mol = mol_data['mol']
            gap = mol_data['predicted_gap']
            
            col_idx = i % 3
            with cols[col_idx]:
                try:
                    # 创建分子图像
                    img = Draw.MolToImage(mol, size=(200, 200))
                    smiles = mol_data['smiles']
                    
                    # 显示分子和预测的能隙
                    st.image(img)
                    st.caption(f"S1-T1 能隙: {gap:.3f} eV")
                    st.code(smiles, language="text")
                except Exception as e:
                    st.error(f"无法显示分子 #{i+1}: {e}")
    
    # 性能分布选项卡
    with tabs[1]:
        st.subheader("分子性能分布")
        
        if results_df is not None:
            # 创建S1-T1能隙分布图
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.histplot(data=results_df, x='predicted_gap', kde=True, ax=ax1)
            ax1.axvline(x=0, color='red', linestyle='--')
            ax1.set_title('S1-T1 能隙分布')
            ax1.set_xlabel('S1-T1 能隙 (eV)')
            st.pyplot(fig1)
            
            # 创建性能关系散点图
            if 'molecular_weight' in results_df.columns and 'donor_groups' in results_df.columns:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                scatter = sns.scatterplot(
                    data=results_df, 
                    x='molecular_weight', 
                    y='donor_groups',
                    hue='predicted_gap',
                    palette='coolwarm',
                    ax=ax2
                )
                ax2.set_title('结构-性能关系')
                ax2.set_xlabel('分子量')
                ax2.set_ylabel('给电子基团数量')
                st.pyplot(fig2)
            
            # 如果有其他图表，从文件加载
            report_dir = "/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports"
            if os.path.exists(os.path.join(report_dir, 'generated_gap_distribution.png')):
                st.image(os.path.join(report_dir, 'generated_gap_distribution.png'))
            
            if os.path.exists(os.path.join(report_dir, 'generated_property_space.png')):
                st.image(os.path.join(report_dir, 'generated_property_space.png'))
            
            if os.path.exists(os.path.join(report_dir, 'top_generated_molecules.png')):
                st.image(os.path.join(report_dir, 'top_generated_molecules.png'))
    
    # 数据表选项卡
    with tabs[2]:
        st.subheader("生成分子数据")
        
        if results_df is not None:
            st.dataframe(results_df)
            
            # 创建下载链接
            csv_path = result.get('report_path')
            if csv_path and os.path.exists(csv_path):
                create_download_link(csv_path, "下载生成分子CSV数据")

def display_design_info():
    """显示分子设计背景信息"""
    st.subheader("反向TADF分子设计原理")
    
    st.markdown("""
    ### 设计策略

    根据最新研究，有效的反向TADF分子设计策略包括：

    1. **Calicene衍生物设计**：在Calicene的三元环上添加强吸电子基团（如-CN），在五元环上添加强给电子基团（如-NMe₂）
    
    2. **轨道调控**：通过取代基调节HOMO和LUMO能级，使它们的重叠减小
    
    3. **电荷转移状态**：创建从给电子基团到吸电子基团的电荷转移状态
    
    ### 关键结构特征

    1. **推拉体系**：分子需要明确的推拉电子体系
    2. **特定环系统**：三元环和五元环的组合是实现负S1-T1能隙的关键
    3. **取代基位置**：取代基位置对轨道能级和重叠有显著影响
    
    ### 设计过程
    
    本系统使用强化学习方法，通过以下步骤设计反向TADF分子：
    
    1. 从基本骨架开始（默认为Calicene）
    2. 通过添加、移除和修改取代基逐步优化分子
    3. 使用预测模型评估每个变化对S1-T1能隙的影响
    4. 根据设计目标给予奖励或惩罚，引导优化方向
    5. 生成多个候选分子并评估其性能
    """)

    # 显示示例图像
    st.subheader("示例分子结构")
    
    example_mols = [
        ("Calicene", "C1=CC=CC=1C=C1C=C1"),
        ("CN-Calicene-NMe2", "C1=C(C#N)C=C1C=C1C(N(C)C)=CC=C1"),
        ("反向TADF候选物", "C1=C(C#N)C=C1C=C1C(N(P(C)(C)C))=CC=C1")
    ]
    
    cols = st.columns(len(example_mols))
    for i, (name, smiles) in enumerate(example_mols):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            with cols[i]:
                img = Draw.MolToImage(mol, size=(200, 200))
                st.image(img)
                st.caption(name)

def create_download_link(file_path, text):
    """创建文件下载链接"""
    with open(file_path, 'rb') as f:
        data = f.read()
        
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def load_design_page(design_agent=None, model_agent=None):
    """加载分子设计页面"""
    return render_design_page(design_agent, model_agent)

if __name__ == "__main__":
    # 用于直接运行测试
    load_design_page()