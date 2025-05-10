# streamlit_ui/TuningPage.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import base64
from PIL import Image
from io import BytesIO
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def render_tuning_page(tuning_agent=None):
    """渲染模型微调页面"""
    st.title("反向TADF分子预测模型微调")
    
    st.markdown("""
    ## 预训练模型微调
    
    本页面用于微调预训练的分子语言模型（如ChemBERTa）以提高S1-T1能隙预测性能。
    
    微调过程:
    
    1. **加载分子数据** - 使用特征工程产生的分子特征和SMILES
    2. **准备训练数据** - 将分子表示转换为模型输入格式
    3. **微调预训练模型** - 基于现有数据调整深度学习模型权重
    4. **评估预测性能** - 测试模型在反向TADF分子上的预测能力
    
    您可以选择预训练模型类型、训练参数和评估指标。
    """)
    
    # 创建侧边栏设置
    st.sidebar.header("微调参数")
    
    # 选择预训练模型
    pretrained_models = {
        "ChemBERTa-zinc-base": "seyonec/ChemBERTa-zinc-base-v1",
        "ChemBERTa-77M-MTR": "DeepChem/ChemBERTa-77M-MTR",
        "MolT5": "laituan245/molt5-base",
        "MolBERT": "seyonec/molbert_100m"
    }
    
    model_name = st.sidebar.selectbox(
        "选择预训练模型",
        list(pretrained_models.keys())
    )
    
    model_path = pretrained_models[model_name]
    
    # 数据选择
    data_source = st.sidebar.radio(
        "数据来源",
        ["使用特征工程结果", "上传新数据"]
    )
    
    feature_file = None
    
    if data_source == "使用特征工程结果":
        # 查找可用的特征文件
        extracted_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/extracted'
        if os.path.exists(extracted_dir):
            feature_files = [f for f in os.listdir(extracted_dir) 
                            if ('feature' in f.lower() or 'processed' in f.lower()) 
                            and f.endswith('.csv')]
            
            if feature_files:
                selected_file = st.sidebar.selectbox(
                    "选择特征文件",
                    feature_files
                )
                feature_file = os.path.join(extracted_dir, selected_file)
            else:
                st.sidebar.warning("未找到特征文件。请先运行特征工程。")
        else:
            st.sidebar.warning("未找到提取的数据目录。请先提取数据并运行特征工程。")
    else:
        # 上传文件选项
        uploaded_file = st.sidebar.file_uploader("上传特征CSV文件", type="csv")
        if uploaded_file:
            # 保存上传的CSV到临时位置
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                feature_file = temp_file.name
    
    # 训练参数
    st.sidebar.subheader("训练参数")
    
    batch_size = st.sidebar.select_slider(
        "批次大小",
        options=[4, 8, 16, 32, 64],
        value=16
    )
    
    epochs = st.sidebar.slider(
        "训练轮数",
        min_value=1,
        max_value=30,
        value=5
    )
    
    learning_rate = st.sidebar.select_slider(
        "学习率",
        options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005],
        value=0.0001,
        format_func=lambda x: f"{x:.5f}"
    )
    
    # 目标列
    target_col = st.sidebar.selectbox(
        "目标预测列",
        ["s1_t1_gap_ev", "is_negative_gap"],
        index=0
    )
    
    # 分子列
    smiles_col = st.sidebar.text_input(
        "分子列名",
        "Molecule"
    )
    
    # 执行微调
    run_button = st.sidebar.button("运行微调")
    
    if run_button:
        if tuning_agent is None:
            st.error("微调代理未初始化。请检查系统配置。")
        elif feature_file is None:
            st.error("请选择或上传特征文件。")
        else:
            # 设置微调代理参数
            tuning_agent.model_name = model_path
            
            # 执行微调
            with st.spinner("正在微调模型...这可能需要几分钟时间..."):
                result = tuning_agent.run_tuning_pipeline(
                    feature_file=feature_file,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    target_col=target_col,
                    smiles_col=smiles_col
                )
                
                if result:
                    st.success("模型微调完成！")
                    display_tuning_results(result)
                else:
                    st.error("模型微调失败。请检查日志以获取更多信息。")
    
    # 如果还没有运行微调，显示一些信息
    if not run_button:
        display_tuning_info()

def display_tuning_results(result):
    """显示模型微调结果"""
    metrics = result.get('metrics', {})
    model_path = result.get('model_path')
    
    st.subheader("微调性能")
    
    # 创建选项卡
    tabs = st.tabs(["性能指标", "学习曲线", "SMILES表示"])
    
    # 性能指标选项卡
    with tabs[0]:
        if metrics:
            # 创建性能指标表格
            st.write("模型评估指标:")
            metrics_df = pd.DataFrame({
                "指标": list(metrics.keys()),
                "值": list(metrics.values())
            })
            st.dataframe(metrics_df)
            
            # 显示关键指标
            cols = st.columns(2)
            
            with cols[0]:
                if 'rmse' in metrics:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                if 'mae' in metrics:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                    
            with cols[1]:
                if 'r2' in metrics:
                    st.metric("R²", f"{metrics['r2']:.4f}")
                if 'accuracy' in metrics:
                    st.metric("准确率", f"{metrics['accuracy']:.4f}")
        else:
            st.write("未找到性能指标")
    
    # 学习曲线选项卡
    with tabs[1]:
        if 'history' in result:
            history = result['history']
            
            # 绘制损失曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history['train_loss'], label='训练损失')
            ax.plot(history['val_loss'], label='验证损失')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('训练和验证损失')
            ax.legend()
            st.pyplot(fig)
            
            # 绘制指标曲线
            if 'train_rmse' in history:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(history['train_rmse'], label='训练RMSE')
                ax2.plot(history['val_rmse'], label='验证RMSE')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('RMSE')
                ax2.set_title('训练和验证RMSE')
                ax2.legend()
                st.pyplot(fig2)
        else:
            st.write("未找到学习曲线数据")
    
    # SMILES表示选项卡
    with tabs[2]:
        st.write("SMILES分子表示可视化")
        
        if 'embeddings' in result:
            embeddings = result['embeddings']
            
            # 创建PCA或t-SNE可视化
            if 'pca' in embeddings:
                st.image(embeddings['pca'], caption="PCA投影的分子嵌入")
                
            if 'tsne' in embeddings:
                st.image(embeddings['tsne'], caption="t-SNE投影的分子嵌入")
                
            if not embeddings:
                st.write("未找到分子嵌入可视化")
        else:
            st.write("未找到分子嵌入数据")
    
    # 显示模型路径
    if model_path:
        st.subheader("保存的模型")
        st.write(f"模型已保存到: {model_path}")
        
        # 添加到全局模型列表（如果适用）
        st.success("微调的模型已添加到可用模型列表，可以在分子设计页面中使用。")

def display_tuning_info():
    """显示模型微调相关信息"""
    st.subheader("深度学习模型微调")
    
    st.markdown("""
    ### 为什么使用预训练模型？

    预训练分子模型如ChemBERTa已经在数百万分子上学习了分子表示能力，可以：

    1. **提高少样本学习**：使用少量反向TADF数据也能获得不错的性能
    2. **捕获复杂模式**：理解分子中原子和键的复杂关系
    3. **泛化能力**：更好地预测未见过的新型分子结构
    
    ### 微调过程

    1. **加载预训练模型**：初始化预训练的分子语言模型
    2. **准备数据**：将SMILES转换为模型可理解的格式
    3. **添加预测层**：在模型顶部添加回归/分类层
    4. **训练**：使用S1-T1能隙数据优化模型权重
    5. **评估**：测试模型在未见过的分子上的性能
    
    ### 建议参数选择
    
    * **批次大小**：小数据集建议使用较小批次(8-16)
    * **训练轮数**：5-10轮通常足够，避免过拟合
    * **学习率**：较小的学习率(1e-5至1e-4)通常效果更好
    
    ### 注意事项
    
    * 确保特征文件包含正确的分子SMILES和S1-T1能隙值
    * 数据量越大，微调效果越好
    * 可以尝试不同的预训练模型以获得最佳性能
    """)
    
    # 添加图示
    st.subheader("模型架构示意图")
    
    # 这里可以添加模型架构图（如果有的话）
    
    # 显示示例模型性能比较
    st.subheader("模型性能比较示例")
    
    # 创建示例性能对比
    model_comparison = pd.DataFrame({
        "模型": ["随机森林", "XGBoost", "ChemBERTa (未微调)", "ChemBERTa (微调)"],
        "RMSE": [0.32, 0.28, 0.25, 0.17],
        "R²": [0.68, 0.72, 0.75, 0.83]
    })
    
    st.table(model_comparison)

def load_tuning_page(tuning_agent=None):
    """加载模型微调页面"""
    return render_tuning_page(tuning_agent)
# 在显示微调性能的函数中，添加学习曲线显示

def display_tuning_results(tuning_results):
    st.subheader("模型微调性能")
    
    # 创建选项卡
    tabs = st.tabs(["性能指标", "学习曲线", "SMILES表示"])
    
    # 性能指标选项卡
    with tabs[0]:
        if 'metrics' in tuning_results and tuning_results['metrics']:
            metrics = tuning_results['metrics']
            
            # 显示指标表
            metrics_data = {
                '指标': ['name', 'value', 'file', 'importance_plot', 'prediction_plot'],
                '值': [
                    metrics.get('name', ''),
                    metrics.get('value', ''),
                    metrics.get('file', ''),
                    metrics.get('importance_plot', ''),
                    metrics.get('prediction_plot', '')
                ]
            }
            
            st.dataframe(pd.DataFrame(metrics_data))
            
            # 显示图表
            if 'importance_plot' in metrics and metrics['importance_plot']:
                st.image(metrics['importance_plot'], caption="特征重要性", use_column_width=True)
                
            if 'prediction_plot' in metrics and metrics['prediction_plot']:
                st.image(metrics['prediction_plot'], caption="预测 vs 实际", use_column_width=True)
        else:
            st.warning("未找到性能指标")
    
    # 学习曲线选项卡
    with tabs[1]:
        if 'metrics' in tuning_results and tuning_results['metrics'] and 'learning_curve_plot' in tuning_results['metrics']:
            # 显示学习曲线图
            st.image(tuning_results['metrics']['learning_curve_plot'], caption="学习曲线", use_column_width=True)
            
            # 可以选择性地显示原始数据
            if 'learning_curve_data' in tuning_results['metrics']:
                with st.expander("显示学习曲线原始数据"):
                    curve_data = tuning_results['metrics']['learning_curve_data']
                    chart_data = pd.DataFrame({
                        '训练集大小 (%)': [x * 100 for x in curve_data['train_sizes']],
                        '训练集得分': curve_data['train_scores'],
                        '测试集得分': curve_data['test_scores']
                    })
                    st.dataframe(chart_data)
        else:
            st.warning("未找到学习曲线数据")
    
    # SMILES表示选项卡
    with tabs[2]:
        if 'smiles_col' in tuning_results and tuning_results['smiles_col']:
            st.info(f"使用的分子列: {tuning_results['smiles_col']}")
            # 这里可以添加关于SMILES表示的更多信息
        else:
            st.warning("未找到SMILES分子表示")

if __name__ == "__main__":
    # 用于直接运行测试
    load_tuning_page()
