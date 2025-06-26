# streamlit_ui/PaperPage.py
import streamlit as st
import os
import base64
import tempfile
import time
import shutil
from io import BytesIO
import sys
import warnings


# 在导入前显示警告但继续执行
warnings.filterwarnings('ignore', message='.*numpy.dtype size changed.*')
warnings.filterwarnings('ignore', message='.*binary incompatibility.*')

# 尝试依次导入各个包，并提供回退机制
try:
    from PIL import Image
    pil_available = True
except ImportError:
    pil_available = False
    st.warning("PIL/Pillow 库不可用，图像预览功能将受限。")

try:
    import pandas as pd
    pandas_available = True
except ImportError:
    pandas_available = False
    st.warning("Pandas 库不可用，数据处理功能将受限。")

try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False
    st.warning("NumPy 库不可用，数据分析功能将受限。")

# 由于 matplotlib 和 seaborn 依赖 numpy，先检查 numpy 再导入
viz_available = False
if numpy_available:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        viz_available = True
    except ImportError:
        st.warning("Matplotlib/Seaborn 库不可用，可视化功能将受限。")

try:
    import markdown
    markdown_available = True
except ImportError:
    markdown_available = False
    st.warning("Markdown 库不可用，可能影响文档格式转换。")

# 尝试导入 WeasyPrint 和 pypandoc
try:
    from weasyprint import HTML
    weasyprint_available = True
except (ImportError, OSError) as e:
    weasyprint_available = False
    st.warning(f"WeasyPrint 库不可用: {str(e)}。PDF 生成将使用替代方法。")

try:
    import pypandoc
    pypandoc_available = True
except (ImportError, OSError):
    pypandoc_available = False
    st.warning("Pypandoc 不可用。DOCX 生成将使用替代方法。")

def render_paper_page(paper_agent=None, model_agent=None, exploration_agent=None, insight_agent=None, multi_model_agent=None):
    """渲染论文生成页面"""
    st.title("学术论文生成")
    
    st.markdown("""
    ## 学术论文生成
    
    本页面允许您基于反向 TADF 分析的结果生成完整的学术论文。
    
    论文将包括：
    
    1. **引言** - 反向 TADF 的背景及其重要性
    2. **方法** - 计算方法和分析技术
    3. **结果** - 探索和模型构建的主要发现
    4. **讨论** - 结果解释与设计原则
    5. **结论** - 总结和未来方向
    
    您可以自定义论文的各个方面并以多种格式下载。
    """)
    
    # 提供环境信息按钮
    if st.button("显示环境信息", help="显示当前 Python 环境和依赖库的状态"):
        show_environment_info()
    
    # 添加多模型选择
    st.sidebar.header("AI模型选择")
    
    use_multi_model = st.sidebar.checkbox("使用多模型协同生成", value=False, help="启用多AI模型协同工作生成高质量论文")
    api_keys = {}
    
    if use_multi_model:
        with st.sidebar.expander("AI模型详情", expanded=False):
            st.info("""
            多模型协同将利用不同AI模型的专长生成更高质量的论文：
            
            - **Claude 3.5 Sonnet**: 负责高质量学术写作，生成引言和参考文献
            - **DeepSeek R1**: 负责数据分析和推理，生成方法部分
            - **GLM + Kimi k1.5**: 负责结果和讨论部分，生成数据图表解释
            - **OpenAI o3 mini**: 负责生成摘要和结论
            """)
            
        with st.sidebar.expander("配置API密钥（可选）"):
            api_keys['anthropic'] = st.text_input("Anthropic API Key", type="password", help="用于Claude 3.5 Sonnet")
            api_keys['openai'] = st.text_input("OpenAI API Key", type="password", help="用于o3 mini")
            api_keys['deepseek'] = st.text_input("DeepSeek API Key", type="password", help="用于DeepSeek R1")
            api_keys['kimi'] = st.text_input("Kimi API Key", type="password", help="用于Kimi k1.5")
            api_keys['glm'] = st.text_input("GLM API Key", type="password", help="用于GLM-4")
    
    # 检查是否有可用的图表和结果
<<<<<<< HEAD
    exploration_results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/exploration'
    modeling_results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
=======
    exploration_results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/exploration'
    modeling_results_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/reports/modeling'
>>>>>>> 0181d62 (update excited)
    has_exploration = os.path.exists(exploration_results_dir) and len(os.listdir(exploration_results_dir)) > 0
    has_modeling = os.path.exists(modeling_results_dir) and len(os.listdir(modeling_results_dir)) > 0
    
    if not (has_exploration or has_modeling):
        st.warning("未找到探索分析或建模结果。建议先运行这些分析以生成包含图表的论文。")
    else:
        exploration_figures = []
        modeling_figures = []
        
        # 收集可用图表
        if has_exploration:
            exploration_figures = [os.path.join(exploration_results_dir, f) for f in os.listdir(exploration_results_dir) if f.endswith('.png')]
            st.success(f"找到 {len(exploration_figures)} 个探索分析图表可用于论文")
            
        if has_modeling:
            modeling_figures = [os.path.join(modeling_results_dir, f) for f in os.listdir(modeling_results_dir) if f.endswith('.png')]
            st.success(f"找到 {len(modeling_figures)} 个建模分析图表可用于论文")
        
        # 显示可用图表预览
        if exploration_figures or modeling_figures:
            with st.expander("预览可用图表", expanded=False):
                all_figures = exploration_figures + modeling_figures
                selected_figures = st.multiselect(
                    "选择要包含在论文中的图表",
                    options=all_figures,
                    default=all_figures[:min(5, len(all_figures))],
                    format_func=lambda x: os.path.basename(x)
                )
                
                # 显示选中的图表预览
                if selected_figures and pil_available:
                    cols = st.columns(2)
                    for i, fig_path in enumerate(selected_figures):
                        try:
                            with cols[i % 2]:
                                img = Image.open(fig_path)
                                st.image(img, caption=os.path.basename(fig_path), use_column_width=True)
                        except Exception as e:
                            st.error(f"无法加载图像 {os.path.basename(fig_path)}: {str(e)}")
                elif selected_figures and not pil_available:
                    st.info("图像预览需要 PIL/Pillow 库。请安装该库以启用图像预览。")
    
    # 论文信息输入
    st.subheader("论文信息")
    
    with st.form("paper_form"):
        title = st.text_input("论文标题", "反向 TADF 分子设计：反转激发态能量排序的计算分析")
        
        authors_input = st.text_input("作者 (逗号分隔)", "作者1, 作者2, 作者3")
        
        abstract = st.text_area("摘要", "This research presents a computational framework for studying reverse Thermally Activated Delayed Fluorescence (TADF) materials, where the energy of the first excited singlet state (S1) is lower than that of the first excited triplet state (T1). Through quantum chemical calculations and machine learning analysis, we identified key molecular descriptors controlling this unusual energy ordering and proposed design principles for developing new reverse TADF candidates.")
        
        # 高级选项
        with st.expander("高级选项"):
            use_figures = st.checkbox("在论文中包含图表", value=True)
            max_figures = st.slider("最大图表数量", min_value=1, max_value=10, value=5)
            
            st.subheader("自定义内容")
            custom_introduction = st.text_area("自定义引言", "", height=200)
            custom_methods = st.text_area("自定义方法", "", height=200)
            custom_results = st.text_area("自定义结果", "", height=200)
            custom_conclusion = st.text_area("自定义结论", "", height=200)
            custom_references = st.text_area("自定义参考文献", "", height=200)
                
        # 输出格式选择
        output_format = st.selectbox(
            "选择输出格式",
            ["markdown", "pdf", "docx"]
        )
                
        # GPT-4 扩展选项（仅在不使用多模型时显示）
        if not use_multi_model:
            use_gpt4 = st.checkbox("使用 GPT-4 增强论文")
            api_key = st.text_input("OpenAI API Key", type="password") if use_gpt4 else None
        else:
            use_gpt4 = False
            api_key = None
        
        # 提交按钮
        submit_button = st.form_submit_button("生成论文")
    
    # 生成论文
    if submit_button:
        # 准备自定义内容
        sections = {
            "include_intro": not bool(custom_introduction),
            "include_methods": not bool(custom_methods),
            "include_results": not bool(custom_results),
            "include_conclusion": not bool(custom_conclusion),
            "include_references": not bool(custom_references),
            "include_figures": use_figures
        }
        
        # 准备自定义部分字典
        custom_sections = {}
        if custom_introduction:
            custom_sections['introduction'] = custom_introduction
        if custom_methods:
            custom_sections['methods'] = custom_methods
        if custom_results:
            custom_sections['results_discussion'] = custom_results
        if custom_conclusion:
            custom_sections['conclusion'] = custom_conclusion
        if custom_references:
            custom_sections['references'] = custom_references
        
        with st.spinner("正在生成论文..."):
            try:
                if use_multi_model:
                    # 使用多模型智能体生成论文
                    # 初始化多模型智能体（如果尚未初始化）
                    if multi_model_agent is None:
                        try:
                            from agents.multi_model_agent import MultiModelAgent
                            multi_model_agent = MultiModelAgent(api_keys=api_keys)
                        except ImportError as e:
                            st.error(f"无法加载多模型智能体: {str(e)}")
                            st.info("正在退回到传统论文生成方法...")
                            use_multi_model = False
                    
                if use_multi_model and multi_model_agent:
                    # 创建进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 加载结果到多模型智能体
                    status_text.text("加载分析结果...")
                    progress_bar.progress(10)
                    
                    # 获取建模结果
                    modeling_results = None
                    if model_agent and hasattr(model_agent, 'results'):
                        modeling_results = model_agent.results
                        
                    # 获取探索结果
                    exploration_results = None
                    if exploration_agent and hasattr(exploration_agent, 'results'):
                        exploration_results = exploration_agent.results
                        
                    # 获取洞察结果
                    insight_results = None
                    if insight_agent and hasattr(insight_agent, 'results'):
                        insight_results = insight_agent.results
                    
                    # 加载结果到多模型智能体
                    multi_model_agent.load_results(
                        modeling_results=modeling_results,
                        exploration_results=exploration_results,
                        insight_results=insight_results
                    )
                    
                    # 收集文献数据（如果有）
                    if paper_agent and hasattr(paper_agent, 'literature_data'):
                        multi_model_agent.load_literature_data(paper_agent.literature_data)
                        status_text.text("已加载文献数据...")
                    progress_bar.progress(20)
                    
                    # 分析和处理图表（如果选择了使用图表）
                    if use_figures and 'selected_figures' in locals() and selected_figures:
                        status_text.text("分析选定的图表...")
                        figures_to_use = selected_figures[:max_figures]
                        visualizations = multi_model_agent.generate_visualizations(
                            data=(modeling_results or exploration_results or {}),
                            figures=figures_to_use
                        )
                        status_text.text("图表分析完成...")
                    progress_bar.progress(30)
                    
                    # 生成论文
                    status_text.text("生成论文内容 (Claude 3.5 Sonnet 处理引言)...")
                    progress_bar.progress(40)
                    
                    # 使用多模型智能体生成完整论文
                    result = multi_model_agent.generate_complete_paper(
                        title=title,
                        custom_sections=custom_sections
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("论文生成完成！")
                    
                    # 如果生成成功，保存并显示结果
                    if result and 'complete_paper' in result:
                        # 创建输出目录
<<<<<<< HEAD
                        papers_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/papers'
=======
                        papers_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/papers'
>>>>>>> 0181d62 (update excited)
                        os.makedirs(papers_dir, exist_ok=True)
                        
                        # 生成输出文件名
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        base_filename = f"reverse_tadf_paper_{timestamp}"
                        
                        # 根据选择的格式保存文件
                        if output_format == "markdown":
                            # 保存为Markdown（纯文本）
                            output_file = os.path.join(papers_dir, f"{base_filename}.md")
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(result['complete_paper'])
                        elif output_format == "pdf":
                            # 保存为PDF
                            output_file = os.path.join(papers_dir, f"{base_filename}.pdf")
                            convert_markdown_to_pdf(result['complete_paper'], output_file)
                        elif output_format == "docx":
                            # 保存为DOCX
                            output_file = os.path.join(papers_dir, f"{base_filename}.docx")
                            convert_markdown_to_docx(result['complete_paper'], output_file)
                        else:
                            # 默认保存为Markdown
                            output_file = os.path.join(papers_dir, f"{base_filename}.md")
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(result['complete_paper'])
                        
                        st.success("论文生成成功！")
                        
                        # 创建标签页用于预览和原始内容
                        tabs = st.tabs(["预览", "原始内容"])
                        
                        with tabs[0]:
                            st.markdown(result['complete_paper'])
                            
                        with tabs[1]:
                            st.code(result['complete_paper'], language="markdown")
                        
                        # 创建下载链接
                        create_download_link(output_file, f"下载论文 ({output_format})")
                    else:
                        st.error("多模型论文生成失败。")
                        if paper_agent:
                            st.info("尝试使用默认论文生成器...")
                            use_multi_model = False
                        else:
                            st.error("无可用的论文生成器。")
                
                # 如果不使用多模型或多模型失败，使用原始论文生成器
                if not use_multi_model:
                    # 初始化论文生成器 (如果尚未初始化)
                    if paper_agent is None:
                        try:
                            from agents.paper_agent import PaperAgent
                            paper_agent = PaperAgent()
                        except ImportError as e:
                            st.error(f"无法加载论文生成器: {str(e)}")
                            st.error("请检查依赖项是否正确安装。")
                            return
                    
                    # 确保结果已加载
                    modeling_results = None
                    if model_agent and hasattr(model_agent, 'results'):
                        modeling_results = model_agent.results
                        
                    exploration_results = None
                    if exploration_agent and hasattr(exploration_agent, 'results'):
                        exploration_results = exploration_agent.results
                        
                    insight_results = None
                    if insight_agent and hasattr(insight_agent, 'results'):
                        insight_results = insight_agent.results
                    
                    # 加载结果到论文生成器
                    paper_agent.load_results(
                        modeling_results=modeling_results,
                        exploration_results=exploration_results,
                        insight_results=insight_results
                    )
                    
                    # 如果使用图表，分析并添加到论文生成器
                    if use_figures and 'selected_figures' in locals() and selected_figures:
                        figures_to_use = selected_figures[:max_figures]
                        with st.spinner("分析选定的图表..."):
                            paper_agent.analyze_figures(figures_to_use)
                    
                    # 首先生成论文文本
                    paper_text = paper_agent.generate_paper(
                        sections=sections,
                        title=title
                    )
                    
                    # 添加自定义内容
                    if custom_introduction:
                        paper_text = paper_text.replace("# Introduction", f"# Introduction\n\n{custom_introduction}")
                    if custom_methods:
                        paper_text = paper_text.replace("# Methods", f"# Methods\n\n{custom_methods}")
                    if custom_results:
                        paper_text = paper_text.replace("# Results and Discussion", f"# Results and Discussion\n\n{custom_results}")
                    if custom_conclusion:
                        paper_text = paper_text.replace("# Conclusion", f"# Conclusion\n\n{custom_conclusion}")
                    if custom_references:
                        paper_text = paper_text.replace("# References", f"# References\n\n{custom_references}")
                    
                    # 然后将其保存到文件
                    result_path = paper_agent.save_paper_to_file(paper_text, output_format)
                    
                    if result_path:
                        st.success("论文生成成功！")
                        
                        # 尝试显示论文内容
                        try:
                            with open(result_path, 'r') as f:
                                paper_content = f.read()
                                
                            # 创建选项卡用于预览和原始内容
                            tabs = st.tabs(["预览", "原始内容"])
                            
                            with tabs[0]:
                                st.markdown(paper_content)
                                
                            with tabs[1]:
                                st.code(paper_content, language="markdown")
                        except Exception as e:
                            st.warning(f"无法显示论文预览: {str(e)}")
                        
                        # 创建下载链接
                        create_download_link(result_path, f"下载论文 ({output_format})")
                    else:
                        st.error("无法生成论文。")
            
            except Exception as e:
                st.error(f"生成论文时出错: {str(e)}")
                st.error(f"错误详情: {type(e).__name__}: {str(e)}")
                st.info("如果遇到依赖相关错误，请参考环境信息并尝试重新安装依赖包。")
                
    # 显示最近生成的论文
<<<<<<< HEAD
    papers_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/papers'
=======
    papers_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/papers'
>>>>>>> 0181d62 (update excited)
    if os.path.exists(papers_dir):
        paper_files = [f for f in os.listdir(papers_dir) if f.endswith('.md') or f.endswith('.pdf') or f.endswith('.docx')]
        
        if paper_files:
            st.subheader("最近生成的论文")
            
            for file in sorted(paper_files, reverse=True)[:5]:  # 只显示最近的5篇
                paper_path = os.path.join(papers_dir, file)
                create_download_link(paper_path, f"下载 {file}")

def show_environment_info():
    """显示当前 Python 环境和依赖库的状态"""
    st.subheader("环境信息")
    
    # Python 版本
    st.code(f"Python 版本: {sys.version}", language="text")
    
    # 依赖库状态
    libs_status = {
        "NumPy": ["numpy_available", "数据处理核心库"],
        "Pandas": ["pandas_available", "表格数据处理"],
        "Matplotlib/Seaborn": ["viz_available", "数据可视化"],
        "PIL/Pillow": ["pil_available", "图像处理"],
        "Markdown": ["markdown_available", "Markdown 解析"],
        "WeasyPrint": ["weasyprint_available", "PDF 生成"],
        "pypandoc": ["pypandoc_available", "文档格式转换"]
    }
    
    status_df = []
    for lib, (var_name, desc) in libs_status.items():
        status = "✅ 可用" if globals().get(var_name, False) else "❌ 不可用"
        status_df.append({"库": lib, "状态": status, "说明": desc})
    
    st.table(status_df)
    
    # 提供修复建议
    st.subheader("修复建议")
    
    if not numpy_available or not pandas_available or not viz_available:
        st.info("""
        ### 解决 NumPy/SciPy/Pandas 兼容性问题:
        ```bash
        # 方法一: 重新安装依赖包
        pip install --force-reinstall scipy pandas matplotlib seaborn
        
        # 方法二: 使用 Conda
        conda update numpy scipy pandas matplotlib seaborn
        ```
        """)
    
    if not weasyprint_available:
        st.info("""
        ### 安装 WeasyPrint 依赖:
        ```bash
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev libffi-dev shared-mime-info
        sudo apt-get install -y libgirepository1.0-dev gir1.2-gtk-3.0
        
        # 然后重新安装
        pip install --force-reinstall weasyprint
        ```
        """)

def create_download_link(file_path, text):
    """创建文件下载链接，根据文件类型设置正确的MIME类型"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            
        b64 = base64.b64encode(data).decode()
        filename = os.path.basename(file_path)
        
        # 根据文件扩展名设置正确的MIME类型
        if filename.lower().endswith('.pdf'):
            mime = "application/pdf"
        elif filename.lower().endswith('.docx'):
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            mime = "text/markdown"
        
        href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{text}</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"创建下载链接时出错: {str(e)}")
        st.info(f"您可以直接从以下路径访问文件: {file_path}")

def convert_markdown_to_pdf(content, output_path):
    """将Markdown内容转换为PDF，带有自动回退机制"""
    # 首先尝试使用WeasyPrint
    if weasyprint_available:
        try:
            # 将Markdown转换为HTML
            html = markdown.markdown(content)
            
            # 添加基本样式
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 2cm; }}
                    h1 {{ color: #333366; }}
                    h2 {{ color: #333366; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """
            
            # 使用WeasyPrint转换HTML为PDF
            HTML(string=styled_html).write_pdf(output_path)
            return True
        except Exception as e:
            st.warning(f"WeasyPrint PDF转换失败: {str(e)}，尝试替代方法...")
    
    # 如果WeasyPrint不可用或失败，尝试使用pypandoc
    # 如果WeasyPrint不可用或失败，尝试使用pypandoc
    if pypandoc_available:
        try:
            # 创建临时Markdown文件
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', encoding='utf-8', delete=False) as temp:
                temp.write(content)
                temp_path = temp.name
            
            # 使用pypandoc转换为PDF (修改转换参数)
            # 方法1：使用默认引擎，不指定pdf-engine
            pypandoc.convert_file(temp_path, 'pdf', outputfile=output_path)
            
            # 如果上述方法失败，可以尝试方法2：明确使用wkhtmltopdf
            # 确保这个命令在运行前被注释掉，只有当方法1失败时才取消注释
            # pypandoc.convert_file(temp_path, 'pdf', outputfile=output_path, extra_args=['--pdf-engine=wkhtmltopdf'])
            
            # 删除临时文件
            os.unlink(temp_path)
            return True
        except Exception as e:
            st.warning(f"Pypandoc PDF转换失败: {str(e)}，尝试最后的替代方法...")
    
    # 如果所有PDF转换方法失败，创建简单的HTML文件并提供HTML下载
    try:
        html_output_path = output_path.replace('.pdf', '.html')
        html = markdown.markdown(content) if markdown_available else f"<pre>{content}</pre>"
        
        # 添加基本样式
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2cm; }}
                h1 {{ color: #333366; }}
                h2 {{ color: #333366; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; }}
                @media print {{ body {{ margin: 2cm; }} }}
            </style>
            <title>导出论文</title>
        </head>
        <body>
            {html}
            <script>
                // 添加打印提示
                document.addEventListener('DOMContentLoaded', function() {{
                    const printMsg = document.createElement('div');
                    printMsg.style.textAlign = 'center';
                    printMsg.style.marginTop = '20px';
                    printMsg.style.padding = '10px';
                    printMsg.style.backgroundColor = '#f0f0f0';
                    printMsg.innerHTML = '<p>您可以通过浏览器的打印功能将此页面保存为PDF</p>';
                    document.body.appendChild(printMsg);
                }});
            </script>
        </body>
        </html>
        """
        
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(styled_html)
        
        # 保存原始Markdown作为备份
        md_output_path = output_path.replace('.pdf', '.md')
        with open(md_output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        st.warning(f"无法直接生成PDF。已创建HTML文件，您可以使用浏览器打印功能将其保存为PDF。")
        st.info(f"已保存HTML格式: {os.path.basename(html_output_path)} 和 Markdown格式: {os.path.basename(md_output_path)}")
        return False
    except Exception as e:
        # 最后的回退：保存为纯Markdown
        fallback_path = output_path.replace('.pdf', '.md')
        with open(fallback_path, 'w', encoding='utf-8') as f:
            f.write(content)
        st.warning(f"所有转换尝试均失败，已保存为Markdown格式: {os.path.basename(fallback_path)}")
        return False

def convert_markdown_to_docx(content, output_path):
    """将Markdown内容转换为DOCX，带有自动回退机制"""
    # 首先尝试使用pypandoc
    if pypandoc_available:
        try:
            # 创建临时Markdown文件
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', encoding='utf-8', delete=False) as temp:
                temp.write(content)
                temp_path = temp.name
            
            # 使用pypandoc转换为DOCX
            result = pypandoc.convert_file(temp_path, 'docx', outputfile=output_path)
            
            # 检查结果和文件是否创建
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                # 删除临时文件
                os.unlink(temp_path)
                return True
            else:
                raise Exception("输出文件创建失败或为空")
                
        except Exception as e:
            st.warning(f"Pypandoc DOCX转换失败: {str(e)}，尝试替代方法...")
    
    # 如果pypandoc不可用或失败，保存为Markdown
    try:
        fallback_path = output_path.replace('.docx', '.md')
        with open(fallback_path, 'w', encoding='utf-8') as f:
            f.write(content)
        st.warning(f"DOCX转换失败。已保存为Markdown格式: {os.path.basename(fallback_path)}")
        
        # 同时创建一个HTML版本，这样用户可以从浏览器复制到Word
        html_path = output_path.replace('.docx', '.html')
        html = markdown.markdown(content) if markdown_available else f"<pre>{content}</pre>"
        
        # 添加增强的样式，使其更像Word文档
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ 
                    font-family: 'Times New Roman', Times, serif; 
                    margin: 2.5cm;
                    line-height: 1.5;
                    font-size: 12pt;
                }}
                h1 {{ 
                    color: #000066; 
                    font-size: 18pt;
                    text-align: center;
                    margin-top: 24pt;
                    margin-bottom: 12pt;
                }}
                h2 {{ 
                    color: #000066; 
                    font-size: 14pt;
                    margin-top: 18pt;
                    margin-bottom: 10pt;
                }}
                h3 {{ 
                    font-size: 12pt;
                    margin-top: 14pt;
                    margin-bottom: 8pt;
                }}
                p {{ text-align: justify; }}
                pre {{ 
                    background-color: #f5f5f5; 
                    padding: 10px; 
                    border-radius: 5px;
                    font-family: Consolas, Monaco, 'Courier New', monospace;
                }}
                img {{ max-width: 100%; height: auto; }}
                .word-instructions {{
                    background-color: #f0f0f0;
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                    text-align: center;
                }}
            </style>
            <title>Word格式论文导出</title>
        </head>
        <body>
            {html}
            <div class="word-instructions">
                <h3>如何将此HTML内容复制到Word</h3>
                <p>1. 按Ctrl+A选择此页面的所有内容</p>
                <p>2. 按Ctrl+C复制内容</p>
                <p>3. 打开新的Word文档</p>
                <p>4. 按Ctrl+V粘贴内容</p>
                <p>5. 根据需要调整格式</p>
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(styled_html)
            
        st.info(f"已创建HTML版本，您可以从中复制内容到Word: {os.path.basename(html_path)}")
        return False
    except Exception as e:
        st.error(f"所有转换尝试均失败: {str(e)}")
        return False
def load_paper_page(paper_agent=None, model_agent=None, exploration_agent=None, insight_agent=None, multi_model_agent=None):
    """加载论文生成页面"""
    return render_paper_page(paper_agent, model_agent, exploration_agent, insight_agent, multi_model_agent)
def check_pandoc_installation():
    """检查pandoc是否正确安装及其版本"""
    try:
        version = pypandoc.get_pandoc_version()
        return f"Pandoc版本: {version}"
    except Exception as e:
        return f"Pandoc检查失败: {str(e)}"

if __name__ == "__main__":
    # 用于直接运行测试
    load_paper_page()