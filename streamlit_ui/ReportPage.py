# streamlit_ui/ReportPage.py
import streamlit as st
import os
import base64

def render_report_page(insight_agent=None, modeling_results=None, exploration_results=None):
    """渲染洞察报告页面"""
    st.title("反向 TADF 洞察报告")
    
    st.markdown("""
    ## 综合洞察与设计原则
    
    本页面呈现对反向 TADF 分子设计原则的综合分析，
    结合了探索分析和预测建模的结果。
    
    报告包括：
    
    1. **量子化学解释** - 为什么某些特征影响 S1-T1 能隙方向
    2. **设计原则** - 开发反向 TADF 材料的策略
    3. **特征重要性分析** - 理解关键分子描述符
    4. **结构-性能关系** - 分子结构与光物理性质之间的联系
    
    您需要在生成此报告之前运行探索和建模分析。
    """)
    
    # 检查是否有建模和探索结果
    has_modeling = os.path.exists('../data/reports/modeling') and len(os.listdir('../data/reports/modeling')) > 0
    has_exploration = os.path.exists('../data/reports/exploration') and len(os.listdir('../data/reports/exploration')) > 0
    
    # 检查报告是否已存在
    report_path = '../data/reports/reverse_tadf_insights_report.md'
    has_report = os.path.exists(report_path)
    
    if has_report:
        st.info("找到现有的洞察报告。")
        
        # 显示报告
        try:
            with open(report_path, 'r') as f:
                report_text = f.read()
                
            st.markdown(report_text)
            
            # 创建报告下载链接
            create_download_link(report_path, "下载洞察报告")
            
            if st.button("重新生成报告"):
                generate_insight_report(insight_agent, modeling_results, exploration_results)
        except Exception as e:
            st.error(f"读取报告文件时出错: {str(e)}")
            
    elif has_modeling and has_exploration:
        if st.button("生成洞察报告"):
            generate_insight_report(insight_agent, modeling_results, exploration_results)
    else:
        missing = []
        if not has_modeling:
            missing.append("建模分析")
        if not has_exploration:
            missing.append("探索分析")
            
        st.warning(f"请先运行{'和'.join(missing)}。")
        
    return None

def generate_insight_report(insight_agent, modeling_results, exploration_results):
    """生成综合洞察报告"""
    with st.spinner("正在生成洞察报告..."):
        try:
            # 执行洞察生成
            if insight_agent:
                # 如果没有传入结果但有之前的结果，使用之前的结果
                model_results = modeling_results
                explor_results = exploration_results
                
                result = insight_agent.run_insight_pipeline(
                    modeling_results=model_results,
                    exploration_results=explor_results
                )
                
                if result and 'report' in result:
                    st.success("洞察报告生成成功。")
                    
                    # 显示报告
                    try:
                        with open(result['report'], 'r') as f:
                            report_text = f.read()
                            
                        st.markdown(report_text)
                        
                        # 创建报告下载链接
                        create_download_link(result['report'], "下载洞察报告")
                        
                        # 返回洞察生成结果
                        return result
                    except Exception as e:
                        st.error(f"读取报告文件时出错: {str(e)}")
                else:
                    st.error("洞察报告生成失败")
            else:
                st.error("洞察生成组件未初始化")
        except Exception as e:
            st.error(f"生成洞察报告过程中出错: {str(e)}")
            
    return None

def create_download_link(file_path, text):
    """创建文件下载链接"""
    with open(file_path, 'rb') as f:
        data = f.read()
        
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def load_report_page(insight_agent=None, modeling_results=None, exploration_results=None):
    """加载洞察报告页面"""
    return render_report_page(insight_agent, modeling_results, exploration_results)

if __name__ == "__main__":
    # 用于直接运行测试
    load_report_page()
