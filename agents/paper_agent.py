# agents/paper_agent.py
import os
import logging
from utils.gatsbi_tools import GenerateGatsbiPromptTool
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

class PaperAgent:
    """
    Agent responsible for generating scientific papers based on the results
    from the Reverse TADF analysis system.
    """
    
    def __init__(self, modeling_results=None, exploration_results=None, insight_results=None):
        """Initialize the PaperAgent."""
        self.modeling_results = modeling_results
        self.exploration_results = exploration_results
        self.insight_results = insight_results
        self.paper_data = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the paper agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/logs/paper_agent.log')
        self.logger = logging.getLogger('PaperAgent')
        
    def load_results(self, modeling_results=None, exploration_results=None, insight_results=None):
        """Load analysis results."""
        if modeling_results:
            self.modeling_results = modeling_results
        if exploration_results:
            self.exploration_results = exploration_results
        if insight_results:
            self.insight_results = insight_results
        return True
    # 添加这个方法到paper_agent.py
    
    def create_gatsbi_prompt(self, input_data, insights, figures):
        """使用GatsbiTool创建格式化论文提示"""
        try:
            from utils.gatsbi_tools import GenerateGatsbiPromptTool
            
            # 提取标题和作者
            title = input_data.get('title', 'Reverse TADF Computational Analysis')
            
            # 处理作者列表
            if 'authors' in input_data:
                if isinstance(input_data['authors'], str):
                    authors = [author.strip() for author in input_data['authors'].split(',')]
                else:
                    authors = input_data['authors']
            else:
                authors = ["Author 1", "Author 2", "Author 3"]
                
            # 提取摘要
            abstract = input_data.get('abstract', 'No abstract provided.')
            
            # 创建工具实例
            gatsbi_prompt = GenerateGatsbiPromptTool(
                title=title,
                authors=authors,
                abstract=abstract,
                introduction=input_data.get('introduction', insights.get('introduction', '')),
                methods=input_data.get('methods', insights.get('methods', '')),
                results=input_data.get('results', insights.get('results', '')),
                discussion=input_data.get('discussion', insights.get('discussion', '')),
                conclusion=input_data.get('conclusion', insights.get('conclusion', '')),
                references=input_data.get('references', insights.get('references', '')),
                keywords=["reverse TADF", "computational chemistry", "molecular design", "excited states", "quantum chemistry"],
                figures=figures
            )
            
            # 生成提示内容
            prompt_content = gatsbi_prompt.generate_prompt()
            
            # 创建输出目录
            output_dir = '../data/papers'
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存到文件
            file_path = gatsbi_prompt.save_to_file(output_dir)
                
            return {
                'content': prompt_content,
                'path': file_path
            }
        except Exception as e:
            self.logger.error(f"创建Gatsbi提示时出错: {str(e)}")
            # 如果出错，回退到使用旧的方法
            return self.generate_gatsbi_prompt(input_data, insights, figures)   
    
    def generate_gatsbi_prompt(self, input_data, insights, figures):
        """备用方法：生成Gatsbi格式的论文提示"""
        try:
            # 创建提示结构
            prompt = "---\n"
            prompt += f"title: \"{input_data.get('title', 'Reverse TADF Computational Analysis')}\"\n"
            
            # 处理作者列表
            authors = input_data.get('authors', 'Author 1, Author 2, Author 3')
            if isinstance(authors, str):
                authors_list = [author.strip() for author in authors.split(',')]
            else:
                authors_list = authors
                
            # 格式化作者，不使用f-string的方式
            author_strings = ['"' + author + '"' for author in authors_list]
            prompt += f"authors: [{', '.join(author_strings)}]\n"
            
            prompt += "keywords: [\"reverse TADF\", \"computational chemistry\", \"molecular design\", \"excited states\", \"quantum chemistry\"]\n"
            prompt += "format: \"academic\"\n"
            prompt += "---\n\n"
            
            # 添加摘要
            prompt += "# Abstract\n\n"
            prompt += input_data.get('abstract', 'No abstract provided.') + "\n\n"
            
            # 添加各部分内容
            section_mapping = {
                'introduction': ('introduction', '# Introduction', '# 介绍'),
                'methods': ('methods', '# Methods', '# 方法'),
                'results': ('results', '# Results', '# 结果'),
                'discussion': ('discussion', '# Discussion', '# 讨论'),
                'conclusion': ('conclusion', '# Conclusion', '# 结论'),
                'references': ('references', '# References', '# 参考文献')
            }
            
            for section, (key, eng_title, cn_title) in section_mapping.items():
                content = input_data.get(key, insights.get(section, f"{eng_title}\n\nNo {section} available."))
                prompt += content.replace(eng_title, eng_title).replace(cn_title, eng_title) + "\n\n"
            
            # 添加图表
            if figures:
                prompt += "# Figures\n\n"
                for i, figure in enumerate(figures[:6]):  # 限制为6个图表
                    prompt += f"**Figure {i+1}: {figure['name']}**\n\n"
                    prompt += f"![{figure['name']}]({figure['path']})\n\n"
            
            # 创建输出目录
            output_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/papers'
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            import re
            from datetime import datetime
            safe_title = re.sub(r'[^\w\s]', '', input_data.get('title', 'reverse_tadf_paper')).replace(' ', '_').lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{safe_title}_{timestamp}.md"
            file_path = os.path.join(output_dir, filename)
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
                
            return {
                'content': prompt,
                'path': file_path
            }
        except Exception as e:
            self.logger.error(f"使用备用方法生成Gatsbi提示时出错: {str(e)}")
            return None
    
    def generate_full_paper_with_gpt4(self, api_key=None, prompt_path=None, api_base="https://api.gptsapi.net/v1"):
        """Use GPT-4 to generate a full paper from the Gatsbi prompt."""
        if api_key is None:
            self.logger.error("需要提供OpenAI API密钥才能使用GPT-4")
            return None

        if prompt_path is None and self.paper_data is None:
            self.logger.error("需要先创建Gatsbi提示或提供提示文件路径")
            return None

        prompt_path = prompt_path or self.paper_data.get('path')

        try:
            # Initialize the LLM with API base URL
            llm = ChatOpenAI(
                temperature=0.3, 
                model_name="gpt-4o-mini", 
                openai_api_key=api_key,
                openai_api_base=api_base  # 添加API基础URL
            )

            # Read the prompt
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()

            # Add specific instructions for the LLM
            enhanced_prompt = f"""你是一位专业的学术论文写作助手。我将为你提供一个科学论文的大纲，请将其扩展为完整的学术论文，保持高度专业性和科学严谨性。

    你需要：
    1. 保持原有的章节结构和主要观点
    2. 扩展和精细化每个部分，增加科学细节和论证
    3. 确保内容流畅连贯，逻辑严密
    4. 使用适当的学术语言和术语
    5. 遵循科学论文的写作规范

    以下是论文大纲：

    {prompt}

    请基于以上内容，撰写一篇完整的学术论文。"""

            # 添加重试逻辑
            import time
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Generate the response
                    response = llm.predict(enhanced_prompt)
                    break
                except Exception as e:
                    retry_count += 1
                    self.logger.warning(f"API调用失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                    if retry_count >= max_retries:
                        self.logger.error(f"达到最大重试次数，放弃生成: {str(e)}")
                        return None
                    # 指数退避策略
                    time.sleep(2 ** retry_count)

            # Save the response
            output_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/paper'
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, "full_paper_gpt4.md")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response)

            self.logger.info(f"完整论文已生成并保存到: {output_path}")
            return {
                'path': output_path,
                'content': response
            }

        except Exception as e:
            self.logger.error(f"使用GPT-4生成论文时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    # 修改run_paper_generation方法
    def run_paper_generation(self, custom_input=None, use_gpt4=False, api_key=None, output_format='markdown'):
        """Run the complete paper generation pipeline."""
        try:
            # 首先获取报告内容和图像
            insights = {}  # 初始化一个空字典用于存放报告内容
            figures = []   # 初始化一个空列表用于存放图像
            
            # 从报告中提取内容
            report_path = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/reverse_tadf_insights_report.md'
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                    
                # 提取各部分内容
                sections = {
                    'introduction': '# Introduction\n\nNo introduction available.',
                    'methods': '# Methods\n\nNo methods available.',
                    'results': '# Results\n\nNo results available.',
                    'discussion': '# Discussion\n\nNo discussion available.',
                    'conclusion': '# Conclusion\n\nNo conclusion available.',
                    'references': '# References\n\nNo references available.'
                }
                
                # 分析报告内容
                current_section = 'introduction'
                section_content = []
                
                for line in report_content.split('\n'):
                    if line.startswith('# ') or line.startswith('## '):
                        # 保存之前的部分
                        if section_content:
                            sections[current_section] = '\n'.join(section_content)
                        
                        # 确定新部分
                        header = line.lower()
                        if 'introduction' in header or 'overview' in header:
                            current_section = 'introduction'
                        elif 'method' in header:
                            current_section = 'methods'
                        elif 'result' in header:
                            current_section = 'results'
                        elif 'discussion' in header or 'analysis' in header:
                            current_section = 'discussion'
                        elif 'conclusion' in header:
                            current_section = 'conclusion'
                        elif 'reference' in header:
                            current_section = 'references'
                        
                        section_content = [line]
                    else:
                        section_content.append(line)
                
                # 保存最后一部分
                if section_content:
                    sections[current_section] = '\n'.join(section_content)
                    
                # 将提取的部分存入insights字典
                insights = sections
            
            # 提取图像
            exploration_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/exploration'
            modeling_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/modeling'
            
            # 检查目录是否存在
            if os.path.exists(exploration_dir):
                for file in os.listdir(exploration_dir):
                    if file.endswith('.png'):
                        figures.append({
                            'path': os.path.join(exploration_dir, file),
                            'name': file.replace('.png', '').replace('_', ' ').title(),
                            'type': 'exploration'
                        })
            
            if os.path.exists(modeling_dir):
                for file in os.listdir(modeling_dir):
                    if file.endswith('.png'):
                        figures.append({
                            'path': os.path.join(modeling_dir, file),
                            'name': file.replace('.png', '').replace('_', ' ').title(),
                            'type': 'modeling'
                        })
            
            # Step 1: Create Gatsbi prompt - 传递所有必要的参数
            prompt_result = self.create_gatsbi_prompt(custom_input, insights, figures)

            if not prompt_result:
                self.logger.error("创建Gatsbi提示失败")
                return None
                
            # 保存生成的提示以便后续使用
            self.paper_data = prompt_result

            # Step 2: Optionally use GPT-4 to expand the paper
            full_paper = None
            if use_gpt4 and api_key:
                full_paper = self.generate_full_paper_with_gpt4(api_key=api_key, prompt_path=prompt_result['path'])

            # 根据输出格式处理结果
            if output_format == 'markdown':
                # 直接返回Markdown格式
                pass
            elif output_format == 'gatsbi':
                # 为Gatsbi网站格式化
                pass
            elif output_format == 'pdf':
                # 可以添加将Markdown转换为PDF的逻辑
                pass

            return {
                'prompt': prompt_result,
                'full_paper': full_paper,
                'format': output_format
            }
        except Exception as e:
            self.logger.error(f"论文生成过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
   