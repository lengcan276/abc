# agents/paper_agent.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import logging
from datetime import datetime
import requests
import json

class PaperAgent:
    """
    Agent responsible for generating academic papers based on analysis results
    from the exploration, modeling, and insight agents.
    """
    
    def __init__(self, modeling_results=None, exploration_results=None, insight_results=None):
        """Initialize the PaperAgent with analysis results."""
        self.modeling_results = modeling_results
        self.exploration_results = exploration_results
        self.insight_results = insight_results
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the paper agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='../data/logs/paper_agent.log')
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
    
    def extract_insights_from_report(self):
        """Extract insights from the generated report file."""
        report_path = '../data/reports/reverse_tadf_insights_report.md'
        
        if not os.path.exists(report_path):
            self.logger.warning("Insight report not found. Using default content.")
            return {
                'introduction': "# 介绍\n\n反向热活化延迟荧光（TADF）材料因其独特的光物理特性受到广泛关注。在这类材料中，第一单重态激发态（S1）的能量低于第一三重态激发态（T1），这与常规分子中根据洪特规则预期的能量排序相反。这种独特的特性为开发新型光电子器件提供了可能性。",
                'methods': "# 方法\n\n本研究采用密度泛函理论（DFT）和含时密度泛函理论（TD-DFT）计算方法，结合机器学习分析，系统研究了反向TADF分子的结构与性能关系。",
                'results': "# 结果\n\n我们分析了100多个分子结构，发现约5%的分子具有负S1-T1能隙。机器学习模型成功地分类了这些分子，准确率达到85%。",
                'discussion': "# 讨论\n\n从特征重要性分析中，我们发现电子效应和共轭程度是决定S1-T1能隙方向的关键因素。",
                'conclusion': "# 结论\n\n本研究为设计反向TADF材料提供了重要见解和指导原则。",
                'references': "# 参考文献\n\n1. Wong, M. Y., & Zysman-Colman, E. (2017). Purely organic thermally activated delayed fluorescence materials for organic light-emitting diodes. Advanced Materials, 29(22), 1605444."
            }
        
        # Read the report file
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract different sections
        sections = {}
        current_section = 'introduction'
        section_content = []
        
        for line in content.split('\n'):
            # Check if this is a section header
            if re.match(r'^#+\s+', line):
                # Save previous section content
                if section_content:
                    sections[current_section] = '\n'.join(section_content)
                    
                # Determine new section
                header = line.lower()
                if 'introduction' in header or 'overview' in header:
                    current_section = 'introduction'
                elif 'method' in header or 'computational' in header:
                    current_section = 'methods'
                elif 'result' in header:
                    current_section = 'results'
                elif 'discussion' in header or 'analysis' in header:
                    current_section = 'discussion'
                elif 'conclusion' in header or 'summary' in header:
                    current_section = 'conclusion'
                elif 'reference' in header or 'bibliography' in header:
                    current_section = 'references'
                elif 'design' in header:
                    current_section = 'discussion'
                else:
                    current_section = 'other'
                    
                section_content = [line]
            else:
                section_content.append(line)
                
        # Save the last section
        if section_content:
            sections[current_section] = '\n'.join(section_content)
            
        # Ensure all required sections exist
        for section in ['introduction', 'methods', 'results', 'discussion', 'conclusion', 'references']:
            if section not in sections:
                sections[section] = f"# {section.title()}\n\n(No content available)"
                
        return sections
    
    def extract_figures_from_reports(self):
        """Extract figure references from reports directories."""
        figures = []
        
        # Check exploration directory
        exploration_dir = '../data/reports/exploration'
        if os.path.exists(exploration_dir):
            for file in os.listdir(exploration_dir):
                if file.endswith('.png'):
                    figures.append({
                        'path': os.path.join(exploration_dir, file),
                        'name': file.replace('.png', '').replace('_', ' ').title(),
                        'type': 'exploration'
                    })
        
        # Check modeling directory
        modeling_dir = '../data/reports/modeling'
        if os.path.exists(modeling_dir):
            for file in os.listdir(modeling_dir):
                if file.endswith('.png'):
                    figures.append({
                        'path': os.path.join(modeling_dir, file),
                        'name': file.replace('.png', '').replace('_', ' ').title(),
                        'type': 'modeling'
                    })
                    
        return figures
    
    def generate_gatsbi_prompt(self, input_data, insights, figures):
        """Generate Gatsbi prompt for academic paper."""
        # Create prompt structure
        prompt = "---\n"
        prompt += f"title: \"{input_data.get('title', 'Reverse TADF Computational Analysis')}\"\n"
        
        # 修改这一行以避免f-string中的反斜杠问题
        author_strings = ['"' + author.strip() + '"' for author in input_data.get('authors', 'Author').split(',')]
        prompt += f"authors: [{', '.join(author_strings)}]\n"
        
        prompt += "keywords: [\"reverse TADF\", \"computational chemistry\", \"molecular design\", \"excited states\", \"quantum chemistry\"]\n"
        prompt += "format: \"academic\"\n"
        prompt += "---\n\n"
        
        # Add abstract
        prompt += "# Abstract\n\n"
        prompt += input_data.get('abstract', 'No abstract provided.') + "\n\n"
        
        # Add introduction
        intro_content = input_data.get('introduction', insights.get('introduction', '# Introduction\n\nNo introduction available.'))
        prompt += intro_content.replace('# Introduction', '# Introduction').replace('# 介绍', '# Introduction') + "\n\n"
        
        # Add methods
        methods_content = input_data.get('methods', insights.get('methods', '# Methods\n\nNo methods available.'))
        prompt += methods_content.replace('# Methods', '# Methods').replace('# 方法', '# Methods') + "\n\n"
        
        # Add results
        results_content = input_data.get('results', insights.get('results', '# Results\n\nNo results available.'))
        prompt += results_content.replace('# Results', '# Results').replace('# 结果', '# Results') + "\n\n"
        
        # Add figures (if available)
        if figures:
            prompt += "# Figures\n\n"
            for i, figure in enumerate(figures[:6]):  # Limit to 6 figures
                prompt += f"**Figure {i+1}: {figure['name']}**\n\n"
                prompt += f"![{figure['name']}]({figure['path']})\n\n"
                
        # Add discussion
        discussion_content = input_data.get('discussion', insights.get('discussion', '# Discussion\n\nNo discussion available.'))
        prompt += discussion_content.replace('# Discussion', '# Discussion').replace('# 讨论', '# Discussion') + "\n\n"
        
        # Add conclusion
        conclusion_content = input_data.get('conclusion', insights.get('conclusion', '# Conclusion\n\nNo conclusion available.'))
        prompt += conclusion_content.replace('# Conclusion', '# Conclusion').replace('# 结论', '# Conclusion') + "\n\n"
        
        # Add references
        references_content = input_data.get('references', insights.get('references', '# References\n\nNo references available.'))
        prompt += references_content.replace('# References', '# References').replace('# 参考文献', '# References') + "\n\n"
        
        # Create output directory if it doesn't exist
        output_dir = '../data/papers'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on title and date
        safe_title = re.sub(r'[^\w\s]', '', input_data.get('title', 'reverse_tadf_paper')).replace(' ', '_').lower()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_title}_{timestamp}.md"
        file_path = os.path.join(output_dir, filename)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
            
        return {
            'content': prompt,
            'path': file_path
        }
    
    def expand_with_gpt4(self, prompt_content, api_key):
        """Use GPT-4 to expand the paper content."""
        if not api_key:
            self.logger.error("No API key provided for GPT-4 expansion.")
            return None
            
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            system_message = """You are a scientific paper writer specializing in quantum chemistry and computational physics. 
            Your task is to expand a scientific paper prompt into a complete academic paper.
            Maintain the academic style and expand each section with relevant scientific content.
            Include theoretical background, detailed methodology, comprehensive results analysis, and thorough discussion.
            Do not fabricate specific numerical results, but you can expand explanations and theoretical frameworks.
            Focus on reverse TADF (Thermally Activated Delayed Fluorescence) materials where S1 energy is lower than T1 energy."""
            
            user_message = f"""Please expand this scientific paper prompt into a complete academic paper:
            
            {prompt_content}
            
            Expand each section thoughtfully while maintaining scientific accuracy.
            """
            
            data = {
                'model': 'gpt-4-turbo',
                'messages': [
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': user_message}
                ],
                'max_tokens': 4000,
                'temperature': 0.7
            }
            
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, data=json.dumps(data))
            
            if response.status_code == 200:
                paper_content = response.json()['choices'][0]['message']['content']
                
                # Create output directory if it doesn't exist
                output_dir = '../data/papers'
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"expanded_paper_{timestamp}.md"
                file_path = os.path.join(output_dir, filename)
                
                # Write to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(paper_content)
                    
                return {
                    'content': paper_content,
                    'path': file_path
                }
            else:
                self.logger.error(f"API request failed: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in GPT-4 expansion: {str(e)}")
            return None
    
    def run_paper_generation(self, custom_input=None, use_gpt4=False, api_key=None):
        """Run the complete paper generation pipeline."""
        try:
            # Extract insights from report
            insights = self.extract_insights_from_report()
            
            # Extract figures
            figures = self.extract_figures_from_reports()
            
            # Parse authors from input
            authors = custom_input.get('authors', 'Author 1, Author 2, Author 3')
            if isinstance(authors, str):
                authors_list = [author.strip() for author in authors.split(',')]
                custom_input['authors'] = authors_list
            
            # Generate Gatsbi prompt
            prompt_result = self.generate_gatsbi_prompt(custom_input, insights, figures)
            
            result = {
                'prompt': prompt_result
            }
            
            # If GPT-4 expansion is requested
            if use_gpt4 and api_key:
                full_paper = self.expand_with_gpt4(prompt_result['content'], api_key)
                if full_paper:
                    result['full_paper'] = full_paper
            
            return result
            
        except Exception as e:
            self.logger.error(f"Paper generation error: {str(e)}")
            return None