# utils/gatsbi_tools.py
import os
import re
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class GenerateGatsbiPromptTool(BaseModel):
    """工具类用于生成Gatsbi格式的论文提示"""
    
    title: str = Field(..., description="论文标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    abstract: str = Field(..., description="论文摘要")
    introduction: str = Field(default="", description="论文引言部分")
    methods: str = Field(default="", description="论文方法部分")
    results: str = Field(default="", description="论文结果部分")
    discussion: str = Field(default="", description="论文讨论部分")
    conclusion: str = Field(default="", description="论文结论部分")
    references: str = Field(default="", description="论文参考文献")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    figures: List[Dict[str, Any]] = Field(default_factory=list, description="图表列表")
    
    class Config:
        arbitrary_types_allowed = True
    
    def generate_prompt(self) -> str:
        """生成Gatsbi格式的提示文本"""
        # 创建YAML前置内容
        prompt = "---\n"
        prompt += f"title: \"{self.title}\"\n"
        
        # 格式化作者列表
        authors_str = ", ".join([f'"{author}"' for author in self.authors])
        prompt += f"authors: [{authors_str}]\n"
        
        # 添加关键词
        if self.keywords:
            keywords_str = ", ".join([f'"{keyword}"' for keyword in self.keywords])
            prompt += f"keywords: [{keywords_str}]\n"
        else:
            prompt += "keywords: [\"reverse TADF\", \"computational chemistry\", \"molecular design\", \"excited states\"]\n"
            
        prompt += "format: \"academic\"\n"
        prompt += "---\n\n"
        
        # 添加摘要
        prompt += "# Abstract\n\n"
        prompt += f"{self.abstract}\n\n"
        
        # 添加引言
        if self.introduction:
            prompt += self.introduction.replace('# Introduction', '# Introduction').replace('# 介绍', '# Introduction') + "\n\n"
        else:
            prompt += "# Introduction\n\n(No introduction available)\n\n"
            
        # 添加方法
        if self.methods:
            prompt += self.methods.replace('# Methods', '# Methods').replace('# 方法', '# Methods') + "\n\n"
        else:
            prompt += "# Methods\n\n(No methods available)\n\n"
            
        # 添加结果
        if self.results:
            prompt += self.results.replace('# Results', '# Results').replace('# 结果', '# Results') + "\n\n"
        else:
            prompt += "# Results\n\n(No results available)\n\n"
            
        # 添加图表
        if self.figures:
            prompt += "# Figures\n\n"
            for i, figure in enumerate(self.figures[:6]):  # 限制为6个图表
                prompt += f"**Figure {i+1}: {figure['name']}**\n\n"
                prompt += f"![{figure['name']}]({figure['path']})\n\n"
                
        # 添加讨论
        if self.discussion:
            prompt += self.discussion.replace('# Discussion', '# Discussion').replace('# 讨论', '# Discussion') + "\n\n"
        else:
            prompt += "# Discussion\n\n(No discussion available)\n\n"
            
        # 添加结论
        if self.conclusion:
            prompt += self.conclusion.replace('# Conclusion', '# Conclusion').replace('# 结论', '# Conclusion') + "\n\n"
        else:
            prompt += "# Conclusion\n\n(No conclusion available)\n\n"
            
        # 添加参考文献
        if self.references:
            prompt += self.references.replace('# References', '# References').replace('# 参考文献', '# References') + "\n\n"
        else:
            prompt += "# References\n\n(No references available)\n\n"
            
        return prompt
        
    def save_to_file(self, output_dir: str = "../data/papers") -> str:
        """将提示保存到文件并返回文件路径"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        safe_title = re.sub(r'[^\w\s]', '', self.title).replace(' ', '_').lower()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_title}_{timestamp}.md"
        file_path = os.path.join(output_dir, filename)
        
        # 生成提示内容
        prompt_content = self.generate_prompt()
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(prompt_content)
            
        return file_path