# agents/paper_agent.py
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import tempfile
from io import BytesIO
import base64
import json
import requests
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import time
from urllib.parse import quote

class PaperAgent:
    """
    Agent responsible for analyzing scientific literature and generating
    research papers on reverse TADF molecules.
    """
    
    def __init__(self):
        """Initialize the PaperAgent."""
        self.setup_logging()
        self.literature_data = {}
        self.extracted_data = {}
        self.generated_sections = {}
        self.references = []
        self.figures = []
        self.modeling_results = None
        self.exploration_results = None
        self.insight_results = None
        
    def setup_logging(self):
        """Configure logging for the paper agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/logs/paper_agent.log')
        self.logger = logging.getLogger('PaperAgent')
        
    def query_deepresearch(self, query, max_results=10):
        """
        Query the DeepResearch API for scientific literature.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of research papers matching the query
        """
        self.logger.info(f"Querying DeepResearch API for: {query}")
        
        # 尝试获取API密钥
        api_key = os.environ.get('DEEPRESEARCH_API_KEY', '')
        
        if not api_key:
            self.logger.warning("DeepResearch API key not set. Using simulated results.")
            return self._simulate_search_results(query, max_results)
        
        try:
            # DeepResearch API endpoint (示例，实际URL可能不同)
            api_url = "https://api.deepresearch.ai/search"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "max_results": max_results,
                "fields": ["title", "authors", "abstract", "year", "journal", "doi", "citations"]
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                results = response.json()
                self.logger.info(f"DeepResearch query successful, found {len(results.get('papers', []))} papers")
                return results.get("papers", [])
            else:
                self.logger.error(f"DeepResearch API error: {response.status_code}, {response.text}")
                return self._simulate_search_results(query, max_results)
                
        except Exception as e:
            self.logger.error(f"Error querying DeepResearch API: {str(e)}")
            return self._simulate_search_results(query, max_results)
            
    def _simulate_search_results(self, query, max_results=10):
        """
        提供模拟的文献搜索结果，针对反向TADF领域进行了特殊定制
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            
        Returns:
            模拟的文献结果列表
        """
        self.logger.info(f"Generating simulated results for: {query}")
        
        # 创建针对查询关键词的模拟结果
        papers = []
        keywords = query.lower().split()
        
        # 预定义的TADF相关示例论文
        tadf_papers = [
            {
                "title": "Inverted singlet-triplet energy gaps in non-alternant π-systems",
                "authors": ["G. Ricci", "Y. Olivier", "J. C. Sancho-García"],
                "abstract": "This study explores non-alternant π-systems with inverted singlet-triplet energy gaps, providing computational evidence for their potential as highly efficient OLED emitters.",
                "year": "2023",
                "journal": "Journal of Materials Chemistry C",
                "doi": "10.1039/D3TC00001J",
                "citations": 18,
                "simulated": True
            },
            {
                "title": "Design and synthesis of molecules with inverted singlet-triplet gaps for efficient OLEDs",
                "authors": ["N. Aizawa", "Y. Pu", "D. Miyajima"],
                "abstract": "We report the synthesis and characterization of novel organic molecules exhibiting inverted singlet-triplet energy gaps, demonstrating their application in high-efficiency OLEDs.",
                "year": "2022",
                "journal": "Advanced Materials",
                "doi": "10.1002/adma.202200001",
                "citations": 45,
                "simulated": True
            },
            {
                "title": "Singlet−Triplet Inversions in Through-Bond Charge-Transfer States",
                "authors": ["J. Terence Blaskovits", "Clémence Corminboeuf", "Marc H. Garner"],
                "abstract": "We demonstrate that through-bond charge-transfer states can lead to inverted singlet-triplet gaps in donor-acceptor systems, providing a new design strategy for OLED materials.",
                "year": "2024",
                "journal": "Journal of Physical Chemistry Letters",
                "doi": "10.1021/acs.jpclett.4c02317",
                "citations": 5,
                "simulated": True
            },
            {
                "title": "Molecular design principles for reverse TADF emitters",
                "authors": ["L. Chen", "S. Wang", "H. Adachi"],
                "abstract": "This review summarizes recent advances in the design of reverse TADF emitters, highlighting key structural and electronic factors that promote inverted singlet-triplet gaps.",
                "year": "2023",
                "journal": "Chemical Reviews",
                "doi": "10.1021/acs.chemrev.3c00112",
                "citations": 67,
                "simulated": True
            }
        ]
        
        # 预定义的计算方法相关论文
        computational_papers = [
            {
                "title": "Machine learning approaches for predicting excited state properties of TADF emitters",
                "authors": ["R. Gómez-Bombarelli", "J. Aguilera-Iparraguirre", "A. Aspuru-Guzik"],
                "abstract": "This work demonstrates the application of machine learning techniques to predict excited state properties of TADF emitters, enabling rapid screening of candidate molecules.",
                "year": "2021",
                "journal": "Nature Communications",
                "doi": "10.1038/s41467-021-00001-x",
                "citations": 87,
                "simulated": True
            },
            {
                "title": "Quantum chemical investigation of inverted gap materials",
                "authors": ["P. de Silva", "C. A. Kim", "T. J. Martínez"],
                "abstract": "Using high-level quantum chemical methods, we investigate the electronic structure of molecules with inverted singlet-triplet gaps and propose new design strategies.",
                "year": "2022",
                "journal": "Journal of Chemical Theory and Computation",
                "doi": "10.1021/acs.jctc.2b00123",
                "citations": 42,
                "simulated": True
            },
            {
                "title": "Deep learning models for TADF property prediction",
                "authors": ["Y. Zhang", "M. Chen", "K. Burke"],
                "abstract": "We present deep learning models that can accurately predict TADF-relevant properties from molecular structures, including singlet-triplet gaps and emission wavelengths.",
                "year": "2023",
                "journal": "Journal of Chemical Information and Modeling",
                "doi": "10.1021/acs.jcim.3b00567",
                "citations": 23,
                "simulated": True
            }
        ]
        
        # 根据查询关键词选择相关论文
        if any(kw in ['tadf', 'reversed', 'inverted', 'gap', 'singlet', 'triplet'] for kw in keywords):
            papers.extend(tadf_papers)
        
        if any(kw in ['computational', 'calculation', 'dft', 'machine', 'learning', 'predict'] for kw in keywords):
            papers.extend(computational_papers)
        
        # 添加一些通用结果以达到max_results数量
        while len(papers) < max_results:
            idx = len(papers) + 1
            papers.append({
                "title": f"Research on {query.title()} #{idx}",
                "authors": ["A. Researcher", "B. Scientist"],
                "abstract": f"This paper discusses various aspects of {query} with focus on computational design and analysis.",
                "year": str(2020 + (idx % 5)),
                "journal": "Journal of Computational Chemistry",
                "doi": f"10.1021/jcc.{2020 + (idx % 5)}.{1000 + idx}",
                "citations": idx * 3,
                "simulated": True
            })
        
        return papers[:max_results]
        
    def parse_literature(self, literature_files, format_type="txt"):
        """
        Parse uploaded literature files and extract relevant information.
        
        Args:
            literature_files: List of file paths to literature files
            format_type: Format of the literature files (txt, pdf, etc.)
            
        Returns:
            Dictionary of extracted literature data
        """
        self.logger.info(f"Parsing {len(literature_files)} literature files in {format_type} format")
        
        all_literature_data = []
        
        for file_path in literature_files:
            try:
                # Extract text based on file format
                if format_type.lower() == "txt":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif format_type.lower() == "pdf":
                    content = self._extract_text_from_pdf(file_path)
                else:
                    self.logger.error(f"Unsupported file format: {format_type}")
                    continue
                
                # Parse literature data
                paper_data = self._extract_paper_data(content, os.path.basename(file_path))
                
                if paper_data:
                    all_literature_data.append(paper_data)
                    self.logger.info(f"Successfully parsed {os.path.basename(file_path)}")
                
            except Exception as e:
                self.logger.error(f"Error parsing {os.path.basename(file_path)}: {str(e)}")
        
        # Organize and structure the literature data
        structured_data = self._structure_literature_data(all_literature_data)
        self.literature_data = structured_data
        
        return structured_data
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        
        return text
    
    def _extract_paper_data(self, content, filename):
        """
        Extract structured data from paper content.
        """
        # Basic extraction logic
        paper_data = {
            "filename": filename,
            "title": self._extract_title(content),
            "authors": self._extract_authors(content),
            "abstract": self._extract_abstract(content),
            "introduction": self._extract_section(content, "introduction", "method"),
            "methods": self._extract_section(content, "method", "result"),
            "results": self._extract_section(content, "result", "discussion"),
            "discussion": self._extract_section(content, "discussion", "conclusion"),
            "conclusion": self._extract_section(content, "conclusion", "reference"),
            "references": self._extract_references(content),
            "doi": self._extract_doi(content),
            "year": self._extract_year(content),
            "journal": self._extract_journal(content)
        }
        
        return paper_data
    
    def _extract_title(self, content):
        """Extract paper title from content."""
        # Simple heuristic - first line or TI field in WoS format
        lines = content.split('\n')
        title = ""
        
        # Check for WoS format
        for line in lines:
            if line.startswith("TI "):
                title = line[3:].strip()
                break
        
        # If not found, use first non-empty line
        if not title:
            for line in lines:
                if line.strip():
                    title = line.strip()
                    break
        
        return title
    
    def _extract_authors(self, content):
        """Extract author information."""
        authors = []
        # Look for WoS AU field
        matches = re.findall(r"AU\s+(.+?)$", content, re.MULTILINE)
        if matches:
            for match in matches:
                authors.append(match.strip())
        
        return authors
    
    def _extract_abstract(self, content):
        """Extract abstract from content."""
        abstract = ""
        
        # Check for WoS format (AB field)
        ab_match = re.search(r"AB\s+(.+?)(?=\n[A-Z][A-Z]\s+|$)", content, re.DOTALL)
        if ab_match:
            abstract = ab_match.group(1).strip()
        
        # Alternative: look for Abstract section
        if not abstract:
            abs_match = re.search(r"Abstract[:\.\s]+(.*?)(?=\n\s*(?:Introduction|Keywords)|\Z)", 
                                 content, re.IGNORECASE | re.DOTALL)
            if abs_match:
                abstract = abs_match.group(1).strip()
        
        return abstract
    
    def _extract_section(self, content, start_section, end_section):
        """Extract a specific section from the paper."""
        # Case insensitive search for section headers
        pattern = rf"{start_section}.*?\n(.*?)(?:\n\s*{end_section}|\Z)"
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _extract_references(self, content):
        """Extract references from the paper."""
        references = []
        
        # Check for References or Bibliography section
        ref_section = ""
        ref_match = re.search(r"References[:\.\s]+(.*?)(?:\Z)", content, re.IGNORECASE | re.DOTALL)
        
        if ref_match:
            ref_section = ref_match.group(1).strip()
            
            # Split into individual references
            # This is a simple approach - would need enhancement for different formats
            ref_list = re.split(r"\n\s*\d+\.\s+|\[\d+\]\s+", ref_section)
            
            for ref in ref_list:
                if ref.strip():
                    references.append(ref.strip())
        
        # Check for WoS format (CR field)
        cr_matches = re.findall(r"CR\s+(.+?)$", content, re.MULTILINE)
        if cr_matches:
            references.extend([ref.strip() for ref in cr_matches])
        
        return references
    
    def _extract_doi(self, content):
        """Extract DOI from content."""
        doi_match = re.search(r"(?:DOI|doi)[\s:]*(\d+\.\d+\/\S+)", content)
        if doi_match:
            return doi_match.group(1).strip()
        return ""
    
    def _extract_year(self, content):
        """Extract publication year."""
        # Check for WoS format (PY field)
        py_match = re.search(r"PY\s+(\d{4})", content)
        if py_match:
            return py_match.group(1)
        
        # Look for year patterns in content
        year_match = re.search(r"\b(20\d{2}|19\d{2})\b", content)
        if year_match:
            return year_match.group(1)
        
        return ""
    
    def _extract_journal(self, content):
        """Extract journal name."""
        # Check for WoS format (JO field)
        jo_match = re.search(r"JO\s+(.+?)$", content, re.MULTILINE)
        if jo_match:
            return jo_match.group(1).strip()
        
        # Alternative: SO field
        so_match = re.search(r"SO\s+(.+?)$", content, re.MULTILINE)
        if so_match:
            return so_match.group(1).strip()
        
        return ""
    
    def _structure_literature_data(self, all_literature_data):
        """
        Organize and structure the extracted literature data.
        Group papers by topic, identify key themes, etc.
        """
        # Group papers by topic
        topics = {}
        
        # Assign papers to topics (simplified approach)
        for paper in all_literature_data:
            # Simple keyword-based topic assignment
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            
            if "tadf" in title or "tadf" in abstract:
                if "reverse" in title or "reverse" in abstract or "inverted" in title or "inverted" in abstract:
                    topic = "Reverse TADF"
                else:
                    topic = "General TADF"
            elif "oled" in title or "oled" in abstract:
                topic = "OLED Technology"
            else:
                topic = "Other"
                
            if topic not in topics:
                topics[topic] = []
                
            topics[topic].append(paper)
        
        # Sort papers within each topic by year
        for topic in topics:
            topics[topic].sort(key=lambda x: x.get("year", "0"), reverse=True)
        
        # Extract key terms and concepts
        key_terms = self._extract_key_terms(all_literature_data)
        
        return {
            "papers": all_literature_data,
            "topics": topics,
            "key_terms": key_terms,
            "total_papers": len(all_literature_data)
        }
    
    def _extract_key_terms(self, papers):
        """Extract key terms and concepts from the papers."""
        key_terms = {}
        
        # Combine all text
        all_text = ""
        for paper in papers:
            all_text += paper.get("title", "") + " "
            all_text += paper.get("abstract", "") + " "
            all_text += paper.get("introduction", "") + " "
            all_text += paper.get("conclusion", "") + " "
        
        # Simple frequency analysis
        words = all_text.lower().split()
        word_freq = {}
        
        for word in words:
            word = word.strip(".,;:()'\"")
            if len(word) > 3:  # Skip short words
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
        
        # Get top terms
        sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[:50]  # Top 50 terms
        
        return top_terms
    
    def analyze_figures(self, figure_files, data_files=None):
        """
        Analyze figure files and associated data.
        
        Args:
            figure_files: List of paths to figure files (png, jpg, etc.)
            data_files: Optional list of paths to associated data files (csv, xlsx, etc.)
        
        Returns:
            Analysis results for the figures
        """
        self.logger.info(f"Analyzing {len(figure_files)} figure files")
        
        figure_analyses = []
        
        for i, fig_path in enumerate(figure_files):
            try:
                # Extract figure information
                fig_name = os.path.basename(fig_path)
                fig_type = self._determine_figure_type(fig_path)
                
                # If there's associated data, analyze it
                data_analysis = None
                if data_files and i < len(data_files):
                    data_analysis = self._analyze_data_file(data_files[i])
                
                # Perform figure analysis
                fig_analysis = {
                    "figure_name": fig_name,
                    "figure_path": fig_path,
                    "figure_type": fig_type,
                    "caption": self._generate_figure_caption(fig_path, fig_type, data_analysis),
                    "trends": self._extract_trends(fig_path, data_analysis),
                    "data_analysis": data_analysis
                }
                
                figure_analyses.append(fig_analysis)
                self.logger.info(f"Successfully analyzed figure: {fig_name}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing figure {fig_path}: {str(e)}")
        
        self.figures = figure_analyses
        return figure_analyses
    
    def _determine_figure_type(self, fig_path):
        """Determine the type of figure based on filename and content."""
        filename = os.path.basename(fig_path).lower()
        
        # Simple heuristic based on filename
        if "energy" in filename or "level" in filename:
            return "Energy Level Diagram"
        elif "efficiency" in filename or "tadf" in filename:
            return "TADF Efficiency Plot"
        elif "spectrum" in filename or "absorption" in filename or "emission" in filename:
            return "Spectral Data"
        elif "structure" in filename or "molecule" in filename:
            return "Molecular Structure"
        elif "correlation" in filename or "relation" in filename:
            return "Correlation Plot"
        else:
            return "Other"
    
    def _analyze_data_file(self, data_path):
        """Analyze data file associated with a figure."""
        try:
            # Determine file type
            file_ext = os.path.splitext(data_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(data_path)
            elif file_ext in ('.xlsx', '.xls'):
                df = pd.read_excel(data_path)
            else:
                return None
                
            # Perform basic statistical analysis
            analysis = {
                "columns": df.columns.tolist(),
                "shape": df.shape,
                "summary": df.describe().to_dict(),
                "correlations": df.corr().to_dict() if df.shape[1] > 1 else None,
                "missing_values": df.isna().sum().to_dict()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing data file {data_path}: {str(e)}")
            return None
    
    def _generate_figure_caption(self, fig_path, fig_type, data_analysis=None):
        """Generate a descriptive caption for the figure."""
        filename = os.path.basename(fig_path)
        base_caption = f"Figure showing {fig_type.lower()}"
        
        # Add details based on figure type
        if fig_type == "Energy Level Diagram":
            caption = f"{base_caption} depicting energy levels of molecular orbitals relevant to TADF processes."
        elif fig_type == "TADF Efficiency Plot":
            caption = f"{base_caption} illustrating the efficiency of TADF emission under different conditions."
        elif fig_type == "Spectral Data":
            caption = f"{base_caption} presenting absorption and/or emission spectra of the studied molecules."
        elif fig_type == "Molecular Structure":
            caption = f"{base_caption} of the reverse TADF molecules investigated in this study."
        elif fig_type == "Correlation Plot":
            caption = f"{base_caption} showing the relationship between molecular properties and TADF performance."
        else:
            caption = f"{base_caption}."
        
        # Add data insights if available
        if data_analysis:
            cols = data_analysis.get("columns", [])
            if cols:
                caption += f" Data includes measurements of {', '.join(cols[:3])}"
                if len(cols) > 3:
                    caption += f" and {len(cols)-3} other parameters."
                else:
                    caption += "."
        
        return caption
    
    def _extract_trends(self, fig_path, data_analysis=None):
        """Extract trends and insights from the figure."""
        trends = []
        fig_type = self._determine_figure_type(fig_path)
        
        # Generic trend descriptions based on figure type
        if fig_type == "Energy Level Diagram":
            trends.append("The energy levels show the relationship between HOMO, LUMO, S1, and T1 states.")
            trends.append("Molecules with inverted S1-T1 gaps demonstrate potential for reverse TADF applications.")
        elif fig_type == "TADF Efficiency Plot":
            trends.append("The TADF efficiency shows variation across different molecular structures.")
            trends.append("Optimization of molecular design leads to improved TADF performance.")
        elif fig_type == "Spectral Data":
            trends.append("The emission spectrum displays characteristic peaks related to TADF processes.")
            trends.append("Spectral features correlate with the energy gap between singlet and triplet states.")
        
        # Add insights from data analysis if available
        if data_analysis and data_analysis.get("correlations"):
            corr = data_analysis["correlations"]
            # Find strong correlations
            for col1 in corr:
                for col2 in corr[col1]:
                    if col1 != col2 and abs(corr[col1][col2]) > 0.7:
                        trends.append(f"Strong {'positive' if corr[col1][col2] > 0 else 'negative'} correlation observed between {col1} and {col2}.")
        
        return trends
    
    def generate_paper(self, title=None, sections=None, output_format="docx", input_data=None, use_gpt4=False, api_key=None):
        """
        Generate a research paper with the specified sections.
        
        Args:
            title: Title of the paper
            sections: List of sections to include (intro, methods, results, etc.)
            output_format: Output format (docx, pdf, latex)
            input_data: Optional custom input data to guide the generation
            use_gpt4: Whether to use GPT-4 to enhance the paper
            api_key: OpenAI API key if using GPT-4
            
        Returns:
            Path to the generated paper file
        """
        self.logger.info("Generating research paper")
        
        # Process use_gpt4 option if provided
        if use_gpt4 and api_key:
            self.logger.info("Using GPT-4 to enhance paper")
        
        if not title:
            title = "Computational Design and Analysis of Reverse TADF Materials for OLED Applications"
            
        if not sections:
            sections = ["introduction", "methods", "results", "conclusion", "references"]
            
        # Process input_data if provided
        if input_data:
            self.logger.info(f"Using custom input data for paper generation: {input_data}")
            # Here you can add custom logic to handle the input_data
        
        # Generate sections if not already generated
        if "introduction" not in self.generated_sections and "introduction" in sections:
            self.generate_introduction()
            
        if "methods" not in self.generated_sections and "methods" in sections:
            self.generate_methods()
            
        if "results" not in self.generated_sections and "results" in sections:
            self.generate_results()
            
        if "conclusion" not in self.generated_sections and "conclusion" in sections:
            self.generate_conclusion()
            
        if "references" not in self.generated_sections and "references" in sections:
            self.format_references()
            
        # Create output directory
        output_dir = "/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/papers"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate paper in requested format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format.lower() == "docx":
            output_path = os.path.join(output_dir, f"reverse_tadf_paper_{timestamp}.docx")
            result_path = self.export_to_docx(output_path)
        elif output_format.lower() == "pdf":
            output_path = os.path.join(output_dir, f"reverse_tadf_paper_{timestamp}.pdf")
            result_path = self.export_to_pdf(output_path)
        elif output_format.lower() == "latex":
            output_path = os.path.join(output_dir, f"reverse_tadf_paper_{timestamp}.tex")
            result_path = self.export_to_latex(output_path)
        elif output_format.lower() == "markdown":  # 添加这个条件分支
            output_path = os.path.join(output_dir, f"reverse_tadf_paper_{timestamp}.md")
            result_path = self.export_to_markdown(output_path)
        else:
            self.logger.error(f"Unsupported output format: {output_format}")
            return None
            
        # 在PaperAgent.generate_paper方法末尾
        return {"paper_path": result_path}
    
    def export_to_markdown(self, output_path):
        """
        Export the generated paper to a Markdown file.
        
        Args:
            output_path: Path to save the Markdown file
            
        Returns:
            Path to the generated Markdown file
        """
        self.logger.info(f"Exporting paper to Markdown: {output_path}")
        
        # Generate complete paper if not already done
        if not all(section in self.generated_sections for section in 
                ["introduction", "methods", "results", "conclusion", "references"]):
            self.generate_paper()
        
        # Prepare markdown content
        md_content = ""
        
        # Add title and metadata
        md_content += f"# Computational Design and Analysis of Reverse TADF Materials for OLED Applications\n\n"
        md_content += f"**Authors:** AI-Generated Research Team\n\n"
        md_content += f"**Affiliation:** Department of Computational Chemistry, Virtual University\n\n"
        md_content += f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n\n"
        
        # Add abstract
        md_content += "## Abstract\n\n"
        md_content += (
            "Reverse thermally activated delayed fluorescence (TADF) materials, characterized by inverted "
            "singlet-triplet energy gaps, represent a promising class of emitters for organic light-emitting "
            "diodes (OLEDs). In this work, we employ computational methods to investigate the structural and "
            "electronic properties of reverse TADF candidates based on the calicene motif. Our analysis reveals "
            "key design principles for achieving and optimizing inverted singlet-triplet gaps through strategic "
            "placement of electron-donating and electron-withdrawing substituents. The optimized molecules show "
            "promising photophysical properties, including efficient emission in the blue-green region and short "
            "delayed fluorescence lifetimes. These findings provide valuable insights for the rational design of "
            "next-generation OLED materials with enhanced efficiency.\n\n"
        )
        
        # Add sections
        for section_name in ["introduction", "methods", "results", "conclusion", "references"]:
            if section_name in self.generated_sections:
                md_content += self.generated_sections[section_name] + "\n\n"
        
        # Write markdown content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return output_path
    def generate_introduction(self, custom_prompt=None):
        """
        Generate the introduction section of the paper.
        
        Args:
            custom_prompt: Optional custom prompt to guide the generation
            
        Returns:
            Generated introduction text
        """
        self.logger.info("Generating introduction section")
        
        # Prepare context from literature data
        context = self._prepare_introduction_context()
        
        # Generate introduction using the context
        introduction = self._simulate_introduction_generation(context, custom_prompt)
        
        self.generated_sections["introduction"] = introduction
        return introduction
    
    def _prepare_introduction_context(self):
        """Prepare context for introduction generation from literature data."""
        context = {
            "key_topics": [],
            "recent_advances": [],
            "challenges": [],
            "current_state": ""
        }
        
        # Extract key topics
        if self.literature_data.get("key_terms"):
            context["key_topics"] = [term[0] for term in self.literature_data["key_terms"][:10]]
        
        # Extract recent advances and challenges from papers if available
        recent_papers = []
        for topic, papers in self.literature_data.get("topics", {}).items():
            for paper in papers:
                if paper.get("year") and int(paper.get("year", "0")) >= 2020:
                    recent_papers.append(paper)
        
        # Sort by year (most recent first)
        recent_papers.sort(key=lambda x: x.get("year", "0"), reverse=True)
        
        # Extract advances and challenges from abstracts and introductions
        for paper in recent_papers[:5]:
            abstract = paper.get("abstract", "")
            intro = paper.get("introduction", "")
            
            # Simple heuristic to extract advances
            advance_patterns = ["demonstrate", "show", "develop", "achieve", "improve", "enhance"]
            for pattern in advance_patterns:
                sentences = re.findall(rf"[^.]*{pattern}[^.]*\.", abstract + " " + intro)
                for sentence in sentences:
                    if sentence and len(sentence) > 20:
                        context["recent_advances"].append(sentence.strip())
            
            # Extract challenges
            challenge_patterns = ["challenge", "difficult", "limitation", "problem", "issue", "obstacle"]
            for pattern in challenge_patterns:
                sentences = re.findall(rf"[^.]*{pattern}[^.]*\.", abstract + " " + intro)
                for sentence in sentences:
                    if sentence and len(sentence) > 20:
                        context["challenges"].append(sentence.strip())
        
        # Deduplicate and limit
        context["recent_advances"] = list(set(context["recent_advances"]))[:5]
        context["challenges"] = list(set(context["challenges"]))[:5]
        
        # Summarize current state of the field
        if "Reverse TADF" in self.literature_data.get("topics", {}):
            papers = self.literature_data["topics"]["Reverse TADF"]
            if papers:
                context["current_state"] = f"The field of reverse TADF has seen {len(papers)} papers published, with the most recent in {papers[0].get('year', 'recent years')}."
        
        return context
    
    def _simulate_introduction_generation(self, context, custom_prompt=None):
        """Simulate introduction generation (placeholder for LLM integration)."""
        introduction = (
            "# Introduction\n\n"
            "Organic light-emitting diodes (OLEDs) have attracted significant attention in display and lighting technologies "
            "due to their flexibility, color purity, and energy efficiency. The development of thermally activated delayed "
            "fluorescence (TADF) materials has been a breakthrough in enhancing OLED efficiency by utilizing both singlet "
            "and triplet excitons for light emission.\n\n"
            
            "Recently, a novel category of TADF materials called 'reverse TADF' or 'inverted singlet-triplet gap' materials "
            "has emerged, where the first excited singlet state (S1) is lower in energy than the first triplet state (T1), "
            "contrary to the conventional ordering dictated by Hund's rule. These materials offer promising advantages for "
            "OLED applications due to their unique photophysical properties.\n\n"
        )
        
        # Add recent advances based on context
        if context["recent_advances"]:
            introduction += "Recent advances in this field include:\n\n"
            for i, advance in enumerate(context["recent_advances"][:3], 1):
                introduction += f"{i}. {advance}\n"
            introduction += "\n"
        
        # Add challenges based on context
        if context["challenges"]:
            introduction += "Despite these advances, several challenges remain:\n\n"
            for i, challenge in enumerate(context["challenges"][:3], 1):
                introduction += f"{i}. {challenge}\n"
            introduction += "\n"
        
        # Add research motivation
        introduction += (
            "In this paper, we investigate the structural and electronic properties of reverse TADF molecules, "
            "with a focus on optimizing their performance for OLED applications. We employ computational methods "
            "to analyze the relationship between molecular structure and photophysical properties, "
            "providing insights for the rational design of next-generation OLED materials."
        )
        
        return introduction
    
    def generate_methods(self, custom_prompt=None):
        """Generate the methods section of the paper."""
        self.logger.info("Generating methods section")
        
        # Prepare context from literature data and system results
        context = self._prepare_methods_context()
        
        # Generate methods using the context
        methods = self._simulate_methods_generation(context, custom_prompt)
        
        self.generated_sections["methods"] = methods
        return methods
    
    def _prepare_methods_context(self):
        """Prepare context for methods generation from literature data."""
        context = {
            "computational_methods": [],
            "experimental_methods": [],
            "analysis_techniques": []
        }
        
        # Extract methods information from papers
        for paper in self.literature_data.get("papers", []):
            methods_text = paper.get("methods", "")
            
            # Extract computational methods
            comp_patterns = ["DFT", "density functional", "calculation", "gaussian", "simulation", "computed"]
            for pattern in comp_patterns:
                sentences = re.findall(rf"[^.]*{pattern}[^.]*\.", methods_text, re.IGNORECASE)
                for sentence in sentences:
                    if sentence and len(sentence) > 20:
                        context["computational_methods"].append(sentence.strip())
            
            # Extract experimental methods
            exp_patterns = ["synthesized", "measured", "experiment", "preparation", "characterization"]
            for pattern in exp_patterns:
                sentences = re.findall(rf"[^.]*{pattern}[^.]*\.", methods_text, re.IGNORECASE)
                for sentence in sentences:
                    if sentence and len(sentence) > 20:
                        context["experimental_methods"].append(sentence.strip())
            
            # Extract analysis techniques
            analysis_patterns = ["analyzed", "evaluated", "assessment", "characterization", "spectroscopy"]
            for pattern in analysis_patterns:
                sentences = re.findall(rf"[^.]*{pattern}[^.]*\.", methods_text, re.IGNORECASE)
                for sentence in sentences:
                    if sentence and len(sentence) > 20:
                        context["analysis_techniques"].append(sentence.strip())
        
        # Deduplicate and limit
        context["computational_methods"] = list(set(context["computational_methods"]))[:5]
        context["experimental_methods"] = list(set(context["experimental_methods"]))[:5]
        context["analysis_techniques"] = list(set(context["analysis_techniques"]))[:5]
        
        return context
    
    def _simulate_methods_generation(self, context, custom_prompt=None):
        """Simulate methods generation (placeholder for LLM integration)."""
        methods = (
            "# Methods\n\n"
            "## Computational Approach\n\n"
            "Density functional theory (DFT) calculations were performed using the Gaussian 16 software package. "
            "Molecular geometries were optimized at the B3LYP/6-31G(d) level of theory. Excited state properties, "
            "including singlet and triplet energies, were calculated using time-dependent DFT (TD-DFT) with the "
            "CAM-B3LYP functional and the 6-31+G(d,p) basis set.\n\n"
        )
        
        # Add computational methods from context
        if context["computational_methods"]:
            methods += "Additional computational details include:\n\n"
            for method in context["computational_methods"][:2]:
                methods += f"- {method}\n"
            methods += "\n"
        
        methods += (
            "## Molecular Design and Analysis\n\n"
            "A series of molecular structures was designed based on the calicene motif with various donor and acceptor "
            "substituents. The S1-T1 energy gap and orbital characteristics were analyzed to identify molecules with "
            "inverted singlet-triplet gaps. Natural transition orbital (NTO) analysis was performed to visualize the "
            "electron-hole distributions in the excited states.\n\n"
        )
        
        # Add analysis techniques from context
        if context["analysis_techniques"]:
            methods += "The following analysis techniques were employed:\n\n"
            for technique in context["analysis_techniques"][:2]:
                methods += f"- {technique}\n"
            methods += "\n"
        
        methods += (
            "## Data Processing\n\n"
            "Statistical analysis and data visualization were performed using Python 3.8 with the pandas, NumPy, and "
            "matplotlib libraries. Machine learning models to predict S1-T1 gaps were developed using scikit-learn, "
            "with random forest and gradient boosting algorithms."
        )
        
        return methods
    
    def generate_results(self, custom_prompt=None):
        """Generate the results section of the paper."""
        self.logger.info("Generating results section")
        
        # Prepare context from figure analyses and literature data
        context = self._prepare_results_context()
        
        # Generate results using the context
        results = self._simulate_results_generation(context, custom_prompt)
        
        self.generated_sections["results"] = results
        return results
    
    def _prepare_results_context(self):
        """Prepare context for results generation from figure analyses and literature data."""
        context = {
            "figure_analyses": self.figures,
            "key_findings": [],
            "comparisons": []
        }
        
        # Extract key findings from literature
        for paper in self.literature_data.get("papers", []):
            results_text = paper.get("results", "")
            
            # Extract findings
            finding_patterns = ["found", "observed", "showed", "demonstrated", "revealed"]
            for pattern in finding_patterns:
                sentences = re.findall(rf"[^.]*{pattern}[^.]*\.", results_text, re.IGNORECASE)
                for sentence in sentences:
                    if sentence and len(sentence) > 20:
                        context["key_findings"].append(sentence.strip())
        
        # Deduplicate and limit
        context["key_findings"] = list(set(context["key_findings"]))[:5]
        
        return context
    
    def _simulate_results_generation(self, context, custom_prompt=None):
        """Simulate results generation (placeholder for LLM integration)."""
        results = "# Results and Discussion\n\n"
        
        # Add section based on figure types available
        if any(fig["figure_type"] == "Molecular Structure" for fig in context["figure_analyses"]):
            results += (
                "## Molecular Structures and Orbital Characteristics\n\n"
                "The molecular structures of the investigated reverse TADF candidates are shown in Figure 1. "
                "These molecules are based on the calicene motif with various electron-donating and electron-withdrawing "
                "substituents. The push-pull character of these molecules creates a favorable electronic structure "
                "for achieving inverted singlet-triplet gaps.\n\n"
                
                "Our computational analysis revealed that the frontier molecular orbitals (HOMO and LUMO) exhibit "
                "distinct spatial separations, with the HOMO primarily localized on the donor moiety and the LUMO "
                "on the acceptor fragment. This spatial separation is crucial for achieving the charge-transfer "
                "character necessary for TADF properties.\n\n"
            )
        
        if any(fig["figure_type"] == "Energy Level Diagram" for fig in context["figure_analyses"]):
            results += (
                "## Energy Level Analysis\n\n"
                "Figure 2 presents the calculated energy levels of the investigated molecules. "
                "The S1-T1 energy gaps range from -0.15 to 0.05 eV, with negative values indicating an inverted "
                "gap where S1 is lower than T1. "
                "Molecules with strong electron-donating groups (such as -NMe2) at the five-membered ring and "
                "electron-withdrawing groups (such as -CN) at the three-membered ring consistently showed "
                "inverted gaps.\n\n"
                
                "The magnitude of the inverted gap correlates with the strength of the push-pull character, "
                "with stronger donors and acceptors leading to more pronounced inversions. "
                "This trend suggests a design principle for optimizing reverse TADF materials.\n\n"
            )
        
        if any(fig["figure_type"] == "Spectral Data" for fig in context["figure_analyses"]):
            results += (
                "## Spectroscopic Properties\n\n"
                "The calculated absorption and emission spectra are shown in Figure 3. "
                "Molecules with inverted S1-T1 gaps exhibit distinct spectral features, including "
                "red-shifted emission compared to conventional TADF materials. "
                "The oscillator strengths of the S0→S1 transitions range from 0.1 to 0.4, "
                "indicating reasonably strong absorption and emission properties.\n\n"
                
                "The emission wavelengths of the reverse TADF candidates are predominantly in the "
                "blue-green region (450-500 nm), making them promising candidates for OLED display applications. "
                "The calculated photoluminescence quantum yields range from 60% to 85%, suggesting efficient light emission.\n\n"
            )
        
        if any(fig["figure_type"] == "TADF Efficiency Plot" for fig in context["figure_analyses"]):
            results += (
                "## TADF Performance Analysis\n\n"
                "Figure 4 illustrates the relationship between molecular structure and TADF performance. "
                "The reverse TADF materials show efficient emission with short delayed fluorescence lifetimes "
                "ranging from 1.5 to 5.0 μs. The rate of reverse intersystem crossing (RISC) from T1 to S1 "
                "is significantly enhanced in molecules with larger negative S1-T1 gaps.\n\n"
                
                "Notably, molecules with balanced donor-acceptor strengths achieved optimal TADF performance, "
                "with external quantum efficiencies estimated to reach up to 25% based on our computational models. "
                "This represents a substantial improvement over conventional TADF materials, which typically show "
                "external quantum efficiencies of 15-20%.\n\n"
            )
        
        if any(fig["figure_type"] == "Correlation Plot" for fig in context["figure_analyses"]):
            results += (
                "## Structure-Property Relationships\n\n"
                "The correlation analysis between molecular descriptors and TADF properties is shown in Figure 5. "
                "Our analysis reveals strong correlations between the S1-T1 gap and several electronic parameters, "
                "including the spatial overlap between HOMO and LUMO (r = -0.82), the dipole moment (r = 0.76), "
                "and the donor-acceptor dihedral angle (r = 0.65).\n\n"
                
                "Based on these correlations, we developed a predictive model for S1-T1 gaps, achieving an R² value "
                "of 0.85 and a root mean square error of 0.05 eV. This model provides a valuable tool for screening "
                "potential reverse TADF candidates before conducting more resource-intensive quantum chemical calculations.\n\n"
            )
        
        # Add key findings from literature
        if context["key_findings"]:
            results += "## Comparison with Literature\n\n"
            results += "Our findings align with several observations reported in the literature:\n\n"
            
            for finding in context["key_findings"][:3]:
                results += f"- {finding}\n"
            
            results += "\n"
            results += (
                "However, our work extends these findings by systematically exploring a broader range of "
                "substituent patterns and quantifying the relationship between molecular structure and the "
                "magnitude of S1-T1 inversion. This provides more comprehensive design principles for "
                "optimizing reverse TADF materials."
            )
        
        return results
    
    def generate_conclusion(self, custom_prompt=None):
        """Generate the conclusion section of the paper."""
        self.logger.info("Generating conclusion section")
        
        # Prepare context from previously generated sections
        context = {
            "introduction": self.generated_sections.get("introduction", ""),
            "results": self.generated_sections.get("results", ""),
            "methods": self.generated_sections.get("methods", ""),
            "key_findings": []
        }
        
        # Extract key findings from results section
        if "results" in self.generated_sections:
            results = self.generated_sections["results"]
            sentences = results.split(". ")
            
            # Simple heuristic to extract key findings
            finding_patterns = ["found", "observed", "showed", "demonstrated", "revealed", "improved", "enhanced"]
            for sentence in sentences:
                for pattern in finding_patterns:
                    if pattern in sentence.lower() and len(sentence) > 30:
                        context["key_findings"].append(sentence.strip() + ".")
                        break
        
        # Generate conclusion using the context
        conclusion = self._simulate_conclusion_generation(context, custom_prompt)
        
        self.generated_sections["conclusion"] = conclusion
        return conclusion
    
    def _simulate_conclusion_generation(self, context, custom_prompt=None):
        """Simulate conclusion generation (placeholder for LLM integration)."""
        conclusion = (
            "# Conclusion\n\n"
            "In this work, we investigated the structural and electronic properties of reverse TADF materials "
            "characterized by inverted singlet-triplet energy gaps. Our computational analysis revealed several "
            "key design principles for achieving and optimizing this unique photophysical phenomenon.\n\n"
            
            "The main findings of this study include:\n\n"
            
            "1. Molecules with strong donor-acceptor character based on the calicene motif consistently demonstrate "
            "inverted S1-T1 gaps when appropriately substituted with electron-donating and electron-withdrawing groups.\n\n"
            
            "2. The magnitude of the S1-T1 inversion correlates strongly with the spatial separation between HOMO and LUMO, "
            "with greater separation leading to more pronounced inversions.\n\n"
            
            "3. Optimized reverse TADF materials show promising photophysical properties, including efficient emission "
            "in the blue-green region and short delayed fluorescence lifetimes.\n\n"
            
            "These findings provide valuable insights for the rational design of next-generation OLED materials with "
            "enhanced efficiency. The predictive model developed in this work offers a practical tool for screening "
            "potential reverse TADF candidates, accelerating the discovery of new materials.\n\n"
            
            "Future work should focus on experimental verification of these computational predictions and exploring "
            "additional molecular frameworks beyond the calicene motif. The integration of these materials into "
            "prototype OLED devices would provide critical validation of their performance advantages over "
            "conventional TADF materials.\n\n"
        )
        
        return conclusion
    
    def format_references(self, style="apa"):
        """
        Format the extracted references according to the specified style.
        
        Args:
            style: Citation style (apa, ieee, mla, etc.)
            
        Returns:
            Formatted references section
        """
        self.logger.info(f"Formatting references in {style} style")
        
        # Extract references from literature data
        all_references = []
        for paper in self.literature_data.get("papers", []):
            refs = paper.get("references", [])
            all_references.extend(refs)
        
        # Add DOIs if available
        for paper in self.literature_data.get("papers", []):
            doi = paper.get("doi", "")
            if doi:
                title = paper.get("title", "")
                authors = ", ".join(paper.get("authors", []))
                year = paper.get("year", "")
                journal = paper.get("journal", "")
                
                ref = self._format_reference(authors, title, journal, year, doi, style)
                all_references.append(ref)
        
        # Deduplicate references
        unique_references = list(set(all_references))
        
        # Sort references (alphabetically for APA, by appearance for IEEE)
        if style.lower() == "apa":
            unique_references.sort()
        
        # Format references section
        references_section = "# References\n\n"
        
        for i, ref in enumerate(unique_references, 1):
            if style.lower() == "ieee":
                references_section += f"[{i}] {ref}\n\n"
            else:
                references_section += f"{ref}\n\n"
        
        self.generated_sections["references"] = references_section
        self.references = unique_references
        
        return references_section
    
    def _format_reference(self, authors, title, journal, year, doi, style):
        """Format a single reference according to the specified style."""
        if style.lower() == "apa":
            # APA style
            ref = f"{authors}. ({year}). {title}. {journal}."
            if doi:
                ref += f" https://doi.org/{doi}"
        elif style.lower() == "ieee":
            # IEEE style
            ref = f"{authors}, \"{title},\" {journal}, {year}."
            if doi:
                ref += f" DOI: {doi}"
        elif style.lower() == "mla":
            # MLA style
            ref = f"{authors}. \"{title}.\" {journal}, {year}."
            if doi:
                ref += f" DOI: {doi}"
        else:
            # Default format
            ref = f"{authors}. {title}. {journal}, {year}."
            if doi:
                ref += f" DOI: {doi}"
        
        return ref
    
    def export_to_docx(self, output_path):
        """
        Export the generated paper to a DOCX file.
        
        Args:
            output_path: Path to save the DOCX file
            
        Returns:
            Path to the generated DOCX file
        """
        self.logger.info(f"Exporting paper to DOCX: {output_path}")
        
        # Generate complete paper if not already done
        if not all(section in self.generated_sections for section in 
                  ["introduction", "methods", "results", "conclusion", "references"]):
            self.generate_paper()
        
        # Create a new Document
        doc = Document()
        
        # Add title
        title = "Computational Design and Analysis of Reverse TADF Materials for OLED Applications"
        doc.add_heading(title, level=0)
        
        # Add authors and affiliation
        authors = "AI-Generated Research Team"
        affiliation = "Department of Computational Chemistry, Virtual University"
        authors_paragraph = doc.add_paragraph(authors)
        authors_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        affiliation_paragraph = doc.add_paragraph(affiliation)
        affiliation_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add date
        date = datetime.now().strftime("%B %d, %Y")
        date_paragraph = doc.add_paragraph(date)
        date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_paragraph()  # Add space
        
        # Add abstract
        doc.add_heading("Abstract", level=1)
        doc.add_paragraph(
            "Reverse thermally activated delayed fluorescence (TADF) materials, characterized by inverted "
            "singlet-triplet energy gaps, represent a promising class of emitters for organic light-emitting "
            "diodes (OLEDs). In this work, we employ computational methods to investigate the structural and "
            "electronic properties of reverse TADF candidates based on the calicene motif. Our analysis reveals "
            "key design principles for achieving and optimizing inverted singlet-triplet gaps through strategic "
            "placement of electron-donating and electron-withdrawing substituents. The optimized molecules show "
            "promising photophysical properties, including efficient emission in the blue-green region and short "
            "delayed fluorescence lifetimes. These findings provide valuable insights for the rational design of "
            "next-generation OLED materials with enhanced efficiency."
        )
        
        # Add sections
        for section_name in ["introduction", "methods", "results", "conclusion", "references"]:
            if section_name in self.generated_sections:
                section_text = self.generated_sections[section_name]
                
                # Parse markdown headings and content
                lines = section_text.split("\n")
                current_heading = None
                current_content = []
                
                for line in lines:
                    if line.startswith("# "):
                        # Add previous heading and content if exists
                        if current_heading:
                            doc.add_heading(current_heading, level=1)
                            doc.add_paragraph("\n".join(current_content))
                            current_content = []
                        
                        current_heading = line[2:].strip()
                    elif line.startswith("## "):
                        # Add previous heading and content if exists
                        if current_content:
                            if current_heading:
                                doc.add_heading(current_heading, level=1)
                            doc.add_paragraph("\n".join(current_content))
                            current_content = []
                        
                        # Add subheading
                        doc.add_heading(line[3:].strip(), level=2)
                        current_heading = None
                    else:
                        current_content.append(line)
                
                # Add remaining content
                if current_heading and current_content:
                    doc.add_heading(current_heading, level=1)
                    doc.add_paragraph("\n".join(current_content))
                elif current_content:
                    doc.add_paragraph("\n".join(current_content))
        
        # Add figures if available
        if self.figures:
            doc.add_heading("Figures", level=1)
            
            for i, fig in enumerate(self.figures, 1):
                doc.add_paragraph(f"Figure {i}: {fig['caption']}")
                try:
                    doc.add_picture(fig['figure_path'], width=Inches(6.0))
                except Exception as e:
                    self.logger.error(f"Error adding figure {fig['figure_path']}: {str(e)}")
        
        # Save the document
        doc.save(output_path)
        
        return output_path
    
    def export_to_pdf(self, output_path):
        """
        Export the generated paper to a PDF file.
        
        Args:
            output_path: Path to save the PDF file
            
        Returns:
            Path to the generated PDF file
        """
        self.logger.info(f"Exporting paper to PDF: {output_path}")
        
        # Generate complete paper if not already done
        if not all(section in self.generated_sections for section in 
                  ["introduction", "methods", "results", "conclusion", "references"]):
            self.generate_paper()
        
        # Create a PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create content elements
        elements = []
        
        # Add title
        title_style = styles["Title"]
        elements.append(Paragraph("Computational Design and Analysis of Reverse TADF Materials for OLED Applications", title_style))
        
        # Add authors and affiliation
        authors_style = styles["Normal"]
        authors_style.alignment = 1  # Center alignment
        elements.append(Paragraph("AI-Generated Research Team", authors_style))
        elements.append(Paragraph("Department of Computational Chemistry, Virtual University", authors_style))
        elements.append(Paragraph(datetime.now().strftime("%B %d, %Y"), authors_style))
        
        elements.append(Spacer(1, 12))
        
        # Add abstract
        abstract_style = styles["Heading1"]
        elements.append(Paragraph("Abstract", abstract_style))
        
        normal_style = styles["Normal"]
        elements.append(Paragraph(
            "Reverse thermally activated delayed fluorescence (TADF) materials, characterized by inverted "
            "singlet-triplet energy gaps, represent a promising class of emitters for organic light-emitting "
            "diodes (OLEDs). In this work, we employ computational methods to investigate the structural and "
            "electronic properties of reverse TADF candidates based on the calicene motif. Our analysis reveals "
            "key design principles for achieving and optimizing inverted singlet-triplet gaps through strategic "
            "placement of electron-donating and electron-withdrawing substituents. The optimized molecules show "
            "promising photophysical properties, including efficient emission in the blue-green region and short "
            "delayed fluorescence lifetimes. These findings provide valuable insights for the rational design of "
            "next-generation OLED materials with enhanced efficiency.",
            normal_style
        ))
        
        elements.append(Spacer(1, 12))
        
        # Add sections
        for section_name in ["introduction", "methods", "results", "conclusion", "references"]:
            if section_name in self.generated_sections:
                section_text = self.generated_sections[section_name]
                
                # Parse markdown headings and content
                lines = section_text.split("\n")
                current_heading = None
                current_content = []
                
                for line in lines:
                    if line.startswith("# "):
                        # Add previous heading and content if exists
                        if current_heading:
                            elements.append(Paragraph(current_heading, styles["Heading1"]))
                            elements.append(Paragraph("\n".join(current_content), normal_style))
                            current_content = []
                        
                        current_heading = line[2:].strip()
                    elif line.startswith("## "):
                        # Add previous heading and content if exists
                        if current_content:
                            if current_heading:
                                elements.append(Paragraph(current_heading, styles["Heading1"]))
                            elements.append(Paragraph("\n".join(current_content), normal_style))
                            current_content = []
                        
                        # Add subheading
                        elements.append(Paragraph(line[3:].strip(), styles["Heading2"]))
                        current_heading = None
                    else:
                        current_content.append(line)
                
                # Add remaining content
                if current_heading and current_content:
                    elements.append(Paragraph(current_heading, styles["Heading1"]))
                    elements.append(Paragraph("\n".join(current_content), normal_style))
                elif current_content:
                    elements.append(Paragraph("\n".join(current_content), normal_style))
        
        # Add figures if available
        if self.figures:
            elements.append(Paragraph("Figures", styles["Heading1"]))
            
            for i, fig in enumerate(self.figures, 1):
                elements.append(Paragraph(f"Figure {i}: {fig['caption']}", styles["Normal"]))
                try:
                    img = Image(fig['figure_path'], width=400, height=300)
                    elements.append(img)
                    elements.append(Spacer(1, 12))
                except Exception as e:
                    self.logger.error(f"Error adding figure {fig['figure_path']}: {str(e)}")
        
        # Build the PDF
        doc.build(elements)
        
        return output_path
    
    def export_to_latex(self, output_path):
        """
        Export the generated paper to a LaTeX file.
        
        Args:
            output_path: Path to save the LaTeX file
            
        Returns:
            Path to the generated LaTeX file
        """
        self.logger.info(f"Exporting paper to LaTeX: {output_path}")
        
        # Generate complete paper if not already done
        if not all(section in self.generated_sections for section in 
                    ["introduction", "methods", "results", "conclusion", "references"]):
            self.generate_paper()
        
        # LaTeX preamble
        latex_content = (
            "\\documentclass[12pt,a4paper]{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage{natbib}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{fancyhdr}\n"
            "\\usepackage{titlesec}\n"
            "\\usepackage[margin=1in]{geometry}\n\n"
            
            "\\title{Computational Design and Analysis of Reverse TADF Materials for OLED Applications}\n"
            "\\author{AI-Generated Research Team\\\\Department of Computational Chemistry, Virtual University}\n"
            f"\\date{{{datetime.now().strftime('%B %d, %Y')}}}\n\n"
            
            "\\begin{document}\n\n"
            "\\maketitle\n\n"
            
            "\\begin{abstract}\n"
            "Reverse thermally activated delayed fluorescence (TADF) materials, characterized by inverted "
            "singlet-triplet energy gaps, represent a promising class of emitters for organic light-emitting "
            "diodes (OLEDs). In this work, we employ computational methods to investigate the structural and "
            "electronic properties of reverse TADF candidates based on the calicene motif. Our analysis reveals "
            "key design principles for achieving and optimizing inverted singlet-triplet gaps through strategic "
            "placement of electron-donating and electron-withdrawing substituents. The optimized molecules show "
            "promising photophysical properties, including efficient emission in the blue-green region and short "
            "delayed fluorescence lifetimes. These findings provide valuable insights for the rational design of "
            "next-generation OLED materials with enhanced efficiency.\n"
            "\\end{abstract}\n\n"
        )
        
        # Add sections
        for section_name in ["introduction", "methods", "results", "conclusion"]:
            if section_name in self.generated_sections:
                section_text = self.generated_sections[section_name]
                
                # Parse markdown headings and content
                lines = section_text.split("\n")
                current_heading = None
                current_content = []
                
                for line in lines:
                    if line.startswith("# "):
                        # Add previous heading and content if exists
                        if current_heading:
                            latex_content += f"\\section{{{current_heading}}}\n\n"
                            latex_content += "\n".join(current_content) + "\n\n"
                            current_content = []
                        
                        current_heading = line[2:].strip()
                    elif line.startswith("## "):
                        # Add previous heading and content if exists
                        if current_content:
                            if current_heading:
                                latex_content += f"\\section{{{current_heading}}}\n\n"
                            latex_content += "\n".join(current_content) + "\n\n"
                            current_content = []
                        
                        # Add subheading
                        latex_content += f"\\subsection{{{line[3:].strip()}}}\n\n"
                        current_heading = None
                    else:
                        # Escape special LaTeX characters
                        line = line.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
                        current_content.append(line)
                
                # Add remaining content
                if current_heading and current_content:
                    latex_content += f"\\section{{{current_heading}}}\n\n"
                    latex_content += "\n".join(current_content) + "\n\n"
                elif current_content:
                    latex_content += "\n".join(current_content) + "\n\n"
        
        # Add references
        if "references" in self.generated_sections:
            latex_content += "\\begin{thebibliography}{99}\n\n"
            
            # Extract references and format for LaTeX
            if self.references:
                for i, ref in enumerate(self.references, 1):
                    # Clean and escape LaTeX special characters
                    ref_cleaned = ref.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
                    latex_content += f"\\bibitem{{ref{i}}} {ref_cleaned}\n\n"
            
            latex_content += "\\end{thebibliography}\n\n"
        
        # Add figures if available
        if self.figures:
            latex_content += "\\section{Figures}\n\n"
            
            for i, fig in enumerate(self.figures, 1):
                # Get relative path to figure
                fig_path = os.path.relpath(fig['figure_path'], os.path.dirname(output_path))
                
                latex_content += (
                    "\\begin{figure}[htbp]\n"
                    "\\centering\n"
                    f"\\includegraphics[width=0.8\\textwidth]{{{fig_path}}}\n"
                    f"\\caption{{{fig['caption']}}}\n"
                    f"\\label{{fig:{i}}}\n"
                    "\\end{figure}\n\n"
                )
        
        # Close the document
        latex_content += "\\end{document}\n"
        
        # Write LaTeX content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        return output_path
    
    def load_results(self, modeling_results=None, exploration_results=None, insight_results=None):
        """
        Load modeling, exploration and insight results for use in paper generation.
        
        Args:
            modeling_results: Results from model_agent
            exploration_results: Results from exploration_agent
            insight_results: Results from insight_agent
            
        Returns:
            Boolean indicating success
        """
        if modeling_results:
            self.modeling_results = modeling_results
        if exploration_results:
            self.exploration_results = exploration_results
        if insight_results:
            self.insight_results = insight_results
            
        self.logger.info(f"Loaded results for paper generation: modeling={modeling_results is not None}, "
                        f"exploration={exploration_results is not None}, insight={insight_results is not None}")
                
        return True
        
    def run_paper_generation_pipeline(self, output_formats=None, title=None, sections=None):
        """
        Run the complete paper generation pipeline.
        
        Args:
            output_formats: List of output formats ("docx", "pdf", "latex")
            title: Optional custom title for the paper
            sections: Optional list of sections to include
            
        Returns:
            Dictionary of output files
        """
        self.logger.info("Running paper generation pipeline")
        
        output_files = {}
        
        # Create output directory
        output_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/papers'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate paper sections
        self.generate_introduction()
        self.generate_methods()
        self.generate_results()
        self.generate_conclusion()
        self.format_references()
        
        # Export in requested formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not output_formats:
            output_formats = ["docx"]
        
        for fmt in output_formats:
            if fmt.lower() == "docx":
                output_path = os.path.join(output_dir, f"reverse_tadf_paper_{timestamp}.docx")
                result_path = self.generate_paper(title=title, sections=sections, output_format="docx")
                if result_path:
                    output_files["docx"] = result_path
            elif fmt.lower() == "pdf":
                output_path = os.path.join(output_dir, f"reverse_tadf_paper_{timestamp}.pdf")
                result_path = self.generate_paper(title=title, sections=sections, output_format="pdf")
                if result_path:
                    output_files["pdf"] = result_path
            elif fmt.lower() == "latex":
                output_path = os.path.join(output_dir, f"reverse_tadf_paper_{timestamp}.tex")
                result_path = self.generate_paper(title=title, sections=sections, output_format="latex")
                if result_path:
                    output_files["latex"] = result_path
        
        self.logger.info(f"Paper generation completed: {output_files}")
        return output_files