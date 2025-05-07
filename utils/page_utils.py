# utils/paper_utils.py
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from io import BytesIO
import base64
import json
import requests
from datetime import datetime
from docx import Document
import PyPDF2
import xml.etree.ElementTree as ET

class PaperUtils:
    """
    Utility functions for processing scientific papers and literature.
    """
    
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {
                # utils/paper_utils.py (继续)
            logging.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        
        return text
    
    @staticmethod
    def extract_text_from_docx(docx_path):
        """
        Extract text content from a DOCX file.
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            Extracted text as a string
        """
        text = ""
        try:
            doc = Document(docx_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            logging.error(f"Error extracting text from DOCX {docx_path}: {str(e)}")
        
        return text
    
    @staticmethod
    def parse_web_of_science_txt(txt_path):
        """
        Parse Web of Science exported TXT file to extract paper information.
        
        Args:
            txt_path: Path to the Web of Science TXT file
            
        Returns:
            List of papers with extracted information
        """
        papers = []
        current_paper = {}
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split content into individual papers
            paper_blocks = content.split("\nER\n")
            
            for block in paper_blocks:
                if not block.strip():
                    continue
                
                # Initialize a new paper
                paper = {}
                
                # Extract fields
                lines = block.split("\n")
                current_field = None
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    # Check if this is a new field
                    if len(line) >= 2 and line[2] == " " and line[:2].strip():
                        field_code = line[:2].strip()
                        field_value = line[3:].strip()
                        
                        current_field = field_code
                        
                        if field_code not in paper:
                            paper[field_code] = []
                            
                        paper[field_code].append(field_value)
                    
                    # Continuation of previous field
                    elif current_field and line.startswith("  "):
                        paper[current_field][-1] += " " + line.strip()
                
                # Convert field codes to readable names
                readable_paper = {}
                field_mapping = {
                    "TI": "title",
                    "AU": "authors",
                    "AF": "author_full",
                    "AB": "abstract",
                    "PY": "year",
                    "JO": "journal",
                    "SO": "source",
                    "DI": "doi",
                    "UT": "unique_id",
                    "DE": "keywords",
                    "ID": "keywords_plus",
                    "CR": "cited_references"
                }
                
                for code, values in paper.items():
                    field_name = field_mapping.get(code, code)
                    
                    # Special handling for single-value fields
                    if field_name in ["title", "year", "doi", "abstract"]:
                        readable_paper[field_name] = values[0] if values else ""
                    else:
                        readable_paper[field_name] = values
                
                papers.append(readable_paper)
                
            return papers
            
        except Exception as e:
            logging.error(f"Error parsing Web of Science TXT file {txt_path}: {str(e)}")
            return []
    
    @staticmethod
    def parse_bibtex(bibtex_path):
        """
        Parse BibTeX file to extract paper information.
        
        Args:
            bibtex_path: Path to the BibTeX file
            
        Returns:
            List of papers with extracted information
        """
        papers = []
        
        try:
            with open(bibtex_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split by entries
            entry_pattern = r'@(\w+)\s*\{([^,]*),([\s\S]*?)\n\}'
            entries = re.findall(entry_pattern, content)
            
            for entry_type, entry_key, entry_content in entries:
                paper = {
                    "entry_type": entry_type,
                    "entry_key": entry_key,
                }
                
                # Extract fields
                field_pattern = r'\s*(\w+)\s*=\s*\{([\s\S]*?)\}(?=,\s*\w+\s*=|\s*\n)'
                fields = re.findall(field_pattern, entry_content + "\n")
                
                for field_name, field_value in fields:
                    field_name = field_name.lower()
                    paper[field_name] = field_value.strip()
                
                # Map to common field names
                mapping = {
                    "title": "title",
                    "author": "authors",
                    "year": "year",
                    "journal": "journal",
                    "abstract": "abstract",
                    "doi": "doi",
                    "volume": "volume",
                    "number": "number",
                    "pages": "pages",
                    "keywords": "keywords"
                }
                
                readable_paper = {}
                for bibtex_field, standard_field in mapping.items():
                    if bibtex_field in paper:
                        readable_paper[standard_field] = paper[bibtex_field]
                
                # Handle authors (split string into list)
                if "authors" in readable_paper:
                    authors_str = readable_paper["authors"]
                    authors = [a.strip() for a in authors_str.split(" and ")]
                    readable_paper["authors"] = authors
                
                papers.append(readable_paper)
                
            return papers
            
        except Exception as e:
            logging.error(f"Error parsing BibTeX file {bibtex_path}: {str(e)}")
            return []
    
    @staticmethod
    def extract_chemical_structures(text):
        """
        Extract chemical structure information from text.
        
        Args:
            text: Text containing chemical information
            
        Returns:
            List of extracted chemical structures
        """
        structures = []
        
        # Extract SMILES notation
        smiles_pattern = r'SMILES[:\s=]+([^\s;,]+)'
        smiles_matches = re.findall(smiles_pattern, text, re.IGNORECASE)
        
        for smiles in smiles_matches:
            structures.append({
                "type": "SMILES",
                "value": smiles
            })
        
        # Extract chemical formulas (simple pattern)
        formula_pattern = r'\b([A-Z][a-z]?\d*)+\b'
        formula_matches = re.findall(formula_pattern, text)
        
        for formula in formula_matches:
            # Filter out common non-chemical terms that might match the pattern
            if len(formula) > 1 and re.search(r'\d', formula):
                structures.append({
                    "type": "Formula",
                    "value": formula
                })
        
        return structures
    
    @staticmethod
    def extract_numerical_data(text):
        """
        Extract numerical data and measurements from text.
        
        Args:
            text: Text containing numerical data
            
        Returns:
            Dictionary of extracted numerical data
        """
        data = {
            "energy_levels": [],
            "wavelengths": [],
            "efficiencies": [],
            "temperatures": [],
            "other_measurements": []
        }
        
        # Extract energy levels (eV)
        energy_pattern = r'(-?\d+\.?\d*)\s*(?:eV|electron\s*volt)'
        energy_matches = re.findall(energy_pattern, text, re.IGNORECASE)
        data["energy_levels"] = [float(e) for e in energy_matches]
        
        # Extract wavelengths (nm)
        wavelength_pattern = r'(\d+\.?\d*)\s*nm'
        wavelength_matches = re.findall(wavelength_pattern, text, re.IGNORECASE)
        data["wavelengths"] = [float(w) for w in wavelength_matches]
        
        # Extract efficiencies (%)
        efficiency_pattern = r'(\d+\.?\d*)\s*%'
        efficiency_matches = re.findall(efficiency_pattern, text)
        data["efficiencies"] = [float(eff) for eff in efficiency_matches]
        
        # Extract temperatures (K, °C)
        temp_pattern = r'(\d+\.?\d*)\s*(?:K|°C|degrees\s*[CK])'
        temp_matches = re.findall(temp_pattern, text, re.IGNORECASE)
        data["temperatures"] = [float(t) for t in temp_matches]
        
        # Extract other measurements with units
        measurement_pattern = r'(\d+\.?\d*)\s*([a-zA-Z]+)'
        measurement_matches = re.findall(measurement_pattern, text)
        
        for value, unit in measurement_matches:
            # Skip if already captured in specific categories
            if unit.lower() in ['ev', 'nm', '%', 'k', 'c', '°c']:
                continue
                
            data["other_measurements"].append({
                "value": float(value),
                "unit": unit
            })
        
        return data
    
    @staticmethod
    def extract_key_phrases(text, num_phrases=10):
        """
        Extract key phrases from text using simple frequency analysis.
        
        Args:
            text: Input text
            num_phrases: Number of key phrases to extract
            
        Returns:
            List of key phrases with scores
        """
        # This is a simplified approach - in a real implementation,
        # you would use NLP techniques like TF-IDF, TextRank, etc.
        
        # Normalize text
        text = text.lower()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-z]{3,}\b', text)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                      'where', 'when', 'how', 'why', 'which', 'who', 'whom', 'this', 'that',
                      'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through',
                      'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in', 'with'}
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
        
        # Extract phrases (adjacent words)
        phrases = []
        for i in range(len(filtered_words) - 1):
            phrase = filtered_words[i] + " " + filtered_words[i + 1]
            phrases.append(phrase)
            
        # Count phrase frequencies
        phrase_counts = {}
        for phrase in phrases:
            if phrase not in phrase_counts:
                phrase_counts[phrase] = 0
            phrase_counts[phrase] += 1
        
        # Combine word and phrase counts
        all_counts = {}
        all_counts.update(word_counts)
        all_counts.update(phrase_counts)
        
        # Sort by frequency
        sorted_items = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top phrases with scores
        top_phrases = []
        for phrase, count in sorted_items[:num_phrases]:
            top_phrases.append({
                "phrase": phrase,
                "score": count
            })
        
        return top_phrases
    
    @staticmethod
    def analyze_figure(figure_path):
        """
        Analyze a figure to extract information.
        
        Args:
            figure_path: Path to the figure file
            
        Returns:
            Dictionary with figure analysis
        """
        # This would be enhanced with image analysis and CV techniques
        
        filename = os.path.basename(figure_path)
        file_extension = os.path.splitext(filename)[1].lower()
        
        # Basic figure info
        figure_info = {
            "filename": filename,
            "path": figure_path,
            "extension": file_extension,
            "size_bytes": os.path.getsize(figure_path) if os.path.exists(figure_path) else 0,
            "type": PaperUtils.determine_figure_type(filename),
            "caption": PaperUtils.generate_figure_caption(filename)
        }
        
        return figure_info
    
    @staticmethod
    def determine_figure_type(filename):
        """Determine the type of figure based on filename."""
        filename_lower = filename.lower()
        
        if any(term in filename_lower for term in ["energy", "level", "orbital", "homo", "lumo"]):
            return "Energy Level Diagram"
        elif any(term in filename_lower for term in ["efficiency", "tadf", "performance", "quantum"]):
            return "TADF Efficiency Plot"
        elif any(term in filename_lower for term in ["spectrum", "absorption", "emission", "fluorescence"]):
            return "Spectral Data"
        elif any(term in filename_lower for term in ["structure", "molecule", "compound", "chemical"]):
            return "Molecular Structure"
        elif any(term in filename_lower for term in ["correlation", "relation", "trend", "comparison"]):
            return "Correlation Plot"
        else:
            return "Other"
    
    @staticmethod
    def generate_figure_caption(filename):
        """Generate a basic caption for the figure based on filename."""
        filename_clean = os.path.splitext(filename)[0]
        # Convert snake_case or camelCase to spaces
        caption = re.sub(r'[_-]', ' ', filename_clean)
        caption = re.sub(r'([a-z])([A-Z])', r'\1 \2', caption)
        
        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        figure_type = PaperUtils.determine_figure_type(filename)
        
        # Add more context based on figure type
        if figure_type == "Energy Level Diagram":
            caption += " showing energy levels of molecular orbitals"
        elif figure_type == "TADF Efficiency Plot":
            caption += " illustrating TADF emission efficiency"
        elif figure_type == "Spectral Data":
            caption += " displaying absorption and emission spectra"
        elif figure_type == "Molecular Structure":
            caption += " showing molecular structure"
        elif figure_type == "Correlation Plot":
            caption += " demonstrating relationship between parameters"
        
        return caption
    
    @staticmethod
    def format_citation(paper, style="apa"):
        """
        Format citation for a paper in the specified style.
        
        Args:
            paper: Dictionary containing paper information
            style: Citation style (apa, ieee, etc.)
            
        Returns:
            Formatted citation string
        """
        if style.lower() == "apa":
            # APA style
            
            # Format authors
            authors = paper.get("authors", [])
            if isinstance(authors, str):
                authors = [authors]
                
            if len(authors) == 0:
                author_str = "Unknown Author"
            elif len(authors) == 1:
                author_str = authors[0]
            elif len(authors) == 2:
                author_str = f"{authors[0]} & {authors[1]}"
            else:
                author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
            
            # Format year
            year = paper.get("year", "n.d.")
            
            # Format title
            title = paper.get("title", "Untitled")
            
            # Format journal
            journal = paper.get("journal", paper.get("source", ""))
            
            # Format volume, issue, pages
            volume = paper.get("volume", "")
            issue = paper.get("number", paper.get("issue", ""))
            pages = paper.get("pages", "")
            
            volume_info = f"{volume}"
            if issue:
                volume_info += f"({issue})"
            if pages:
                volume_info += f", {pages}"
            
            # Format DOI
            doi = paper.get("doi", "")
            doi_str = f" https://doi.org/{doi}" if doi else ""
            
            # Combine all elements
            citation = f"{author_str}. ({year}). {title}. "
            
            if journal:
                citation += f"{journal}, {volume_info}."
                
            citation += doi_str
            
        elif style.lower() == "ieee":
            # IEEE style
            
            # Format authors
            authors = paper.get("authors", [])
            if isinstance(authors, str):
                authors = [authors]
                
            if len(authors) == 0:
                author_str = "Unknown Author"
            elif len(authors) == 1:
                author_str = authors[0]
            else:
                author_str = ", ".join(authors)
            
            # Format title
            title = paper.get("title", "Untitled")
            
            # Format journal
            journal = paper.get("journal", paper.get("source", ""))
            
            # Format volume, issue, pages
            volume = paper.get("volume", "")
            issue = paper.get("number", paper.get("issue", ""))
            pages = paper.get("pages", "")
            year = paper.get("year", "")
            
            volume_info = f"vol. {volume}" if volume else ""
            if issue:
                volume_info += f", no. {issue}"
            if pages:
                volume_info += f", pp. {pages}"
            if year:
                volume_info += f", {year}"
            
            # Format DOI
            doi = paper.get("doi", "")
            doi_str = f", doi: {doi}" if doi else ""
            
            # Combine all elements
            citation = f"{author_str}, \"{title}\", "
            
            if journal:
                citation += f"{journal}, {volume_info}"
                
            citation += doi_str + "."
            
        else:
            # Default format
            authors = paper.get("authors", [])
            if isinstance(authors, str):
                authors = [authors]
            
            author_str = ", ".join(authors) if authors else "Unknown Author"
            title = paper.get("title", "Untitled")
            journal = paper.get("journal", paper.get("source", ""))
            year = paper.get("year", "")
            
            citation = f"{author_str}. {title}. {journal}, {year}."
        
        return citation
    
    @staticmethod
    def create_download_link(content, filename, text):
        """
        Create an HTML download link for file content.
        
        Args:
            content: File content as bytes
            filename: Name of the file
            text: Link text
            
        Returns:
            HTML link string
        """
        b64 = base64.b64encode(content).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
        return href