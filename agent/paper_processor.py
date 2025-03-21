mport re
import PyPDF2
from typing import Dict, List, Any

class PaperProcessor:
    """Handles the processing of research papers: reading, decomposing and filtering."""
    
    def read_paper(self, paper_file) -> str:
        """Read a research paper from a file"""
        if paper_file.name.endswith('.pdf'):
            return self._read_pdf(paper_file)
        else:
            with open(paper_file.name, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _read_pdf(self, pdf_file) -> str:
        """Extract text from a PDF file"""
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def decompose_paper(self, paper_content: str) -> Dict[str, str]:
        """Decompose a paper into its components (abstract, intro, methods, etc.)"""
        sections = {
            "title": "",
            "abstract": "",
            "introduction": "",
            "related_work": "",
            "methodology": "",
            "architecture": "",
            "implementation": "",
            "training": "",
            "evaluation": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": ""
        }
        
        # Simple regex-based extraction of sections
        # This is a basic implementation; a more sophisticated approach would be needed for real papers
        section_patterns = {
            "abstract": r"(?i)abstract\s*\n(.*?)(?=\n\s*\d+\.?\s*Introduction|\n\s*\d+\.?\s*Related Work|\Z)",
            "introduction": r"(?i)(?:\d+\.?\s*)?introduction\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "related_work": r"(?i)(?:\d+\.?\s*)?related\s*work\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "methodology": r"(?i)(?:\d+\.?\s*)?(methodology|method|approach)\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "architecture": r"(?i)(?:\d+\.?\s*)?(architecture|design|model|system design)\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "implementation": r"(?i)(?:\d+\.?\s*)?(implementation|experimental setup)\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "training": r"(?i)(?:\d+\.?\s*)?(training|learning)\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "evaluation": r"(?i)(?:\d+\.?\s*)?(evaluation|experiments|results)\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "results": r"(?i)(?:\d+\.?\s*)?(results|findings)\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "discussion": r"(?i)(?:\d+\.?\s*)?(discussion|analysis)\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "conclusion": r"(?i)(?:\d+\.?\s*)?(conclusion|future work)\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)",
            "references": r"(?i)(?:\d+\.?\s*)?(references|bibliography)\s*\n(.*?)(?=\n\s*\d+\.?\s*|$)"
        }
        
        # Extract the title (typically at the beginning of the paper)
        title_match = re.search(r"\A\s*(.*?)\n", paper_content)
        if title_match:
            sections["title"] = title_match.group(1).strip()
        
        # Extract each section
        for section, pattern in section_patterns.items():
            match = re.search(pattern, paper_content, re.DOTALL)
            if match:
                # The last group contains the section content
                group_idx = len(match.groups())
                sections[section] = match.group(group_idx).strip()
        
        return sections
    
    def filter_relevant_content(self, decomposed_paper: Dict[str, str]) -> Dict[str, str]:
        """Filter the decomposed paper to keep only task and evaluation relevant parts"""
        # For replication studies, we focus on methodology, architecture, implementation,
        # training, and evaluation sections
        relevant_sections = [
            "methodology", "architecture", "implementation", "training", "evaluation",
            "results"  # including results for comparison
        ]
        
        return {section: content for section, content in decomposed_paper.items() 
                if section in relevant_sections and content}