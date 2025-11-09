"""
Resume and Job Description Parsing Module

Extracts structured data from resumes (PDF, DOCX) and job descriptions.
Handles various formats and edge cases.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pdfplumber
import docx
from docx2txt import process as docx2txt_process

logger = logging.getLogger(__name__)


class ResumeParser:
    """Parse resumes from various formats (PDF, DOCX, TXT)."""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
    
    def parse(self, file_path: str) -> Dict:
        """
        Parse a resume file and extract structured information.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Dictionary with parsed resume data:
            {
                'name': str,
                'email': str,
                'phone': str,
                'skills': List[str],
                'experience': List[Dict],
                'education': List[Dict],
                'certifications': List[str],
                'raw_text': str
            }
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Extract raw text
        raw_text = self._extract_text(file_path, file_ext)
        
        # Parse structured data
        parsed_data = {
            'name': self._extract_name(raw_text),
            'email': self._extract_email(raw_text),
            'phone': self._extract_phone(raw_text),
            'skills': self._extract_skills(raw_text),
            'experience': self._extract_experience(raw_text),
            'education': self._extract_education(raw_text),
            'certifications': self._extract_certifications(raw_text),
            'raw_text': raw_text
        }
        
        logger.info(f"Parsed resume: {file_path.name}")
        return parsed_data
    
    def _extract_text(self, file_path: Path, file_ext: str) -> str:
        """Extract raw text from file based on format."""
        if file_ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self._extract_from_docx(file_path)
        elif file_ext == '.txt':
            return file_path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise
        return text
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            text = docx2txt_process(str(file_path))
            return text if text else ""
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name (usually first line or header)."""
        lines = text.split('\n')[:5]  # Check first 5 lines
        for line in lines:
            line = line.strip()
            if line and len(line.split()) <= 4:  # Name typically 1-4 words
                # Check if it looks like a name (not email, phone, etc.)
                if '@' not in line and not re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', line):
                    return line
        return None
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number."""
        phone_patterns = [
            r'\d{3}[-.]?\d{3}[-.]?\d{4}',
            r'\(\d{3}\)\s?\d{3}[-.]?\d{4}',
            r'\+\d{1,3}[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}'
        ]
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        return None
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills section."""
        skills = []
        
        # Look for "Skills" section
        skills_pattern = r'(?:skills?|technical\s+skills?|competencies?)[:]\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        match = re.search(skills_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            skills_text = match.group(1)
            # Split by common delimiters
            skills = [s.strip() for s in re.split(r'[,;•\n]', skills_text) if s.strip()]
        
        return skills[:50]  # Limit to 50 skills
    
    def _extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience."""
        experience = []
        
        # Look for experience section
        exp_pattern = r'(?:experience|work\s+history|employment|professional\s+experience)[:]\s*(.+?)(?:\n\n(?:education|skills|$))'
        match = re.search(exp_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            exp_text = match.group(1)
            # Split by job entries (look for dates or company names)
            entries = re.split(r'\n(?=\w+.*\d{4}|\w+\s+[-–—])', exp_text)
            
            for entry in entries[:10]:  # Limit to 10 entries
                entry_dict = self._parse_experience_entry(entry)
                if entry_dict:
                    experience.append(entry_dict)
        
        return experience
    
    def _parse_experience_entry(self, entry: str) -> Optional[Dict]:
        """Parse a single experience entry."""
        lines = [l.strip() for l in entry.split('\n') if l.strip()]
        if not lines:
            return None
        
        # Extract dates
        date_pattern = r'(\d{4}|\w+\s+\d{4})\s*[-–—]\s*(\d{4}|present|current)'
        dates = re.findall(date_pattern, entry, re.IGNORECASE)
        
        # Extract company and title (usually first line)
        title_company = lines[0] if lines else ""
        
        # Extract bullets
        bullets = [l for l in lines[1:] if l.startswith(('•', '-', '*')) or l[0].islower()]
        
        return {
            'title': title_company,
            'company': title_company,  # Simplified - can be improved
            'start_date': dates[0][0] if dates else None,
            'end_date': dates[0][1] if dates else None,
            'description': '\n'.join(bullets) if bullets else entry
        }
    
    def _extract_education(self, text: str) -> List[Dict]:
        """Extract education section."""
        education = []
        
        # Look for education section
        edu_pattern = r'(?:education|academic|qualifications)[:]\s*(.+?)(?:\n\n(?:experience|skills|certifications|$))'
        match = re.search(edu_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            edu_text = match.group(1)
            entries = re.split(r'\n(?=\w+.*\d{4})', edu_text)
            
            for entry in entries[:5]:  # Limit to 5 entries
                entry_dict = self._parse_education_entry(entry)
                if entry_dict:
                    education.append(entry_dict)
        
        return education
    
    def _parse_education_entry(self, entry: str) -> Optional[Dict]:
        """Parse a single education entry."""
        lines = [l.strip() for l in entry.split('\n') if l.strip()]
        if not lines:
            return None
        
        # Extract degree and institution
        degree_pattern = r'(bachelor|master|phd|doctorate|mba|bs|ba|ms|ma|ph\.?d\.?)'
        degree_match = re.search(degree_pattern, entry, re.IGNORECASE)
        
        return {
            'degree': degree_match.group(0) if degree_match else lines[0],
            'institution': lines[0] if lines else None,
            'year': re.search(r'\d{4}', entry).group(0) if re.search(r'\d{4}', entry) else None
        }
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications."""
        certifications = []
        
        # Look for certifications section
        cert_pattern = r'(?:certifications?|certificates?|licenses?)[:]\s*(.+?)(?:\n\n|$)'
        match = re.search(cert_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            cert_text = match.group(1)
            certifications = [c.strip() for c in re.split(r'[,;•\n]', cert_text) if c.strip()]
        
        return certifications


class JobDescriptionParser:
    """Parse job descriptions from text or files."""
    
    def parse(self, text: str) -> Dict:
        """
        Parse a job description and extract structured information.
        
        Args:
            text: Job description text
            
        Returns:
            Dictionary with parsed JD data:
            {
                'title': str,
                'company': str,
                'required_skills': List[str],
                'preferred_skills': List[str],
                'responsibilities': List[str],
                'requirements': List[str],
                'education_requirements': str,
                'experience_years': int,
                'raw_text': str
            }
        """
        parsed_data = {
            'title': self._extract_title(text),
            'company': self._extract_company(text),
            'required_skills': self._extract_required_skills(text),
            'preferred_skills': self._extract_preferred_skills(text),
            'responsibilities': self._extract_responsibilities(text),
            'requirements': self._extract_requirements(text),
            'education_requirements': self._extract_education_requirements(text),
            'experience_years': self._extract_experience_years(text),
            'raw_text': text
        }
        
        logger.info("Parsed job description")
        return parsed_data
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract job title."""
        lines = text.split('\n')[:3]
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:
                return line
        return None
    
    def _extract_company(self, text: str) -> Optional[str]:
        """Extract company name."""
        # Look for "Company:" or similar patterns
        company_pattern = r'(?:company|employer|organization)[:]\s*([^\n]+)'
        match = re.search(company_pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _extract_required_skills(self, text: str) -> List[str]:
        """Extract required skills."""
        skills = []
        
        # Look for "Required Skills" or "Must Have" sections
        required_pattern = r'(?:required\s+skills?|must\s+have|required\s+qualifications?)[:]\s*(.+?)(?:\n\n|preferred|$)'
        match = re.search(required_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            skills_text = match.group(1)
            skills = [s.strip() for s in re.split(r'[,;•\n]', skills_text) if s.strip()]
        
        return skills
    
    def _extract_preferred_skills(self, text: str) -> List[str]:
        """Extract preferred skills."""
        skills = []
        
        # Look for "Preferred Skills" or "Nice to Have" sections
        preferred_pattern = r'(?:preferred\s+skills?|nice\s+to\s+have|preferred\s+qualifications?)[:]\s*(.+?)(?:\n\n|$)'
        match = re.search(preferred_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            skills_text = match.group(1)
            skills = [s.strip() for s in re.split(r'[,;•\n]', skills_text) if s.strip()]
        
        return skills
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract job responsibilities."""
        responsibilities = []
        
        # Look for "Responsibilities" or "Key Duties" sections
        resp_pattern = r'(?:responsibilities?|key\s+duties?|what\s+you\'?ll\s+do)[:]\s*(.+?)(?:\n\n(?:requirements?|qualifications?|$))'
        match = re.search(resp_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            resp_text = match.group(1)
            responsibilities = [r.strip() for r in re.split(r'[•\n]', resp_text) if r.strip() and len(r.strip()) > 10]
        
        return responsibilities[:20]  # Limit to 20
    
    def _extract_requirements(self, text: str) -> List[str]:
        """Extract job requirements."""
        requirements = []
        
        # Look for "Requirements" section
        req_pattern = r'(?:requirements?|qualifications?)[:]\s*(.+?)(?:\n\n|$)'
        match = re.search(req_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            req_text = match.group(1)
            requirements = [r.strip() for r in re.split(r'[•\n]', req_text) if r.strip() and len(r.strip()) > 10]
        
        return requirements[:20]  # Limit to 20
    
    def _extract_education_requirements(self, text: str) -> Optional[str]:
        """Extract education requirements."""
        edu_pattern = r'(?:education|degree|qualification)[:]\s*([^\n]+)'
        match = re.search(edu_pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _extract_experience_years(self, text: str) -> Optional[int]:
        """Extract required years of experience."""
        # Look for patterns like "3+ years", "5 years experience"
        exp_pattern = r'(\d+)\+?\s*years?\s*(?:of\s+)?experience'
        match = re.search(exp_pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

