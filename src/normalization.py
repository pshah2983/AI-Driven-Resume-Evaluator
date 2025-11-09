"""
Normalization and Enrichment Module

Normalizes skills, dates, job titles, and other resume/JD data.
Handles canonicalization and standardization.
"""

import re
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
try:
    from dateutil import parser as date_parser
except ImportError:
    # Fallback if dateutil not available
    date_parser = None

logger = logging.getLogger(__name__)


class SkillNormalizer:
    """Normalize and canonicalize skill names."""
    
    def __init__(self):
        # Common skill synonyms and variations
        self.skill_synonyms = {
            'python': ['python3', 'python 3', 'py'],
            'javascript': ['js', 'ecmascript', 'node.js', 'nodejs'],
            'machine learning': ['ml', 'machine-learning', 'machinelearning'],
            'artificial intelligence': ['ai', 'artificial-intelligence'],
            'natural language processing': ['nlp', 'natural-language-processing'],
            'deep learning': ['dl', 'deep-learning'],
            'data science': ['data-science', 'datascience'],
            'sql': ['sql server', 'mysql', 'postgresql', 'postgres'],
            'aws': ['amazon web services', 'amazon aws'],
            'docker': ['docker container', 'dockerization'],
            'kubernetes': ['k8s', 'kube'],
            'react': ['react.js', 'reactjs'],
            'vue': ['vue.js', 'vuejs'],
            'angular': ['angular.js', 'angularjs'],
            'git': ['git version control', 'git vcs'],
            'rest api': ['rest', 'restful api', 'rest api'],
            'graphql': ['graph ql', 'graph-ql'],
        }
        
        # Build reverse lookup
        self.skill_map = {}
        for canonical, variants in self.skill_synonyms.items():
            self.skill_map[canonical.lower()] = canonical
            for variant in variants:
                self.skill_map[variant.lower()] = canonical
    
    def normalize(self, skill: str) -> str:
        """
        Normalize a skill name to its canonical form.
        
        Args:
            skill: Raw skill name
            
        Returns:
            Normalized skill name
        """
        skill_lower = skill.lower().strip()
        
        # Check if skill exists in map
        if skill_lower in self.skill_map:
            return self.skill_map[skill_lower]
        
        # Try fuzzy matching (simple substring match)
        for canonical, variants in self.skill_synonyms.items():
            if skill_lower in canonical or canonical in skill_lower:
                return canonical
            for variant in variants:
                if skill_lower in variant or variant in skill_lower:
                    return canonical
        
        # Return original if no match found
        return skill.strip()
    
    def normalize_list(self, skills: List[str]) -> List[str]:
        """Normalize a list of skills."""
        normalized = [self.normalize(skill) for skill in skills]
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in normalized:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique_skills.append(skill)
        return unique_skills


class DateNormalizer:
    """Normalize and parse dates from various formats."""
    
    def __init__(self):
        self.date_formats = [
            '%Y-%m-%d',
            '%m/%Y',
            '%Y',
            '%B %Y',
            '%b %Y',
            '%m-%Y',
            '%d/%m/%Y',
            '%d-%m-%Y'
        ]
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse a date string to datetime object.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Datetime object or None if parsing fails
        """
        if not date_str or date_str.lower() in ['present', 'current', 'now']:
            return datetime.now()
        
        date_str = date_str.strip()
        
        # Try dateutil parser first (most flexible)
        if date_parser:
            try:
                return date_parser.parse(date_str, fuzzy=True)
            except:
                pass
        
        # Try specific formats
        for fmt in self.date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def calculate_years_experience(self, start_date: Optional[datetime], 
                                   end_date: Optional[datetime] = None) -> float:
        """
        Calculate years of experience between two dates.
        
        Args:
            start_date: Start date
            end_date: End date (defaults to now)
            
        Returns:
            Years of experience as float
        """
        if not start_date:
            return 0.0
        
        if not end_date:
            end_date = datetime.now()
        
        delta = end_date - start_date
        years = delta.days / 365.25
        return round(years, 2)
    
    def extract_dates_from_text(self, text: str) -> List[datetime]:
        """Extract all dates from text."""
        dates = []
        
        # Pattern for dates like "Jan 2020 - Dec 2022" or "2020-2022"
        date_patterns = [
            r'\d{4}[-–—]\d{4}',
            r'\w+\s+\d{4}\s*[-–—]\s*\w+\s+\d{4}',
            r'\d{1,2}/\d{4}\s*[-–—]\s*\d{1,2}/\d{4}',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Split by separator and parse each date
                parts = re.split(r'[-–—]', match)
                for part in parts:
                    date_obj = self.parse_date(part.strip())
                    if date_obj:
                        dates.append(date_obj)
        
        return dates


class TitleNormalizer:
    """Normalize job titles to standard categories."""
    
    def __init__(self):
        # Title categories and their variations
        self.title_categories = {
            'software engineer': ['software developer', 'programmer', 'coder', 'dev'],
            'data scientist': ['data analyst', 'data engineer', 'ml engineer'],
            'product manager': ['pm', 'product owner', 'product lead'],
            'data analyst': ['business analyst', 'analyst', 'data specialist'],
            'machine learning engineer': ['ml engineer', 'ml developer', 'ai engineer'],
            'devops engineer': ['devops', 'sre', 'site reliability engineer'],
            'full stack developer': ['fullstack developer', 'full-stack', 'full stack'],
            'backend developer': ['backend engineer', 'server-side developer'],
            'frontend developer': ['frontend engineer', 'front-end developer', 'ui developer'],
        }
        
        # Build reverse lookup
        self.title_map = {}
        for canonical, variants in self.title_categories.items():
            self.title_map[canonical.lower()] = canonical
            for variant in variants:
                self.title_map[variant.lower()] = canonical
    
    def normalize(self, title: str) -> str:
        """
        Normalize a job title to its canonical form.
        
        Args:
            title: Raw job title
            
        Returns:
            Normalized job title
        """
        title_lower = title.lower().strip()
        
        if title_lower in self.title_map:
            return self.title_map[title_lower]
        
        # Try fuzzy matching
        for canonical, variants in self.title_categories.items():
            if canonical in title_lower or title_lower in canonical:
                return canonical
            for variant in variants:
                if variant in title_lower or title_lower in variant:
                    return canonical
        
        return title.strip()
    
    def get_seniority_level(self, title: str) -> str:
        """
        Extract seniority level from job title.
        
        Returns:
            'junior', 'mid', 'senior', or 'unknown'
        """
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['senior', 'sr', 'lead', 'principal', 'staff']):
            return 'senior'
        elif any(word in title_lower for word in ['junior', 'jr', 'entry', 'intern', 'internship']):
            return 'junior'
        elif any(word in title_lower for word in ['mid', 'intermediate']):
            return 'mid'
        else:
            return 'unknown'


class ResumeNormalizer:
    """Main normalizer for resume data."""
    
    def __init__(self):
        self.skill_normalizer = SkillNormalizer()
        self.date_normalizer = DateNormalizer()
        self.title_normalizer = TitleNormalizer()
    
    def normalize(self, resume_data: Dict) -> Dict:
        """
        Normalize all fields in resume data.
        
        Args:
            resume_data: Raw parsed resume data
            
        Returns:
            Normalized resume data
        """
        normalized = resume_data.copy()
        
        # Normalize skills
        if 'skills' in normalized:
            normalized['skills'] = self.skill_normalizer.normalize_list(
                normalized['skills']
            )
        
        # Normalize experience
        if 'experience' in normalized:
            normalized['experience'] = self._normalize_experience(
                normalized['experience']
            )
        
        # Calculate total years of experience
        normalized['total_years_experience'] = self._calculate_total_experience(
            normalized.get('experience', [])
        )
        
        return normalized
    
    def _normalize_experience(self, experience: List[Dict]) -> List[Dict]:
        """Normalize experience entries."""
        normalized_exp = []
        
        for exp in experience:
            normalized_exp_entry = exp.copy()
            
            # Normalize title
            if 'title' in exp:
                normalized_exp_entry['normalized_title'] = self.title_normalizer.normalize(
                    exp['title']
                )
                normalized_exp_entry['seniority_level'] = self.title_normalizer.get_seniority_level(
                    exp['title']
                )
            
            # Parse dates
            if 'start_date' in exp and exp['start_date']:
                start_dt = self.date_normalizer.parse_date(exp['start_date'])
                normalized_exp_entry['start_date_parsed'] = start_dt
            
            if 'end_date' in exp and exp['end_date']:
                end_dt = self.date_normalizer.parse_date(exp['end_date'])
                normalized_exp_entry['end_date_parsed'] = end_dt
            
            # Calculate years for this role
            if 'start_date_parsed' in normalized_exp_entry:
                start_dt = normalized_exp_entry.get('start_date_parsed')
                end_dt = normalized_exp_entry.get('end_date_parsed')
                if start_dt:  # Only calculate if start date exists
                    years = self.date_normalizer.calculate_years_experience(start_dt, end_dt)
                    normalized_exp_entry['years_in_role'] = years
            
            normalized_exp.append(normalized_exp_entry)
        
        return normalized_exp
    
    def _calculate_total_experience(self, experience: List[Dict]) -> float:
        """Calculate total years of experience across all roles."""
        total_years = 0.0
        
        for exp in experience:
            if 'years_in_role' in exp:
                years = exp['years_in_role']
                # Ensure years is a number
                if isinstance(years, (int, float)):
                    total_years += float(years)
        
        return round(total_years, 2)


class JobDescriptionNormalizer:
    """Normalizer for job description data."""
    
    def __init__(self):
        self.skill_normalizer = SkillNormalizer()
        self.title_normalizer = TitleNormalizer()
    
    def normalize(self, jd_data: Dict) -> Dict:
        """
        Normalize job description data.
        
        Args:
            jd_data: Raw parsed JD data
            
        Returns:
            Normalized JD data
        """
        normalized = jd_data.copy()
        
        # Normalize skills
        if 'required_skills' in normalized:
            normalized['required_skills'] = self.skill_normalizer.normalize_list(
                normalized['required_skills']
            )
        
        if 'preferred_skills' in normalized:
            normalized['preferred_skills'] = self.skill_normalizer.normalize_list(
                normalized['preferred_skills']
            )
        
        # Normalize title
        if 'title' in normalized:
            normalized['normalized_title'] = self.title_normalizer.normalize(
                normalized['title']
            )
            normalized['seniority_level'] = self.title_normalizer.get_seniority_level(
                normalized['title']
            )
        
        return normalized

