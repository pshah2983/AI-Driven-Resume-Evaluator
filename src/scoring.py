"""
Scoring Engine Module

Computes match scores between resumes and job descriptions using
a weighted rubric combining multiple factors.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Score resume-JD matches using a weighted rubric."""
    
    def __init__(self, 
                 weights: Optional[Dict[str, float]] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize scoring engine.
        
        Args:
            weights: Dictionary of scoring weights (must sum to 100)
            embedding_generator: EmbeddingGenerator instance for similarity
        """
        # Default weights
        self.weights = weights or {
            'skill_match': 40.0,
            'experience_match': 25.0,
            'role_responsibility_match': 15.0,
            'education_certifications': 10.0,
            'ats_friendliness': 10.0
        }
        
        # Validate weights sum to 100
        total_weight = sum(self.weights.values())
        if abs(total_weight - 100.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 100")
            self.weights = {k: v * 100.0 / total_weight for k, v in self.weights.items()}
        
        self.embedding_generator = embedding_generator
    
    def score(self, resume_data: Dict, resume_embeddings: Dict[str, np.ndarray],
              jd_data: Dict, jd_embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Compute overall match score and detailed breakdown.
        
        Args:
            resume_data: Normalized resume data
            resume_embeddings: Resume embeddings dictionary
            jd_data: Normalized JD data
            jd_embeddings: JD embeddings dictionary
            
        Returns:
            Dictionary with scores and breakdown:
            {
                'overall_score': float (0-100),
                'skill_match_score': float (0-100),
                'experience_match_score': float (0-100),
                'role_responsibility_score': float (0-100),
                'education_score': float (0-100),
                'ats_score': float (0-100),
                'confidence': float (0-1),
                'missing_skills': List[str],
                'weak_areas': List[str],
                'evidence': Dict
            }
        """
        scores = {}
        
        # Skill match score
        skill_score, skill_details = self._score_skills(
            resume_data, resume_embeddings, jd_data, jd_embeddings
        )
        scores['skill_match_score'] = skill_score
        scores['skill_details'] = skill_details
        
        # Experience match score
        exp_score, exp_details = self._score_experience(resume_data, jd_data)
        scores['experience_match_score'] = exp_score
        scores['experience_details'] = exp_details
        
        # Role/responsibility match score
        role_score, role_details = self._score_role_responsibility(
            resume_embeddings, jd_embeddings
        )
        scores['role_responsibility_score'] = role_score
        scores['role_details'] = role_details
        
        # Education score
        edu_score, edu_details = self._score_education(resume_data, jd_data)
        scores['education_score'] = edu_score
        scores['education_details'] = edu_details
        
        # ATS friendliness score
        ats_score, ats_details = self._score_ats_friendliness(resume_data)
        scores['ats_score'] = ats_score
        scores['ats_details'] = ats_details
        
        # Compute weighted overall score
        overall_score = (
            skill_score * self.weights['skill_match'] / 100.0 +
            exp_score * self.weights['experience_match'] / 100.0 +
            role_score * self.weights['role_responsibility_match'] / 100.0 +
            edu_score * self.weights['education_certifications'] / 100.0 +
            ats_score * self.weights['ats_friendliness'] / 100.0
        )
        # Convert to native Python float to avoid numpy float32 issues with Streamlit
        scores['overall_score'] = float(round(overall_score, 2))
        
        # Confidence score (based on data completeness)
        confidence = self._compute_confidence(resume_data, jd_data)
        scores['confidence'] = confidence
        
        # Missing skills and weak areas
        scores['missing_skills'] = skill_details.get('missing_skills', [])
        scores['weak_areas'] = self._identify_weak_areas(scores)
        
        # Evidence for explainability
        scores['evidence'] = {
            'matched_skills': skill_details.get('matched_skills', []),
            'missing_skills': skill_details.get('missing_skills', []),
            'experience_gap': exp_details.get('gap_years', 0),
            'role_match': role_details.get('similarity', 0)
        }
        
        return scores
    
    def _score_skills(self, resume_data: Dict, resume_embeddings: Dict,
                     jd_data: Dict, jd_embeddings: Dict) -> Tuple[float, Dict]:
        """Score skill match (0-100)."""
        resume_skills = set(s.lower() for s in resume_data.get('skills', []))
        required_skills = set(s.lower() for s in jd_data.get('required_skills', []))
        preferred_skills = set(s.lower() for s in jd_data.get('preferred_skills', []))
        
        # Exact matches
        required_matches = resume_skills.intersection(required_skills)
        preferred_matches = resume_skills.intersection(preferred_skills)
        
        # Semantic matches (using embeddings if available)
        semantic_matches = 0
        if self.embedding_generator and 'skills' in resume_embeddings:
            if 'required_skills' in jd_embeddings:
                similarity = self.embedding_generator.compute_similarity(
                    resume_embeddings['skills'],
                    jd_embeddings['required_skills']
                )
                semantic_matches = similarity * len(required_skills)
        
        # Calculate score
        total_required = len(required_skills) if required_skills else 1
        required_score = (len(required_matches) / total_required) * 70  # 70% weight for required
        
        total_preferred = len(preferred_skills) if preferred_skills else 1
        preferred_score = (len(preferred_matches) / total_preferred) * 30  # 30% weight for preferred
        
        semantic_score = min(semantic_matches / total_required * 20, 20)  # Bonus for semantic match
        
        total_score = min(required_score + preferred_score + semantic_score, 100.0)
        
        missing_skills = list(required_skills - resume_skills)
        
        details = {
            'matched_required': len(required_matches),
            'total_required': len(required_skills),
            'matched_preferred': len(preferred_matches),
            'total_preferred': len(preferred_skills),
            'matched_skills': list(required_matches | preferred_matches),
            'missing_skills': missing_skills,
            'semantic_similarity': semantic_matches
        }
        
        return float(round(total_score, 2)), details
    
    def _score_experience(self, resume_data: Dict, jd_data: Dict) -> Tuple[float, Dict]:
        """Score experience match (0-100)."""
        # Safely get years with None handling
        resume_years = resume_data.get('total_years_experience') or 0.0
        if not isinstance(resume_years, (int, float)):
            resume_years = 0.0
        
        required_years = jd_data.get('experience_years') or 0
        if required_years is None or not isinstance(required_years, (int, float)):
            required_years = 0
        
        # Convert to float for calculations
        resume_years = float(resume_years)
        required_years = float(required_years)
        
        if required_years == 0:
            return 100.0, {'gap_years': 0.0, 'resume_years': float(resume_years)}
        
        # Calculate gap
        gap_years = required_years - resume_years
        
        # Score based on gap
        if gap_years <= 0:
            score = 100.0  # Meets or exceeds requirement
        elif gap_years <= 1:
            score = 80.0  # Close
        elif gap_years <= 2:
            score = 60.0  # Moderate gap
        elif gap_years <= 3:
            score = 40.0  # Significant gap
        else:
            score = 20.0  # Large gap
        
        # Check seniority level match
        resume_seniority = self._get_resume_seniority(resume_data)
        jd_seniority = jd_data.get('seniority_level', 'unknown')
        
        seniority_bonus = 0
        if resume_seniority == jd_seniority:
            seniority_bonus = 10
        elif (resume_seniority == 'senior' and jd_seniority in ['mid', 'junior']) or \
             (resume_seniority == 'mid' and jd_seniority == 'junior'):
            seniority_bonus = 5  # Overqualified is better than underqualified
        
        total_score = min(score + seniority_bonus, 100.0)
        
        details = {
            'resume_years': float(resume_years),
            'required_years': float(required_years),
            'gap_years': float(gap_years),
            'resume_seniority': resume_seniority,
            'jd_seniority': jd_seniority,
            'seniority_match': resume_seniority == jd_seniority
        }
        
        return float(round(total_score, 2)), details
    
    def _score_role_responsibility(self, resume_embeddings: Dict,
                                  jd_embeddings: Dict) -> Tuple[float, Dict]:
        """Score role/responsibility match using semantic similarity (0-100)."""
        if not self.embedding_generator:
            return 50.0, {'similarity': 0.5, 'note': 'Embeddings not available'}
        
        # Compare experience bullets with responsibilities
        if 'experience_bullets' in resume_embeddings and 'responsibilities' in jd_embeddings:
            similarities = []
            for bullet_emb in resume_embeddings['experience_bullets']:
                sim = self.embedding_generator.compute_similarity(
                    bullet_emb,
                    jd_embeddings['responsibilities']
                )
                similarities.append(sim)
            
            avg_similarity = float(np.mean(similarities)) if similarities else 0.0
        else:
            # Fallback to overall similarity
            if 'overall' in resume_embeddings and 'overall' in jd_embeddings:
                avg_similarity = self.embedding_generator.compute_similarity(
                    resume_embeddings['overall'],
                    jd_embeddings['overall']
                )
            else:
                avg_similarity = 0.0
        
        # Convert similarity (0-1) to score (0-100)
        score = avg_similarity * 100.0
        
        details = {
            'similarity': float(round(avg_similarity, 3)),
            'num_bullets_matched': len(resume_embeddings.get('experience_bullets', []))
        }
        
        return float(round(score, 2)), details
    
    def _score_education(self, resume_data: Dict, jd_data: Dict) -> Tuple[float, Dict]:
        """Score education match (0-100)."""
        jd_edu_req = jd_data.get('education_requirements', '')
        if not jd_edu_req:
            return 100.0, {'note': 'No education requirement specified'}
        
        resume_education = resume_data.get('education', [])
        if not resume_education:
            return 0.0, {'note': 'No education found in resume'}
        
        # Simple keyword matching
        jd_edu_lower = jd_edu_req.lower()
        matched = False
        
        for edu in resume_education:
            degree = edu.get('degree', '').lower()
            if any(keyword in degree for keyword in ['bachelor', 'master', 'phd', 'doctorate']):
                if any(keyword in jd_edu_lower for keyword in ['bachelor', 'master', 'phd', 'doctorate']):
                    matched = True
                    break
        
        score = 100.0 if matched else 50.0
        
        details = {
            'resume_education': resume_education,
            'jd_requirement': jd_edu_req,
            'matched': matched
        }
        
        return float(round(score, 2)), details
    
    def _score_ats_friendliness(self, resume_data: Dict) -> Tuple[float, Dict]:
        """Score ATS friendliness (0-100)."""
        score = 0.0
        details = {}
        
        # Check for contact info
        has_email = bool(resume_data.get('email'))
        has_phone = bool(resume_data.get('phone'))
        has_name = bool(resume_data.get('name'))
        
        contact_score = (has_email + has_phone + has_name) / 3.0 * 30
        score += contact_score
        details['has_contact_info'] = has_email and has_phone and has_name
        
        # Check for skills section
        has_skills = len(resume_data.get('skills', [])) > 0
        skills_score = 20 if has_skills else 0
        score += skills_score
        details['has_skills_section'] = has_skills
        
        # Check for experience section
        has_experience = len(resume_data.get('experience', [])) > 0
        exp_score = 20 if has_experience else 0
        score += exp_score
        details['has_experience'] = has_experience
        
        # Check for education section
        has_education = len(resume_data.get('education', [])) > 0
        edu_score = 15 if has_education else 0
        score += edu_score
        details['has_education'] = has_education
        
        # Check for keywords near top (first 500 chars)
        raw_text = resume_data.get('raw_text', '')
        top_section = raw_text[:500].lower()
        has_keywords = any(keyword in top_section for keyword in 
                          ['experience', 'skills', 'education', 'summary'])
        keyword_score = 15 if has_keywords else 0
        score += keyword_score
        details['has_keywords_near_top'] = has_keywords
        
        return float(round(score, 2)), details
    
    def _get_resume_seniority(self, resume_data: Dict) -> str:
        """Get overall seniority level from resume."""
        experience = resume_data.get('experience', [])
        if not experience:
            return 'unknown'
        
        # Check most recent role
        seniority_levels = [exp.get('seniority_level', 'unknown') for exp in experience]
        
        # Return most senior level found
        if 'senior' in seniority_levels:
            return 'senior'
        elif 'mid' in seniority_levels:
            return 'mid'
        elif 'junior' in seniority_levels:
            return 'junior'
        else:
            return 'unknown'
    
    def _compute_confidence(self, resume_data: Dict, jd_data: Dict) -> float:
        """Compute confidence score based on data completeness."""
        resume_completeness = (
            (1.0 if resume_data.get('name') else 0.0) +
            (1.0 if resume_data.get('email') else 0.0) +
            (1.0 if resume_data.get('skills') else 0.0) +
            (1.0 if resume_data.get('experience') else 0.0) +
            (1.0 if resume_data.get('education') else 0.0)
        ) / 5.0
        
        jd_completeness = (
            (1.0 if jd_data.get('title') else 0.0) +
            (1.0 if jd_data.get('required_skills') else 0.0) +
            (1.0 if jd_data.get('responsibilities') else 0.0)
        ) / 3.0
        
        confidence = (resume_completeness + jd_completeness) / 2.0
        return float(round(confidence, 3))
    
    def _identify_weak_areas(self, scores: Dict) -> List[str]:
        """Identify weak areas based on scores."""
        weak_areas = []
        
        if scores.get('skill_match_score', 0) < 60:
            weak_areas.append('Skill match is below threshold')
        
        if scores.get('experience_match_score', 0) < 60:
            weak_areas.append('Experience level may not meet requirements')
        
        if scores.get('role_responsibility_score', 0) < 50:
            weak_areas.append('Role/responsibility alignment is low')
        
        if scores.get('education_score', 0) < 50:
            weak_areas.append('Education requirements may not be met')
        
        if scores.get('ats_score', 0) < 70:
            weak_areas.append('Resume may not be ATS-friendly')
        
        return weak_areas

