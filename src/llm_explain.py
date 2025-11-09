"""
LLM Explainability Module

Uses LLMs to generate human-readable explanations, bullet rewrites,
and interview preparation questions.
"""

import logging
import os
from typing import Dict, List, Optional
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class LLMExplainer:
    """Generate explanations and recommendations using LLMs."""
    
    def __init__(self, provider: str = "openai", model_name: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize LLM explainer.
        
        Args:
            provider: LLM provider ("openai", "huggingface", "local")
            model_name: Model name/identifier
            api_key: API key for provider (if needed)
        """
        self.provider = provider
        self.model_name = model_name or self._get_default_model(provider)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if provider == "openai":
            if not self.api_key:
                logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            else:
                openai.api_key = self.api_key
        elif provider in ["huggingface", "local"]:
            self._load_local_model()
        
        logger.info(f"Initialized LLM explainer: {provider}, model={self.model_name}")
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model name for provider."""
        defaults = {
            "openai": "gpt-3.5-turbo",
            "huggingface": "google/flan-t5-base",
            "local": "microsoft/DialoGPT-medium"
        }
        return defaults.get(provider, "gpt-3.5-turbo")
    
    def _load_local_model(self):
        """Load local Hugging Face model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            logger.info(f"Loaded local model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            raise
    
    def generate_summary(self, scores: Dict, resume_data: Dict, jd_data: Dict) -> str:
        """
        Generate human-readable summary of evaluation.
        
        Args:
            scores: Scoring results dictionary
            resume_data: Resume data
            jd_data: Job description data
            
        Returns:
            Summary text
        """
        overall_score = scores.get('overall_score', 0)
        missing_skills = scores.get('missing_skills', [])
        weak_areas = scores.get('weak_areas', [])
        
        prompt = f"""Summarize this resume evaluation in 2-3 sentences:

Overall Match Score: {overall_score}/100
Missing Required Skills: {', '.join(missing_skills[:5]) if missing_skills else 'None'}
Weak Areas: {', '.join(weak_areas[:3]) if weak_areas else 'None'}

Provide a concise, actionable summary."""
        
        return self._generate_text(prompt, max_tokens=150)
    
    def generate_gap_analysis(self, scores: Dict, resume_data: Dict, jd_data: Dict) -> List[str]:
        """
        Generate gap analysis with top 3 gaps.
        
        Args:
            scores: Scoring results
            resume_data: Resume data
            jd_data: Job description data
            
        Returns:
            List of gap descriptions
        """
        gaps = []
        
        # Missing skills
        missing_skills = scores.get('missing_skills', [])
        if missing_skills:
            gaps.append(f"Missing required skills: {', '.join(missing_skills[:3])}")
        
        # Experience gap
        exp_details = scores.get('experience_details', {})
        gap_years = exp_details.get('gap_years', 0)
        if gap_years > 0:
            gaps.append(f"Experience gap: {gap_years} years below requirement")
        
        # Education gap
        edu_details = scores.get('education_details', {})
        if not edu_details.get('matched', True):
            gaps.append("Education requirements may not be fully met")
        
        # Generate LLM explanation for gaps
        if gaps:
            prompt = f"""Explain these resume gaps in a helpful way:

{chr(10).join(gaps)}

Provide 2-3 actionable suggestions to address these gaps."""
            
            explanation = self._generate_text(prompt, max_tokens=200)
            gaps.append(f"\nRecommendations: {explanation}")
        
        return gaps
    
    def rewrite_bullet(self, bullet: str, jd_responsibilities: List[str]) -> str:
        """
        Rewrite a resume bullet to be more impactful and match JD.
        
        Args:
            bullet: Original bullet point
            jd_responsibilities: Job description responsibilities
            
        Returns:
            Rewritten bullet point
        """
        jd_context = ', '.join(jd_responsibilities[:3])
        
        prompt = f"""Rewrite this resume bullet to be more impactful, include metrics if possible, and align with these job responsibilities: {jd_context}

Original bullet: {bullet}

Rewritten bullet (STAR format - Situation, Task, Action, Result):"""
        
        rewritten = self._generate_text(prompt, max_tokens=100)
        return rewritten.strip()
    
    def generate_interview_questions(self, scores: Dict, resume_data: Dict, 
                                    jd_data: Dict, num_questions: int = 5) -> List[str]:
        """
        Generate interview questions based on gaps and strengths.
        
        Args:
            scores: Scoring results
            resume_data: Resume data
            jd_data: Job description data
            num_questions: Number of questions to generate
            
        Returns:
            List of interview questions
        """
        missing_skills = scores.get('missing_skills', [])
        weak_areas = scores.get('weak_areas', [])
        
        prompt = f"""Generate {num_questions} interview questions that a recruiter might ask based on:

Missing skills: {', '.join(missing_skills[:3]) if missing_skills else 'None'}
Weak areas: {', '.join(weak_areas[:2]) if weak_areas else 'None'}
Job title: {jd_data.get('title', 'Unknown')}

Generate specific, relevant interview questions (one per line):"""
        
        questions_text = self._generate_text(prompt, max_tokens=200)
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and '?' in q]
        
        return questions[:num_questions]
    
    def generate_action_items(self, scores: Dict, resume_data: Dict, 
                             jd_data: Dict) -> List[str]:
        """
        Generate actionable improvement suggestions.
        
        Args:
            scores: Scoring results
            resume_data: Resume data
            jd_data: Job description data
            
        Returns:
            List of action items
        """
        action_items = []
        
        # Skill-related actions
        missing_skills = scores.get('missing_skills', [])
        if missing_skills:
            action_items.append(f"Add missing required skills: {', '.join(missing_skills[:3])}")
        
        # Experience-related actions
        exp_details = scores.get('experience_details', {})
        gap_years = exp_details.get('gap_years', 0)
        if gap_years > 0:
            action_items.append(f"Highlight relevant experience to address {gap_years}-year gap")
        
        # Bullet rewriting
        if scores.get('role_responsibility_score', 0) < 60:
            action_items.append("Rewrite experience bullets to better match job responsibilities")
        
        # ATS improvements
        ats_details = scores.get('ats_details', {})
        if not ats_details.get('has_contact_info', False):
            action_items.append("Add complete contact information (email, phone)")
        if not ats_details.get('has_skills_section', False):
            action_items.append("Add a dedicated skills section")
        
        # Generate LLM suggestions
        prompt = f"""Based on this resume evaluation, provide 3 specific, actionable improvement suggestions:

Overall Score: {scores.get('overall_score', 0)}/100
Missing Skills: {', '.join(missing_skills[:3]) if missing_skills else 'None'}
Weak Areas: {', '.join(scores.get('weak_areas', [])[:2]) if scores.get('weak_areas') else 'None'}

Provide 3 concrete action items (one per line):"""
        
        llm_suggestions = self._generate_text(prompt, max_tokens=150)
        llm_items = [item.strip() for item in llm_suggestions.split('\n') if item.strip()]
        action_items.extend(llm_items[:3])
        
        return action_items[:10]  # Limit to 10 items
    
    def _generate_text(self, prompt: str, max_tokens: int = 200, 
                      temperature: float = 0.7) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            if self.provider == "openai":
                return self._generate_openai(prompt, max_tokens, temperature)
            elif self.provider in ["huggingface", "local"]:
                return self._generate_local(prompt, max_tokens, temperature)
            else:
                logger.error(f"Unsupported provider: {self.provider}")
                return ""
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"[Error generating explanation: {str(e)}]"
    
    def _generate_openai(self, prompt: str, max_tokens: int, 
                        temperature: float) -> str:
        """Generate text using OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful resume evaluation assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"[OpenAI API error: {str(e)}]"
    
    def _generate_local(self, prompt: str, max_tokens: int, 
                       temperature: float) -> str:
        """Generate text using local model."""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the output
            return generated[len(prompt):].strip()
        except Exception as e:
            logger.error(f"Local model error: {e}")
            return f"[Local model error: {str(e)}]"

