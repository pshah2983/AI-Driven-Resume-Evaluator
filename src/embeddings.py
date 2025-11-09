"""
Embedding Generation Module

Creates dense embeddings for resumes and job descriptions using SentenceTransformers.
Handles batch processing and caching.
"""

import logging
from typing import List, Dict, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using SentenceTransformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", 
                 device: Optional[str] = None, batch_size: int = 32):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to use ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Cache for embeddings (simple in-memory cache)
        self._cache = {}
    
    def encode(self, texts: Union[str, List[str]], 
               normalize: bool = True, 
               show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            normalize: Whether to normalize embeddings to unit length
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        cache_key = tuple(texts) if len(texts) == 1 else None
        if cache_key and cache_key in self._cache:
            logger.debug("Using cached embedding")
            return self._cache[cache_key]
        
        # Generate embeddings
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Cache single text embeddings
            if cache_key:
                self._cache[cache_key] = embeddings
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def encode_resume(self, resume_data: Dict) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for different sections of a resume.
        
        Args:
            resume_data: Normalized resume data dictionary
            
        Returns:
            Dictionary with embeddings for each section:
            {
                'overall': np.ndarray,
                'skills': np.ndarray,
                'experience_bullets': List[np.ndarray],
                'education': np.ndarray
            }
        """
        embeddings = {}
        
        # Overall resume embedding
        overall_text = resume_data.get('raw_text', '') or ''
        if overall_text and len(overall_text.strip()) > 0:
            try:
                embeddings['overall'] = self.encode(overall_text)[0]
            except Exception as e:
                logger.warning(f"Error encoding overall text: {e}")
        
        # Skills embedding
        skills = resume_data.get('skills', []) or []
        if skills and len(skills) > 0:
            try:
                skills_text = ', '.join(str(s) for s in skills if s)
                if skills_text:
                    embeddings['skills'] = self.encode(skills_text)[0]
            except Exception as e:
                logger.warning(f"Error encoding skills: {e}")
        
        # Experience bullets
        experience = resume_data.get('experience', []) or []
        experience_bullets = []
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            desc = exp.get('description', '') or ''
            if desc:
                try:
                    # Split into individual bullets
                    bullets = [b.strip() for b in desc.split('\n') if b.strip()]
                    for bullet in bullets:
                        if len(bullet) > 10:  # Only meaningful bullets
                            bullet_emb = self.encode(bullet)[0]
                            experience_bullets.append(bullet_emb)
                except Exception as e:
                    logger.warning(f"Error encoding experience bullet: {e}")
        embeddings['experience_bullets'] = experience_bullets
        
        # Education embedding
        education = resume_data.get('education', []) or []
        if education and len(education) > 0:
            try:
                edu_text = ' '.join([
                    f"{e.get('degree', '')} {e.get('institution', '')}" 
                    for e in education if isinstance(e, dict)
                ])
                if edu_text.strip():
                    embeddings['education'] = self.encode(edu_text)[0]
            except Exception as e:
                logger.warning(f"Error encoding education: {e}")
        
        return embeddings
    
    def encode_job_description(self, jd_data: Dict) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for different sections of a job description.
        
        Args:
            jd_data: Normalized JD data dictionary
            
        Returns:
            Dictionary with embeddings for each section:
            {
                'overall': np.ndarray,
                'required_skills': np.ndarray,
                'preferred_skills': np.ndarray,
                'responsibilities': np.ndarray,
                'requirements': np.ndarray
            }
        """
        embeddings = {}
        
        # Overall JD embedding
        overall_text = jd_data.get('raw_text', '') or ''
        if overall_text and len(overall_text.strip()) > 0:
            try:
                embeddings['overall'] = self.encode(overall_text)[0]
            except Exception as e:
                logger.warning(f"Error encoding overall JD text: {e}")
        
        # Required skills
        required_skills = jd_data.get('required_skills', []) or []
        if required_skills and len(required_skills) > 0:
            try:
                skills_text = ', '.join(str(s) for s in required_skills if s)
                if skills_text:
                    embeddings['required_skills'] = self.encode(skills_text)[0]
            except Exception as e:
                logger.warning(f"Error encoding required skills: {e}")
        
        # Preferred skills
        preferred_skills = jd_data.get('preferred_skills', []) or []
        if preferred_skills and len(preferred_skills) > 0:
            try:
                skills_text = ', '.join(str(s) for s in preferred_skills if s)
                if skills_text:
                    embeddings['preferred_skills'] = self.encode(skills_text)[0]
            except Exception as e:
                logger.warning(f"Error encoding preferred skills: {e}")
        
        # Responsibilities
        responsibilities = jd_data.get('responsibilities', []) or []
        if responsibilities and len(responsibilities) > 0:
            try:
                resp_text = ' '.join(str(r) for r in responsibilities if r)
                if resp_text.strip():
                    embeddings['responsibilities'] = self.encode(resp_text)[0]
            except Exception as e:
                logger.warning(f"Error encoding responsibilities: {e}")
        
        # Requirements
        requirements = jd_data.get('requirements', []) or []
        if requirements and len(requirements) > 0:
            try:
                req_text = ' '.join(str(r) for r in requirements if r)
                if req_text.strip():
                    embeddings['requirements'] = self.encode(req_text)[0]
            except Exception as e:
                logger.warning(f"Error encoding requirements: {e}")
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Ensure embeddings are normalized
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")


class BatchEmbeddingGenerator:
    """Generate embeddings for multiple resumes/JDs efficiently."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize batch embedding generator.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
        """
        self.embedding_generator = embedding_generator
    
    def encode_resumes_batch(self, resumes: List[Dict]) -> List[Dict[str, np.ndarray]]:
        """
        Generate embeddings for multiple resumes.
        
        Args:
            resumes: List of normalized resume data dictionaries
            
        Returns:
            List of embedding dictionaries (one per resume)
        """
        embeddings_list = []
        
        for resume in resumes:
            try:
                embeddings = self.embedding_generator.encode_resume(resume)
                embeddings_list.append(embeddings)
            except Exception as e:
                logger.error(f"Error encoding resume: {e}")
                embeddings_list.append({})
        
        return embeddings_list
    
    def encode_jds_batch(self, jds: List[Dict]) -> List[Dict[str, np.ndarray]]:
        """
        Generate embeddings for multiple job descriptions.
        
        Args:
            jds: List of normalized JD data dictionaries
            
        Returns:
            List of embedding dictionaries (one per JD)
        """
        embeddings_list = []
        
        for jd in jds:
            try:
                embeddings = self.embedding_generator.encode_job_description(jd)
                embeddings_list.append(embeddings)
            except Exception as e:
                logger.error(f"Error encoding JD: {e}")
                embeddings_list.append({})
        
        return embeddings_list

