"""
FAISS Vector Index Module

Manages FAISS indices for efficient similarity search across resumes and JDs.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Manage FAISS indices for vector search."""
    
    def __init__(self, dimension: int = 768, index_type: str = "L2"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Dimension of embedding vectors
            index_type: Type of index ("L2" for L2 distance, "IP" for inner product)
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Metadata storage (map index to document info)
        self.metadata = []
        logger.info(f"Initialized FAISS index: {index_type}, dimension={dimension}")
    
    def add(self, embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Numpy array of embeddings (shape: [n_vectors, dimension])
            metadata: Optional list of metadata dictionaries (one per embedding)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        if metadata:
            if len(metadata) != embeddings.shape[0]:
                raise ValueError("Metadata length must match number of embeddings")
            self.metadata.extend(metadata)
        else:
            # Create default metadata
            self.metadata.extend([{}] * embeddings.shape[0])
        
        logger.info(f"Added {embeddings.shape[0]} vectors to index. Total: {self.index.ntotal}")
    
    def search(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_embeddings: Query embeddings (shape: [n_queries, dimension])
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices):
            - distances: Distances to nearest neighbors (shape: [n_queries, k])
            - indices: Indices of nearest neighbors (shape: [n_queries, k])
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, cannot search")
            return np.array([]), np.array([])
        
        if query_embeddings.shape[1] != self.dimension:
            raise ValueError(f"Query dimension mismatch: expected {self.dimension}, got {query_embeddings.shape[1]}")
        
        k = min(k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_embeddings.astype('float32'), k)
        
        logger.debug(f"Searched for {query_embeddings.shape[0]} queries, returning top {k} results")
        return distances, indices
    
    def get_metadata(self, indices: np.ndarray) -> List[Dict]:
        """
        Get metadata for given indices.
        
        Args:
            indices: Array of indices
            
        Returns:
            List of metadata dictionaries
        """
        metadata_list = []
        for idx in indices.flatten():
            if 0 <= idx < len(self.metadata):
                metadata_list.append(self.metadata[idx])
            else:
                metadata_list.append({})
        return metadata_list
    
    def save(self, index_path: str, metadata_path: Optional[str] = None):
        """
        Save index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata (defaults to index_path + '.metadata')
        """
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        if metadata_path is None:
            metadata_path = str(index_path) + '.metadata'
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: Optional[str] = None):
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file (defaults to index_path + '.metadata')
        """
        index_path = Path(index_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d
        logger.info(f"Loaded FAISS index from {index_path}. Dimension: {self.dimension}, Vectors: {self.index.ntotal}")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = str(index_path) + '.metadata'
        
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}, using empty metadata")
            self.metadata = [{}] * self.index.ntotal
    
    def reset(self):
        """Reset the index (remove all vectors)."""
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        
        self.metadata = []
        logger.info("Reset FAISS index")
    
    def size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal


class ResumeJDIndex:
    """Specialized index for resume-JD matching."""
    
    def __init__(self, dimension: int = 768):
        """
        Initialize resume-JD index.
        
        Args:
            dimension: Dimension of embedding vectors
        """
        self.dimension = dimension
        self.resume_index = FAISSIndex(dimension, "L2")
        self.jd_index = FAISSIndex(dimension, "L2")
        
        logger.info("Initialized Resume-JD index")
    
    def add_resumes(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add resume embeddings to index."""
        self.resume_index.add(embeddings, metadata)
    
    def add_jds(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add JD embeddings to index."""
        self.jd_index.add(embeddings, metadata)
    
    def find_similar_resumes(self, jd_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Find resumes similar to a job description.
        
        Args:
            jd_embedding: Job description embedding
            k: Number of similar resumes to return
            
        Returns:
            List of metadata dictionaries for similar resumes
        """
        distances, indices = self.resume_index.search(jd_embedding.reshape(1, -1), k)
        metadata = self.resume_index.get_metadata(indices[0])
        
        # Add similarity scores
        for i, meta in enumerate(metadata):
            meta['similarity_score'] = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
            meta['distance'] = float(distances[0][i])
        
        return metadata
    
    def find_similar_jds(self, resume_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Find job descriptions similar to a resume.
        
        Args:
            resume_embedding: Resume embedding
            k: Number of similar JDs to return
            
        Returns:
            List of metadata dictionaries for similar JDs
        """
        distances, indices = self.jd_index.search(resume_embedding.reshape(1, -1), k)
        metadata = self.jd_index.get_metadata(indices[0])
        
        # Add similarity scores
        for i, meta in enumerate(metadata):
            meta['similarity_score'] = 1.0 / (1.0 + distances[0][i])
            meta['distance'] = float(distances[0][i])
        
        return metadata
    
    def save(self, base_path: str):
        """Save both indices."""
        self.resume_index.save(f"{base_path}_resumes.index")
        self.jd_index.save(f"{base_path}_jds.index")
    
    def load(self, base_path: str):
        """Load both indices."""
        self.resume_index.load(f"{base_path}_resumes.index")
        self.jd_index.load(f"{base_path}_jds.index")

