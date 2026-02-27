"""
Similarity Model Module

This module implements multiple similarity approaches:
1. TF-IDF + Cosine Similarity
2. Word Embeddings (GloVe/Word2Vec style)
3. Sentence Transformers (BERT)

The final score is computed as the average of these three approaches.
"""

import logging
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model for sentence transformers
BERT_MODEL = None


def initialize_bert_model():
    """Initialize the sentence transformer model."""
    global BERT_MODEL
    try:
        if BERT_MODEL is None:
            logger.info("Loading BERT model (all-MiniLM-L6-v2)...")
            BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("BERT model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading BERT model: {e}")
        raise


def tfidf_similarity(resume_text: str, job_text: str) -> float:
    """
    Calculate similarity using TF-IDF + Cosine Similarity.
    
    This approach:
    - Converts text to TF-IDF vectors
    - Computes cosine similarity between vectors
    - Handles importance of term frequency
    
    Args:
        resume_text (str): Resume text
        job_text (str): Job description text
        
    Returns:
        float: Similarity score (0-1)
    """
    try:
        if not resume_text or not job_text:
            logger.warning("Empty text provided to TF-IDF similarity")
            return 0.0
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        # Fit and transform documents
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        logger.debug(f"TF-IDF Similarity: {similarity:.4f}")
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error in TF-IDF similarity calculation: {e}")
        return 0.0


def bert_similarity(resume_text: str, job_text: str) -> float:
    """
    Calculate similarity using BERT sentence embeddings.
    
    This approach:
    - Uses pre-trained BERT model (all-MiniLM-L6-v2)
    - Generates sentence embeddings
    - Computes cosine similarity on semantic vectors
    
    Args:
        resume_text (str): Resume text
        job_text (str): Job description text
        
    Returns:
        float: Similarity score (0-1)
    """
    try:
        if not resume_text or not job_text:
            logger.warning("Empty text provided to BERT similarity")
            return 0.0
        
        # Initialize model if not already done
        initialize_bert_model()
        
        # Encode texts to embeddings
        embeddings = BERT_MODEL.encode([resume_text, job_text], convert_to_numpy=True)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            embeddings[0:1], 
            embeddings[1:2]
        )[0][0]
        
        logger.debug(f"BERT Similarity: {similarity:.4f}")
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error in BERT similarity calculation: {e}")
        return 0.0


def word_embedding_similarity(resume_text: str, job_text: str) -> float:
    """
    Calculate similarity using word embeddings.
    
    This approach:
    - Splits text into words
    - Uses average word vectors (simple approach)
    - Computes cosine similarity
    
    Args:
        resume_text (str): Resume text
        job_text (str): Job description text
        
    Returns:
        float: Similarity score (0-1)
    """
    try:
        if not resume_text or not job_text:
            logger.warning("Empty text provided to word embedding similarity")
            return 0.0
        
        # Initialize BERT model for embeddings
        initialize_bert_model()
        
        # Split into sentences
        resume_sentences = resume_text.split('.')
        job_sentences = job_text.split('.')
        
        # Remove empty sentences
        resume_sentences = [s.strip() for s in resume_sentences if s.strip()]
        job_sentences = [s.strip() for s in job_sentences if s.strip()]
        
        if not resume_sentences or not job_sentences:
            return 0.0
        
        # Encode sentences and get average embedding
        resume_embeddings = BERT_MODEL.encode(resume_sentences, convert_to_numpy=True)
        job_embeddings = BERT_MODEL.encode(job_sentences, convert_to_numpy=True)
        
        resume_avg = np.mean(resume_embeddings, axis=0)
        job_avg = np.mean(job_embeddings, axis=0)
        
        # Calculate similarity
        similarity = cosine_similarity([resume_avg], [job_avg])[0][0]
        
        logger.debug(f"Word Embedding Similarity: {similarity:.4f}")
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error in word embedding similarity calculation: {e}")
        return 0.0


def compute_similarity(resume_text: str, job_text: str) -> Tuple[float, dict]:
    """
    Compute overall similarity score using all three methods.
    
    Final score = Average of:
    - TF-IDF Similarity (weight: 1/3)
    - BERT Similarity (weight: 1/3)
    - Word Embedding Similarity (weight: 1/3)
    
    Args:
        resume_text (str): Resume text
        job_text (str): Job description text
        
    Returns:
        Tuple[float, dict]: Overall similarity score and breakdown of individual scores
    """
    try:
        logger.info("Computing overall similarity score...")
        
        # Calculate individual similarities
        tfidf_score = tfidf_similarity(resume_text, job_text)
        bert_score = bert_similarity(resume_text, job_text)
        embed_score = word_embedding_similarity(resume_text, job_text)
        
        # Compute weighted average (equal weights)
        overall_score = (tfidf_score + bert_score + embed_score) / 3
        
        # Round to 4 decimal places
        overall_score = round(overall_score, 4)
        
        breakdown = {
            'tfidf': round(tfidf_score, 4),
            'bert': round(bert_score, 4),
            'word_embedding': round(embed_score, 4),
            'overall': overall_score
        }
        
        logger.info(f"Similarity Breakdown: {breakdown}")
        return overall_score, breakdown
        
    except Exception as e:
        logger.error(f"Error computing overall similarity: {e}")
        return 0.0, {
            'tfidf': 0.0,
            'bert': 0.0,
            'word_embedding': 0.0,
            'overall': 0.0
        }


def normalize_similarity(score: float) -> float:
    """
    Normalize similarity score to 0-100 range for display.
    
    Args:
        score (float): Similarity score (0-1)
        
    Returns:
        float: Normalized score (0-100)
    """
    return round(score * 100, 2)
