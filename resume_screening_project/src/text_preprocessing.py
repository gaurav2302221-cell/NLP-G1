"""
Text Preprocessing Module

This module handles text normalization, cleaning, and preprocessing.
Uses spaCy and NLTK for advanced NLP preprocessing.
"""

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logger.warning("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None


def preprocess_text(text: str) -> str:
    """
    Preprocess and clean text for NLP analysis.
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs and email addresses
    3. Remove special characters and numbers
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Cleaned and preprocessed text
    """
    try:
        if not text or not isinstance(text, str):
            logger.warning("Invalid text input")
            return ""
        
        # Step 1: Convert to lowercase
        text = text.lower()
        
        # Step 2: Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Step 3: Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Step 4: Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 5: Tokenize
        tokens = word_tokenize(text)
        
        # Step 6: Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        
        # Step 7: Lemmatize using spaCy
        if nlp:
            lemmatized_tokens = []
            doc = nlp(' '.join(tokens))
            for token in doc:
                lemmatized_tokens.append(token.lemma_)
            text = ' '.join(lemmatized_tokens)
        else:
            text = ' '.join(tokens)
        
        logger.info("Text preprocessing completed successfully")
        return text
        
    except Exception as e:
        logger.error(f"Error during text preprocessing: {e}")
        raise


def normalize_text(text: str) -> str:
    """
    Simple text normalization without stopword removal.
    Useful for maintaining important context.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        raise


def clean_text(text: str) -> str:
    """
    Aggressive text cleaning for specific use cases.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    try:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise


def extract_key_phrases(text: str) -> list:
    """
    Extract key noun phrases from text using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of key phrases
    """
    try:
        if not nlp:
            logger.warning("spaCy model not available")
            return []
        
        doc = nlp(text)
        phrases = [chunk.text for chunk in doc.noun_chunks]
        return phrases
        
    except Exception as e:
        logger.error(f"Error extracting key phrases: {e}")
        return []


def tokenize_text(text: str) -> list:
    """
    Tokenize text into words.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        list: List of tokens
    """
    try:
        tokens = word_tokenize(text.lower())
        return tokens
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return []
