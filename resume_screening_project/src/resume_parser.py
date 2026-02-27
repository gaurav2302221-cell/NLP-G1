"""
Resume Parser Module

This module handles PDF resume extraction and text parsing.
Uses PyMuPDF for efficient PDF text extraction.
"""

import os
import fitz  # PyMuPDF
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_resume(file_path: str) -> str:
    """
    Extract text from a single PDF resume file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the resume
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If PDF parsing fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume file not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be PDF format: {file_path}")
        
        # Open PDF file
        pdf_document = fitz.open(file_path)
        text = ""
        
        # Extract text from all pages
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
        
        pdf_document.close()
        
        logger.info(f"Successfully parsed resume: {file_path}")
        return text.strip()
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid file format: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing resume {file_path}: {e}")
        raise RuntimeError(f"Failed to parse resume: {str(e)}")


def parse_multiple_resumes(directory: str) -> Dict[str, str]:
    """
    Extract text from multiple PDF resume files in a directory.
    
    Args:
        directory (str): Path to directory containing PDF files
        
    Returns:
        Dict[str, str]: Dictionary with file names as keys and extracted text as values
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    try:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        resumes = {}
        
        # Iterate through all PDF files in directory
        for file_name in os.listdir(directory):
            if file_name.lower().endswith('.pdf'):
                file_path = os.path.join(directory, file_name)
                try:
                    text = parse_resume(file_path)
                    resumes[file_name] = text
                    logger.info(f"Parsed: {file_name}")
                except Exception as e:
                    logger.warning(f"Skipped {file_name}: {str(e)}")
                    continue
        
        logger.info(f"Successfully parsed {len(resumes)} resumes from {directory}")
        return resumes
        
    except FileNotFoundError as e:
        logger.error(f"Directory error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing multiple resumes: {e}")
        raise


def get_resume_filename(file_path: str) -> str:
    """
    Extract candidate name from resume file path.
    
    Args:
        file_path (str): Path to the resume file
        
    Returns:
        str: Candidate name extracted from filename
    """
    file_name = os.path.basename(file_path)
    # Remove .pdf extension and underscores/hyphens
    candidate_name = file_name.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    return candidate_name
