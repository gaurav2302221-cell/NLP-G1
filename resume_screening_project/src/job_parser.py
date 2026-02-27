"""
Job Description Parser Module

This module processes job descriptions and extracts relevant information
including keywords and required skills.
"""

import re
import logging
from typing import Set, List, Dict
from src.text_preprocessing import preprocess_text, extract_key_phrases
from src.skill_extractor import extract_skills

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_job_description(job_text: str) -> Dict:
    """
    Process and analyze a job description.
    
    Args:
        job_text (str): Raw job description text
        
    Returns:
        Dict: Contains preprocessed text, skills, and keywords
    """
    try:
        if not job_text or not isinstance(job_text, str):
            raise ValueError("Invalid job description text")
        
        logger.info("Processing job description...")
        
        # Preprocess the job description
        processed_text = preprocess_text(job_text)
        
        # Extract skills from job description
        required_skills = extract_skills(job_text)
        
        # Extract key phrases/keywords
        keywords = extract_key_phrases(job_text)
        
        # Extract job title if possible
        job_title = extract_job_title(job_text)
        
        result = {
            'original_text': job_text,
            'processed_text': processed_text,
            'required_skills': required_skills,
            'keywords': keywords,
            'job_title': job_title,
            'num_required_skills': len(required_skills)
        }
        
        logger.info(f"Job description processed. Skills found: {len(required_skills)}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing job description: {e}")
        raise


def extract_job_title(job_text: str) -> str:
    """
    Try to extract job title from job description text.
    
    Args:
        job_text (str): Job description text
        
    Returns:
        str: Extracted or placeholder job title
    """
    try:
        # Look for common patterns
        patterns = [
            r'(?:Position|Job Title|Role)[:\s]+([^.\n]+)',
            r'^([^.\n]+)\s*-\s*(?:Job|Position)',
            r'(?:We are looking for|We seek)[:\s]+(?:a|an)?\s*([^.\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, job_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Extract first line as fallback
        first_line = job_text.split('\n')[0].strip()
        if first_line and len(first_line) < 100:
            return first_line
        
        return "Data Science/ML Position"
        
    except Exception as e:
        logger.warning(f"Could not extract job title: {e}")
        return "Position"


def extract_job_requirements(job_text: str) -> Dict[str, List[str]]:
    """
    Extract different types of job requirements from text.
    
    Args:
        job_text (str): Job description text
        
    Returns:
        Dict: Different categories of requirements
    """
    try:
        requirements = {
            'required': [],
            'preferred': [],
            'education': [],
            'experience': []
        }
        
        # Look for required section
        required_pattern = r'(?:Required|Must Have)[:\s]*\n?(.*?)(?=\n\n|\nPreferred|$)'
        match = re.search(required_pattern, job_text, re.IGNORECASE | re.DOTALL)
        if match:
            requirements['required'] = extract_bullet_points(match.group(1))
        
        # Look for preferred section
        preferred_pattern = r'(?:Preferred|Nice to Have)[:\s]*\n?(.*?)(?=\n\n|$)'
        match = re.search(preferred_pattern, job_text, re.IGNORECASE | re.DOTALL)
        if match:
            requirements['preferred'] = extract_bullet_points(match.group(1))
        
        # Look for education requirements
        education_pattern = r'(?:Education|Degree)[:\s]*\n?(.*?)(?=\n\n|$)'
        match = re.search(education_pattern, job_text, re.IGNORECASE | re.DOTALL)
        if match:
            requirements['education'] = extract_bullet_points(match.group(1))
        
        # Look for experience requirements
        experience_pattern = r'(?:Experience)[:\s]*\n?(.*?)(?=\n\n|$)'
        match = re.search(experience_pattern, job_text, re.IGNORECASE | re.DOTALL)
        if match:
            requirements['experience'] = extract_bullet_points(match.group(1))
        
        logger.info(f"Extracted requirements: {len(requirements['required'])} required, "
                   f"{len(requirements['preferred'])} preferred")
        return requirements
        
    except Exception as e:
        logger.error(f"Error extracting job requirements: {e}")
        return {'required': [], 'preferred': [], 'education': [], 'experience': []}


def extract_bullet_points(text: str) -> List[str]:
    """
    Extract bullet points from text.
    
    Args:
        text (str): Text containing bullet points
        
    Returns:
        List[str]: Extracted bullet points
    """
    try:
        # Match common bullet point patterns
        pattern = r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)'
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        
        bullet_points = [point.strip() for point in matches if point.strip()]
        return bullet_points
        
    except Exception as e:
        logger.debug(f"Error extracting bullet points: {e}")
        return []


def get_job_keywords(job_text: str) -> Set[str]:
    """
    Extract important keywords from job description.
    
    Args:
        job_text (str): Job description text
        
    Returns:
        Set[str]: Important keywords
    """
    try:
        # Preprocess and extract keywords
        processed = preprocess_text(job_text)
        keywords = set(processed.split())
        
        # Remove very common words that aren't meaningful
        common_words = {'job', 'position', 'role', 'work', 'team', 'will', 'must', 'should'}
        keywords = keywords - common_words
        
        return keywords
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return set()
