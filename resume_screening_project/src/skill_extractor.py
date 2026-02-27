"""
Skill Extraction Module

This module extracts technical and professional skills from resumes
using a predefined skill dictionary and pattern matching.
"""

import re
import logging
from typing import List, Set, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive skill dictionary
SKILL_DICTIONARY = {
    # Programming Languages
    'python': ['python', 'py'],
    'java': ['java'],
    'cpp': ['c++', 'cpp', 'c plus plus'],
    'javascript': ['javascript', 'js', 'node.js', 'nodejs'],
    'csharp': ['c#', 'csharp', 'c sharp'],
    'php': ['php'],
    'ruby': ['ruby'],
    'go': ['golang', 'go'],
    'rust': ['rust'],
    'kotlin': ['kotlin'],
    'swift': ['swift'],
    'r': ['r programming'],
    'scala': ['scala'],
    'typescript': ['typescript', 'ts'],
    
    # Machine Learning & AI
    'machine learning': ['machine learning', 'ml', 'machine-learning'],
    'deep learning': ['deep learning', 'deep-learning', 'neural networks', 'cnn', 'rnn', 'lstm'],
    'nlp': ['nlp', 'natural language processing', 'text mining'],
    'computer vision': ['computer vision', 'cv', 'image processing'],
    'reinforcement learning': ['reinforcement learning', 'rl'],
    
    # Data Science & Analysis
    'data science': ['data science', 'data scientist', 'data-science'],
    'data analysis': ['data analysis', 'data analytics', 'analytics'],
    'big data': ['big data', 'hadoop', 'spark', 'apache spark'],
    'statistics': ['statistics', 'statistical analysis'],
    
    # Databases
    'sql': ['sql', 'mysql', 'postgresql', 'oracle', 'mssql', 'tsql'],
    'mongodb': ['mongodb', 'mongo'],
    'cassandra': ['cassandra'],
    'redis': ['redis'],
    'dynamodb': ['dynamodb'],
    'elasticsearch': ['elasticsearch'],
    
    # Data Processing & Analysis Libraries
    'pandas': ['pandas'],
    'numpy': ['numpy', 'numpy array'],
    'scipy': ['scipy'],
    'scikit-learn': ['scikit-learn', 'sklearn', 'scikit learn'],
    'matplotlib': ['matplotlib'],
    'seaborn': ['seaborn'],
    'plotly': ['plotly'],
    
    # Deep Learning Frameworks
    'tensorflow': ['tensorflow', 'tensorflow.js'],
    'pytorch': ['pytorch', 'torch'],
    'keras': ['keras'],
    'mxnet': ['mxnet', 'apache mxnet'],
    
    # Cloud Platforms
    'aws': ['aws', 'amazon web services', 'amazon aws', 'ec2', 's3'],
    'azure': ['azure', 'microsoft azure'],
    'gcp': ['gcp', 'google cloud platform', 'google cloud'],
    'heroku': ['heroku'],
    'ibm cloud': ['ibm cloud'],
    
    # Containerization & Orchestration
    'docker': ['docker', 'containerization'],
    'kubernetes': ['kubernetes', 'k8s'],
    'docker compose': ['docker-compose', 'docker compose'],
    
    # Web Frameworks
    'django': ['django'],
    'flask': ['flask'],
    'fastapi': ['fastapi'],
    'spring': ['spring framework', 'spring boot', 'spring'],
    'react': ['react', 'react.js'],
    'angular': ['angular', 'angularjs'],
    'vue': ['vue', 'vue.js'],
    
    # Version Control
    'git': ['git', 'github', 'gitlab', 'bitbucket'],
    
    # Visualization & BI
    'tableau': ['tableau'],
    'powerbi': ['powerbi', 'power bi', 'power-bi'],
    'looker': ['looker'],
    'qlik': ['qlik', 'qlikview'],
    
    # CI/CD
    'jenkins': ['jenkins'],
    'gitlab ci': ['gitlab ci'],
    'github actions': ['github actions'],
    'circleci': ['circleci'],
    
    # Other Tools
    'jupyter': ['jupyter', 'ipython'],
    'anaconda': ['anaconda', 'conda'],
    'pip': ['pip'],
    'linux': ['linux', 'unix'],
    'windows': ['windows'],
    'api': ['api', 'rest api', 'restful', 'graphql'],
    'json': ['json'],
    'xml': ['xml'],
    'html': ['html', 'html5'],
    'css': ['css', 'css3'],
    'excel': ['excel', 'vba'],
    'regex': ['regex', 'regular expressions'],
}


def extract_skills(text: str) -> Set[str]:
    """
    Extract technical skills from resume text using skill dictionary.
    
    Args:
        text (str): Resume text
        
    Returns:
        Set[str]: Set of detected skills
    """
    try:
        text_lower = text.lower()
        detected_skills = set()
        
        # Search for each skill and its variations
        for skill_name, variations in SKILL_DICTIONARY.items():
            for variation in variations:
                # Use word boundaries to match whole words/phrases
                pattern = r'\b' + re.escape(variation) + r'\b'
                if re.search(pattern, text_lower):
                    detected_skills.add(skill_name)
                    break  # Found this skill, move to next
        
        logger.info(f"Extracted {len(detected_skills)} skills from text")
        return detected_skills
        
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        return set()


def extract_skills_with_frequency(text: str) -> Dict[str, int]:
    """
    Extract skills and count their frequency in the text.
    
    Args:
        text (str): Resume text
        
    Returns:
        Dict[str, int]: Dictionary of skills and their frequency
    """
    try:
        text_lower = text.lower()
        skill_frequency = {}
        
        for skill_name, variations in SKILL_DICTIONARY.items():
            count = 0
            for variation in variations:
                pattern = r'\b' + re.escape(variation) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                count += matches
            
            if count > 0:
                skill_frequency[skill_name] = count
        
        logger.info(f"Extracted skills with frequency: {skill_frequency}")
        return skill_frequency
        
    except Exception as e:
        logger.error(f"Error extracting skills with frequency: {e}")
        return {}


def get_missing_skills(required_skills: Set[str], candidate_skills: Set[str]) -> Set[str]:
    """
    Identify skills required for the job that candidate doesn't have.
    
    Args:
        required_skills (Set[str]): Skills required by the job
        candidate_skills (Set[str]): Skills from candidate resume
        
    Returns:
        Set[str]: Missing skills
    """
    return required_skills - candidate_skills


def get_matching_skills(required_skills: Set[str], candidate_skills: Set[str]) -> Set[str]:
    """
    Identify skills that match between job requirements and candidate.
    
    Args:
        required_skills (Set[str]): Skills required by the job
        candidate_skills (Set[str]): Skills from candidate resume
        
    Returns:
        Set[str]: Matching skills
    """
    return required_skills & candidate_skills


def calculate_skill_match_percentage(required_skills: Set[str], 
                                     candidate_skills: Set[str]) -> float:
    """
    Calculate the percentage of required skills that candidate has.
    
    Args:
        required_skills (Set[str]): Skills required by the job
        candidate_skills (Set[str]): Skills from candidate resume
        
    Returns:
        float: Match percentage (0-100)
    """
    if not required_skills:
        return 0.0
    
    matching_skills = len(get_matching_skills(required_skills, candidate_skills))
    total_required = len(required_skills)
    
    match_percentage = (matching_skills / total_required) * 100
    return round(match_percentage, 2)


def get_skill_summary(required_skills: Set[str], 
                     candidate_skills: Set[str]) -> Dict:
    """
    Get a detailed summary of skill matching.
    
    Args:
        required_skills (Set[str]): Skills required by the job
        candidate_skills (Set[str]): Skills from candidate resume
        
    Returns:
        Dict: Summary information about skill matching
    """
    matching = get_matching_skills(required_skills, candidate_skills)
    missing = get_missing_skills(required_skills, candidate_skills)
    additional = candidate_skills - required_skills
    
    return {
        'matching_skills': list(matching),
        'missing_skills': list(missing),
        'additional_skills': list(additional),
        'match_percentage': calculate_skill_match_percentage(required_skills, candidate_skills),
        'total_required': len(required_skills),
        'total_matched': len(matching)
    }
