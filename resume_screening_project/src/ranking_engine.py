"""
Ranking Engine Module

This module ranks candidates based on combined scoring:
- Similarity Score: 70% weight
- Skill Match Percentage: 30% weight
"""

import logging
from typing import List, Dict, Tuple
from src.similarity_model import compute_similarity
from src.skill_extractor import calculate_skill_match_percentage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rank_candidates(candidate_resumes: Dict[str, Tuple[str, set]], 
                   job_description: str,
                   job_skills: set,
                   similarity_weight: float = 0.7,
                   skill_weight: float = 0.3) -> List[Dict]:
    """
    Rank candidates based on similarity and skill matching.
    
    Scoring Formula:
    Final Score = (Similarity Score * 0.7) + (Skill Match % / 100 * 0.3)
    
    Args:
        candidate_resumes (Dict): Dictionary with candidate names as keys and
                                 tuples of (resume_text, extracted_skills)
        job_description (str): Job description text
        job_skills (set): Set of skills required for the job
        similarity_weight (float): Weight for similarity score (default 0.7)
        skill_weight (float): Weight for skill match (default 0.3)
        
    Returns:
        List[Dict]: Ranked candidates with scores and rankings
    """
    try:
        if not candidate_resumes:
            logger.warning("No candidate resumes provided")
            return []
        
        epsilon = 1e-9
        if abs(similarity_weight + skill_weight - 1.0) > epsilon:
            logger.warning("Weights don't sum to 1.0, normalizing...")
            total = similarity_weight + skill_weight
            similarity_weight /= total
            skill_weight /= total
        
        candidate_scores = []
        
        for candidate_name, (resume_text, resume_skills) in candidate_resumes.items():
            try:
                # Calculate similarity score
                similarity_score, _ = compute_similarity(resume_text, job_description)
                
                # Calculate skill match percentage
                skill_match_pct = calculate_skill_match_percentage(job_skills, resume_skills)
                skill_match_normalized = skill_match_pct / 100
                
                # Calculate final score
                final_score = (similarity_score * similarity_weight) + \
                             (skill_match_normalized * skill_weight)
                
                candidate_scores.append({
                    'candidate_name': candidate_name,
                    'similarity_score': round(similarity_score * 100, 2),  # Convert to 0-100
                    'skill_match_percentage': skill_match_pct,
                    'final_score': round(final_score * 100, 2),  # Convert to 0-100
                    'extracted_skills': resume_skills,
                    'matching_skills': job_skills & resume_skills,
                    'missing_skills': job_skills - resume_skills
                })
                
                logger.info(f"Scored candidate: {candidate_name} - "
                          f"Similarity: {similarity_score:.4f}, "
                          f"Skill Match: {skill_match_pct:.2f}%")
                
            except Exception as e:
                logger.warning(f"Error scoring candidate {candidate_name}: {e}")
                continue
        
        # Sort by final score in descending order
        ranked_candidates = sorted(candidate_scores, 
                                  key=lambda x: x['final_score'], 
                                  reverse=True)
        
        # Add ranking number
        for idx, candidate in enumerate(ranked_candidates, 1):
            candidate['rank'] = idx
        
        logger.info(f"Ranking complete. Top candidate: "
                   f"{ranked_candidates[0]['candidate_name'] if ranked_candidates else 'None'}")
        
        return ranked_candidates
        
    except Exception as e:
        logger.error(f"Error during candidate ranking: {e}")
        raise


def get_top_candidates(ranked_candidates: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Get top N candidates from ranking results.
    
    Args:
        ranked_candidates (List[Dict]): Ranked candidate list
        top_n (int): Number of top candidates to return
        
    Returns:
        List[Dict]: Top N candidates
    """
    return ranked_candidates[:top_n]


def get_candidate_summary(ranked_candidates: List[Dict]) -> Dict:
    """
    Get summary statistics of candidate ranking.
    
    Args:
        ranked_candidates (List[Dict]): Ranked candidate list
        
    Returns:
        Dict: Summary statistics
    """
    if not ranked_candidates:
        return {
            'total_candidates': 0,
            'average_score': 0.0,
            'max_score': 0.0,
            'min_score': 0.0,
            'top_candidate': None
        }
    
    scores = [c['final_score'] for c in ranked_candidates]
    
    return {
        'total_candidates': len(ranked_candidates),
        'average_score': round(sum(scores) / len(scores), 2),
        'max_score': max(scores),
        'min_score': min(scores),
        'top_candidate': ranked_candidates[0]['candidate_name'],
        'top_candidate_score': ranked_candidates[0]['final_score']
    }


def filter_candidates_by_threshold(ranked_candidates: List[Dict], 
                                   threshold: float = 60.0) -> List[Dict]:
    """
    Filter candidates by minimum score threshold.
    
    Args:
        ranked_candidates (List[Dict]): Ranked candidate list
        threshold (float): Minimum score to qualify (0-100)
        
    Returns:
        List[Dict]: Candidates meeting threshold
    """
    if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 100:
        logger.warning(f"Invalid threshold {threshold}, using default 60")
        threshold = 60.0
    
    qualified = [c for c in ranked_candidates if c['final_score'] >= threshold]
    
    logger.info(f"Filtered {len(qualified)} candidates above threshold {threshold}")
    return qualified


def export_rankings_to_dict(ranked_candidates: List[Dict]) -> List[Dict]:
    """
    Convert rankings to exportable format.
    
    Args:
        ranked_candidates (List[Dict]): Ranked candidate list
        
    Returns:
        List[Dict]: Exportable format
    """
    export_data = []
    
    for candidate in ranked_candidates:
        export_data.append({
            'Rank': candidate['rank'],
            'Candidate': candidate['candidate_name'],
            'Final Score': candidate['final_score'],
            'Similarity Score': candidate['similarity_score'],
            'Skill Match %': candidate['skill_match_percentage'],
            'Matched Skills': ', '.join(candidate['matching_skills']) if candidate['matching_skills'] else 'None',
            'Missing Skills': ', '.join(candidate['missing_skills']) if candidate['missing_skills'] else 'None',
            'Total Skills': len(candidate['extracted_skills'])
        })
    
    return export_data
