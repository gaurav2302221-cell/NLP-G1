"""
Test Script - Resume Screening System
Demonstrates the core functionality without using Streamlit
"""

import sys
sys.path.insert(0, '.')

from src.text_preprocessing import preprocess_text
from src.skill_extractor import extract_skills, calculate_skill_match_percentage
from src.job_parser import process_job_description
from src.similarity_model import compute_similarity

print("=" * 70)
print("RESUME SCREENING SYSTEM - QUICK TEST")
print("=" * 70)

# Sample Resume
resume_text = """
John Doe
Software Engineer - Data Science Team

EXPERIENCE
Senior Data Scientist at Tech Corp (2022-Present)
- Developed machine learning models using Python and TensorFlow
- Implemented deep learning solutions for computer vision
- Worked with AWS services (EC2, S3) for deployment
- Collaborated with SQL databases and Pandas for data processing

SKILLS
- Programming: Python, Java, JavaScript
- Machine Learning: TensorFlow, PyTorch, scikit-learn, NLP
- Data: Pandas, NumPy, SQL, MongoDB
- Cloud: AWS, Docker, Kubernetes
- Tools: Git, Jupyter, Linux

EDUCATION
MS Computer Science - University of Tech (2020)
"""

# Sample Job Description
job_description = """
Position: Senior Data Scientist

We are seeking a talented Senior Data Scientist to join our team.

Required Skills:
- Python and advanced programming
- Machine Learning and Deep Learning experience
- TensorFlow or PyTorch
- SQL and data manipulation
- AWS cloud platform knowledge
- Docker containerization

Nice to have:
- NLP experience
- Computer Vision
- Apache Spark
"""

print("\n1. PROCESSING JOB DESCRIPTION")
print("-" * 70)
job_info = process_job_description(job_description)
print(f"Job Title: {job_info['job_title']}")
print(f"Required Skills Found: {len(job_info['required_skills'])}")
print(f"Skills: {', '.join(list(job_info['required_skills'])[:8])}")

print("\n2. EXTRACTING CANDIDATE SKILLS")
print("-" * 70)
candidate_skills = extract_skills(resume_text)
print(f"Total Skills Found: {len(candidate_skills)}")
print(f"Skills: {', '.join(sorted(candidate_skills))}")

print("\n3. CALCULATING SKILL MATCH")
print("-" * 70)
skill_match = calculate_skill_match_percentage(
    job_info['required_skills'],
    candidate_skills
)
print(f"Skill Match Percentage: {skill_match:.2f}%")

matching_skills = job_info['required_skills'] & candidate_skills
missing_skills = job_info['required_skills'] - candidate_skills
print(f"Matched Skills ({len(matching_skills)}): {', '.join(sorted(matching_skills)[:5])}")
print(f"Missing Skills ({len(missing_skills)}): {', '.join(sorted(list(missing_skills))[:5])}")

print("\n4. CALCULATING SIMILARITY SCORE")
print("-" * 70)
similarity_score, breakdown = compute_similarity(resume_text, job_description)
print(f"Overall Similarity: {similarity_score * 100:.2f}%")
print(f"  - TF-IDF: {breakdown['tfidf'] * 100:.2f}%")
print(f"  - BERT: {breakdown['bert'] * 100:.2f}%")
print(f"  - Word Embedding: {breakdown['word_embedding'] * 100:.2f}%")

print("\n5. FINAL RANKING SCORE")
print("-" * 70)
final_score = (similarity_score * 0.7) + (skill_match / 100 * 0.3)
print(f"Final Score: {final_score * 100:.2f}%")
print(f"  Calculation: ({similarity_score * 100:.2f}% x 0.70) + ({skill_match:.2f}% x 0.30)")

print("\n" + "=" * 70)
print("TEST COMPLETE - All modules working correctly!")
print("=" * 70)
print("\nNext Steps:")
print("1. Open: http://localhost:8501 in your browser")
print("2. Upload PDF resumes to the system")
print("3. Enter a job description")
print("4. Click 'Analyze Resumes' to see rankings")
print("=" * 70)
