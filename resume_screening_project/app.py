"""
Resume Screening System - Streamlit Web Application

Interactive dashboard for automated resume screening and candidate ranking
based on job description matching using advanced NLP techniques.

Author: AI/ML Engineering Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import logging
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from src.resume_parser import parse_resume, get_resume_filename
from src.text_preprocessing import preprocess_text
from src.skill_extractor import extract_skills, get_skill_summary
from src.job_parser import process_job_description
from src.similarity_model import compute_similarity, initialize_bert_model
from src.ranking_engine import (
    rank_candidates, 
    get_top_candidates, 
    get_candidate_summary,
    filter_candidates_by_threshold,
    export_rankings_to_dict
)
from src.evaluation import evaluate_model

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="document",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling - No emojis in CSS
st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
        }
        
        .main {
            padding: 2rem;
            background-color: #f8f9fa;
        }
        
        .header-container {
            background: linear-gradient(135deg, #1f77b4 0%, #2b5db8 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header-container h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }
        
        .header-container p {
            font-size: 1.1rem;
            opacity: 0.95;
        }
        
        .input-section {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }
        
        .section-title {
            color: #1f77b4;
            font-size: 1.3rem;
            font-weight: bold;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #1f77b4;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #1f77b4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .top-candidate {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            padding: 2rem;
            border-radius: 8px;
            border-left: 5px solid #28a745;
            margin: 1rem 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .top-candidate h3 {
            color: #155724;
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }
        
        .top-candidate p {
            margin: 0.5rem 0;
            color: #155724;
        }
        
        .results-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }
        
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #e0e0e0;
        }
        
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            border-left: 4px solid #28a745;
        }
        
        .info-message {
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            border-left: 4px solid #17a2b8;
        }
        
        .chart-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }
        
        .export-button {
            background-color: #1f77b4;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-weight: bold;
            margin-right: 0.5rem;
        }
        
        .stButton > button {
            width: 100%;
            padding: 0.75rem;
            font-size: 1rem;
            font-weight: bold;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'resumes_data' not in st.session_state:
    st.session_state.resumes_data = {}
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'ranked_results' not in st.session_state:
    st.session_state.ranked_results = None
if 'job_skills' not in st.session_state:
    st.session_state.job_skills = set()
if 'models_initialized' not in st.session_state:
    st.session_state.models_initialized = False


def initialize_models():
    """Initialize NLP models on startup."""
    try:
        if not st.session_state.models_initialized:
            with st.spinner("Loading NLP models (this may take 10-15 seconds on first run)..."):
                initialize_bert_model()
                st.session_state.models_initialized = True
            st.success("Models ready! You can now analyze resumes.")
    except Exception as e:
        st.error(f"Error loading models: {str(e)[:100]}")
        logger.error(f"Model initialization error: {e}")


def process_uploaded_resumes(uploaded_files):
    """
    Process uploaded resume files.
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        Dict: Dictionary with candidate data
    """
    try:
        resumes_data = {}
        progress_container = st.container()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                progress_container.info(f"Processing file {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Save temporary file
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Parse resume
                resume_text = parse_resume(temp_path)
                
                # Preprocess text
                processed_text = preprocess_text(resume_text)
                
                # Extract skills
                skills = extract_skills(resume_text)
                
                # Get candidate name from filename
                candidate_name = get_resume_filename(temp_path).replace('temp_', '')
                
                resumes_data[candidate_name] = {
                    'original_text': resume_text,
                    'processed_text': processed_text,
                    'skills': skills,
                    'file_name': uploaded_file.name
                }
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
            except Exception as e:
                st.warning(f"Could not process {uploaded_file.name}: {str(e)[:80]}")
                logger.error(f"Error processing file {uploaded_file.name}: {e}")
                continue
        
        return resumes_data
        
    except Exception as e:
        st.error(f"Resume processing error: {str(e)[:100]}")
        logger.error(f"Resume processing error: {e}")
        return {}


def display_ranking_results(ranked_candidates):
    """
    Display comprehensive ranking results.
    
    Args:
        ranked_candidates: List of ranked candidate dictionaries
    """
    if not ranked_candidates:
        st.warning("No candidates to display")
        return
    
    with st.container():
        st.markdown('<div class="section-title">Candidate Rankings</div>', unsafe_allow_html=True)
        
        # Create results DataFrame
        results_df = pd.DataFrame([{
            'Rank': c['rank'],
            'Candidate': c['candidate_name'],
            'Final Score': f"{c['final_score']:.2f}%",
            'Similarity': f"{c['similarity_score']:.2f}%",
            'Skill Match': f"{c['skill_match_percentage']:.2f}%",
            'Matched': len(c['matching_skills']),
            'Missing': len(c['missing_skills'])
        } for c in ranked_candidates])
        
        # Display table
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 35 * len(results_df) + 50)
        )
    
    # Top candidate highlight
    if ranked_candidates:
        top_candidate = ranked_candidates[0]
        st.markdown(
            f"""
            <div class="top-candidate">
                <h3>Top Candidate - {top_candidate['candidate_name']}</h3>
                <p><strong>Final Score:</strong> {top_candidate['final_score']:.2f}%</p>
                <p><strong>Similarity:</strong> {top_candidate['similarity_score']:.2f}%</p>
                <p><strong>Skill Match:</strong> {top_candidate['skill_match_percentage']:.2f}%</p>
                <p><strong>Matched Skills:</strong> {', '.join(sorted(top_candidate['matching_skills'])[:5]) if top_candidate['matching_skills'] else 'None'}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )


def display_metrics(ranked_candidates):
    """
    Display summary metrics.
    
    Args:
        ranked_candidates: List of ranked candidate dictionaries
    """
    if not ranked_candidates:
        return
    
    with st.container():
        st.markdown('<div class="section-title">Summary Metrics</div>', unsafe_allow_html=True)
        
        # Calculate metrics
        scores = [c['final_score'] for c in ranked_candidates]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Candidates",
                len(ranked_candidates)
            )
        
        with col2:
            st.metric(
                "Average Score",
                f"{np.mean(scores):.2f}%"
            )
        
        with col3:
            st.metric(
                "Highest Score",
                f"{max(scores):.2f}%"
            )
        
        with col4:
            st.metric(
                "Lowest Score",
                f"{min(scores):.2f}%"
            )


def display_visualizations(ranked_candidates):
    """
    Display visualization charts.
    
    Args:
        ranked_candidates: List of ranked candidate dictionaries
    """
    if not ranked_candidates:
        return
    
    with st.container():
        st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Final Scores Chart
        with col1:
            st.markdown("##### Final Scores")
            
            chart_data = pd.DataFrame({
                'Candidate': [c['candidate_name'][:20] for c in ranked_candidates],
                'Score': [c['final_score'] for c in ranked_candidates]
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(chart_data)))
            ax.barh(chart_data['Candidate'], chart_data['Score'], color=colors)
            ax.set_xlabel('Score (%)', fontsize=10)
            ax.set_xlim(0, 100)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(chart_data['Score']):
                ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        # Skill Match vs Similarity Chart
        with col2:
            st.markdown("##### Similarity vs Skill Match")
            
            chart_data = pd.DataFrame({
                'Candidate': [c['candidate_name'][:20] for c in ranked_candidates],
                'Similarity': [c['similarity_score'] for c in ranked_candidates],
                'Skill Match': [c['skill_match_percentage'] for c in ranked_candidates]
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(chart_data))
            width = 0.35
            
            ax.bar(x - width/2, chart_data['Similarity'], width, label='Similarity', color='steelblue', alpha=0.8)
            ax.bar(x + width/2, chart_data['Skill Match'], width, label='Skill Match', color='orange', alpha=0.8)
            
            ax.set_ylabel('Score (%)', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(chart_data['Candidate'], rotation=45, ha='right', fontsize=9)
            ax.legend()
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)


def display_skill_analysis(ranked_candidates, job_skills):
    """
    Display detailed skill analysis.
    
    Args:
        ranked_candidates: List of ranked candidate dictionaries
        job_skills: Set of required job skills
    """
    if not ranked_candidates:
        return
    
    with st.container():
        st.markdown('<div class="section-title">Skill Analysis</div>', unsafe_allow_html=True)
        
        # Skill summary for top candidates
        for candidate in ranked_candidates[:3]:
            with st.expander(f"{candidate['candidate_name']} - Skills"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Matching:**")
                    if candidate['matching_skills']:
                        for skill in sorted(candidate['matching_skills']):
                            st.write(f"- {skill}")
                    else:
                        st.write("None")
                
                with col2:
                    st.write("**Missing:**")
                    if candidate['missing_skills']:
                        for skill in sorted(list(candidate['missing_skills'])[:10]):
                            st.write(f"- {skill}")
                    else:
                        st.write("All present!")
                
                with col3:
                    st.write("**Summary:**")
                    st.write(f"Matched: {len(candidate['matching_skills'])} / {len(job_skills)}")
                    st.write(f"Rate: {candidate['skill_match_percentage']:.1f}%")
                    st.write(f"Total: {len(candidate['extracted_skills'])}")


def display_job_analysis(job_info):
    """
    Display job description analysis.
    
    Args:
        job_info: Dictionary with job processing results
    """
    with st.container():
        st.markdown('<div class="section-title">Job Description Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Required Skills",
                len(job_info['required_skills'])
            )
            
            if job_info['required_skills']:
                st.write("**Top 10 Skills:**")
                for skill in list(job_info['required_skills'])[:10]:
                    st.write(f"• {skill}")
        
        with col2:
            st.write("**Job Title:**")
            st.info(job_info['job_title'])
            
            st.write("**Keywords:**")
            keywords = list(job_info['keywords'])[:10]
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.write("No keywords identified")


def export_results(ranked_candidates):
    """
    Export ranking results to Excel/CSV.
    
    Args:
        ranked_candidates: List of ranked candidate dictionaries
        
    Returns:
        BytesIO: Exportable file object
    """
    try:
        # Convert to DataFrame
        export_data = export_rankings_to_dict(ranked_candidates)
        df = pd.DataFrame(export_data)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Rankings')
        
        output.seek(0)
        return output
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return None


def main():
    """Main application function."""
    
    # Header
    st.markdown(
        """
        <div class="header-container">
            <h1>Resume Screening System</h1>
            <p>Automated Candidate Ranking using Advanced NLP</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Configuration")
        
        # Initialize models
        if st.button("Initialize Models"):
            initialize_models()
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        if st.session_state.resumes_data:
            st.info(f"Resumes Loaded: {len(st.session_state.resumes_data)}")
        if st.session_state.ranked_results:
            st.info(f"Candidates Ranked: {len(st.session_state.ranked_results)}")
    
    # Main content
    st.markdown('<div class="section-title">Input Data</div>', unsafe_allow_html=True)
    
    input_col1, input_col2 = st.columns([1, 1], gap="medium")
    
    # Resume upload
    with input_col1:
        st.markdown("**Upload Resumes**")
        uploaded_files = st.file_uploader(
            "Select PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("Process Resumes", use_container_width=True):
                with st.spinner("Processing resumes..."):
                    st.session_state.resumes_data = process_uploaded_resumes(uploaded_files)
                    if st.session_state.resumes_data:
                        st.success(f"Processed {len(st.session_state.resumes_data)} resume(s)")
        
        # Display loaded resumes
        if st.session_state.resumes_data:
            with st.expander("View Loaded Resumes"):
                for idx, candidate_name in enumerate(st.session_state.resumes_data.keys(), 1):
                    st.write(f"{idx}. {candidate_name}")
    
    # Job description input
    with input_col2:
        st.markdown("**Job Description**")
        job_text = st.text_area(
            "Enter the job description",
            height=250,
            placeholder="Paste job description here...",
            label_visibility="collapsed"
        )
        st.session_state.job_description = job_text
    
    st.markdown("---")
    
    # Analysis section
    if st.session_state.resumes_data and st.session_state.job_description:
        if st.button("Analyze Resumes", use_container_width=True, key="analyze_btn"):
            try:
                with st.spinner("Analyzing resumes (this may take 30-60 seconds on first run)..."):
                    # Process job description
                    job_info = process_job_description(st.session_state.job_description)
                    st.session_state.job_skills = job_info['required_skills']
                    
                    # Prepare candidate data for ranking
                    candidate_dict = {
                        name: (data['original_text'], data['skills'])
                        for name, data in st.session_state.resumes_data.items()
                    }
                    
                    # Rank candidates
                    ranked_candidates = rank_candidates(
                        candidate_dict,
                        st.session_state.job_description,
                        st.session_state.job_skills,
                        similarity_weight=0.7,
                        skill_weight=0.3
                    )
                    
                    st.session_state.ranked_results = ranked_candidates
                
                st.success("Analysis complete!")
                
                # Display results
                st.divider()
                
                # Job analysis
                display_job_analysis(job_info)
                
                st.divider()
                
                # Metrics
                display_metrics(st.session_state.ranked_results)
                
                st.divider()
                
                # Rankings
                display_ranking_results(st.session_state.ranked_results)
                
                st.divider()
                
                # Visualizations
                display_visualizations(st.session_state.ranked_results)
                
                st.divider()
                
                # Skill analysis
                display_skill_analysis(
                    st.session_state.ranked_results,
                    st.session_state.job_skills
                )
                
                st.divider()
                
                # Export section
                st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1], gap="medium")
                
                with col1:
                    export_file = export_results(st.session_state.ranked_results)
                    if export_file:
                        st.download_button(
                            label="Download Results (Excel)",
                            data=export_file,
                            file_name=f"ranking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                
                with col2:
                    csv_data = pd.DataFrame(
                        export_rankings_to_dict(st.session_state.ranked_results)
                    ).to_csv(index=False)
                    
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv_data,
                        file_name=f"ranking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
            except Exception as e:
                error_msg = str(e)[:150]
                st.error(f"Error during analysis: {error_msg}")
                logger.error(f"Analysis error: {e}")
    
    else:
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            if not st.session_state.resumes_data:
                st.info("Please upload resume files (PDF) using the uploader above")
        with col2:
            if not st.session_state.job_description:
                st.info("Please enter a job description in the text area above")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div class="footer">
            <p>Resume Screening System v1.0.0</p>
            <p>Built with Streamlit, spaCy, BERT, and scikit-learn</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
