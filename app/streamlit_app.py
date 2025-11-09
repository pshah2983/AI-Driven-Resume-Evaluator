"""
Streamlit Application - AI-Driven Resume Evaluator

Main UI for resume evaluation against job descriptions.
"""

import sys
from pathlib import Path
import tempfile
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import logging
import yaml
from typing import Dict, List, Optional
import pandas as pd

# Import core modules
from src.parsing import ResumeParser, JobDescriptionParser
from src.normalization import ResumeNormalizer, JobDescriptionNormalizer
from src.embeddings import EmbeddingGenerator
from src.scoring import ScoringEngine
from src.llm_explain import LLMExplainer
from components import (
    render_resume_upload,
    render_jd_upload,
    render_score_display,
    render_detailed_breakdown,
    render_recommendations,
    render_batch_results
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI-Driven Resume Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# Initialize session state
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'jd_data' not in st.session_state:
    st.session_state.jd_data = None
if 'scores' not in st.session_state:
    st.session_state.scores = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None


def initialize_components(config: Dict):
    """Initialize core components with configuration."""
    # Embedding generator
    embedding_config = config.get('embeddings', {})
    embedding_generator = EmbeddingGenerator(
        model_name=embedding_config.get('model_name', 'sentence-transformers/all-mpnet-base-v2'),
        device=embedding_config.get('device', 'cpu'),
        batch_size=embedding_config.get('batch_size', 32)
    )
    
    # Scoring engine
    scoring_config = config.get('scoring', {})
    weights = scoring_config.get('weights', {})
    scoring_engine = ScoringEngine(weights=weights, embedding_generator=embedding_generator)
    
    # LLM explainer
    llm_config = config.get('llm', {})
    llm_explainer = LLMExplainer(
        provider=llm_config.get('provider', 'openai'),
        model_name=llm_config.get('model_name', 'gpt-3.5-turbo'),
        api_key=llm_config.get('api_key')
    )
    
    return embedding_generator, scoring_engine, llm_explainer


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location and return path."""
    # Get file extension
    file_ext = Path(uploaded_file.name).suffix
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        # Write uploaded file content to temp file
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    return tmp_path


def process_single_resume(resume_file, jd_text: str, config: Dict):
    """Process a single resume against a job description."""
    tmp_path = None
    try:
        # Initialize components
        embedding_generator, scoring_engine, llm_explainer = initialize_components(config)
        
        # Save uploaded file temporarily
        tmp_path = save_uploaded_file(resume_file)
        
        # Parse resume
        with st.spinner("Parsing resume..."):
            resume_parser = ResumeParser()
            resume_data = resume_parser.parse(tmp_path)
            
            # Normalize resume
            resume_normalizer = ResumeNormalizer()
            resume_data = resume_normalizer.normalize(resume_data)
        
        # Parse JD
        with st.spinner("Parsing job description..."):
            jd_parser = JobDescriptionParser()
            jd_data = jd_parser.parse(jd_text)
            
            # Normalize JD
            jd_normalizer = JobDescriptionNormalizer()
            jd_data = jd_normalizer.normalize(jd_data)
        
        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            resume_embeddings = embedding_generator.encode_resume(resume_data)
            jd_embeddings = embedding_generator.encode_job_description(jd_data)
        
        # Score
        with st.spinner("Computing scores..."):
            scores = scoring_engine.score(
                resume_data, resume_embeddings,
                jd_data, jd_embeddings
            )
        
        # Generate explanations
        with st.spinner("Generating recommendations..."):
            summary = llm_explainer.generate_summary(scores, resume_data, jd_data)
            gap_analysis = llm_explainer.generate_gap_analysis(scores, resume_data, jd_data)
            action_items = llm_explainer.generate_action_items(scores, resume_data, jd_data)
            interview_questions = llm_explainer.generate_interview_questions(
                scores, resume_data, jd_data, num_questions=5
            )
        
        # Store in session state
        st.session_state.resume_data = resume_data
        st.session_state.jd_data = jd_data
        st.session_state.scores = {
            **scores,
            'summary': summary,
            'gap_analysis': gap_analysis,
            'action_items': action_items,
            'interview_questions': interview_questions
        }
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        st.error(f"Error processing resume: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


def process_batch_resumes(resume_files: List, jd_text: str, config: Dict):
    """Process multiple resumes against a job description."""
    tmp_paths = []
    try:
        # Initialize components
        embedding_generator, scoring_engine, llm_explainer = initialize_components(config)
        
        # Parse JD once
        with st.spinner("Parsing job description..."):
            jd_parser = JobDescriptionParser()
            jd_data = jd_parser.parse(jd_text)
            jd_normalizer = JobDescriptionNormalizer()
            jd_data = jd_normalizer.normalize(jd_data)
            jd_embeddings = embedding_generator.encode_job_description(jd_data)
        
        # Process each resume
        results = []
        progress_bar = st.progress(0)
        
        for i, resume_file in enumerate(resume_files):
            tmp_path = None
            try:
                # Save uploaded file temporarily
                tmp_path = save_uploaded_file(resume_file)
                tmp_paths.append(tmp_path)
                
                # Parse resume
                resume_parser = ResumeParser()
                resume_data = resume_parser.parse(tmp_path)
                resume_normalizer = ResumeNormalizer()
                resume_data = resume_normalizer.normalize(resume_data)
                
                # Generate embeddings
                resume_embeddings = embedding_generator.encode_resume(resume_data)
                
                # Score
                scores = scoring_engine.score(
                    resume_data, resume_embeddings,
                    jd_data, jd_embeddings
                )
                
                # Store result
                results.append({
                    'name': resume_data.get('name', f'Resume {i+1}'),
                    'email': resume_data.get('email', 'N/A'),
                    'overall_score': scores.get('overall_score', 0),
                    'skill_score': scores.get('skill_match_score', 0),
                    'experience_score': scores.get('experience_match_score', 0),
                    'years_experience': resume_data.get('total_years_experience', 0),
                    'matched_skills': len(scores.get('evidence', {}).get('matched_skills', [])),
                    'missing_skills': len(scores.get('missing_skills', [])),
                    'resume_data': resume_data,
                    'scores': scores
                })
                
            except Exception as e:
                logger.error(f"Error processing resume {i+1}: {e}")
                results.append({
                    'name': f'Resume {i+1}',
                    'error': str(e)
                })
            finally:
                # Clean up temporary file for this resume
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                        if tmp_path in tmp_paths:
                            tmp_paths.remove(tmp_path)
                    except:
                        pass
            
            progress_bar.progress((i + 1) / len(resume_files))
        
        # Sort by overall score
        results.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        
        st.session_state.batch_results = results
        st.session_state.jd_data = jd_data
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        st.error(f"Error processing batch: {str(e)}")
        return False
    finally:
        # Clean up any remaining temporary files
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass


def main():
    """Main application function."""
    # Load configuration
    config = load_config()
    
    # Title and description
    st.title("📄 AI-Driven Resume Evaluator")
    st.markdown("""
    Evaluate resumes against job descriptions using AI-powered semantic matching.
    Get detailed scores, gap analysis, and actionable recommendations.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Evaluation Mode",
            ["Single Resume", "Batch Processing"],
            help="Choose single resume evaluation or batch processing for multiple resumes"
        )
        
        # Feature toggles
        st.subheader("Features")
        show_recommendations = st.checkbox("Show Recommendations", value=True)
        show_interview_prep = st.checkbox("Show Interview Prep", value=True)
        show_bullet_rewrites = st.checkbox("Show Bullet Rewrites", value=False)
        
        # Privacy settings
        st.subheader("Privacy")
        persist_data = st.checkbox(
            "Save evaluation data",
            value=config.get('privacy', {}).get('persist_data', False),
            help="If enabled, evaluation data will be saved (with encryption)"
        )
    
    # Main content area
    if mode == "Single Resume":
        # Single resume evaluation
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Job Description")
            jd_text = render_jd_upload()
        
        with col2:
            st.subheader("📄 Resume")
            resume_file = render_resume_upload()
        
        # Process button
        if st.button("🚀 Evaluate Resume", type="primary", use_container_width=True):
            if jd_text and resume_file:
                with st.spinner("Processing..."):
                    success = process_single_resume(resume_file, jd_text, config)
                    if success:
                        st.success("Evaluation complete!")
                        st.rerun()
            else:
                st.warning("Please upload both a job description and a resume.")
        
        # Display results
        if st.session_state.scores:
            st.divider()
            
            # Overall score
            render_score_display(st.session_state.scores)
            
            # Detailed breakdown
            with st.expander("📊 Detailed Breakdown", expanded=True):
                render_detailed_breakdown(st.session_state.scores)
            
            # Recommendations
            if show_recommendations:
                with st.expander("💡 Recommendations", expanded=True):
                    render_recommendations(
                        st.session_state.scores,
                        st.session_state.resume_data,
                        st.session_state.jd_data
                    )
            
            # Interview prep
            if show_interview_prep and st.session_state.scores.get('interview_questions'):
                with st.expander("❓ Interview Preparation", expanded=False):
                    st.write("**Likely Interview Questions:**")
                    for i, question in enumerate(st.session_state.scores['interview_questions'], 1):
                        st.write(f"{i}. {question}")
            
            # Download report
            st.download_button(
                label="📥 Download Report (PDF)",
                data="Report generation coming soon...",
                file_name="resume_evaluation_report.pdf",
                mime="application/pdf"
            )
    
    else:
        # Batch processing
        st.subheader("📋 Job Description")
        jd_text = render_jd_upload()
        
        st.subheader("📄 Resumes (Batch Upload)")
        resume_files = st.file_uploader(
            "Upload multiple resumes",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple resume files for batch evaluation"
        )
        
        # Process button
        if st.button("🚀 Evaluate All Resumes", type="primary", use_container_width=True):
            if jd_text and resume_files:
                with st.spinner(f"Processing {len(resume_files)} resumes..."):
                    success = process_batch_resumes(resume_files, jd_text, config)
                    if success:
                        st.success(f"Processed {len(resume_files)} resumes!")
                        st.rerun()
            else:
                st.warning("Please upload a job description and at least one resume.")
        
        # Display batch results
        if st.session_state.batch_results:
            st.divider()
            render_batch_results(st.session_state.batch_results)
            
            # Download CSV
            df = pd.DataFrame([
                {
                    'Name': r.get('name', 'N/A'),
                    'Email': r.get('email', 'N/A'),
                    'Overall Score': r.get('overall_score', 0),
                    'Skill Score': r.get('skill_score', 0),
                    'Experience Score': r.get('experience_score', 0),
                    'Years Experience': r.get('years_experience', 0),
                    'Matched Skills': r.get('matched_skills', 0),
                    'Missing Skills': r.get('missing_skills', 0)
                }
                for r in st.session_state.batch_results
                if 'error' not in r
            ])
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="batch_evaluation_results.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()

