"""
Streamlit UI Components

Reusable UI components for the resume evaluator application.
"""

import streamlit as st
from typing import Dict, List, Optional, Union
import io
import numpy as np


def to_float(value: Union[int, float, np.number, None]) -> float:
    """
    Convert value to native Python float.
    
    Args:
        value: Value to convert (can be int, float, numpy number, or None)
        
    Returns:
        Native Python float
    """
    if value is None:
        return 0.0
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def render_jd_upload() -> Optional[str]:
    """Render job description upload component."""
    jd_input_method = st.radio(
        "Input method",
        ["Text Input", "File Upload"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    jd_text = None
    
    if jd_input_method == "Text Input":
        jd_text = st.text_area(
            "Enter job description",
            height=200,
            placeholder="Paste the job description here...",
            help="Enter or paste the complete job description"
        )
    else:
        jd_file = st.file_uploader(
            "Upload job description",
            type=['txt', 'pdf', 'docx'],
            help="Upload a job description file"
        )
        
        if jd_file:
            if jd_file.type == "text/plain":
                jd_text = str(jd_file.read(), "utf-8")
            else:
                st.info("PDF and DOCX parsing for JD coming soon. Please use text input for now.")
    
    return jd_text


def render_resume_upload():
    """Render resume upload component."""
    resume_file = st.file_uploader(
        "Upload resume",
        type=['pdf', 'docx', 'doc', 'txt'],
        help="Upload a resume file (PDF, DOCX, or TXT)"
    )
    
    return resume_file


def render_score_display(scores: Dict):
    """Render overall score display."""
    overall_score = to_float(scores.get('overall_score', 0))
    confidence = to_float(scores.get('confidence', 0))
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric("Overall Match Score", f"{overall_score:.1f}/100")
    
    with col2:
        # Score gauge
        score_color = (
            "🟢" if overall_score >= 80 else
            "🟡" if overall_score >= 60 else
            "🔴"
        )
        st.metric("Rating", score_color)
    
    with col3:
        st.metric("Confidence", f"{confidence*100:.1f}%")
    
    # Progress bar - ensure value is between 0 and 1
    progress_value = max(0.0, min(1.0, overall_score / 100.0))
    st.progress(progress_value)


def render_detailed_breakdown(scores: Dict):
    """Render detailed score breakdown."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        skill_score = to_float(scores.get('skill_match_score', 0))
        st.metric("Skill Match", f"{skill_score:.1f}/100")
        st.progress(max(0.0, min(1.0, skill_score / 100.0)))
    
    with col2:
        exp_score = to_float(scores.get('experience_match_score', 0))
        st.metric("Experience", f"{exp_score:.1f}/100")
        st.progress(max(0.0, min(1.0, exp_score / 100.0)))
    
    with col3:
        role_score = to_float(scores.get('role_responsibility_score', 0))
        st.metric("Role Match", f"{role_score:.1f}/100")
        st.progress(max(0.0, min(1.0, role_score / 100.0)))
    
    with col4:
        edu_score = to_float(scores.get('education_score', 0))
        st.metric("Education", f"{edu_score:.1f}/100")
        st.progress(max(0.0, min(1.0, edu_score / 100.0)))
    
    with col5:
        ats_score = to_float(scores.get('ats_score', 0))
        st.metric("ATS Friendly", f"{ats_score:.1f}/100")
        st.progress(max(0.0, min(1.0, ats_score / 100.0)))
    
    # Detailed information
    st.subheader("📋 Details")
    
    # Skill details
    skill_details = scores.get('skill_details', {})
    if skill_details:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Matched Skills:**")
            matched_skills = skill_details.get('matched_skills', [])
            if matched_skills:
                st.write(", ".join(matched_skills[:10]))
            else:
                st.write("None")
        
        with col2:
            st.write("**Missing Required Skills:**")
            missing_skills = scores.get('missing_skills', [])
            if missing_skills:
                st.write(", ".join(missing_skills[:10]))
            else:
                st.write("None")
    
    # Experience details
    exp_details = scores.get('experience_details', {})
    if exp_details:
        st.write("**Experience:**")
        resume_years = exp_details.get('resume_years', 0)
        required_years = exp_details.get('required_years', 0)
        gap_years = exp_details.get('gap_years', 0)
        
        if gap_years > 0:
            st.warning(f"Resume: {resume_years} years | Required: {required_years} years | Gap: {gap_years} years")
        else:
            st.success(f"Resume: {resume_years} years | Required: {required_years} years | Meets requirement ✓")


def render_recommendations(scores: Dict, resume_data: Dict, jd_data: Dict):
    """Render recommendations and action items."""
    # Summary
    summary = scores.get('summary', '')
    if summary:
        st.subheader("📝 Summary")
        st.write(summary)
    
    # Gap analysis
    gap_analysis = scores.get('gap_analysis', [])
    if gap_analysis:
        st.subheader("🔍 Gap Analysis")
        for gap in gap_analysis:
            st.write(f"• {gap}")
    
    # Action items
    action_items = scores.get('action_items', [])
    if action_items:
        st.subheader("✅ Action Items")
        for i, item in enumerate(action_items, 1):
            st.write(f"{i}. {item}")
    
    # Evidence
    evidence = scores.get('evidence', {})
    if evidence:
        st.subheader("📊 Evidence")
        with st.expander("View evidence", expanded=False):
            st.json(evidence)


def render_batch_results(results: List[Dict]):
    """Render batch processing results."""
    st.subheader("📊 Batch Evaluation Results")
    
    # Results table
    import pandas as pd
    
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        df = pd.DataFrame([
            {
                'Rank': i + 1,
                'Name': r.get('name', 'N/A'),
                'Overall Score': r.get('overall_score', 0),
                'Skill Score': r.get('skill_score', 0),
                'Experience Score': r.get('experience_score', 0),
                'Years Exp.': r.get('years_experience', 0),
                'Matched Skills': r.get('matched_skills', 0),
                'Missing Skills': r.get('missing_skills', 0)
            }
            for i, r in enumerate(valid_results)
        ])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Individual candidate details
        st.subheader("👤 Candidate Details")
        selected_candidate = st.selectbox(
            "Select candidate to view details",
            options=range(len(valid_results)),
            format_func=lambda i: f"{valid_results[i].get('name', 'Unknown')} - Score: {valid_results[i].get('overall_score', 0)}"
        )
        
        if selected_candidate is not None:
            candidate = valid_results[selected_candidate]
            scores = candidate.get('scores', {})
            
            # Display candidate details
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Contact:**")
                st.write(f"Name: {candidate.get('name', 'N/A')}")
                st.write(f"Email: {candidate.get('email', 'N/A')}")
            
            with col2:
                st.write("**Scores:**")
                st.write(f"Overall: {candidate.get('overall_score', 0):.1f}/100")
                st.write(f"Skills: {candidate.get('skill_score', 0):.1f}/100")
                st.write(f"Experience: {candidate.get('experience_score', 0):.1f}/100")
            
            # Detailed breakdown
            with st.expander("View detailed breakdown", expanded=False):
                render_detailed_breakdown(scores)
    
    # Errors
    error_results = [r for r in results if 'error' in r]
    if error_results:
        st.warning(f"⚠️ {len(error_results)} resume(s) had errors during processing:")
        for r in error_results:
            st.write(f"• {r.get('name', 'Unknown')}: {r.get('error', 'Unknown error')}")

