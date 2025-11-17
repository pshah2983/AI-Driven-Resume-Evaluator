"""
Flask Application - AI-Driven Resume Evaluator

Production-ready Flask web application for resume evaluation.
"""

import sys
from pathlib import Path
import tempfile
import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify, session, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import yaml

# Import core modules
from src.parsing import ResumeParser, JobDescriptionParser
from src.normalization import ResumeNormalizer, JobDescriptionNormalizer
from src.embeddings import EmbeddingGenerator
from src.scoring import ScoringEngine
from src.llm_explain import LLMExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
# Set template and static folders explicitly
template_dir = Path(__file__).parent / 'templates'
static_dir = Path(__file__).parent / 'static'
app = Flask(__name__, 
            template_folder=str(template_dir),
            static_folder=str(static_dir))
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc', 'txt'}

# Load configuration
def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

# Global config
config = load_config()

# Initialize components (lazy loading)
_embedding_generator = None
_scoring_engine = None
_llm_explainer = None

def get_components():
    """Get or initialize core components."""
    global _embedding_generator, _scoring_engine, _llm_explainer
    
    if _embedding_generator is None:
        embedding_config = config.get('embeddings', {})
        _embedding_generator = EmbeddingGenerator(
            model_name=embedding_config.get('model_name', 'sentence-transformers/all-mpnet-base-v2'),
            device=embedding_config.get('device', 'cpu'),
            batch_size=embedding_config.get('batch_size', 32)
        )
    
    if _scoring_engine is None:
        scoring_config = config.get('scoring', {})
        weights = scoring_config.get('weights', {})
        _scoring_engine = ScoringEngine(weights=weights, embedding_generator=_embedding_generator)
    
    if _llm_explainer is None:
        llm_config = config.get('llm', {})
        _llm_explainer = LLMExplainer(
            provider=llm_config.get('provider', 'openai'),
            model_name=llm_config.get('model_name', 'gpt-3.5-turbo'),
            api_key=llm_config.get('api_key')
        )
    
    return _embedding_generator, _scoring_engine, _llm_explainer

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file) -> str:
    """Save uploaded file to temporary location and return path."""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_ext = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            file.save(tmp_file.name)
            return tmp_file.name
    raise ValueError("Invalid file type")

# Routes
@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate a single resume against a job description."""
    try:
        # Get form data
        jd_text = request.form.get('jd_text', '').strip()
        resume_file = request.files.get('resume_file')
        
        if not jd_text:
            return jsonify({'error': 'Job description is required'}), 400
        
        if not resume_file or resume_file.filename == '':
            return jsonify({'error': 'Resume file is required'}), 400
        
        # Initialize components
        embedding_generator, scoring_engine, llm_explainer = get_components()
        
        # Save uploaded file temporarily
        tmp_path = None
        try:
            tmp_path = save_uploaded_file(resume_file)
            
            # Parse resume
            resume_parser = ResumeParser()
            resume_data = resume_parser.parse(tmp_path)
            
            # Normalize resume
            resume_normalizer = ResumeNormalizer()
            resume_data = resume_normalizer.normalize(resume_data)
            
            # Parse JD
            jd_parser = JobDescriptionParser()
            jd_data = jd_parser.parse(jd_text)
            
            # Normalize JD
            jd_normalizer = JobDescriptionNormalizer()
            jd_data = jd_normalizer.normalize(jd_data)
            
            # Generate embeddings
            resume_embeddings = embedding_generator.encode_resume(resume_data)
            jd_embeddings = embedding_generator.encode_job_description(jd_data)
            
            # Score
            scores = scoring_engine.score(
                resume_data, resume_embeddings,
                jd_data, jd_embeddings
            )
            
            # Generate explanations
            summary = llm_explainer.generate_summary(scores, resume_data, jd_data)
            gap_analysis = llm_explainer.generate_gap_analysis(scores, resume_data, jd_data)
            action_items = llm_explainer.generate_action_items(scores, resume_data, jd_data)
            interview_questions = llm_explainer.generate_interview_questions(
                scores, resume_data, jd_data, num_questions=5
            )
            
            # Prepare response
            result = {
                'success': True,
                'scores': {
                    **scores,
                    'summary': summary,
                    'gap_analysis': gap_analysis,
                    'action_items': action_items,
                    'interview_questions': interview_questions
                },
                'resume_data': {
                    'name': resume_data.get('name'),
                    'email': resume_data.get('email'),
                    'phone': resume_data.get('phone'),
                    'skills': resume_data.get('skills', [])[:20],  # Limit for response
                    'total_years_experience': resume_data.get('total_years_experience', 0)
                },
                'jd_data': {
                    'title': jd_data.get('title'),
                    'company': jd_data.get('company'),
                    'required_skills': jd_data.get('required_skills', [])[:20]
                }
            }
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Error evaluating resume: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_evaluate():
    """Evaluate multiple resumes against a job description."""
    try:
        # Get form data
        jd_text = request.form.get('jd_text', '').strip()
        resume_files = request.files.getlist('resume_files')
        
        if not jd_text:
            return jsonify({'error': 'Job description is required'}), 400
        
        if not resume_files or len(resume_files) == 0:
            return jsonify({'error': 'At least one resume file is required'}), 400
        
        # Initialize components
        embedding_generator, scoring_engine, llm_explainer = get_components()
        
        # Parse JD once
        jd_parser = JobDescriptionParser()
        jd_data = jd_parser.parse(jd_text)
        jd_normalizer = JobDescriptionNormalizer()
        jd_data = jd_normalizer.normalize(jd_data)
        jd_embeddings = embedding_generator.encode_job_description(jd_data)
        
        # Process each resume
        results = []
        tmp_paths = []
        
        for i, resume_file in enumerate(resume_files):
            tmp_path = None
            try:
                if not resume_file or resume_file.filename == '':
                    continue
                
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
                    'overall_score': float(scores.get('overall_score', 0)),
                    'skill_score': float(scores.get('skill_match_score', 0)),
                    'experience_score': float(scores.get('experience_match_score', 0)),
                    'years_experience': float(resume_data.get('total_years_experience', 0)),
                    'matched_skills': len(scores.get('evidence', {}).get('matched_skills', [])),
                    'missing_skills': len(scores.get('missing_skills', [])),
                    'scores': {k: float(v) if isinstance(v, (int, float)) else v 
                              for k, v in scores.items() if k not in ['evidence', 'skill_details']}
                })
                
            except Exception as e:
                logger.error(f"Error processing resume {i+1}: {e}")
                results.append({
                    'name': f'Resume {i+1}',
                    'error': str(e)
                })
            finally:
                # Clean up temporary file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                        if tmp_path in tmp_paths:
                            tmp_paths.remove(tmp_path)
                    except:
                        pass
        
        # Sort by overall score
        valid_results = [r for r in results if 'error' not in r]
        valid_results.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        error_results = [r for r in results if 'error' in r]
        
        return jsonify({
            'success': True,
            'results': valid_results,
            'errors': error_results,
            'total': len(results),
            'successful': len(valid_results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch evaluation: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up any remaining temporary files
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Use port 5001 by default (5000 is often used by AirPlay on macOS)
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)

