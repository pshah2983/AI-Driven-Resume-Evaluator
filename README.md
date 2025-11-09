# AI-Driven Resume Evaluator

An intelligent web application that evaluates resumes against job descriptions using semantic similarity, keyword matching, and LLM-powered explainability. Built for recruiters and candidates to improve resume quality and matching accuracy.

## 🎯 Features

- **Intelligent Scoring**: Multi-dimensional scoring (0-100) based on skills, experience, role match, education, and ATS compatibility
- **Semantic Matching**: Uses SentenceTransformers embeddings for deep semantic understanding beyond keyword matching
- **Gap Analysis**: Identifies missing skills, weak areas, and improvement opportunities
- **Actionable Recommendations**: LLM-powered bullet rewrites, interview prep questions, and specific action items
- **Batch Processing**: Evaluate multiple resumes against a single job description and rank candidates
- **Explainable AI**: Every recommendation includes evidence lines from the resume/JD
- **Privacy-First**: GDPR-compliant with opt-in data storage and encryption

## 🏗️ Architecture

```
┌─────────────┐
│  Streamlit  │  Frontend UI
│     App     │
└──────┬──────┘
       │
┌──────▼─────────────────────────────────────┐
│           Core Processing Pipeline          │
├─────────────────────────────────────────────┤
│  1. Parsing (PDF/DOCX → Structured Data)   │
│  2. Normalization (Skills, Dates, Titles)   │
│  3. Embeddings (SentenceTransformers)       │
│  4. Vector Search (FAISS/Chroma)           │
│  5. Scoring Engine (Weighted Rubric)        │
│  6. LLM Explainability (Rewrites & Qs)     │
└─────────────────────────────────────────────┘
```

## 📋 Requirements

- Python 3.9+
- 4GB+ RAM (8GB recommended for batch processing)
- Optional: GPU for faster embeddings (CPU works fine)

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "AI Driven Resume Evaluator"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your API keys if using OpenAI/GPT models
   ```

### Running the Application

**Option 1: Streamlit (Recommended for development)**
```bash
streamlit run app/streamlit_app.py
```

**Option 2: Docker**
```bash
docker build -t resume-evaluator .
docker run -p 8501:8501 resume-evaluator
```

**Option 3: Hugging Face Spaces / Streamlit Cloud**
- Push to repository and connect to Streamlit Cloud
- Or deploy directly to Hugging Face Spaces

## 📁 Project Structure

```
resume-evaluator/
├── app/                    # Streamlit application
│   ├── streamlit_app.py    # Main Streamlit UI
│   ├── components.py       # Reusable UI components
│   └── templates/          # Report templates
├── src/                    # Core processing modules
│   ├── parsing.py          # Resume/JD parsing (PDF, DOCX)
│   ├── normalization.py    # Skill normalization, date parsing
│   ├── embeddings.py       # Embedding generation
│   ├── scoring.py          # Scoring engine
│   ├── faiss_index.py      # Vector search
│   └── llm_explain.py      # LLM-powered explanations
├── data/                   # Sample data
│   ├── sample_resumes/     # Example resumes for testing
│   └── sample_jds/         # Example job descriptions
├── notebooks/              # Analysis notebooks
│   └── eval_analysis.ipynb # Evaluation metrics
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── config.yaml            # Configuration file
└── README.md              # This file
```

## 🔧 Configuration

Edit `config.yaml` to customize:

- **Scoring weights**: Adjust skill/experience/role match weights
- **Model settings**: Choose embedding models, LLM providers
- **Feature flags**: Enable/disable advanced features
- **Privacy settings**: Data retention, encryption options

## 📊 Scoring Rubric

The final score (0-100) combines:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Skill Match** | 40% | Keyword + semantic skill matching |
| **Experience Match** | 25% | Years of experience, level alignment |
| **Role/Responsibility** | 15% | Semantic similarity of responsibilities |
| **Education & Certs** | 10% | Degree/certification requirements |
| **ATS Friendliness** | 10% | Format, keywords, structure |

## 🎨 Usage

### Single Resume Evaluation

1. Upload a job description (text or file)
2. Upload a candidate resume (PDF or DOCX)
3. View the match score and detailed breakdown
4. Review recommendations and rewritten bullets
5. Download the evaluation report (PDF)

### Batch Processing

1. Upload a job description
2. Upload multiple resumes (bulk upload)
3. View ranked candidate list with top-line scores
4. Click any candidate for detailed analysis
5. Export ranked CSV for ATS integration

## 🔒 Privacy & Security

- **No data persistence by default**: Resumes are processed in-memory
- **Opt-in storage**: Users can choose to save evaluations
- **Encryption**: All stored data encrypted at rest
- **GDPR compliance**: Data deletion on request
- **PII masking**: Personal information masked in development logs

## 🧪 Evaluation & Metrics

The system is evaluated on:

- **Correlation with human scores**: Spearman ρ > 0.7 target
- **Ranking accuracy**: Precision@10, NDCG metrics
- **Explainability quality**: Human-rated recommendation usefulness
- **Time savings**: A/B tests with recruiters

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ app/
isort src/ app/
```

### Type Checking
```bash
mypy src/ app/
```

## 📈 Roadmap

- [x] Project structure and core architecture
- [ ] Resume parsing (PDF, DOCX)
- [ ] Skill normalization and enrichment
- [ ] Embedding generation and vector search
- [ ] Scoring engine implementation
- [ ] LLM explainability layer
- [ ] Streamlit UI development
- [ ] Batch processing and ranking
- [ ] Evaluation metrics and testing
- [ ] Deployment and productionization

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SentenceTransformers for embedding models
- FAISS for efficient vector search
- Streamlit for rapid UI development
- O*NET for skill taxonomies

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for better hiring decisions**

