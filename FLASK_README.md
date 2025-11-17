# Flask Application - AI-Driven Resume Evaluator

Production-ready Flask web application for resume evaluation against job descriptions.

## 🚀 Quick Start

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (optional)
   ```bash
   export SECRET_KEY="your-secret-key-here"
   export OPENAI_API_KEY="your-openai-key"  # If using OpenAI for LLM
   ```

3. **Run the Flask application**
   ```bash
   python app/flask_app.py
   ```

   Or using Flask CLI:
   ```bash
   export FLASK_APP=app/flask_app.py
   flask run --port=5001
   ```

   **Note**: The app uses port 5001 by default (instead of 5000) because macOS AirPlay Receiver often uses port 5000. You can change it by setting the `PORT` environment variable:
   ```bash
   PORT=8080 python app/flask_app.py
   ```

4. **Access the application**
   - Open your browser and navigate to: `http://localhost:5001`

## 📋 Features

### Interface Design Options

The Flask application provides a **modern dashboard-style interface** with:

1. **Clean, Responsive Design**
   - Mobile-friendly layout
   - Modern card-based UI
   - Smooth animations and transitions

2. **Single Page Application (SPA) Features**
   - AJAX-based async processing
   - No page reloads
   - Real-time progress updates
   - Interactive visualizations

3. **Two Processing Modes**
   - **Single Resume**: Evaluate one resume against a job description
   - **Batch Processing**: Evaluate multiple resumes and rank candidates

4. **Rich Results Display**
   - Overall match score with visual gauge
   - Detailed breakdown by category (Skills, Experience, Role, Education, ATS)
   - Matched and missing skills visualization
   - AI-powered recommendations and gap analysis
   - Interview preparation questions
   - Export functionality (CSV for batch results)

## 🎨 Interface Design Highlights

### Modern Dashboard Style
- **Card-based layout** for easy information scanning
- **Color-coded score indicators** (Green/Yellow/Red)
- **Progress bars** for visual score representation
- **Skill tags** with color coding (matched vs missing)
- **Responsive grid layouts** that adapt to screen size

### User Experience Features
- **Tab-based navigation** between single and batch modes
- **Drag-and-drop file upload** (visual feedback)
- **Loading overlays** with progress indicators
- **Smooth scrolling** to results
- **Error handling** with user-friendly messages

## 🔧 Configuration

The application uses the same `config.yaml` file as the Streamlit version. Key settings:

- **Scoring weights**: Adjust in `config.yaml` under `scoring.weights`
- **Embedding model**: Configure in `config.yaml` under `embeddings`
- **LLM provider**: Set in `config.yaml` under `llm`

## 📡 API Endpoints

### POST `/evaluate`
Evaluate a single resume against a job description.

**Request:**
- `jd_text` (form data): Job description text
- `resume_file` (file): Resume file (PDF, DOCX, DOC, TXT)

**Response:**
```json
{
  "success": true,
  "scores": {
    "overall_score": 85.5,
    "skill_match_score": 90.0,
    ...
  },
  "resume_data": {...},
  "jd_data": {...}
}
```

### POST `/batch`
Evaluate multiple resumes against a job description.

**Request:**
- `jd_text` (form data): Job description text
- `resume_files` (files): Multiple resume files

**Response:**
```json
{
  "success": true,
  "results": [...],
  "errors": [...],
  "total": 10,
  "successful": 9
}
```

### GET `/health`
Health check endpoint.

## 🏗️ Project Structure

```
app/
├── flask_app.py          # Main Flask application
├── templates/
│   └── index.html        # Main HTML template
└── static/
    ├── css/
    │   └── style.css     # Modern CSS styling
    └── js/
        └── main.js       # Frontend JavaScript
```

## 🚢 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app.flask_app:app
```

### Using Docker

Create a `Dockerfile.flask`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app.flask_app:app"]
```

Build and run:
```bash
docker build -f Dockerfile.flask -t resume-evaluator-flask .
docker run -p 5000:5000 resume-evaluator-flask
```

### Environment Variables for Production

```bash
export SECRET_KEY="your-production-secret-key"
export FLASK_ENV=production
export FLASK_DEBUG=False
```

## 🔒 Security Considerations

1. **Secret Key**: Change the default secret key in production
2. **File Upload Limits**: Currently set to 16MB (configurable in `flask_app.py`)
3. **CORS**: Add CORS headers if needed for API access
4. **Rate Limiting**: Consider adding rate limiting for production
5. **HTTPS**: Use HTTPS in production (via reverse proxy like Nginx)

## 🎯 Alternative Interface Designs

If you want to customize the interface, here are some ideas:

### 1. Multi-Step Wizard
- Step 1: Upload Job Description
- Step 2: Upload Resume(s)
- Step 3: Review Results
- Step 4: Download Report

### 2. Dashboard with Charts
- Add Chart.js or D3.js for visualizations
- Skill matching radar chart
- Score distribution histograms
- Timeline view of experience

### 3. Admin Panel
- User authentication
- History of evaluations
- Saved job descriptions
- Candidate database

### 4. RESTful API Only
- Separate frontend (React/Vue)
- Pure API backend
- JWT authentication
- WebSocket for real-time updates

## 📝 Notes

- The Flask app reuses all existing backend modules (`src/`)
- No changes needed to core processing logic
- Same configuration file (`config.yaml`) works for both Streamlit and Flask
- Temporary files are automatically cleaned up after processing

## 🤝 Comparison: Flask vs Streamlit

| Feature | Flask | Streamlit |
|---------|-------|-----------|
| **Production Ready** | ✅ Yes | ⚠️ Limited |
| **Customization** | ✅ Full control | ⚠️ Limited |
| **Performance** | ✅ Better | ⚠️ Slower |
| **Deployment** | ✅ Standard WSGI | ⚠️ Specialized |
| **Development Speed** | ⚠️ Slower | ✅ Faster |
| **UI Flexibility** | ✅ Complete | ⚠️ Constrained |

For production deployment, Flask is the recommended choice.

