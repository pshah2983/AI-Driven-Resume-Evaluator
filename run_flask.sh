#!/bin/bash

# Run Flask Application
# Make sure you're in the project root directory

echo "Starting AI-Driven Resume Evaluator (Flask)..."
echo "Make sure you have activated your virtual environment!"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Virtual environment not detected."
    echo "Activate it with: source venv/bin/activate"
    echo ""
fi

# Set Flask app
export FLASK_APP=app/flask_app.py
export FLASK_ENV=development

# Use port 5001 by default (5000 is often used by AirPlay on macOS)
PORT=${PORT:-5001}

# Run Flask
python -m flask run --host=0.0.0.0 --port=$PORT

# Alternative: Run directly
# PORT=$PORT python app/flask_app.py

