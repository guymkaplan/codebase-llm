#!/bin/bash
set -e

# Print current environment info
echo "Python: $(which python)"
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Warning: No Conda environment detected. It is recommended to run this in a conda environment."
else
    echo "Conda Env: $CONDA_DEFAULT_ENV"
fi

# Define paths
REPO_DIR="research-llm-exercise/repos"

# Create directories
echo "📂 Creating directories..."
mkdir -p "$REPO_DIR"

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt
pip install -e .

# Clone repositories
echo "⬇️  Cloning repositories..."

# Jsign (Java Authenticode signing)
if [ ! -d "$REPO_DIR/jsign" ]; then
    echo "   Cloning jsign..."
    git clone https://github.com/ebourg/jsign.git "$REPO_DIR/jsign"
else
    echo "   ✅ jsign already exists"
fi

# Signify (Python Authenticode verification)
if [ ! -d "$REPO_DIR/signify" ]; then
    echo "   Cloning signify..."
    git clone https://github.com/ralphje/signify.git "$REPO_DIR/signify"
else
    echo "   ✅ signify already exists"
fi

# Run the app
echo "🚀 Starting Streamlit app..."
streamlit run app.py
