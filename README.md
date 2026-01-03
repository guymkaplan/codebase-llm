# Codebase LLM Assistant

A research tool for analyzing code repositories using LLMs.

## Quick Start

1. **Configure Environment**
   Ensure you have the required environment variables set (AZURE_OPENAI_ENDPOINT, etc.) in a `.env` file or your shell.

2. **Run the App**
   The `run.sh` script will install dependencies, download the required repositories, and start the Streamlit app.

   ```bash
   ./run.sh
   ```

## Architecture

- **Frontend**: Streamlit
- **Agent**: LangChain with ReACT pattern
- **Search**: Semantic search using HyDE and Azure OpenAI Embeddings
- **Models**: Azure OpenAI GPT-4o
