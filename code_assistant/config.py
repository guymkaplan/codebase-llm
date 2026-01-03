"""
Configuration for the research LLM exercise tools
"""

# Semantic Search Configuration
SEMANTIC_SEARCH_CONFIG = {
    # Chunking configuration
    "chunk_size": 800,  # Number of tokens per chunk
    "chunk_overlap": 100,  # Number of tokens to overlap between chunks
    
    # Search configuration
    "max_files_to_search": 20,  # Maximum number of files to process
    "top_k_results": 10,  # Number of top results to return to orchestrator agent
    "max_results_with_lines": 10,  # Maximum file paths with line numbers to return
    
    # Keyword extraction
    "num_keywords": 5,  # Number of keywords to extract (increased for broader search)
    "max_keyword_retries": 3,  # Maximum retries if no results found
    
    # File types to search
    "searchable_extensions": [
        '.py', '.java', '.js', '.ts', '.tsx', '.jsx',
        '.go', '.rs', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.rb', '.php', '.swift', '.kt', '.scala',
        '.sh', '.bash', '.sql', '.yaml', '.yml', '.json',
        '.xml', '.html', '.css', '.md', '.txt', '.rst'
    ],
}

# Repository paths
REPOS_BASE_PATH = "./research-llm-exercise/repos"

# Embeddings configuration
EMBEDDING_CONFIG = {
    # Embedding provider: "azure" or "nomic"
    "provider": "azure",  # Options: "azure", "nomic"
    
    # Azure OpenAI settings (used when provider="azure")
    "azure_batch_size": 100,
    
    # Nomic settings (used when provider="nomic")
    "nomic_model": "nomic-ai/nomic-embed-text-v1.5",  # HuggingFace model name
    "nomic_device": "auto",  # Options: "auto", "cpu", "cuda", "mps" - auto detects best device
    "nomic_batch_size": 32,  # Smaller batches for local inference
    "nomic_max_length": 8192,  # Nomic supports up to 8192 tokens
}

# Code reading configuration
CODE_READING_CONFIG = {
    "max_lines": 200,  # Maximum number of lines to return in read_code tool
}
