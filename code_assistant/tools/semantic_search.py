"""
Tool for semantic file search using HyDE and embeddings
"""
import os
import tiktoken
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from azure.identity import get_bearer_token_provider, EnvironmentCredential
from openai import AzureOpenAI
from dotenv import load_dotenv
from ..prompts import (
    HYDE_GENERATION_PROMPT,
    KEYWORD_EXTRACTION_PROMPT,
    KEYWORD_EXTRACTION_RETRY_PROMPT
)
from ..config import SEMANTIC_SEARCH_CONFIG, REPOS_BASE_PATH, EMBEDDING_CONFIG
from .keyword_search import search_keyword
from .repo_documentation import get_repo_documentation

load_dotenv(override=True)

# Global variable to store subagent metadata for the last semantic search
_last_semantic_search_metadata = []

# Cache for Nomic embedding model (lazy loading)
_nomic_model = None
_nomic_tokenizer = None


def get_last_semantic_search_metadata() -> List[Dict]:
    """Get metadata from the last semantic search operation"""
    return _last_semantic_search_metadata


def _get_nomic_model():
    """Lazy load Nomic embedding model with automatic device selection"""
    global _nomic_model, _nomic_tokenizer
    
    if _nomic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for Nomic embeddings. "
                "Install it with: pip install sentence-transformers einops"
            )
        
        model_name = EMBEDDING_CONFIG["nomic_model"]
        device = EMBEDDING_CONFIG["nomic_device"]
        
        # Auto-detect best device if set to "auto"
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            print(f"🔍 Auto-detected device: {device}")
        
        print(f"Loading Nomic model: {model_name} on {device}...")
        _nomic_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        print(f"✅ Nomic model loaded successfully on {device.upper()}!")
    
    return _nomic_model


def _get_internal_llm() -> AzureChatOpenAI:
    """Get LLM for internal use (HyDE and keyword extraction)"""
    token_provider = get_bearer_token_provider(
        EnvironmentCredential(), 
        "https://cognitiveservices.azure.com/.default"
    )
    
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL_GPT4o"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_ad_token_provider=token_provider,
        temperature=0.0,
    )


def _get_embedding_client() -> AzureOpenAI:
    """Get Azure OpenAI client for embeddings"""
    token_provider = get_bearer_token_provider(
        EnvironmentCredential(), 
        "https://cognitiveservices.azure.com/.default"
    )
    
    return AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


def _generate_hyde_query(query: str, repo_context: str = "") -> tuple:
    """
    Generate a hypothetical document (HyDE) that would answer the query.
    This helps with semantic search by creating a richer embedding target.
    
    Args:
        query: User's question about the code
        repo_context: Repository documentation for context
    
    Returns:
        tuple: (hyde_answer, metadata with prompt and response)
    """
    llm = _get_internal_llm()
    
    prompt = HYDE_GENERATION_PROMPT.format(query=query, repo_context=repo_context)
    
    response = llm.invoke([("human", prompt)])
    
    metadata = {
        "agent": "HyDE Generator",
        "prompt": prompt,
        "response": response.content
    }
    
    return response.content, metadata


def _extract_keywords(hyde_answer: str, attempt: int = 0) -> tuple:
    """
    Extract technical keywords from HyDE answer.
    Different attempts may produce different keywords for retry logic.
    
    Returns:
        tuple: (keywords, metadata with prompt and response)
    """
    llm = _get_internal_llm()
    
    num_keywords = SEMANTIC_SEARCH_CONFIG["num_keywords"]
    
    if attempt > 0:
        prompt = KEYWORD_EXTRACTION_RETRY_PROMPT.format(
            num_keywords=num_keywords,
            hyde_answer=hyde_answer,
            attempt=attempt + 1
        )
    else:
        prompt = KEYWORD_EXTRACTION_PROMPT.format(
            num_keywords=num_keywords,
            hyde_answer=hyde_answer
        )
    
    response = llm.invoke([("human", prompt)])
    
    # Parse comma-separated keywords
    keywords = [kw.strip() for kw in response.content.split(',')]
    keywords = [kw for kw in keywords if kw]  # Remove empty strings
    keywords = keywords[:num_keywords]
    
    metadata = {
        "agent": f"Keyword Extractor (Attempt {attempt + 1})",
        "prompt": prompt,
        "response": response.content,
        "extracted_keywords": keywords
    }
    
    return keywords, metadata


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _chunk_file_by_tokens(
    file_path: Path, 
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Dict]:
    """
    Chunk a file into segments based on token count.
    Returns list of dicts with: text, file_path, start_line, end_line, tokens
    """
    if chunk_size is None:
        chunk_size = SEMANTIC_SEARCH_CONFIG["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = SEMANTIC_SEARCH_CONFIG["chunk_overlap"]
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return []
    
    lines = content.split('\n')
    encoding = tiktoken.get_encoding("cl100k_base")
    
    chunks = []
    current_chunk_lines = []
    current_chunk_tokens = 0
    start_line = 1
    
    for i, line in enumerate(lines, start=1):
        line_tokens = len(encoding.encode(line + '\n'))
        
        # If adding this line would exceed chunk_size, finalize current chunk
        if current_chunk_tokens + line_tokens > chunk_size and current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append({
                'text': chunk_text,
                'file_path': str(file_path),
                'start_line': start_line,
                'end_line': i - 1,
                'tokens': current_chunk_tokens
            })
            
            # Start new chunk with overlap
            # Calculate how many lines to keep for overlap
            overlap_lines = []
            overlap_tokens = 0
            for line in reversed(current_chunk_lines):
                line_tok = len(encoding.encode(line + '\n'))
                if overlap_tokens + line_tok <= chunk_overlap:
                    overlap_lines.insert(0, line)
                    overlap_tokens += line_tok
                else:
                    break
            
            current_chunk_lines = overlap_lines + [line]
            current_chunk_tokens = overlap_tokens + line_tokens
            start_line = i - len(overlap_lines)
        else:
            current_chunk_lines.append(line)
            current_chunk_tokens += line_tokens
    
    # Add final chunk
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines)
        chunks.append({
            'text': chunk_text,
            'file_path': str(file_path),
            'start_line': start_line,
            'end_line': len(lines),
            'tokens': current_chunk_tokens
        })
    
    return chunks


def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed texts using configured embedding provider (Azure OpenAI or Nomic local).
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        numpy array of embeddings with shape (len(texts), embedding_dim)
    """
    provider = EMBEDDING_CONFIG["provider"]
    
    if provider == "nomic":
        return _embed_texts_nomic(texts)
    elif provider == "azure":
        return _embed_texts_azure(texts)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use 'azure' or 'nomic'")


def _embed_texts_azure(texts: List[str]) -> np.ndarray:
    """Embed texts using Azure OpenAI Ada-2"""
    client = _get_embedding_client()
    model = os.getenv("AZURE_OPENAI_MODEL_ADA2")
    batch_size = EMBEDDING_CONFIG["azure_batch_size"]
    
    embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


def _embed_texts_nomic(texts: List[str]) -> np.ndarray:
    """Embed texts using local Nomic model"""
    model = _get_nomic_model()
    batch_size = EMBEDDING_CONFIG["nomic_batch_size"]
    
    # Prefix for code embedding (Nomic's recommendation)
    prefixed_texts = [f"search_document: {text}" for text in texts]
    
    embeddings = []
    
    # Process in batches
    for i in range(0, len(prefixed_texts), batch_size):
        batch = prefixed_texts[i:i + batch_size]
        
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Nomic recommends normalization
        )
        
        embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    return np.vstack(embeddings) if len(embeddings) > 1 else embeddings[0]


def _calculate_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between query and chunks"""
    # Normalize vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    chunks_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    similarities = np.dot(chunks_norm, query_norm)
    
    return similarities


@tool
def semantic_file_search(
    query: str,
    repo_name: str
) -> str:
    """
    Semantic search for code files based on natural language query.
    Uses HyDE (Hypothetical Document Embeddings) and keyword extraction to find relevant code.
    Returns relevant file paths with line ranges.
    
    All search parameters (chunk_size, top_k_results, max_results_with_lines, etc.) are configured in config.py.
    
    Args:
        query: Natural language question about code (e.g., "How does authentication work?")
        repo_name: Name of the repository to search in
    """
    global _last_semantic_search_metadata
    _last_semantic_search_metadata = []
    
    # Get all parameters from config
    top_k = SEMANTIC_SEARCH_CONFIG["top_k_results"]
    max_results = SEMANTIC_SEARCH_CONFIG["max_results_with_lines"]
    chunk_size = SEMANTIC_SEARCH_CONFIG["chunk_size"]
    repos_base_path = REPOS_BASE_PATH
    
    repo_path = Path(repos_base_path) / repo_name
    
    if not repo_path.exists() or not repo_path.is_dir():
        return f"Error: Repository '{repo_name}' not found at {repo_path}"
    
    try:
        # Step 0: Get repository documentation for context
        repo_doc = get_repo_documentation.invoke({"repo_name": repo_name})
        
        # Truncate documentation if too long (keep first 2000 chars for context)
        repo_context = repo_doc[:2000] + "..." if len(repo_doc) > 2000 else repo_doc
        
        # Step 1: Generate HyDE query with repository context
        hyde_answer, hyde_metadata = _generate_hyde_query(query, repo_context)
        _last_semantic_search_metadata.append(hyde_metadata)
        
        # Step 2: Extract keywords and search (with retry)
        max_retries = SEMANTIC_SEARCH_CONFIG["max_keyword_retries"]
        file_paths = []
        
        for attempt in range(max_retries):
            keywords, keyword_metadata = _extract_keywords(hyde_answer, attempt)
            _last_semantic_search_metadata.append(keyword_metadata)
            
            # Use keyword search tool
            search_result = search_keyword.invoke({
                'keywords': keywords,
                'repo_name': repo_name
            })
            
            # Parse file paths from result
            if "No files found" not in search_result:
                # Extract file paths from the result string
                # Format: "- path/to/file.py (N occurrences)"
                lines = search_result.split('\n')
                file_paths = []
                for line in lines:
                    if line.strip().startswith('-'):
                        # Extract just the path, removing occurrence count
                        path_part = line.strip('- ').strip()
                        # Remove occurrence count if present (everything after last '(')
                        if '(' in path_part:
                            path_part = path_part.rsplit('(', 1)[0].strip()
                        file_paths.append(path_part)
                break
        
        if not file_paths:
            return f"No relevant files found for query: {query}"
        
        # Step 3: Chunk all files
        all_chunks = []
        for fp in file_paths:
            full_path = repo_path / fp
            if full_path.exists():
                chunks = _chunk_file_by_tokens(full_path, chunk_size=chunk_size)
                for chunk in chunks:
                    chunk['relative_path'] = fp
                all_chunks.extend(chunks)
        
        if not all_chunks:
            return f"Found {len(file_paths)} files but could not read their content."
        
        # Step 4: Decision point - embed or return paths?
        if len(all_chunks) <= 5:
            # Simple case: return file paths only
            unique_files = list(set(chunk['relative_path'] for chunk in all_chunks))
            result = f"Found {len(unique_files)} relevant files:\n\n"
            result += "\n".join(f"- {fp}" for fp in unique_files)
            return result
        
        # Step 5: Embed and rank by similarity
        # Embed HyDE query
        query_embedding = _embed_texts([hyde_answer])[0]
        
        # Embed all chunks
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        chunk_embeddings = _embed_texts(chunk_texts)
        
        # Calculate similarities
        similarities = _calculate_similarity(query_embedding, chunk_embeddings)
        
        # Get top_k chunks for ranking, but limit final output to max_results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Group by file and get best chunk per file
        file_best_chunks = {}
        for idx in top_indices:
            chunk = all_chunks[idx]
            file_path = chunk['relative_path']
            
            if file_path not in file_best_chunks:
                file_best_chunks[file_path] = {
                    'chunk': chunk,
                    'similarity': similarities[idx],
                    'index': idx
                }
            # Keep the highest similarity chunk for each file
            elif similarities[idx] > file_best_chunks[file_path]['similarity']:
                file_best_chunks[file_path] = {
                    'chunk': chunk,
                    'similarity': similarities[idx],
                    'index': idx
                }
        
        # Sort files by best similarity and limit to max_results
        sorted_files = sorted(
            file_best_chunks.items(), 
            key=lambda x: x[1]['similarity'], 
            reverse=True
        )[:max_results]
        
        # Format results
        result = f"Found {len(all_chunks)} code sections across {len(file_paths)} files.\n"
        result += f"Returning top {len(sorted_files)} files with most relevant sections:\n\n"
        
        for i, (file_path, data) in enumerate(sorted_files, 1):
            chunk = data['chunk']
            similarity = data['similarity']
            result += f"{i}. {file_path} (lines {chunk['start_line']}-{chunk['end_line']}) [similarity: {similarity:.3f}]\n"
        
        return result
        
    except Exception as e:
        return f"Error during semantic search: {str(e)}"
