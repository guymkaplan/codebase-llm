"""
Tool for fast keyword-based file search
"""
import re
from pathlib import Path
from typing import List
from langchain_core.tools import tool
from ..config import SEMANTIC_SEARCH_CONFIG, REPOS_BASE_PATH


@tool
def search_keyword(
    keywords: List[str],
    repo_name: str
) -> str:
    """
    Fast keyword-based search in repository files. Returns file paths containing any of the keywords.
    Useful for finding files related to specific technical terms or concepts.
    
    All search parameters (max_files, searchable_extensions) are configured in config.py.
    
    Args:
        keywords: List of keywords to search for (case-insensitive)
        repo_name: Name of the repository to search in
    """
    repos_base_path = REPOS_BASE_PATH
    max_files = SEMANTIC_SEARCH_CONFIG["max_files_to_search"]
    
    repo_path = Path(repos_base_path) / repo_name
    
    if not repo_path.exists() or not repo_path.is_dir():
        return f"Error: Repository '{repo_name}' not found at {repo_path}"
    
    # Get searchable extensions from config
    extensions = SEMANTIC_SEARCH_CONFIG["searchable_extensions"]
    
    # Compile regex patterns for each keyword (case-insensitive)
    patterns = [re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords]
    
    matching_files = []
    files_searched = 0
    
    try:
        # Walk through all files in repository
        for file_path in repo_path.rglob('*'):
            # Skip directories and hidden files
            if file_path.is_dir() or file_path.name.startswith('.'):
                continue
            
            # Check if file extension is searchable
            if file_path.suffix not in extensions:
                continue
            
            # Skip common directories
            if any(part.startswith('.') or part in {'__pycache__', 'node_modules', 'venv', '.venv'}
                   for part in file_path.parts):
                continue
            
            files_searched += 1
            
            try:
                # Read file content
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Count total keyword occurrences in this file
                total_count = 0
                for pattern in patterns:
                    total_count += len(pattern.findall(content))
                
                # If any keyword matches, store with count
                if total_count > 0:
                    # Store relative path from repo root
                    relative_path = file_path.relative_to(repo_path)
                    matching_files.append({
                        'path': str(relative_path),
                        'count': total_count
                    })
                        
            except Exception as e:
                # Skip files that can't be read
                continue
        
        if not matching_files:
            return f"No files found containing keywords: {', '.join(keywords)}\nSearched {files_searched} files."
        
        # Sort by occurrence count (descending)
        matching_files.sort(key=lambda x: x['count'], reverse=True)
        
        # Limit to max_files
        if len(matching_files) > max_files:
            matching_files = matching_files[:max_files]
        
        result = f"Found {len(matching_files)} files containing keywords [{', '.join(keywords)}]:\n"
        result += "(Sorted by keyword occurrence count)\n\n"
        
        for file_info in matching_files:
            result += f"- {file_info['path']} ({file_info['count']} occurrences)\n"
        
        if len(matching_files) >= max_files:
            result += f"\n(Limited to {max_files} files. {files_searched} total files searched.)"
        
        return result
        
    except Exception as e:
        return f"Error during search: {str(e)}"
