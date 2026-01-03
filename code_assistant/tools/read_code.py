"""
Tool for reading code files
"""
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from ..config import REPOS_BASE_PATH, CODE_READING_CONFIG


@tool
def read_code(
    file_path: str,
    repo_name: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> str:
    """
    Read code from a file in a repository. Returns the file content with line numbers.
    Use this tool to examine specific code sections identified by semantic_file_search.
    
    The max_lines parameter is configured in config.py.
    
    Args:
        file_path: Relative path to the file within the repository
        repo_name: Name of the repository
        start_line: Optional starting line number (1-indexed). If not provided, starts from line 1.
        end_line: Optional ending line number (1-indexed, inclusive). If not provided, reads to end of file.
    """
    repos_base_path = REPOS_BASE_PATH
    max_lines = CODE_READING_CONFIG["max_lines"]
    
    # Construct full path
    repo_path = Path(repos_base_path) / repo_name
    full_file_path = repo_path / file_path
    
    # Validate repository exists
    if not repo_path.exists() or not repo_path.is_dir():
        return f"Error: Repository '{repo_name}' not found at {repo_path}"
    
    # Validate file exists
    if not full_file_path.exists() or not full_file_path.is_file():
        return f"Error: File '{file_path}' not found in repository '{repo_name}'"
    
    try:
        # Read file content
        with open(full_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Handle line range
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = total_lines
        
        # Validate line numbers
        if start_line < 1:
            start_line = 1
        if end_line > total_lines:
            end_line = total_lines
        if start_line > end_line:
            return f"Error: start_line ({start_line}) cannot be greater than end_line ({end_line})"
        
        # Calculate actual range considering max_lines
        requested_lines = end_line - start_line + 1
        truncated = False
        
        if requested_lines > max_lines:
            end_line = start_line + max_lines - 1
            truncated = True
        
        # Extract requested lines (convert to 0-indexed)
        selected_lines = lines[start_line - 1:end_line]
        
        # Format output with line numbers
        result = f"File: {file_path}\n"
        result += f"Repository: {repo_name}\n"
        result += f"Lines: {start_line}-{end_line} (Total file lines: {total_lines})\n"
        
        if truncated:
            result += f"⚠️  TRUNCATED: Requested {requested_lines} lines, showing first {max_lines} lines.\n"
        
        result += "=" * 80 + "\n\n"
        
        # Add line numbers and content
        for i, line in enumerate(selected_lines, start=start_line):
            result += f"{i:6d} | {line.rstrip()}\n"
        
        result += "\n" + "=" * 80
        
        if truncated:
            result += f"\n\n⚠️  Output truncated at {max_lines} lines. "
            result += f"Use read_code with start_line={end_line + 1} to continue reading."
        
        return result
        
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"
