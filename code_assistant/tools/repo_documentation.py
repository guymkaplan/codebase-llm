"""
Tool for fetching repository documentation
"""
import os
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool
from ..config import REPOS_BASE_PATH


def _generate_tree(directory: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0, max_files: int = 100) -> List[str]:
    """
    Generate a directory tree structure
    
    Args:
        directory: Path to directory
        prefix: Prefix for tree formatting
        max_depth: Maximum depth to traverse
        current_depth: Current depth in traversal
        max_files: Maximum number of files to show
        
    Returns:
        List of strings representing the tree structure
    """
    if current_depth >= max_depth:
        return []
    
    tree_lines = []
    try:
        # Get all items and sort (directories first, then files)
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        
        # Filter out common directories to ignore
        ignore_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv', '.idea', '.vscode'}
        items = [item for item in items if item.name not in ignore_dirs]
        
        # Limit total items shown
        if len(items) > max_files:
            items = items[:max_files]
            truncated = True
        else:
            truncated = False
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            
            # Tree characters
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            
            if item.is_dir():
                tree_lines.append(f"{prefix}{connector}{item.name}/")
                # Recursively add subdirectory contents
                subtree = _generate_tree(item, prefix + extension, max_depth, current_depth + 1, max_files)
                tree_lines.extend(subtree)
            else:
                tree_lines.append(f"{prefix}{connector}{item.name}")
        
        if truncated:
            tree_lines.append(f"{prefix}... (truncated)")
            
    except PermissionError:
        tree_lines.append(f"{prefix}[Permission Denied]")
    except Exception as e:
        tree_lines.append(f"{prefix}[Error: {str(e)}]")
    
    return tree_lines


@tool
def get_repo_documentation(repo_name: str) -> str:
    """Fetch documentation for a repository. ALWAYS call this tool first when asked about any repository.
    
    This tool will look for documentation in the following order:
    1. llms.txt file (LLM-optimized documentation)
    2. README file (with various extensions: .md, .rst, .txt, or no extension)
    
    Args:
        repo_name: Name of the repository to fetch documentation for
    """
    repos_base_path = REPOS_BASE_PATH
    
    # Construct the repository path
    repo_path = Path(repos_base_path) / repo_name
    
    # Check if repository exists
    if not repo_path.exists() or not repo_path.is_dir():
        return f"Error: Repository '{repo_name}' not found at {repo_path}"
    
    # Generate directory tree
    tree_lines = _generate_tree(repo_path)
    directory_tree = "\n".join(tree_lines)
    
    # First, try to find llms.txt
    llms_txt = repo_path / "llms.txt"
    if llms_txt.exists() and llms_txt.is_file():
        try:
            with open(llms_txt, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"# Documentation for repository: {repo_name}\n\n## Directory Structure\n```\n{repo_name}/\n{directory_tree}\n```\n\n## Documentation (from llms.txt)\n{content}"
        except Exception as e:
            return f"Error reading llms.txt: {str(e)}"
    
    # If llms.txt not found, look for README with various extensions
    readme_extensions = ['.md', '.rst', '.txt', ''] 
    readme_base_names = ['README', 'Readme', 'readme']
    
    for base_name in readme_base_names:
        for ext in readme_extensions:
            readme_path = repo_path / f"{base_name}{ext}"
            if readme_path.exists() and readme_path.is_file():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return f"# Documentation for repository: {repo_name}\n\n## Directory Structure\n```\n{repo_name}/\n{directory_tree}\n```\n\n## Documentation (from {readme_path.name})\n{content}"
                except Exception as e:
                    return f"Error reading {readme_path.name}: {str(e)}"
    
    # If nothing found, just return the directory tree
    return f"# Repository: {repo_name}\n\n## Directory Structure\n```\n{repo_name}/\n{directory_tree}\n```\n\nNo llms.txt or README found for this repository."
