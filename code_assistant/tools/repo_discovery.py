"""
Tool for discovering available repositories
"""
from pathlib import Path
from langchain_core.tools import tool
from ..config import REPOS_BASE_PATH


@tool
def list_available_repositories() -> str:
    """List all available repositories that can be queried.
    """
    repos_base_path = REPOS_BASE_PATH
    repos_path = Path(repos_base_path)
    
    if not repos_path.exists():
        return f"Error: Repositories path not found at {repos_path}"
    
    try:
        repos = [d.name for d in repos_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if repos:
            return "Available repositories:\n" + "\n".join(f"- {repo}" for repo in sorted(repos))
        else:
            return "No repositories found"
    except Exception as e:
        return f"Error listing repositories: {str(e)}"
