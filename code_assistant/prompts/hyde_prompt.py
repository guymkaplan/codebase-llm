"""
HyDE (Hypothetical Document Embeddings) prompt for semantic search
"""

HYDE_GENERATION_PROMPT = """You are an expert software developer. Given a question about a code repository and its documentation, generate a hypothetical code implementation with docstrings that would answer the question.

**Repository Context:**
{repo_context}

**Question:** {query}

Based on the repository context above, generate a hypothetical implementation that:
1. Uses the same programming language as the repository
2. Follows similar patterns and conventions shown in the documentation
3. Includes a detailed docstring explaining the functionality
4. Shows hypothetical code/pseudocode with realistic function/class names
5. References technical details: parameters, return types, common libraries

Write in a technical, code-focused style. Maximum 500 words.

**Hypothetical Implementation:**"""

