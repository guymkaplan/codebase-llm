"""
Keyword extraction retry prompt for semantic search (when initial search returns no results)
"""

KEYWORD_EXTRACTION_RETRY_PROMPT = """Extract {num_keywords} SHORT and GENERAL keywords from the following code/text for file searching.

**IMPORTANT:** This is attempt {attempt}. The previous keywords found no results.
Use EVEN MORE GENERAL and SHORTER terms this time!

RULES:
- Use VERY SHORT words (3-6 characters preferred)
- Break everything into smallest parts
- Use extremely common programming terms
- Think about basic operations and concepts
- Prefer generic terms over technical jargon

Examples for this retry:
- "auth", "user", "login" (instead of authentication-specific terms)
- "sign", "cert", "key" (instead of signature/certificate terms)
- "read", "write", "file" (instead of I/O operation names)
- "http", "api", "request" (instead of specific HTTP handlers)

Return ONLY a comma-separated list of keywords, nothing else.

Text:
{hyde_answer}

Keywords:"""
