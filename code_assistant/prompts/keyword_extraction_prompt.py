"""
Keyword extraction prompt for semantic search
"""

KEYWORD_EXTRACTION_PROMPT = """Extract {num_keywords} SHORT and GENERAL keywords from the following code/text for file searching.

IMPORTANT RULES:
- Use SHORT words (3-8 characters preferred)
- Break compound words into parts (e.g., "JWTAuth" → "JWT", "auth")
- Prefer common technical terms over specific class/function names
- Use lowercase when appropriate
- Think about what words would actually appear in code

GOOD examples:
- "auth" (not "JWTAuthenticator")
- "verify", "signature" (not "verify_signature")  
- "token", "JWT" (not "TokenManager")
- "hash", "digest" (not "HashDigestCalculator")

BAD examples:
- "JWTAuthenticator" (too specific)
- "verify_signature" (too long, use separate words)
- "CompleteAuthenticationHandler" (way too specific)

Focus on finding the most searchable, general terms.

Return ONLY a comma-separated list of keywords, nothing else.

Text:
{hyde_answer}

Keywords:"""
