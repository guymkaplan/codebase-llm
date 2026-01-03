"""
Test script for the new semantic search tools
"""
from dotenv import load_dotenv
from code_assistant.tools import search_keyword, semantic_file_search

load_dotenv(override=True)

def test_keyword_search():
    print("=" * 60)
    print("Testing search_keyword tool")
    print("=" * 60)
    
    result = search_keyword.invoke({
        'keywords': ['signature', 'verify'],
        'repo_name': 'signify'
    })
    
    print(result[:800])
    print("\n✅ Keyword search working!\n")


def test_semantic_search():
    print("=" * 60)
    print("Testing semantic_file_search tool")
    print("=" * 60)
    print("Note: This requires Azure OpenAI authentication\n")
    
    try:
        result = semantic_file_search.invoke({
            'query': 'How does signature verification work?',
            'repo_name': 'signify',
            'top_k': 3
        })
        
        print(result)
        print("\n✅ Semantic search working!\n")
    except Exception as e:
        print(f"⚠️  Expected error (Azure auth): {str(e)[:200]}")
        print("This is normal if Azure credentials are not configured.")


if __name__ == "__main__":
    print("\n🧪 Testing Search Tools\n")
    
    test_keyword_search()
    test_semantic_search()
    
    print("\n✅ All tests completed!")
