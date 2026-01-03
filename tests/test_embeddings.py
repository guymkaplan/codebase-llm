"""
Test script to compare Azure OpenAI and Nomic embeddings
"""
import time
import numpy as np
from code_assistant.config import EMBEDDING_CONFIG

def test_embedding_provider(provider: str):
    """Test embedding with specified provider"""
    print(f"\n{'='*70}")
    print(f"Testing {provider.upper()} Embeddings")
    print('='*70)
    
    # Temporarily set provider
    original_provider = EMBEDDING_CONFIG["provider"]
    EMBEDDING_CONFIG["provider"] = provider
    
    try:
        from code_assistant.tools.semantic_search import _embed_texts
        
        # Test texts
        test_texts = [
            "def authenticate_user(username: str, password: str) -> bool:",
            "class SignatureVerifier:",
            "import cryptography.hazmat.primitives.asymmetric.rsa as rsa"
        ]
        
        print(f"\nEmbedding {len(test_texts)} code snippets...")
        print(f"Provider: {provider}")
        
        start = time.time()
        embeddings = _embed_texts(test_texts)
        elapsed = time.time() - start
        
        print(f"\n✅ Success!")
        print(f"   Embedding shape: {embeddings.shape}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        print(f"   Time taken: {elapsed:.3f}s")
        print(f"   Time per text: {elapsed/len(test_texts):.3f}s")
        
        # Calculate similarity between first two texts
        similarity = np.dot(embeddings[0], embeddings[1])
        print(f"\nSimilarity between text 1 and text 2: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if provider == "nomic":
            print("\nTo use Nomic embeddings, install sentence-transformers:")
            print("  pip install sentence-transformers")
        return False
        
    finally:
        # Restore original provider
        EMBEDDING_CONFIG["provider"] = original_provider


def main():
    print("Embedding Provider Comparison Test")
    print("="*70)
    
    # Test Azure
    azure_success = test_embedding_provider("azure")
    
    # Test Nomic
    nomic_success = test_embedding_provider("nomic")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print('='*70)
    print(f"Azure OpenAI (Ada-2): {'✅ Available' if azure_success else '❌ Failed'}")
    print(f"Nomic Embed (Local):  {'✅ Available' if nomic_success else '❌ Failed'}")
    print()
    
    if azure_success and nomic_success:
        print("💡 Both embedding providers are working!")
        print("   Switch between them by changing EMBEDDING_CONFIG['provider'] in config.py")
    elif azure_success:
        print("💡 Only Azure embeddings available.")
        print("   Install sentence-transformers to use Nomic: pip install sentence-transformers")
    elif nomic_success:
        print("💡 Only Nomic embeddings available.")
        print("   Configure Azure credentials to use Azure OpenAI embeddings.")
    else:
        print("⚠️  No embedding providers available. Please configure at least one.")
    
    print()
    print("Configuration in config.py:")
    print(f"  Current provider: {EMBEDDING_CONFIG['provider']}")
    print(f"  Nomic model: {EMBEDDING_CONFIG['nomic_model']}")
    print(f"  Nomic device: {EMBEDDING_CONFIG['nomic_device']}")


if __name__ == "__main__":
    main()
