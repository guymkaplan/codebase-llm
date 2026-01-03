"""
Simple test script to verify the agent works
"""
from dotenv import load_dotenv
from code_assistant.agent import CodeAssistantAgent

load_dotenv(override=True)

def main():
    print("Initializing agent...")
    try:
        agent = CodeAssistantAgent()
        print("✅ Agent initialized successfully\n")
        
        # Test simple question
        question = "What is 2+2? Respond with just the number."
        print(f"Question: {question}")
        
        response = agent.chat(question)
        print(f"Response: {response}\n")
        
        print("✅ Agent is working! You can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
