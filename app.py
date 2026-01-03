import streamlit as st
from dotenv import load_dotenv
from code_assistant.agent import CodeAssistantAgent
from datetime import datetime
import sys

def log(message):
    """Print to stderr to avoid buffering issues with Streamlit"""
    print(message, file=sys.stderr, flush=True)

log("=" * 80)
log("🚀 STREAMLIT APP INITIALIZING")
log("=" * 80)

load_dotenv(override=True)
log("✅ Environment variables loaded")

st.set_page_config(
    page_title="Code Repository Assistant",
    page_icon="🤖",
    layout="wide"
)

log("✅ Streamlit page config set")

st.title("🤖 Code Repository Assistant")
st.markdown("Ask questions about code repositories using Azure OpenAI")

def generate_flow_dump(user_query: str, assistant_message: str, full_trace: str, tool_calls: list, iterations: int) -> str:
    """
    Generate a complete text dump of the conversation flow
    
    Args:
        user_query: The user's question
        assistant_message: The assistant's response
        full_trace: The full reasoning trace
        tool_calls: List of tool call information
        iterations: Number of iterations
        
    Returns:
        Formatted text string with complete flow
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    dump = f"""{'=' * 80}
CODE REPOSITORY ASSISTANT - FULL FLOW DUMP
{'=' * 80}
Timestamp: {timestamp}
{'=' * 80}

USER QUERY:
{user_query}

{'=' * 80}
FULL REASONING TRACE:
{'=' * 80}
{full_trace}

{'=' * 80}
TOOL CALLS SUMMARY:
{'=' * 80}
Total Iterations: {iterations}
Total Tool Calls: {len(tool_calls)}

"""
    
    # Group tool calls by iteration
    from collections import defaultdict
    calls_by_iteration = defaultdict(list)
    for tool_call in tool_calls:
        iter_num = tool_call.get("iteration", 1)
        calls_by_iteration[iter_num].append(tool_call)
    
    # Add tool call details
    for iter_num in sorted(calls_by_iteration.keys()):
        iter_time = sum(tc.get("execution_time", 0) for tc in calls_by_iteration[iter_num])
        dump += f"\n{'—' * 80}\n"
        dump += f"ITERATION {iter_num} (Execution Time: {iter_time:.3f}s)\n"
        dump += f"{'—' * 80}\n"
        
        for i, tool_call in enumerate(calls_by_iteration[iter_num], 1):
            exec_time = tool_call.get("execution_time", 0)
            dump += f"\n  Tool {i}: {tool_call['name']} (⏱️ {exec_time:.3f}s)\n"
            dump += f"  {'-' * 76}\n"
            dump += f"  Arguments:\n"
            for key, value in tool_call["args"].items():
                if isinstance(value, str) and len(value) > 100:
                    dump += f"    {key}: {value[:100]}... (truncated)\n"
                else:
                    dump += f"    {key}: {value}\n"
            
            # Add subagent info if present
            if "subagents" in tool_call:
                dump += f"\n  Subagents:\n"
                for j, subagent in enumerate(tool_call["subagents"], 1):
                    dump += f"    {j}. {subagent['agent']}\n"
                    if "extracted_keywords" in subagent:
                        dump += f"       Keywords: {subagent['extracted_keywords']}\n"
            
            dump += f"\n  Result:\n"
            result = tool_call["result"]
            if len(result) > 1000:
                dump += f"    {result[:1000]}...\n    (Result truncated, {len(result)} total characters)\n"
            else:
                # Indent each line of result
                for line in result.split('\n'):
                    dump += f"    {line}\n"
            dump += f"\n"
    
    # Add final answer
    dump += f"\n{'=' * 80}\n"
    dump += f"FINAL ANSWER:\n"
    dump += f"{'=' * 80}\n"
    dump += f"{assistant_message}\n"
    
    dump += f"\n{'=' * 80}\n"
    dump += f"END OF FLOW DUMP\n"
    dump += f"{'=' * 80}\n"
    
    return dump

@st.cache_resource
def create_agent():
    """Initialize the Code Assistant Agent"""
    log("\n" + "=" * 80)
    log("🤖 CREATING AGENT")
    log("=" * 80)
    try:
        agent = CodeAssistantAgent()
        log("✅ Agent created successfully")
        return agent
    except ValueError as e:
        log(f"❌ Agent creation failed: {str(e)}")
        st.error(f"Configuration error: {str(e)}")
        st.stop()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Parse and display answer if present
        if message["role"] == "assistant":
            import re
            answer_match = re.search(r'<answer>(.*?)</answer>', message["content"], re.DOTALL)
            
            if answer_match:
                # Display only the answer prominently
                answer_content = answer_match.group(1).strip()
                st.markdown(answer_content)
                
                # Show full trace in expander if available
                if message.get("full_trace"):
                    with st.expander("🧠 View Full Reasoning Trace", expanded=False):
                        st.markdown("### Agent's Reasoning Process")
                        st.markdown(message["full_trace"])
                        st.markdown("### Complete Response with Answer")
                        st.markdown(message["content"])
            else:
                # No answer tags - show full content
                st.markdown(message["content"])
                
                # Show trace if available
                if message.get("full_trace"):
                    with st.expander("🧠 View Full Reasoning Trace", expanded=False):
                        st.markdown(message["full_trace"])
        else:
            # User messages - just show content
            st.markdown(message["content"])
        
        # Display tool calls if they exist in the message
        if "tool_calls" in message and message["tool_calls"]:
            iterations = message.get("iterations", 1)
            total_time = sum(tc.get("execution_time", 0) for tc in message["tool_calls"])
            
            # Add download button for this message
            if message["role"] == "assistant" and message.get("full_trace"):
                # Find corresponding user query
                msg_idx = st.session_state.messages.index(message)
                user_query = ""
                if msg_idx > 0 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                    user_query = st.session_state.messages[msg_idx - 1]["content"]
                
                flow_dump = generate_flow_dump(
                    user_query=user_query,
                    assistant_message=message["content"],
                    full_trace=message["full_trace"],
                    tool_calls=message["tool_calls"],
                    iterations=iterations
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"flow_dump_{timestamp}.txt"
                
                st.download_button(
                    label="📥 Download Full Flow as Text",
                    data=flow_dump,
                    file_name=filename,
                    mime="text/plain",
                    key=f"download_history_{msg_idx}"
                )
            
            with st.expander(f"🔧 Tool Calls ({len(message['tool_calls'])} calls across {iterations} iteration{'s' if iterations > 1 else ''}, total: {total_time:.2f}s)", expanded=False):
                # Group tool calls by iteration
                calls_by_iteration = {}
                for tool_call in message["tool_calls"]:
                    iter_num = tool_call.get("iteration", 1)
                    if iter_num not in calls_by_iteration:
                        calls_by_iteration[iter_num] = []
                    calls_by_iteration[iter_num].append(tool_call)
                
                # Display each iteration
                for iter_num in sorted(calls_by_iteration.keys()):
                    iter_time = sum(tc.get("execution_time", 0) for tc in calls_by_iteration[iter_num])
                    st.markdown(f"### Iteration {iter_num} ({iter_time:.2f}s)")
                    
                    for i, tool_call in enumerate(calls_by_iteration[iter_num], 1):
                        exec_time = tool_call.get("execution_time", 0)
                        st.markdown(f"**Tool {i}: `{tool_call['name']}`** ⏱️ {exec_time:.2f}s")
                        st.json(tool_call["args"])
                        
                        # Show subagents if they exist
                        if "subagents" in tool_call:
                            st.markdown("**🤖 Subagents:**")
                            for j, subagent in enumerate(tool_call["subagents"], 1):
                                with st.expander(f"Subagent {j}: {subagent['agent']}", expanded=False):
                                    st.markdown("**Prompt:**")
                                    st.code(subagent["prompt"], language="text")
                                    st.markdown("**Response:**")
                                    st.text(subagent["response"])
                                    if "extracted_keywords" in subagent:
                                        st.markdown("**Extracted Keywords:**")
                                        st.write(subagent["extracted_keywords"])
                        
                        st.markdown("**Result:**")
                        st.code(tool_call["result"], language="text")
                        st.markdown("---")

# Handle user input
if prompt := st.chat_input("Ask a question about the code..."):
    log("\n" + "=" * 80)
    log(f"📝 USER INPUT RECEIVED: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    log("=" * 80)
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                log("🤖 Initializing agent...")
                agent = create_agent()
                
                log(f"🔄 Processing query with {len(st.session_state.messages) - 1} messages in history")
                
                # Use chat_with_history to maintain conversation context
                result = agent.chat_with_history_and_tools_info(
                    message=prompt,
                    history=st.session_state.messages[:-1]  # Exclude the current message
                )
                
                log(f"✅ Agent response received")
                log(f"   - Iterations: {result.get('iterations', 0)}")
                log(f"   - Tool calls: {len(result.get('tool_calls', []))}")
                
                # Display the final response
                assistant_message = result["response"]
                full_trace = result.get("full_trace", "")
                
                # Parse <answer> tags
                import re
                answer_match = re.search(r'<answer>(.*?)</answer>', assistant_message, re.DOTALL)
                
                if answer_match:
                    log("📄 Response contains <answer> tags")
                    # Extract answer and display prominently
                    answer_content = answer_match.group(1).strip()
                    st.markdown(answer_content)
                    
                    # Show full reasoning trace in expander
                    with st.expander("🧠 View Full Reasoning Trace", expanded=False):
                        st.markdown("### Agent's Reasoning Process")
                        st.markdown(full_trace)
                        st.markdown("### Complete Response with Answer")
                        st.markdown(assistant_message)
                else:
                    # No answer tags - show everything
                    log("📄 Response without <answer> tags")
                    if assistant_message:
                        st.markdown(assistant_message)
                    else:
                        st.warning("⚠️ No response generated from the agent.")
                    
                    # Still show trace if available
                    if full_trace:
                        with st.expander("🧠 View Full Reasoning Trace", expanded=False):
                            st.markdown(full_trace)
                
                # Display tool calls if any
                if result.get("tool_calls"):
                    iterations = result.get("iterations", 1)
                    total_time = sum(tc.get("execution_time", 0) for tc in result["tool_calls"])
                    
                    log(f"🔧 Displaying {len(result['tool_calls'])} tool calls (total time: {total_time:.2f}s)")
                    
                    # Add download button for current flow
                    flow_dump = generate_flow_dump(
                        user_query=prompt,
                        assistant_message=assistant_message,
                        full_trace=full_trace,
                        tool_calls=result["tool_calls"],
                        iterations=iterations
                    )
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"flow_dump_{timestamp}.txt"
                    
                    st.download_button(
                        label="📥 Download Full Flow as Text",
                        data=flow_dump,
                        file_name=filename,
                        mime="text/plain",
                        key="download_current"
                    )
                    
                    with st.expander(f"🔧 Tool Calls ({len(result['tool_calls'])} calls across {iterations} iteration{'s' if iterations > 1 else ''}, total: {total_time:.2f}s)", expanded=False):
                        # Group tool calls by iteration
                        calls_by_iteration = {}
                        for tool_call in result["tool_calls"]:
                            iter_num = tool_call.get("iteration", 1)
                            if iter_num not in calls_by_iteration:
                                calls_by_iteration[iter_num] = []
                            calls_by_iteration[iter_num].append(tool_call)
                        
                        # Display each iteration
                        for iter_num in sorted(calls_by_iteration.keys()):
                            iter_time = sum(tc.get("execution_time", 0) for tc in calls_by_iteration[iter_num])
                            st.markdown(f"### Iteration {iter_num} ({iter_time:.2f}s)")
                            
                            for i, tool_call in enumerate(calls_by_iteration[iter_num], 1):
                                exec_time = tool_call.get("execution_time", 0)
                                st.markdown(f"**Tool {i}: `{tool_call['name']}`** ⏱️ {exec_time:.2f}s")
                                st.json(tool_call["args"])
                                
                                # Show subagents if they exist
                                if "subagents" in tool_call:
                                    st.markdown("**🤖 Subagents:**")
                                    for j, subagent in enumerate(tool_call["subagents"], 1):
                                        with st.expander(f"Subagent {j}: {subagent['agent']}", expanded=False):
                                            st.markdown("**Prompt:**")
                                            st.code(subagent["prompt"], language="text")
                                            st.markdown("**Response:**")
                                            st.text(subagent["response"])
                                            if "extracted_keywords" in subagent:
                                                st.markdown("**Extracted Keywords:**")
                                                st.write(subagent["extracted_keywords"])
                                
                                st.markdown("**Result:**")
                                st.code(tool_call["result"], language="text")
                                st.markdown("---")
                
                # Store message with tool calls and trace for history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message,
                    "tool_calls": result.get("tool_calls", []),
                    "iterations": result.get("iterations", 1),
                    "full_trace": result.get("full_trace", "")
                })
                
                log(f"✅ Response stored in session state")
                log(f"   - Total messages in history: {len(st.session_state.messages)}")
                log("=" * 80 + "\n")
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                log(f"❌ ERROR: {error_message}")
                log("=" * 80 + "\n")
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.info("✅ Using Azure AD authentication (no API key needed)")
    
    if st.button("Clear Chat History"):
        log("\n🗑️  CLEARING CHAT HISTORY")
        st.session_state.messages = []
        log(f"✅ Chat history cleared\n")
        st.rerun()
    
    if st.button("Clear Agent Cache"):
        log("\n🗑️  CLEARING AGENT CACHE")
        st.cache_resource.clear()
        log(f"✅ Agent cache cleared\n")
        st.success("Agent cache cleared! Reload the page to reinitialize.")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses:
    - **Azure OpenAI** for LLM inference
    - **LangChain** for LLM integration
    - **Azure AD** for authentication
    - **Streamlit** for the UI
    
    ### Available Tools:
    - 📄 **get_repo_documentation** - Get README + directory tree
    - 📋 **list_available_repositories** - List all repos
    - 🔍 **search_keyword** - Fast keyword search
    - 🧠 **semantic_file_search** - Smart semantic search with HyDE
    """)
