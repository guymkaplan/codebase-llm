"""
Agent implementation for Azure OpenAI with Azure AD authentication
"""
import os
import time
from azure.identity import get_bearer_token_provider, EnvironmentCredential
from langchain_openai import AzureChatOpenAI
from .prompts import MAIN_AGENT_SYSTEM_PROMPT
from .tools import get_repo_documentation, list_available_repositories, search_keyword, semantic_file_search, read_code
from .tools.semantic_search import get_last_semantic_search_metadata


class CodeAssistantAgent:
    """Agent for assisting with code repository questions"""
    
    def __init__(
        self,
        azure_endpoint: str = None,
        model_name: str = None,
        api_version: str = None,
        temperature: float = 0.7
    ):
        """
        Initialize the Code Assistant Agent
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL (reads from AZURE_OPENAI_ENDPOINT if not provided)
            model_name: Model deployment name (reads from AZURE_OPENAI_MODEL_GPT4o if not provided)
            api_version: API version (reads from AZURE_OPENAI_API_VERSION if not provided)
            temperature: Temperature for response generation
        """
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.model_name = model_name or os.getenv("AZURE_OPENAI_MODEL_GPT4o")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.temperature = temperature
        
        # Validate required parameters
        if not all([self.azure_endpoint, self.model_name, self.api_version]):
            raise ValueError(
                "Missing required parameters. Provide azure_endpoint, model_name, and api_version "
                "or set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL_GPT4o, and AZURE_OPENAI_API_VERSION "
                "environment variables."
            )
        
        # Use Azure AD authentication (no API key needed!)
        self.token_provider = get_bearer_token_provider(
            EnvironmentCredential(), 
            "https://cognitiveservices.azure.com/.default"
        )
        
        # Create LangChain Azure OpenAI LLM with Azure AD auth
        self.llm = AzureChatOpenAI(
            azure_deployment=self.model_name,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_ad_token_provider=self.token_provider,
            temperature=self.temperature,
        )
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools([
            get_repo_documentation,
            list_available_repositories,
            search_keyword,
            semantic_file_search,
            read_code
        ])
        
        # System prompt for the agent
        self.system_prompt = MAIN_AGENT_SYSTEM_PROMPT
    
    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response
        
        Args:
            message: User message/question
            
        Returns:
            Agent's response as string
        """
        messages = [
            ("system", self.system_prompt),
            ("human", message)
        ]
        
        response = self.llm_with_tools.invoke(messages)
        
        # Check if the model wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute tool calls
            messages.append(response)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Execute the tool
                if tool_name == "get_repo_documentation":
                    tool_result = get_repo_documentation.invoke(tool_args)
                elif tool_name == "list_available_repositories":
                    tool_result = list_available_repositories.invoke(tool_args)
                elif tool_name == "search_keyword":
                    tool_result = search_keyword.invoke(tool_args)
                elif tool_name == "semantic_file_search":
                    tool_result = semantic_file_search.invoke(tool_args)
                elif tool_name == "read_code":
                    tool_result = read_code.invoke(tool_args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"]
                })
            
            # Get final response after tool execution
            final_response = self.llm_with_tools.invoke(messages)
            return final_response.content
        
        return response.content
    
    def chat_with_history(self, message: str, history: list) -> str:
        """
        Send a message with conversation history
        
        Args:
            message: Current user message
            history: List of previous messages in format [{"role": "user"/"assistant", "content": "..."}]
            
        Returns:
            Agent's response as string
        """
        messages = [("system", self.system_prompt)]
        
        # Add history
        for msg in history:
            if msg["role"] == "user":
                messages.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(("ai", msg["content"]))
        
        # Add current message
        messages.append(("human", message))
        
        response = self.llm_with_tools.invoke(messages)
        
        # Check if the model wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute tool calls
            messages.append(response)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Execute the tool
                if tool_name == "get_repo_documentation":
                    tool_result = get_repo_documentation.invoke(tool_args)
                elif tool_name == "list_available_repositories":
                    tool_result = list_available_repositories.invoke(tool_args)
                elif tool_name == "search_keyword":
                    tool_result = search_keyword.invoke(tool_args)
                elif tool_name == "semantic_file_search":
                    tool_result = semantic_file_search.invoke(tool_args)
                elif tool_name == "read_code":
                    tool_result = read_code.invoke(tool_args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"]
                })
            
            # Get final response after tool execution
            final_response = self.llm_with_tools.invoke(messages)
            return final_response.content
        
        return response.content
    
    def chat_with_history_and_tools_info(self, message: str, history: list, max_iterations: int = 10) -> dict:
        """
        Send a message with conversation history and return both response and tool call information.
        Supports multiple tool calls in a loop until agent provides final answer.
        
        Args:
            message: Current user message
            history: List of previous messages in format [{"role": "user"/"assistant", "content": "..."}]
            max_iterations: Maximum number of tool call iterations to prevent infinite loops
            
        Returns:
            Dictionary with 'response' (str), 'tool_calls' (list), 'full_trace' (str), and 'iterations' (int) keys
        """
        messages = [("system", self.system_prompt)]
        
        # Add history
        for msg in history:
            if msg["role"] == "user":
                messages.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(("ai", msg["content"]))
        
        # Add current message
        messages.append(("human", message))
        
        tool_calls_info = []
        full_trace = []  # Store complete reasoning trace
        iteration = 0
        
        # Loop to allow multiple tool calls
        while iteration < max_iterations:
            iteration += 1
            
            # Get response from LLM (with tools)
            response = self.llm_with_tools.invoke(messages)
            
            # Add to trace
            if response.content:
                full_trace.append(f"**Agent Response (Iteration {iteration}):**\n{response.content}\n")
            
            # Check if the model wants to use tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Add the assistant's response (with tool calls) to messages
                messages.append(response)
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Execute the tool
                    if tool_name == "get_repo_documentation":
                        tool_result = get_repo_documentation.invoke(tool_args)
                    elif tool_name == "list_available_repositories":
                        tool_result = list_available_repositories.invoke(tool_args)
                    elif tool_name == "search_keyword":
                        tool_result = search_keyword.invoke(tool_args)
                    elif tool_name == "semantic_file_search":
                        tool_result = semantic_file_search.invoke(tool_args)
                    elif tool_name == "read_code":
                        tool_result = read_code.invoke(tool_args)
                    else:
                        tool_result = f"Unknown tool: {tool_name}"
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Store tool call info
                    tool_calls_info.append({
                        "name": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                        "iteration": iteration,
                        "execution_time": execution_time
                    })
                    
                    # Add to trace
                    full_trace.append(f"**Tool Call:** `{tool_name}` (took {execution_time:.2f}s)\n**Args:** {tool_args}\n**Result:** {tool_result[:500]}{'...' if len(tool_result) > 500 else ''}\n")
                    
                    # If this was semantic_file_search, add subagent metadata
                    if tool_name == "semantic_file_search":
                        subagent_metadata = get_last_semantic_search_metadata()
                        if subagent_metadata:
                            tool_calls_info[-1]["subagents"] = subagent_metadata
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call["id"]
                    })
                
                # Continue loop - agent will see tool results and can call more tools or provide final answer
                continue
            
            else:
                # No tool calls - this is the final response
                full_trace_text = "\n".join(full_trace)
                
                return {
                    "response": response.content,
                    "tool_calls": tool_calls_info,
                    "iterations": iteration,
                    "full_trace": full_trace_text
                }
        
        # Max iterations reached
        full_trace_text = "\n".join(full_trace)
        
        return {
            "response": f"Maximum iterations ({max_iterations}) reached. Last response: {response.content if 'response' in locals() else 'No response'}",
            "tool_calls": tool_calls_info,
            "iterations": iteration,
            "full_trace": full_trace_text
        }
