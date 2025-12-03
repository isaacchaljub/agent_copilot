import os
import uuid
from typing import TypedDict
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_litellm import ChatLiteLLM
from langchain_tavily import TavilySearch
from numpy.linalg import norm
from numpy import dot
import numpy as np
import googlemaps
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import build_gmail_service, get_google_credentials

# Load the environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("MAPS_API_KEY")


#Initialize the Gmail Toolkit
gmail_scopes = ["https://mail.google.com/"]
creds = get_google_credentials(scopes=gmail_scopes)
resource_service = build_gmail_service(credentials=creds)
gmail_toolkit = GmailToolkit(api_resource=resource_service)
email_tools = gmail_toolkit.get_tools()

#Initialize the Tavily Client
search_tool = TavilySearch(
    max_results=3,
    include_answer=True,
    tavily_api_key=TAVILY_API_KEY
)

# Define global variables
cache=[]
cache_size=20
similarity_threshold=0.9
global_embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
maps=googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# LLM to define tools
tool_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY,
)

# LLM for agent execution
agent_llm = ChatLiteLLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    max_tokens=2000,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API_KEY)

# Define the agent state
class AgentState(TypedDict):
    query: str
    cache_hit:bool
    answer: str
    agent: str
    need_more_info: bool
    more_info_query: str


# Define function to check cosine similarity
def cosine_similarity(query_embedding, cached_embedding):
    """Calculate cosine similarity between two embeddings"""
    return dot(query_embedding, cached_embedding) / (norm(query_embedding) * norm(cached_embedding))

#Define function to check cache
def check_cache(state: AgentState) -> AgentState:
    """Check if query is in semantic cache"""
    try:
        query = state.get("query", "")
        if not query:
            return {**state, "cache_hit": False}
        
        if len(cache) == 0:
            return {**state, "cache_hit": False}
        
        embedded_query = global_embeddings.embed_query(query)
        similarity_scores = [cosine_similarity(embedded_query, cached_item[0]) for cached_item in cache]
        
        if np.max(similarity_scores) > similarity_threshold:
            cached_answer = cache[np.argmax(similarity_scores)][2]
            return {
                **state,
                "cache_hit": True,
                "answer": cached_answer
            }
        
        return {**state, "cache_hit": False}
    except Exception as e:
        # If cache check fails, continue without cache
        return {**state, "cache_hit": False}


#Define function to choose the agent to use
def define_agent(state: AgentState) -> AgentState:
    """Use LLM to choose the agent to use"""
    try:
        query = state.get("query", "")
        if not query:
            return {**state, "agent": "", "answer": "Error: No query provided"}
        
        prompt = '''Role: AI Agent Choice Assistant
Task: Choose the agent for the user's question.
Instructions:
    - Analyze the user's question and identify the agent that is needed to answer the question.
    - Provide a clear and concise response indicating the agent that is needed to answer the question.
    - Your response should include only the agent that is needed to answer the question. Nothing else, no other text, information, header/footer.
    - The agents available are:
        - find_places: to find places nearby
        - search_web: to search and scrape the web
        - send_email: to send an email
Output Format:
    - agent: name of the agent
Examples:
    Input: 
        User Query: Find me bars that are close to Calle de Clara del Rey, 5, in Madrid, Spain.
    Expected Output:
        agent: find_places
    Input: 
        User Query: What is the population of China?
    Expected Output:
        agent: search_web
    Input: 
        User Query: Send an email to John Doe with the subject "Meeting" and the body "We need to discuss the project".
    Expected Output:
        agent: send_email
    Input:
        User Query: {query}
'''
        formatted_prompt = prompt.format(query=query)
        response = tool_llm.invoke(formatted_prompt)
        agent = response.content.strip().lower()

        agent_name = agent
        if "agent:" in agent:
            agent_name = agent.split("agent:", 1)[1].strip()
        
        return {**state, "agent": agent_name}
    except Exception as e:
        return {**state, "agent": "", "answer": f"Error selecting agent: {str(e)}"}


#Define a function to revew if the query has enough information to answer and fix it
def review_query(state: AgentState) -> AgentState:
    """Review if the query has enough information to answer"""
    try:
        query = state.get("query", "")
        if not query:
            return {**state, "need_more_info": True, "more_info_query": "Please provide a query"}

        prompt = '''Role: You are a helpful AI Query Reviewer that determines if the user's query has enough information to answer it.
Task: Determine if the user's query has enough information to answer.
Instructions:
    - Read the user's query and identify if it is clear and has enough information to answer.
    - Provide a clear and concise response indicating whether the query has all the needed information or not.
    - Provide a brief explanation of what information is missing.
    - Your response should include only the answer. Nothing else, no other text, information, header/footer.
    - If the query has enough information to answer, provide the answer as "Yes" and don't provide an explanation.
    - If the query doesn't have enough information to answer, provide the answer as "No" and provide a brief explanation of what information is missing.
Output Format:
    - answer: Yes/No
    - explanation: brief explanation of what information is missing
Examples:
    Input:
        User Query: Find me bars that are close to Calle de Clara del Rey, 5.
    Expected Output:
        answer: No
        explanation: The query is missing the city and country.
    Input:
        User Query: Find me bars that are close to Calle de Clara del Rey, 5, in Madrid, Spain.
    Expected Output:
        answer: Yes
    Input: 
        User Query: What is the population of China?
    Expected Output:
        answer: Yes
    Input:
        User Query: What is the altitude of Colombia?
    Expected Output:
        answer: No
        explanation: The query is missing the city to search for.
    Input: 
        User Query: Send an email to John Doe with the subject "Meeting" and the body "We need to discuss the project".
    Expected Output:
        answer: No
        explanation: The query is missing the recipient email.
    Input:
        User Query: Send an email to devtestchal@gmail.com with the subject "Meeting" and the body "We need to discuss the project".
    Expected Output:
        answer: Yes
    Input:
        User Query: {query}
'''
        formatted_prompt = prompt.format(query=query)
        response = tool_llm.invoke(formatted_prompt)
        answer = response.content.strip()

        is_enough_info = ""
        explanation = ""

        for line in answer.splitlines():
            line = line.strip().lower()
            if line.startswith("answer:"):
                is_enough_info = line.split("answer:", 1)[1].strip().lower()
            elif line.startswith("explanation:"):
                explanation = line.split("explanation:", 1)[1].strip()

        if is_enough_info == "no":
            return {**state, "need_more_info": True, "more_info_query": explanation}
        else:
            return {**state, "need_more_info": False}
    except Exception as e:
        # If review fails, assume we have enough info and continue
        return {**state, "need_more_info": False}
    

#Define function to ask for more information
def ask_for_more_info(state: AgentState) -> AgentState:
    """Ask for more information"""
    more_info_query=state["more_info_query"]

    # print(f"Please provide the following information to answer the question: {more_info_query}")
    user_response = interrupt(more_info_query)
    current_query = state["query"]
    updated_query = f"{current_query}. User provided details: {user_response}"
    
    return {**state, "query": updated_query}

#Place finder function
@tool
def place_finder(address: str, keyword: str) -> str:
    """Find places nearby"""
    try:
        #Define the radius to search for
        radius = 500

        if not address or not keyword:
            return "Error: Address and keyword are required"

        #Convert the address to a latitude and longitude using googlemaps geocode
        location = maps.geocode(address)
        if not location:
            return f"Error: Location not found for address: {address}"

        # Extract latitude and longitude from the first result
        location_coords = location[0]['geometry']['location']
        lat = location_coords['lat']
        lng = location_coords['lng']

        #Use the location (as coordinates) and the keyword to find the places nearby using googlemaps places_nearby
        places_list = maps.places_nearby(location=(lat, lng), radius=radius, keyword=keyword)

        if not places_list or 'results' not in places_list or not places_list['results']:
            return f"No places found near {address} matching '{keyword}'"

        answer = "\n".join([f"{place.get('name', 'Unknown')}: {place.get('rating', 'No rating available')}, {place.get('vicinity', 'No address available')}" for place in places_list.get('results', [])])
        return answer
    except Exception as e:
        return f"Error finding places: {str(e)}"


#Place finder agent
def places_finder_agent():
    "Create a places finder agent using Langchain"
    tools=[place_finder]
    places_finder_agent=create_agent(
        model=agent_llm,
        tools=tools,
        system_prompt='''
        You are an expert places finder agent.
        You are given a user's query and you need to find the places nearby based on the user's query.
        You have a tool called place_finder. Simply extract the human address and the keyword and pass them directly to the tool.
        Do not attempt to geocode it yourself.
        Use the available tool to find the places nearby. Be thorough and precise in your search.
        Output Format:
        - answer: list of places nearby
        Examples:
        Input:
        User Query: Find me bars that are close to Calle de Clara del Rey, 5, in Madrid, Spain.
        Expected Output:
        answer: [Bar 1, Bar 2, Bar 3]
        Input:
        User Query: Find me restaurants that are close to Calle Hermosilla, 87, in Madrid, Spain.
        '''
    )
    return places_finder_agent

#Define function to find places nearby
def find_places_nearby(state: AgentState) -> AgentState:
    """Find places nearby"""
    try:
        query = state.get("query", "")
        if not query:
            return {**state, "answer": "Error: No query provided"}

        #Get the agent
        agent = places_finder_agent()

        # Create a message from the query
        user_message = HumanMessage(
            content=f"Find places nearby based on the user's query: {query}"
        )

        # Invoke the agent with message
        result = agent.invoke({"messages": [user_message]})

        #Extract the answer from the result
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            answer = "No response from places finder agent"

        return {**state, "answer": answer}
    except Exception as e:
        return {**state, "answer": f"Error finding places: {str(e)}"}


#Web searcher agent
def get_web_agent():
    """Get or create the LangChain agent using create_agent"""

    tools = [search_tool]

    # Create the agent using langchain.agents.create_agent
    web_agent = create_agent(
        model=agent_llm,
        tools=tools,
        system_prompt="""You are an expert web research assistant.

TOOLS:
- search_tool: TavilySearch: Use this to search the web with the user's query and get relevant results.

ALWAYS FOLLOW THIS PATTERN:
1. Call search_tool with the user's question.
2. Look at the returned results.
3. Read and synthesize everything into a clear, concise, factual answer.

Focus on giving a final, well-structured answer to the user."""
    )
    return web_agent

web_agent = get_web_agent()

#Define function to search the web
def search_web(state: AgentState) -> AgentState:
    """Search the web for information using LangChain agent"""
    try:
        query = state.get("query", "")
        if not query:
            return {**state, "answer": "Error: No query provided"}
        
        # Get the web agent
        agent = web_agent
        
        # Create a message from the query
        user_message = HumanMessage(
            content=f"Search the web and find comprehensive information about: {query}"
        )
        
        # Invoke the agent with messages
        result = agent.invoke({"messages": [user_message]})
        
        # Extract the final message content from the agent's response
        messages = result.get("messages", [])
        if messages:
            # Get the last message (agent's response)
            last_message = messages[-1]
            answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            answer = "No response from web agent"
        
        return {**state, "answer": answer}
    except Exception as e:
        return {**state, "answer": f"Error searching the web: {str(e)}"}


#Email sender agent
def get_email_agent():
    'Create a email sender agent using Langchain and the Gmail Toolkit'
    email_agent = create_agent(
        model=agent_llm, 
        tools=email_tools,
        system_prompt='''
        You are an expert Email Assistant.
        Your primary job is to send emails based on the user's instructions.
        
        Important Instructions:
        1. Always create a draft first if the user asks to "draft" an email.
        2. Use the `send_message` tool if the user explicitly asks to SEND.
        3. If the user provides a recipient (to), subject, and body, use them exactly.''')

    return email_agent


#Define function to send email  
def send_email(state: AgentState) -> AgentState:
    """Send an email using LangChain agent"""
    try:
        #Get the query
        query = state.get("query", "")
        if not query:
            return {**state, "answer": "Error: No query provided"}

        #Get the email agent
        agent = get_email_agent()

        # Create a message from the query
        user_message = HumanMessage(
            content=f"Send an email according to the user's instructions: {query}"
        )

        # Invoke the agent with messages
        result = agent.invoke({"messages": [user_message]})

        # Extract the answer from the result
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            answer = "No response from email agent"
        
        return {**state, "answer": answer}
    except Exception as e:
        return {**state, "answer": f"Error sending email: {str(e)}"}


#Define function to generate final answer
def generate_answer(state: AgentState) -> AgentState:
    """Generate final answer using LLM"""
    try:
        answer = state.get("answer", "")
        query = state.get("query", "")
        
        if not answer or not query:
            return {**state, "answer": "Error: Missing answer or query"}
        
        # Check if answer contains error messages - if so, return as-is
        if "Error:" in answer or answer.startswith("Error"):
            return {**state, "answer": answer}
        
        # Format the answer nicely - extract only the relevant information
        messages = [
            SystemMessage(content="You are a helpful assistant. Your task is to take the information provided and format it into a clear, direct answer to the user's question. Ignore any error messages, technical details, or metadata. Only provide the factual answer to the question."),
            SystemMessage(content=f"Information retrieved: {answer}"),
            HumanMessage(content=f"User question: {query}\n\nBased on the information above, provide a clear and direct answer. Do not mention errors, technical details, or that you're using provided information - just answer the question."),
        ]
        
        # Use tool_llm
        response = tool_llm.invoke(messages)
        final_answer = response.content
        
        # Update cache
        try:
            embedded_query = global_embeddings.embed_query(query)
            cache.append((embedded_query, query, final_answer))
            
            # Prune cache if too large
            if len(cache) > cache_size:
                cache.pop(0)
        except Exception:
            # If cache update fails, continue without caching
            pass
        
        return {**state, "answer": final_answer}
    except Exception as e:
        return {**state, "answer": f"Error generating answer: {str(e)}"}


#Define routing function for agent selection
def route_agent(state: AgentState) -> str:
    """Route to the appropriate agent based on the selected agent type"""
    agent_type = state.get("agent", "").strip().lower()
    
    # Map agent types to their corresponding nodes
    agent_routes = {
        "find_places": "Find Places Nearby",
        "search_web": "Search Web",
        "send_email": "Send Email"
    }
    
    # Return the route if found, otherwise end the graph
    return agent_routes.get(agent_type, END)


#Define the graph
def create_graph():
    """Create the agentic pipeline graph"""
    workflow=StateGraph(AgentState)

    #Add nodes to the graph
    workflow.add_node("Check Cache", check_cache)
    workflow.add_node("Review Query", review_query)
    workflow.add_node("Ask for More Info", ask_for_more_info)
    workflow.add_node("Define Agent", define_agent)
    workflow.add_node("Find Places Nearby", find_places_nearby)
    workflow.add_node("Search Web", search_web)
    workflow.add_node("Send Email", send_email)
    workflow.add_node("Generate Answer", generate_answer)

    #Set the entry point
    workflow.set_entry_point("Check Cache")

    #Add Edges to the graph
    workflow.add_conditional_edges("Check Cache", lambda state: END if state.get("cache_hit", False) else "Review Query")
    workflow.add_conditional_edges("Review Query", lambda state: "Ask for More Info" if state.get("need_more_info", False) else "Define Agent")
    workflow.add_edge("Ask for More Info", "Review Query")
    workflow.add_conditional_edges("Define Agent", route_agent)
    workflow.add_edge("Find Places Nearby", "Generate Answer")
    workflow.add_edge("Search Web", "Generate Answer")
    workflow.add_edge("Send Email", "Generate Answer")
    workflow.add_edge("Generate Answer", END)

    #Add the checkpointer to the graph
    checkpointer = MemorySaver()
    app=workflow.compile(checkpointer=checkpointer)

    return app

#Initialize the graph once at startup
_graph=create_graph()

#Define the function to run the graph
def process_query(query: str, thread_id: str | None = None) -> str | dict:
    """Process the query through the graph
    
    Args:
        query: The user's query to process
        thread_id: Optional thread ID for state persistence. If None, generates a new UUID.
    
    Returns:
        The final answer (str), or a dict with interrupt information if an interrupt occurs
    """
    try:
        if not query or not query.strip():
            return "Error: Empty query provided"

        # Generate or use provided thread_id for state persistence
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        # Create config with thread_id for checkpointer
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        #Define the initial state
        initial_state = {
            "query": query.strip(),
            "cache_hit": False,
            "answer": "",
            "agent": "",
            "need_more_info": False,
            "more_info_query": ""
        }

        # Use stream to handle interrupts properly
        # Collect all chunks to get final state
        final_state = None
        for chunk in _graph.stream(initial_state, config=config):
            # Check for interrupt in chunk
            if "__interrupt__" in chunk:
                interrupt_info = chunk["__interrupt__"]
                # Extract interrupt value
                interrupt_value = interrupt_info[0].value if interrupt_info and len(interrupt_info) > 0 else "Need more information"
                return {"__interrupt__": True, "message": interrupt_value, "thread_id": thread_id, "config": config}
            
            # Store the latest state (usually the last chunk contains the final state)
            final_state = chunk

        # Extract answer from final state
        if final_state:
            # The final state might be in different formats, try to get answer
            for key, value in final_state.items():
                if isinstance(value, dict) and "answer" in value:
                    answer = value.get("answer", "")
                    if answer:
                        return answer
        
        return "Error: No answer generated from the pipeline"
    except Exception as e:
        return f"Error processing query: {str(e)}"


#Define function to resume after interrupt
def resume_query(user_response: str, thread_id: str, config: dict) -> str | dict:
    """Resume the graph execution after an interrupt
    
    Args:
        user_response: The user's response to the interrupt question
        thread_id: The thread ID from the interrupt
        config: The config from the interrupt
    
    Returns:
        The final answer (str), or a dict with interrupt information if another interrupt occurs
    """
    try:
        # Resume the graph with the user's response using Command
        final_state = None
        for chunk in _graph.stream(Command(resume=[user_response]), config=config):
            # Check for another interrupt
            if "__interrupt__" in chunk:
                interrupt_info = chunk["__interrupt__"]
                interrupt_value = interrupt_info[0].value if interrupt_info and len(interrupt_info) > 0 else "Need more information"
                return {"__interrupt__": True, "message": interrupt_value, "thread_id": thread_id, "config": config}
            
            # Store the latest state
            final_state = chunk
        
        # Extract answer from final state
        if final_state:
            for key, value in final_state.items():
                if isinstance(value, dict) and "answer" in value:
                    answer = value.get("answer", "")
                    if answer:
                        return answer
        
        return "Error: No answer generated from the pipeline"
    except Exception as e:
        return f"Error resuming query: {str(e)}"

#Define main to test the pipeline
def main():
    """Main function to test the pipeline with interrupt handling"""
    query = input("Enter a query to process: ")
    thread_id = None
    config = None
    
    # Initial query processing
    result = process_query(query, thread_id)
    
    # Handle interrupt loop
    while isinstance(result, dict) and result.get("__interrupt__"):
        # Extract interrupt information
        interrupt_message = result.get("message", "Need more information")
        thread_id = result.get("thread_id")
        config = result.get("config")
        
        # Get user response
        user_response = input(f"{interrupt_message}\nYour response: ")
        
        # Resume the graph
        result = resume_query(user_response, thread_id, config)
    
    print(f"Answer: {result}")

if __name__ == "__main__":
    main()