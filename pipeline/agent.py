import os
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_litellm import ChatLiteLLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from numpy.linalg import norm
from numpy import dot
import numpy as np
import googlemaps
from googlemaps.places import find_place, places_nearby
import requests
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import build_resource_service, get_gmail_credentials

# Load the environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("MAPS_API_KEY")

#Initialize the Gmail Toolkit
gmail_toolkit=GmailToolkit()
email_tools=gmail_toolkit.get_tools()

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
    query = state["query"]
    
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


#Define function to choose the agent to use
def define_agent(state: AgentState) -> AgentState:
    """Use LLM to choose the agent to use"""
    query = state["query"]
    
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
    return {**state, "agent": agent}


#Define a function to revew if the query has enough information to answer and fix it
def review_query(state: AgentState) -> AgentState:
    """Review if the query has enough information to answer"""
    query=state["query"]

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
    answer = response.content.strip().lower()
    
    is_enough_info = answer.get("answer")
    explanation = answer.get("explanation")

    if is_enough_info=="No":
        return {**state, "need_more_info": True, "more_info_query": explanation}
    else:
        return {**state, "need_more_info": False}
    

#Define function to ask for more information
def ask_for_more_info(state: AgentState) -> AgentState:
    """Ask for more information"""
    query=state["query"]
    more_info_query=state["more_info_query"]

    # print(f"Please provide the following information to answer the question: {more_info_query}")

    additional_info=input("Please provide the following information to answer the question: {more_info_query}")

    state["query"]=f"{query} {additional_info}"
    return state

#Place finder function
@tool
def place_finder(address: str, keyword: str) -> list:
    """Find places nearby"""

    #Define the radius to search for
    radius=500

    #Convert the address to a latitude and longitude using googlemaps geocode
    location=maps.geocode(address)
    if location is None:
        raise ValueError("Location not found")

    # Extract latitude and longitude from the first result
    location = location[0]['geometry']['location']
    lat = location['lat']
    lng = location['lng']


    #Use the location (as coordinates) and the keyword to find the places nearby using googlemaps places_nearby
    places_list=places_nearby(client=maps, location=(lat, lng), keyword=keyword, radius=radius)

    answer="\n".join([f"{place.get('name', 'Unknown')}: {place.get('rating', 'No rating available')}, {place.get('vicinity', 'No address available')}" for place in places_list['results']])
    return answer


#Place finder agent
def places_finder_agent():
    "Create a places finder agent using Langchain"
    tools=[place_finder]
    places_finder_agent=create_agent(
        llm=agent_llm,
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

#Initialize the places finder agent
maps_agent_executor=places_finder_agent()

#Define function to find places nearby
def find_places_nearby(state: AgentState) -> AgentState:
    """Find places nearby"""
    query=state["query"]
    
    #Define the radius to search for
    radius=500

    #Get the agent
    agent=maps_agent_executor

    # Create a message from the query
    user_message = HumanMessage(
        content=f"Find places nearby based on the user's query: {query}"
    )

    # Invoke the agent with message
    result = agent.invoke({"messages": [user_message]})

    #Extract the answer from the result
    messages=result.get("messages", [])
    if messages:
        last_message=messages[-1]
        answer=last_message.content if hasattr(last_message, 'content') else str(last_message)
    else:
        raise ValueError("No places found for the answer")

    return {**state, "answer": answer}

#Web searcher agent
def get_web_agent():
    """Get or create the LangChain agent using create_agent"""

    tools = [SerperDevTool(), ScrapeWebsiteTool()]

    # Create the agent using langchain.agents.create_agent
    web_agent = create_agent(
        model=agent_llm,
        tools=tools,
        system_prompt="""You are an expert web research assistant. Your task is to:
1. Search the web for relevant information about the user's query
2. Scrape and analyze the most relevant web pages
3. Provide a comprehensive summary of the findings

Use the available tools to search and scrape websites. Be thorough and accurate."""
        )
    return web_agent

#Initialize the web agent
web_agent_executor=get_web_agent()

def search_web(state: AgentState) -> AgentState:
    """Search the web for information using LangChain agent"""
    query = state["query"]
    
    # Get the web agent
    agent = web_agent_executor
    
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


#Email sender agent
def get_email_agent():
    'Create a email sender agent using Langchain and the Gmail Toolkit'
    email_agent = create_agent(
        llm=agent_llm, 
        tools=email_tools,
        system_prompt='''
        You are an expert Email Assistant.
        Your primary job is to send emails based on the user's instructions.
        
        Important Instructions:
        1. Always create a draft first if the user asks to "draft" an email.
        2. Use the `send_message` tool if the user explicitly asks to SEND.
        3. If the user provides a recipient (to), subject, and body, use them exactly.''')

    return email_agent

#Initialize the email agent
email_agent_executor=get_email_agent()

#Define function to send email  
def send_email(state: AgentState) -> AgentState:
    """Send an email using LangChain agent"""

    #Get the query
    query=state["query"]

    #Get the email agent
    agent=email_agent_executor

    # Create a message from the query
    user_message = HumanMessage(
        content=f"Send an email according to the user's instructions: {query}"
    )

    # Invoke the agent with messages
    result=agent.invoke({"messages": [user_message]})

    # Extract the answer from the result
    messages=result.get("messages", [])
    if messages:
        last_message=messages[-1]
        answer=last_message.content if hasattr(last_message, 'content') else str(last_message)
    else:
        answer="No response from email agent"
    
    return {**state, "answer": answer}


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

    #Set the entry point
    workflow.set_entry_point("Check Cache")

    #Add Edges to the graph
    workflow.add_conditional_edges("Check Cache", lambda state: END if state.get("cache_hit", False) else "Review Query")
    workflow.add_conditional_edges("Review Query", lambda state: "Ask for More Info" if state.get("need_more_info", False) else "Define Agent")
    workflow.add_edges("Ask for More Info", "Review Query")
    workflow.add_conditional_edges("Define Agent", lambda state: "Find Places Nearby" if state.get("agent", "") == "find_places" else "Search Web" if state.get("agent", "") == "search_web" else "Send Email" if state.get("agent", "") == "send_email" else END)
    workflow.add_edges("Find Places Nearby", END)
    workflow.add_edges("Search Web", END)
    workflow.add_edges("Send Email", END)

    #Compile the graph
    app=workflow.compile()

    return app

#Initialize the graph once at startup
_graph=create_graph()

#Define the function to run the graph
def process_query(query: str) -> str:
    """Process the query through the graph"""

    #Deifne the initial state
    initial_state={
        "query": query,
        "cache_hit": False,
        "answer": "",
        "agent": "",
        "need_more_info": False,
        "more_info_query": ""
    }

    #Run the graph
    result=_graph.invoke(initial_state)

    return result.get("answer", "")

#Define main to test the pipeline
def main():
    """Main function to test the pipeline"""

    query=input("Enter a query to process: ")
    result=process_query(query)

    print(f"Answer: {result}")

if __name__ == "__main__":
    main()