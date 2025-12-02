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

# Load the environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("MAPS_API_KEY")

# Define global variables
cache=[]
cache_size=20
similarity_threshold=0.85
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
    is_cache_good:bool
    final_answer: str
    agent: str
    need_more_info: bool
    more_info_query: str
    more_info_answer: str

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


#Define function to define tools
def define_agent(state: AgentState) -> AgentState:
    """Use LLM to define agent"""
    query = state["query"]
    
    prompt = '''Role: AI Agent Definition Assistant
Task: Define the agent for the user's question.
Instructions:
    - Analyze the user's question and identify the agent that is needed to answer the question.
    - Provide a clear and concise response indicating the agent that is needed to answer the question.
    - Your response should include only the agent that is needed to answer the question. Nothing else, no other text, information, header/footer. 
    - The agents available are:
        - find_places_nearby: to find places nearby
        - web_search: to search and scrape the web
        - mail_sender: to send an email
Output Format:
    - agent: name of the agent
Examples:
    Input: 
        User Query: Find me bars that are close to Calle de Clara del Rey, 5, in Madrid, Spain.
    Expected Output:
        agent: places_finder
    Input: 
        User Query: What is the population of China?
    Expected Output:
        agent: web_searcher
    Input: 
        User Query: Send an email to John Doe with the subject "Meeting" and the body "We need to discuss the project".
    Expected Output:
        agent: email_sender
    Input:
        User Query: {query}
        agent: {agent}
'''
    formatted_prompt = prompt.format(query=query)
    response = tool_llm.invoke(formatted_prompt)
    agent = response.content.strip().lower()
    return {**state, "agent": agent}


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























































#     prompt = '''Role: AI Agent to use googlemaps Places Nearby Tool
# Task: Extract the address and places to search for from the user's query.
# Instructions:
#     - Read the user's query and determine the location and places to search for.
#     - Extract the address from the user's query
#     - Extract the places to search for from the user's query
# Output Format:
#     - address: address of the location
#     - places: places to search for
# Examples:
#     Input: 
#         User Query: Find me bars that are close to Calle de Clara del Rey, 5, in Madrid, Spain.
#     Expected Output:
#         address: Calle de Clara del Rey, 5, Madrid, Spain
#         places: bars
#     Input: 
#         User Query: Find me restaurants that are close to Calle Hermosilla, 87, in Madrid, Spain.
#     Expected Output:
#         address: Calle Hermosilla, 87, Madrid, Spain
#         places: restaurants
#     Input: 
#         User Query: Find me hotels that are close to Carrera 7 #16-50, in Pereira, Colombia.
#     Expected Output:
#         address: Carrera 7 #16-50, Pereira, Colombia
#         places: hotels
#     Input: 
#         User Query: {query}
#     Expected Output:
#         address: {address}
#         places: {places}
#     '''