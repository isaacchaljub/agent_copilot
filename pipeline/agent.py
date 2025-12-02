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
from langchain_litellm import ChatLiteLLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from numpy.linalg import norm
from numpy import dot
import numpy as np
import googlemaps
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
web_llm = ChatLiteLLM(
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
    tools: list[str]
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
def define_tools(state: AgentState) -> AgentState:
    """Use LLM to define tools"""
    query = state["query"]
    
    prompt = '''Role: AI Agent Tool Definition Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer. 
Output Format:
    - Answer: Yes/No
Study the below examples and based on that, respond to the last question. 
Examples:
    Input: 
        Text: The capital of France is Paris.
        User Question: What is the capital of France?
    Expected Output:
        Answer: Yes
    Input: 
        Text: The population of the United States is over 330 million.
        User Question: What is the population of China?
    Expected Output:
        Answer: No
    Input:
        User Question: {query}
        Text: {text}
'''
    formatted_prompt = prompt.format(text=local_context, query=query)
    response = llm.invoke(formatted_prompt)
    can_answer_locally = response.content.strip().lower() == "yes"
    
    return {**state, "can_answer_locally": can_answer_locally}