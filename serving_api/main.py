from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..pipeline.agent import process_query, resume_query
import asyncio
from logging import getLogger
from contextlib import asynccontextmanager
import os
from pathlib import Path


#Logger for the serving API
logger = getLogger(__name__)

# Global ready flag
ready = False

# Response models
class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: Optional[str] = None
    interrupt: Optional[bool] = False
    message: Optional[str] = None 
    thread_id: Optional[str] = None 

class ResumeRequest(BaseModel):
    user_response: str
    thread_id: str
    config: dict  # Config from the interrupt response

# Lifespan for the serving API
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ready
    try:
        # Test that the graph is compiled and ready
        logger.info("Initializing agent pipeline...")
        # The graph is compiled at module import, so if we get here, it's ready
        ready = True
        logger.info("Agent pipeline loaded successfully (graph compiled at module import)")
    except Exception as e:
        ready = False 
        logger.error(f"Error initializing agents: {e}")
        print(f"LIFESPAN ERROR: {e}")
        raise Exception(f"Error initializing agents: {e}")
        

    yield

    #Shutdown cleanup
    logger.info("Shutting down agent pipeline")
    logger.info("Agent pipeline shut down successfully")

#Create the FastAPI app
app=FastAPI(lifespan=lifespan,
title="Agentic RAG",
description="A simple API for the copilot AI agent",
version="1.0.0")

@app.get("/health")
async def get_health():
    if ready:
        return Response(content="OK", status_code=200)
    else:
        return Response(content="NOT OK", status_code=500)

@app.post("/query", response_model=QueryResponse)
async def answer_query(request: QueryRequest):
    """
    Process a user query through the agent pipeline.
    
    Returns either:
    - A complete answer if the query can be processed without additional input
    - An interrupt response if more information is needed from the user
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Run sync function in thread pool
        result = await asyncio.to_thread(process_query, request.query, request.thread_id)
        
        # Check if result is an interrupt
        if isinstance(result, dict) and result.get("__interrupt__"):
            return QueryResponse(
                interrupt=True,
                message=result.get("message", "Need more information"),
                thread_id=result.get("thread_id"),
                answer=None
            )
        
        # Otherwise, it's a complete answer
        return QueryResponse(
            answer=result if isinstance(result, str) else str(result),
            interrupt=False,
            message=None,
            thread_id=None
        )
    
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        print(f"QUERY ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resume", response_model=QueryResponse)
async def resume_query_endpoint(request: ResumeRequest):
    """
    Resume a query after an interrupt.
    
    Use this endpoint when you receive an interrupt response from /query.
    Provide the user's response to the interrupt question along with the thread_id and config.
    """
    try:
        if not request.user_response or not request.user_response.strip():
            raise HTTPException(status_code=400, detail="User response cannot be empty")
        
        if not request.thread_id:
            raise HTTPException(status_code=400, detail="Thread ID is required")
        
        if not request.config:
            raise HTTPException(status_code=400, detail="Config is required")
        
        # Run sync function in thread pool
        result = await asyncio.to_thread(
            resume_query,
            request.user_response,
            request.thread_id,
            request.config
        )
        
        # Check if result is another interrupt
        if isinstance(result, dict) and result.get("__interrupt__"):
            return QueryResponse(
                interrupt=True,
                message=result.get("message", "Need more information"),
                thread_id=result.get("thread_id"),
                answer=None
            )
        
        # Otherwise, it's a complete answer
        return QueryResponse(
            answer=result if isinstance(result, str) else str(result),
            interrupt=False,
            message=None,
            thread_id=None
        )
    
    except Exception as e:
        logger.error(f"Error resuming query: {e}")
        print(f"RESUME ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# Handle HTTP exceptions with consistent error format
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format"""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return Response(status_code=exc.status_code, content=f"HTTP error: {exc.status_code} - {exc.detail}")