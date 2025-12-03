# Assistant Copilot

An intelligent agentic system that can understand user queries, determine if additional information is needed, select appropriate tools, and execute tasks using multiple specialized agents. Built with LangGraph, LangChain, and FastAPI.

## 🚀 Features

- **Multi-Agent System**: Specialized agents for different tasks (places search, web search, email)
- **Intelligent Query Analysis**: Automatically determines if a query has enough information
- **Human-in-the-Loop**: Asks for clarification when information is missing using LangGraph interrupts
- **Semantic Caching**: Caches previous queries and answers for faster responses
- **State Persistence**: Uses checkpointer to maintain conversation state
- **Multiple Interfaces**:
  - CLI interface for direct interaction
  - FastAPI REST API for programmatic access
  - Streamlit web interface for user-friendly interaction

## 🏗️ Architecture

The system uses a **LangGraph workflow** with the following pipeline:

```
User Query
    ↓
[Check Cache] → If hit → Return cached answer
    ↓ (if miss)
[Review Query] → Check if enough information
    ↓
    ├─→ Need More Info → [Ask for More Info] → Wait for user → Loop back
    ↓
[Define Agent] → Choose appropriate agent:
    ├─→ find_places → [Find Places Nearby]
    ├─→ search_web → [Search Web]
    └─→ send_email → [Send Email]
    ↓
[Generate Answer] → Format final response
    ↓
Return Answer
```

### Agents

1. **Places Finder Agent**: Finds nearby places using Google Maps API
2. **Web Search Agent**: Searches and summarizes web information using Tavily
3. **Email Agent**: Sends emails using Gmail API

## 📋 Prerequisites

- Python 3.12 or higher
- `uv` package manager (recommended) or `pip`
- API Keys for:
  - Groq (for LLM)
  - Google Gemini (for agent execution)
  - Tavily (for web search)
  - Google Maps (for places search)
  - Google OAuth (for Gmail - optional)

## 🔧 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd agent_copilot
```

### 2. Create virtual environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using Python
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
# LLM API Keys
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Search API
TAVILY_API_KEY=your_tavily_api_key_here

# Google Services
MAPS_API_KEY=your_google_maps_api_key_here

# Gmail (optional - for email functionality)
# OAuth credentials will be created on first run
```

### 5. Configure Google OAuth (for Gmail)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project or select existing one
3. Enable Gmail API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download `credentials.json` and place it in project root
6. Add your email as a test user in OAuth consent screen
7. On first run, the app will open a browser for authentication

## 🚀 Usage

### CLI Interface

Run the agent directly from the command line:

```bash
python pipeline/agent.py
```

Example interactions:

```
Enter a query to process: What is the highest city in Colombia?
Answer: The highest city in Colombia is...

Enter a query to process: Find me bars
Please provide the following information to answer the question: The query is missing the city and country.
Your response: in Madrid, Spain
Answer: Here are bars in Madrid...
```

### FastAPI Server

Start the API server:

```bash
uvicorn serving_api.main:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**POST `/query`**

- Process a query
- Request body:
  ```json
  {
    "query": "What is Agentic RAG?",
    "thread_id": "optional-thread-id"
  }
  ```
- Response (complete answer):
  ```json
  {
    "answer": "Agentic RAG is...",
    "interrupt": false,
    "message": null,
    "thread_id": null
  }
  ```
- Response (interrupt):
  ```json
  {
    "answer": null,
    "interrupt": true,
    "message": "The query is missing the city and country.",
    "thread_id": "abc-123-def-456"
  }
  ```

**POST `/resume`**

- Resume after an interrupt
- Request body:
  ```json
  {
    "user_response": "in Madrid, Spain",
    "thread_id": "abc-123-def-456",
    "config": {
      "configurable": {
        "thread_id": "abc-123-def-456"
      }
    }
  }
  ```
- Response: Same format as `/query`

**GET `/health`**

- Health check endpoint
- Returns `200 OK` if service is ready

### Streamlit Web Interface

Start the Streamlit app:

```bash
streamlit run app/main.py
```

The web interface will open at `http://localhost:8501`

Features:

- Interactive query input
- Automatic interrupt handling
- Visual feedback for answers and errors
- Health check button

## 📁 Project Structure

```
agent_copilot/
├── pipeline/
│   └── agent.py          # Main agent pipeline with LangGraph workflow
├── serving_api/
│   └── main.py           # FastAPI REST API server
├── app/
│   └── main.py           # Streamlit web interface
├── pyproject.toml         # Project dependencies and configuration
├── uv.lock                # Locked dependencies (if using uv)
├── .env                   # Environment variables (not in git)
├── credentials.json       # Google OAuth credentials (not in git)
├── token.json             # OAuth token (not in git)
└── README.md              # This file
```

## 🔑 API Keys Setup

### Groq API Key

1. Sign up at [console.groq.com](https://console.groq.com/)
2. Create an API key
3. Add to `.env` as `GROQ_API_KEY`

### Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. **Important**: In Google Cloud Console, set Application restrictions to "None" (not HTTP referrers) for CLI/server use
4. Add to `.env` as `GEMINI_API_KEY`

### Tavily API Key

1. Sign up at [tavily.com](https://tavily.com/)
2. Get your API key
3. Add to `.env` as `TAVILY_API_KEY`

### Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Maps JavaScript API and Places API
3. Create an API key
4. Add to `.env` as `MAPS_API_KEY`

## 💡 Example Queries

### Places Search

```
Find me bars near Calle de Clara del Rey, 85, in Madrid, Spain
```

### Web Search

```
What is Agentic RAG?
What is the population of China?
```

### Email

```
Send an email to example@gmail.com with subject "Meeting" and body "We need to discuss the project"
```

### Queries Requiring More Info

```
Find me bars          → Will ask for location
Send an email         → Will ask for recipient, subject, body
```

## 🛠️ Development

### Running Tests

```bash
# Test the pipeline directly
python pipeline/agent.py

# Test the API
uvicorn serving_api.main:app --reload
# Then test with curl or Postman

# Test the Streamlit app
streamlit run app/main.py
```

### Code Structure

- **`pipeline/agent.py`**: Core agent logic

  - State management with `AgentState` TypedDict
  - Graph workflow with conditional routing
  - Interrupt handling for user input
  - Semantic caching implementation
- **`serving_api/main.py`**: FastAPI server

  - RESTful endpoints for query processing
  - Interrupt handling via `/resume` endpoint
  - Async execution using thread pool
- **`app/main.py`**: Streamlit interface

  - Session state management
  - Interactive interrupt handling
  - User-friendly UI

## 🐛 Troubleshooting

### Gmail OAuth Issues

**Error: "Access blocked: Python has not completed the Google verification process"**

- Solution: Add your email as a test user in Google Cloud Console → OAuth consent screen → Test users

### Gemini API Key Issues

**Error: "API_KEY_HTTP_REFERRER_BLOCKED"**

- Solution: In Google Cloud Console, edit your API key and set "Application restrictions" to "None" (not "HTTP referrers")

### Import Errors

**Error: "cannot import from pipeline.agent"**

- Solution: Make sure you're running from the project root directory
- Or use: `python -m pipeline.agent` instead of `python pipeline/agent.py`

### Interrupt Not Working

- Make sure checkpointer is enabled (it is by default)
- Verify thread_id is being passed correctly
- Check that `interrupt()` is being called in the node

## 📚 Technologies Used

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: Agent framework and tool integration
- **FastAPI**: REST API server
- **Streamlit**: Web interface
- **Groq**: Fast LLM inference
- **Google Gemini**: Agent execution LLM
- **Tavily**: Web search API
- **Google Maps**: Places search
- **Gmail API**: Email functionality
