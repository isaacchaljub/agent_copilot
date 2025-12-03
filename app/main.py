import streamlit as st
import requests

#Create the Streamlit app

URL = "http://localhost:8000"

# Initialize session state for interrupt handling
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "config" not in st.session_state:
    st.session_state.config = None
if "interrupt_message" not in st.session_state:
    st.session_state.interrupt_message = None
if "waiting_for_input" not in st.session_state:
    st.session_state.waiting_for_input = False

st.title("Assistant Copilot")

st.subheader("Welcome to the your copilot agent app")

def check_health():
    try:
        response = requests.get(URL + "/health")
        if response.status_code == 200:
            st.success("Health check passed")
        else:
            st.error("Health check failed")
    except Exception as e:
        st.error(f"Error checking health: {e}")

if st.button("Check Health", key="health_button"):
    check_health()

def answer_query(query, thread_id=None):
    """Process a query and handle interrupts"""
    try:
        payload = {"query": query}
        if thread_id:
            payload["thread_id"] = thread_id
        
        response = requests.post(URL + "/query", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if interrupt occurred
            if data.get("interrupt", False):
                # Store interrupt info in session state
                st.session_state.thread_id = data.get("thread_id")
                st.session_state.interrupt_message = data.get("message", "Need more information")
                st.session_state.waiting_for_input = True
                st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Show interrupt message
                st.warning(f"⚠️ {st.session_state.interrupt_message}")
                return None
            else:
                # Complete answer received
                answer = data.get("answer", "")
                # Clear interrupt state
                st.session_state.waiting_for_input = False
                st.session_state.thread_id = None
                st.session_state.config = None
                st.session_state.interrupt_message = None
                return answer
        else:
            st.error(f"Error answering query: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error answering query: {e}")
        return None

def resume_query(user_response):
    """Resume query after interrupt"""
    try:
        if not st.session_state.thread_id or not st.session_state.config:
            st.error("Missing thread information. Please start a new query.")
            return None
        
        payload = {
            "user_response": user_response,
            "thread_id": st.session_state.thread_id,
            "config": st.session_state.config
        }
        
        response = requests.post(URL + "/resume", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if another interrupt occurred
            if data.get("interrupt", False):
                # Update interrupt info
                st.session_state.thread_id = data.get("thread_id")
                st.session_state.interrupt_message = data.get("message", "Need more information")
                st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Show interrupt message
                st.warning(f"⚠️ {st.session_state.interrupt_message}")
                return None
            else:
                # Complete answer received
                answer = data.get("answer", "")
                # Clear interrupt state
                st.session_state.waiting_for_input = False
                st.session_state.thread_id = None
                st.session_state.config = None
                st.session_state.interrupt_message = None
                return answer
        else:
            st.error(f"Error resuming query: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error resuming query: {e}")
        return None

# Main UI
if st.session_state.waiting_for_input:
    # Show interrupt UI
    st.info("🔔 The agent needs more information to continue.")
    
    with st.form("resume_form", clear_on_submit=True):
        st.write(f"**Question:** {st.session_state.interrupt_message}")
        user_response = st.text_input("Your response:", key="resume_input")
        submitted = st.form_submit_button("Continue")
        
        if submitted:
            if user_response:
                with st.spinner("Processing your response..."):
                    answer = resume_query(user_response)
                    if answer:
                        st.success("✅ Answer received!")
                        st.write(answer)
            else:
                st.error("Please provide a response")
    
    # Option to cancel and start new query
    if st.button("Cancel & Start New Query", key="cancel_button"):
        st.session_state.waiting_for_input = False
        st.session_state.thread_id = None
        st.session_state.config = None
        st.session_state.interrupt_message = None
        st.rerun()

else:
    # Normal query form
    with st.form("query_form", clear_on_submit=False):
        query = st.text_input("Enter your query here", key="query_form_input")
        submitted = st.form_submit_button("Answer Query")
        
        if submitted:
            if query:
                with st.spinner("Processing your query..."):
                    answer = answer_query(query)
                    if answer:
                        st.success("✅ Answer received!")
                        st.write(answer)
            else:
                st.error("Please enter a query")
