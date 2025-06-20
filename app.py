import os
import base64
import io
import logging
import time
from typing import Optional, List, Dict, Any
import hashlib

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Configure matplotlib before other imports for Streamlit Cloud compatibility
import matplotlib
matplotlib.use('Agg')

from instruction import INSTRUCTIONS
from code_executor import SecureCodeExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Authentication Configuration
VALID_USERS = {
    "admin": "adminadmin123",  # Change these credentials in production
}

def hash_password(password: str) -> str:
    """Hash password for secure comparison (for future enhancement)"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(username: str, password: str) -> bool:
    """Verify username and password"""
    if username in VALID_USERS:
        return VALID_USERS[username] == password
    return False

def login_form():
    """Display login form"""
    st.markdown("# 🔐 Login to InsightBot")
    st.markdown("*Please enter your credentials to access the AI data analysis platform*")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### 👤 User Authentication")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            login_button = st.form_submit_button("🔑 Login", use_container_width=True)
            
            if login_button:
                if username and password:
                    if verify_credentials(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.error("⚠️ Please enter both username and password")

def logout():
    """Handle user logout"""
    st.session_state.authenticated = False
    st.session_state.username = None
    # Clear all session data
    for key in list(st.session_state.keys()):
        if key not in ['authenticated', 'username']:
            del st.session_state[key]
    st.rerun()

# Initialize OpenAI Client (official API, GPT-4o nano)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Streamlit Page
st.set_page_config(
    page_title="InsightBot - AI Data Analysis", 
    page_icon="🔍", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

# Check authentication before showing main app
if not st.session_state.authenticated:
    login_form()
    st.stop()

# Initialize all session states
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "code_executor" not in st.session_state:
    st.session_state.code_executor = SecureCodeExecutor()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Helper functions
def upload_file_locally(file) -> bool:
    """Upload file and load into the code executor"""
    try:
        file_content = file.read()
        success = st.session_state.code_executor.load_dataframe(file_content, file.name)
        if success:
            st.session_state.uploaded_files.append({
                'name': file.name,
                'size': len(file_content),
                'uploaded_at': time.time()
            })
            logger.info("File uploaded successfully: %s", file.name)
            return True
        return False
    except Exception as e:
        logger.error("File upload failed: %s", e)
        st.error(f"File upload failed: {e}")
        return False

def delete_all_files():
    """Clear all uploaded files"""
    st.session_state.uploaded_files = []
    st.session_state.code_executor = SecureCodeExecutor()
    logger.info("All files deleted and executor reset.")

def generate_llm_response(user_message: str, context: str = "") -> str:
    """Generate response using OpenAI GPT-4o nano model"""
    try:
        # Prepare the conversation context
        system_prompt = f"""
{INSTRUCTIONS}

You are analyzing a dataset. Here's the current context:
{context}

Respond naturally as a data analyst. Focus on insights and explanations. 
If a visualization would help, include ONLY the plotting code (matplotlib/seaborn).
Never mention code execution or that you're generating code.
Make visualizations feel like a natural part of your explanation.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history (last 5 exchanges to keep context manageable)
        recent_history = st.session_state.conversation_history[-10:]  # Last 5 exchanges
        messages.extend(recent_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            stream=False,
            max_tokens=2048,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error("LLM response generation failed: %s", e)
        return f"Sorry, I encountered an error generating a response: {str(e)}"

def parse_response_with_inline_visualizations(response_text: str):
    """Parse response text and create inline content segments with visualizations"""
    import re
    
    # Pattern to match code blocks with their positions
    pattern = r'```(?:python)?\s*(.*?)```'
    
    content_segments = []
    last_end = 0
    
    # Find all code blocks and their positions
    for match in re.finditer(pattern, response_text, re.DOTALL):
        start_pos = match.start()
        end_pos = match.end()
        code_content = match.group(1).strip()
        
        # Add text before this code block
        if start_pos > last_end:
            text_before = response_text[last_end:start_pos].strip()
            if text_before:
                content_segments.append({
                    'type': 'text',
                    'content': text_before
                })
        
        # Execute the code block if it's visualization code
        if code_content and st.session_state.code_executor.is_visualization_code(code_content):
            logger.info("Executing visualization code silently")
            try:
                output, image_data = st.session_state.code_executor.execute_code(code_content)
                if image_data:
                    content_segments.append({
                        'type': 'image',
                        'content': image_data
                    })
                else:
                    # If execution failed, add a more user-friendly error message
                    logger.warning(f"Visualization failed: {output}")
                    content_segments.append({
                        'type': 'text',
                        'content': f"*[Unable to generate visualization: {output.split(':')[-1].strip() if ':' in output else 'Technical issue with chart generation'}]*"
                    })
            except Exception as e:
                logger.error(f"Visualization execution error: {e}")
                content_segments.append({
                    'type': 'text',
                    'content': "*[Visualization temporarily unavailable due to technical issues]*"
                })
        else:
            # Non-visualization code - show as error (shouldn't happen with proper instructions)
            content_segments.append({
                'type': 'text',
                'content': f"*[Non-visualization code block removed]*"
            })
        
        last_end = end_pos
    
    # Add remaining text after the last code block
    if last_end < len(response_text):
        remaining_text = response_text[last_end:].strip()
        if remaining_text:
            content_segments.append({
                'type': 'text',
                'content': remaining_text
            })
    
    # If no code blocks found, return the entire text as one segment
    if not content_segments:
        content_segments.append({
            'type': 'text',
            'content': response_text
        })
    
    return content_segments

def process_user_message(user_message: str) -> Dict[str, Any]:
    """Process user message and execute any code if needed with inline visualization placement"""
    # Get dataset context
    context = st.session_state.code_executor.get_dataframe_info()
    
    # Generate LLM response
    llm_response = generate_llm_response(user_message, context)
    
    # Parse response and create inline content segments
    content_segments = parse_response_with_inline_visualizations(llm_response)
    
    # Check if any visualizations were generated
    has_visualizations = any(segment['type'] == 'image' for segment in content_segments)
    
    result = {
        'content_segments': content_segments,
        'has_visualization': has_visualizations,
        'original_response': llm_response  # Keep for conversation history
    }
    
    # Update conversation history with original response (including code for context)
    st.session_state.conversation_history.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": llm_response}
    ])
    
    return result

# Sidebar for File Upload and Management
st.sidebar.markdown("# 🔍 InsightBot")
st.sidebar.markdown("*AI-powered data analysis through natural conversation*")
st.sidebar.markdown(f"**👤 Welcome, {st.session_state.username}!**")

# Logout button in sidebar
if st.sidebar.button("🚪 Logout", use_container_width=True):
    logout()

st.sidebar.markdown("")

st.sidebar.header("📁 File Management")
st.sidebar.markdown("Upload your CSV file to start analyzing your data with AI insights.")

file_uploaded = st.sidebar.file_uploader(
    "Choose a CSV file", 
    type=["csv"],
    help="Upload a CSV file to begin your data analysis conversation"
)

# Upload Button
if st.sidebar.button("📤 Upload File", use_container_width=True):
    if file_uploaded:
        if upload_file_locally(file_uploaded):
            st.sidebar.success(f"✅ File '{file_uploaded.name}' uploaded successfully!")
            st.session_state.start_chat = True
        else:
            st.sidebar.error("❌ Failed to upload file.")
    else:
        st.sidebar.error("⚠️ Please select a file to upload.")

st.sidebar.markdown("")

# Display uploaded files
if st.session_state.uploaded_files:
    st.sidebar.markdown("### 📊 Current Dataset")
    for file_info in st.session_state.uploaded_files:
        file_size_kb = round(file_info['size'] / 1024, 1)
        st.sidebar.markdown(f"📄 **{file_info['name']}**")
        st.sidebar.markdown(f"   📏 Size: {file_size_kb} KB")
    st.sidebar.markdown("")

# Dataset Info Button
if st.session_state.uploaded_files and st.sidebar.button("📈 View Dataset Info", use_container_width=True):
    info = st.session_state.code_executor.get_dataframe_info()
    st.sidebar.text_area("Dataset Information", info, height=200)

# Button to Delete All Files
if st.session_state.uploaded_files:
    st.sidebar.markdown("")
    if st.sidebar.button("🗑️ Delete All Files", use_container_width=True):
        delete_all_files()
        st.session_state.start_chat = False
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.sidebar.success("✅ All files deleted and chat reset.")

# Reset Chat Button
if st.session_state.messages:
    st.sidebar.markdown("")
    if st.sidebar.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        logger.info("Chat history cleared.")
        st.sidebar.success("✅ Chat history cleared.")

# Main Chat Section
st.title("🔍 InsightBot")
st.markdown("*AI-Powered Data Analysis through Natural Conversation*")


if not st.session_state.uploaded_files:
    st.info("👆 Please upload a CSV file in the sidebar to start analyzing your data!")
    
    st.markdown("## Welcome to InsightBot! 🤖")
    st.markdown("""
    **InsightBot** is a chatbot-based web application that enables users to interact with their data intuitively. 
    Users can upload CSV files, ask natural language questions about the content, receive insights generated via AI, 
    visualize data trends in charts, and even export insights — all within a conversational chat interface.
    
    ### 🚀 How to get started:
    1. **📤 Upload** a CSV file using the sidebar
    2. **💬 Ask** questions about your data in natural language
    3. **📊 Get** insights and visualizations in a natural conversation flow
    4. **🔍 Explore** deeper with follow-up questions
    
    ### 💡 Example questions you can ask:
    - *"What does this dataset look like?"*
    - *"Show me the distribution of key variables"*
    - *"How are different columns related?"*
    - *"What patterns do you see in the data?"*
    - *"Are there any interesting trends over time?"*
    - *"Can you identify any outliers or anomalies?"*
    
    **Ready to unlock insights from your data? Upload a file to begin! 🎯**
    """)

# Display Messages
for message in st.session_state.messages:
    if message["type"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["type"] == "assistant":
        with st.chat_message("assistant"):
            # Handle new content segments structure for inline visualization
            if "content_segments" in message:
                for segment in message["content_segments"]:
                    if segment["type"] == "text":
                        st.markdown(segment["content"])
                    elif segment["type"] == "image":
                        try:
                            image_bytes = base64.b64decode(segment["content"])
                            st.image(image_bytes, use_container_width=True)
                        except Exception as e:
                            logger.error(f"Error displaying inline image: {e}")
                            st.warning("⚠️ Visualization could not be displayed due to technical issues.")
            else:
                # Fallback for old message format (backward compatibility)
                st.markdown(message["content"])
                if message.get("image"):
                    try:
                        image_bytes = base64.b64decode(message["image"])
                        st.image(image_bytes, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error displaying image: {e}")

# User Input and Chat Process
if st.session_state.start_chat:
    if prompt := st.chat_input("💬 Ask me anything about your data! What insights would you like to discover?"):
        # Add User Message
        st.session_state.messages.append({"type": "user", "content": prompt})
        logger.info("User input received: %s", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the message
        result = process_user_message(prompt)
        
        # Prepare assistant message with new content segments structure
        assistant_message = {
            "type": "assistant",
            "content_segments": result['content_segments'],
            "has_visualization": result['has_visualization']
        }
        
        # Add assistant message to session
        st.session_state.messages.append(assistant_message)
        
        # Display assistant response with inline visualizations
        with st.chat_message("assistant"):
            for segment in result['content_segments']:
                if segment["type"] == "text":
                    st.markdown(segment["content"])
                elif segment["type"] == "image":
                    try:
                        image_bytes = base64.b64decode(segment["content"])
                        st.image(image_bytes, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error displaying inline image: {e}")
                        st.warning("⚠️ Visualization could not be displayed due to technical issues.")