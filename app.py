"""
InsightBot: AI-Powered Data Analysis Chatbot

This is the main application file for InsightBot, a Streamlit-based web application
that enables users to interact with their data through natural language conversations.
The application provides AI-powered insights, visualizations, and analysis capabilities
through a secure and user-friendly chat interface.

Author: InsightBot Team
Version: 2.0
Created: 2025
"""

# Standard library imports
import os
import base64
import io
import logging
import time
import re
from typing import Optional, List, Dict, Any
import hashlib
from datetime import datetime

# Third-party imports
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Configure matplotlib before other imports for Streamlit Cloud compatibility
import matplotlib
matplotlib.use('Agg')

# Local imports
from instruction import INSTRUCTIONS  # AI prompt instructions
from code_executor import SecureCodeExecutor  # Secure Python code execution
from chat_storage import SQLiteChatStorage  # Database operations

# ReportLab imports for PDF export functionality
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from PIL import Image as PILImage

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# ========================
# AUTHENTICATION SYSTEM
# ========================

# Authentication Configuration - Simple username/password system
# TODO: In production, replace with proper authentication system (OAuth, LDAP, etc.)
VALID_USERS = {
    "ihsankhan": "ergb4809uibewf",  # Change these credentials in production
}

def hash_password(password: str) -> str:
    """
    Hash password using SHA-256 for secure comparison.
    
    Args:
        password (str): Plain text password to hash
        
    Returns:
        str: Hashed password in hexadecimal format
        
    Note:
        This is a basic implementation for demonstration.
        In production, use bcrypt or similar with salt.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(username: str, password: str) -> bool:
    """
    Verify username and password against stored credentials.
    
    Args:
        username (str): Username to verify
        password (str): Password to verify
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    if username in VALID_USERS:
        return VALID_USERS[username] == password
    return False

def login_form():
    """
    Display the login form interface for user authentication.
    
    Creates a centered login form with username and password fields.
    Handles authentication validation and session state management.
    """
    # Hero Section with Project Branding
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem; color: white;'>
        <h1 style='margin: 0; font-size: 3rem; font-weight: bold;'>🔍 InsightBot</h1>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>AI-Powered Data Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login Section
    st.markdown("## 🔐 Sign In to Continue")
    st.markdown("*Please enter your credentials to access the AI data analysis platform*")
    
    # Create centered columns for better UI layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        
        with st.form("login_form"):
            st.markdown("#### 👤 User Authentication")
            st.markdown("*Enter your credentials to access the platform*")
            
            username = st.text_input(
                "Username", 
                placeholder="Enter your username",
                help="Use your assigned username to access the platform"
            )
            password = st.text_input(
                "Password", 
                type="password", 
                placeholder="Enter your password",
                help="Enter your secure password"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            login_button = st.form_submit_button(
                "🔑 Sign In to InsightBot", 
                use_container_width=True,
                type="primary"
            )
            
            if login_button:
                if username and password:
                    if verify_credentials(username, password):
                        # Set authentication state
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        # Set query param for session persistence across browser refreshes
                        st.query_params["auth_user"] = username
                        st.success("✅ Login successful! Redirecting to your dashboard...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")
                else:
                    st.error("⚠️ Please enter both username and password")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Welcome message and features - Moved below login
    st.markdown("---")
    st.markdown("## About InsightBot 🚀")
    st.markdown("""
    Transform your data into actionable insights through natural conversation. Upload your datasets and let AI guide you through comprehensive analysis.
    """)
    
    # Feature highlights in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; border: 2px solid #ffffff; border-radius: 10px; margin-bottom: 1rem; background: transparent;'>
            <h3>💬 Chat Interface</h3>
            <p style='font-size: 0.9rem; color: #666;'>Ask questions in natural language and get intelligent responses</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; border: 2px solid #ffffff; border-radius: 10px; margin-bottom: 1rem; background: transparent;'>
            <h3>📊 Smart Visualizations</h3>
            <p style='font-size: 0.9rem; color: #666;'>Automatic charts and graphs based on your data patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; border: 2px solid #ffffff; border-radius: 10px; margin-bottom: 1rem; background: transparent;'>
            <h3>📈 Deep Insights</h3>
            <p style='font-size: 0.9rem; color: #666;'>Discover hidden patterns and trends in your datasets</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem 0;'>
        <p><strong>Supported File Formats:</strong> CSV, Excel (.xlsx/.xls), JSON, XML, TXT, PDF</p>
        <p><strong>Features:</strong> Natural Language Queries • Automatic Visualizations • Statistical Analysis • Export Reports</p>
        <p style='margin-top: 1rem; font-size: 0.8rem;'>InsightBot v2.0 - AI-Powered Data Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

def logout():
    """
    Handle user logout and clean up session state.
    
    Clears all authentication data, session variables, and redirects to login.
    Ensures secure cleanup of sensitive information.
    """
    st.session_state.authenticated = False
    st.session_state.username = None
    # Clear query params for session persistence
    st.query_params.clear()
    # Clear all session data except authentication keys
    for key in list(st.session_state.keys()):
        if key not in ['authenticated', 'username']:
            del st.session_state[key]
    st.rerun()

# ========================
# CHAT MANAGEMENT SYSTEM
# ========================

def create_new_chat():
    """
    Create a new chat session and initialize its state.
    
    Saves current chat if it exists, creates new chat in database,
    and resets message history for fresh conversation.
    """
    # Save current chat if it exists and has messages
    if st.session_state.current_chat_id and st.session_state.messages:
        save_current_chat_messages()
    
    # Create new chat with metadata about uploaded files
    metadata = {
        "dataset_files": [f["name"] for f in st.session_state.uploaded_files],
        "created_from": "new_chat_button"
    }
    
    # Generate default chat title with timestamp
    chat_title = f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
    st.session_state.current_chat_id = st.session_state.chat_storage.create_chat(
        st.session_state.username, 
        chat_title, 
        metadata
    )
    
    # Save file information if files are uploaded
    if st.session_state.uploaded_files:
        st.session_state.chat_storage.save_chat_files(
            st.session_state.current_chat_id,
            st.session_state.uploaded_files
        )
    
    # Clear current messages and conversation history for new chat
    st.session_state.messages = []
    st.session_state.conversation_history = []
    
    logger.info(f"Created new chat {st.session_state.current_chat_id}")

def save_current_chat_messages():
    """
    Save current chat messages to the database.
    
    Only saves new messages that haven't been saved yet to avoid duplicates.
    Handles both user and assistant messages with proper content formatting.
    """
    if not st.session_state.current_chat_id or not st.session_state.messages:
        return
    
    try:
        # Get existing message count from database to avoid duplicates
        existing_messages = st.session_state.chat_storage.load_chat_messages(st.session_state.current_chat_id)
        existing_count = len(existing_messages)
        current_count = len(st.session_state.messages)
        
        # Only save new messages that haven't been saved yet
        if current_count > existing_count:
            new_messages = st.session_state.messages[existing_count:]
            
            for message in new_messages:
                if message["type"] == "user":
                    st.session_state.chat_storage.save_message(
                        st.session_state.current_chat_id,
                        "user",
                        message["content"]
                    )
                elif message["type"] == "assistant":
                    # Extract text content from content segments for storage
                    content_segments = message.get("content_segments", [])
                    text_content = ""
                    for segment in content_segments:
                        if segment["type"] == "text":
                            text_content += segment["content"] + "\n"
                    
                    st.session_state.chat_storage.save_message(
                        st.session_state.current_chat_id,
                        "assistant",
                        text_content.strip(),
                        content_segments
                    )
            
            logger.info(f"Saved {len(new_messages)} new messages for chat {st.session_state.current_chat_id}")
        else:
            logger.info(f"No new messages to save for chat {st.session_state.current_chat_id}")
            
    except Exception as e:
        logger.error(f"Failed to save chat messages: {e}")
        st.error("Failed to save chat messages")

def load_chat(chat_id: int):
    """
    Load a specific chat and its messages from the database.
    
    Args:
        chat_id (int): Database ID of the chat to load
        
    Restores chat messages, files, and conversation history to session state.
    """
    try:
        # Save current chat first to avoid losing data
        if st.session_state.current_chat_id and st.session_state.messages:
            save_current_chat_messages()
        
        # Load chat info to verify it exists
        chat_info = st.session_state.chat_storage.get_chat_info(chat_id)
        if not chat_info:
            st.error("Chat not found")
            return
        
        # Load messages from database
        messages = st.session_state.chat_storage.load_chat_messages(chat_id)
        
        # Load associated files from database
        files = st.session_state.chat_storage.load_chat_files(chat_id)
        
        # Update session state with loaded data
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = messages
        
        # Rebuild conversation history for LLM context
        st.session_state.conversation_history = []
        for msg in messages:
            if msg["type"] == "user":
                st.session_state.conversation_history.append({
                    "role": "user", 
                    "content": msg["content"]
                })
            elif msg["type"] == "assistant":
                # Use the original content for conversation history
                st.session_state.conversation_history.append({
                    "role": "assistant", 
                    "content": msg["content"]
                })
        
        # Update uploaded files (though files themselves aren't reloaded into executor)
        st.session_state.uploaded_files = files
        
        logger.info(f"Loaded chat {chat_id} with {len(messages)} messages")
        st.success(f"Loaded chat: {chat_info['title']}")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Failed to load chat {chat_id}: {e}")
        st.error("Failed to load chat")

def delete_chat(chat_id: int):
    """
    Delete a chat from the database (soft delete).
    
    Args:
        chat_id (int): Database ID of the chat to delete
        
    Marks chat as inactive rather than permanently deleting data.
    """
    try:
        success = st.session_state.chat_storage.delete_chat(chat_id, st.session_state.username)
        if success:
            # If we're deleting the current chat, reset to new chat state
            if st.session_state.current_chat_id == chat_id:
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.session_state.conversation_history = []
            st.success("Chat deleted successfully")
        else:
            st.error("Failed to delete chat")
    except Exception as e:
        logger.error(f"Failed to delete chat {chat_id}: {e}")
        st.error("Failed to delete chat")

# ========================
# AI AND DATA PROCESSING
# ========================

# Initialize OpenAI Client for GPT-4o nano model
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========================
# STREAMLIT CONFIGURATION
# ========================

# Initialize Streamlit Page Configuration
st.set_page_config(
    page_title="InsightBot - AI Data Analysis", 
    page_icon="🔍", 
    layout="centered",  # Use centered layout for better UX
    initial_sidebar_state="expanded"  # Keep sidebar open by default
)

# ========================
# SESSION STATE INITIALIZATION
# ========================

# Initialize authentication state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

# Check for existing authentication in browser session
# This allows users to stay logged in across browser refreshes
if not st.session_state.authenticated:
    # Try to restore authentication from query params or session
    query_params = st.query_params
    if "auth_user" in query_params:
        stored_username = query_params["auth_user"]
        if stored_username in VALID_USERS:
            st.session_state.authenticated = True
            st.session_state.username = stored_username
            # Clean up the URL for security
            st.query_params.clear()
            st.rerun()

# Check authentication before showing main app
if not st.session_state.authenticated:
    login_form()
    st.stop()

# Initialize all other session states after authentication
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

# Initialize chat storage database connection
if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = SQLiteChatStorage()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "show_chat_manager" not in st.session_state:
    st.session_state.show_chat_manager = False

# ========================
# HELPER FUNCTIONS
# ========================
def upload_file_locally(file) -> bool:
    """
    Upload file and load it into the code executor for analysis.
    
    Args:
        file: Streamlit file upload object
        
    Returns:
        bool: True if upload successful, False otherwise
        
    Supports multiple file formats: CSV, Excel, JSON, XML, TXT, PDF
    """
    try:
        file_content = file.read()
        success = st.session_state.code_executor.load_dataframe(file_content, file.name)
        if success:
            # Store file metadata in session state
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
    """
    Clear all uploaded files and reset the code executor.
    
    Removes all file references and reinitializes the execution environment.
    """
    st.session_state.uploaded_files = []
    st.session_state.code_executor = SecureCodeExecutor()
    logger.info("All files deleted and executor reset.")

def generate_llm_response(user_message: str, context: str = "") -> str:
    """
    Generate AI response using OpenAI GPT-4o nano model.
    
    Args:
        user_message (str): User's question or request
        context (str): Dataset context information
        
    Returns:
        str: AI-generated response with analysis and/or code
        
    Combines user message with dataset context and conversation history
    to provide contextually relevant responses.
    """
    try:
        # Prepare the conversation context with dataset information
        system_prompt = f"""
{INSTRUCTIONS}

You are analyzing a dataset. Here's the current context:
{context}

Respond naturally as a data analyst. Focus on insights and explanations. 
If a visualization would help, always include the actual matplotlib or seaborn code block to generate it, using the columns you reference. Never say you can't generate a plot. Never mention code execution or that you're generating code. Make visualizations feel like a natural part of your explanation.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history (last 5 exchanges to keep context manageable)
        recent_history = st.session_state.conversation_history[-10:]  # Last 5 exchanges
        messages.extend(recent_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response using OpenAI API
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
    """
    Parse AI response text and create inline content segments with executed code outputs.
    
    Args:
        response_text (str): Raw AI response containing text and code blocks
        
    Returns:
        list: Content segments with text, code outputs, and images
        
    Extracts Python code blocks from response, executes them safely,
    and creates structured content segments for display.
    """
    import re
    content_segments = []
    last_end = 0
    code_executor = st.session_state.code_executor
    
    # Pattern to match code blocks with their positions
    pattern = r'```(?:python)?\s*(.*?)```'
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
        
        # Execute the code block (data analysis or visualization)
        if code_content:
            output, image_b64 = code_executor.execute_code(code_content)
            
            # Only add code_output if it's meaningful
            if output and output.strip() and output.strip().lower() != 'code executed successfully.':
                content_segments.append({
                    'type': 'code_output',
                    'content': output.strip()
                })
            
            # Always add all images if present
            if image_b64:
                # Handle both single images and lists of images
                if isinstance(image_b64, list):
                    if image_b64:  # Only add if not empty
                        content_segments.append({
                            'type': 'image',
                            'content': image_b64
                        })
                else:
                    content_segments.append({
                        'type': 'image',
                        'content': [image_b64]
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
    """
    Process user message and execute any code with inline visualization placement.
    
    Args:
        user_message (str): User's question or analysis request
        
    Returns:
        Dict[str, Any]: Processed response with content segments and metadata
        
    This function orchestrates the entire AI response process:
    1. Gets dataset context information
    2. Generates AI response using LLM
    3. Parses response for code execution
    4. Updates conversation history
    """
    # Get dataset context for AI model
    context = st.session_state.code_executor.get_dataframe_info()
    
    # Generate LLM response with context
    llm_response = generate_llm_response(user_message, context)
    
    # Parse response and create inline content segments with code execution
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

def export_chat_to_pdf(messages, filename="InsightBot_Report.pdf"):
    """
    Generate a comprehensive PDF report from chat messages.
    
    Args:
        messages (list): List of chat messages to include in PDF
        filename (str): Name for the PDF file
        
    Returns:
        BytesIO: PDF buffer ready for download
        
    Creates a professionally formatted PDF report containing:
    - Chat transcript with user questions and AI responses
    - Embedded visualizations and charts
    - Code outputs and analysis results
    - First Look Dashboard if available
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title and header
    story.append(Paragraph("<b>InsightBot Chat Transcript & Analysis Report</b>", styles['Title']))
    story.append(Spacer(1, 18))
    
    # Add First Look Dashboard to PDF if present
    if "first_look_segments" in st.session_state and st.session_state.first_look_segments:
        story.append(Paragraph("<b>First Look Dashboard</b>", styles['Heading2']))
        for seg in st.session_state.first_look_segments:
            if seg["type"] == "text":
                story.append(Paragraph(seg["content"], styles['Normal']))
                story.append(Spacer(1, 6))
            elif seg["type"] == "code_output":
                story.append(Paragraph(f"<font face='Courier'>{seg['content']}</font>", styles['Code']))
                story.append(Spacer(1, 6))
            elif seg["type"] == "image":
                try:
                    for img_b64 in seg["content"]:
                        imgdata = base64.b64decode(img_b64)
                        img = PILImage.open(BytesIO(imgdata))
                        img_io = BytesIO()
                        img.save(img_io, format='PNG')
                        img_io.seek(0)
                        story.append(Image(img_io, width=400, height=250))
                        story.append(Spacer(1, 8))
                except Exception as e:
                    story.append(Paragraph("[Image could not be rendered in PDF]", styles['Italic']))
                    story.append(Spacer(1, 6))
    
    # Add chat messages to PDF
    for idx, message in enumerate(messages):
        if message["type"] == "user":
            story.append(Paragraph(f"<b>User:</b> {message['content']}", styles['Normal']))
            story.append(Spacer(1, 8))
        elif message["type"] == "assistant":
            segments = message.get("content_segments")
            if not segments:
                story.append(Paragraph(f"<b>InsightBot:</b> {message['content']}", styles['Normal']))
                story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(f"<b>InsightBot:</b>", styles['Normal']))
                for seg in segments:
                    if seg["type"] == "text":
                        story.append(Paragraph(seg["content"], styles['Normal']))
                        story.append(Spacer(1, 6))
                    elif seg["type"] == "code_output":
                        story.append(Paragraph(f"<font face='Courier'>{seg['content']}</font>", styles['Code']))
                        story.append(Spacer(1, 6))
                    elif seg["type"] == "image":
                        try:
                            for img_b64 in seg["content"]:
                                imgdata = base64.b64decode(img_b64)
                                img = PILImage.open(BytesIO(imgdata))
                                img_io = BytesIO()
                                img.save(img_io, format='PNG')
                                img_io.seek(0)
                                story.append(Image(img_io, width=400, height=250))
                                story.append(Spacer(1, 8))
                        except Exception as e:
                            story.append(Paragraph("[Image could not be rendered in PDF]", styles['Italic']))
                            story.append(Spacer(1, 6))
    
    # Build and return PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ========================
# STREAMLIT USER INTERFACE
# ========================

# Sidebar for File Upload and Management
st.sidebar.markdown("# 🔍 InsightBot")
st.sidebar.markdown("*AI-powered data analysis through natural conversation*")
st.sidebar.markdown(f"**👤 Welcome, {st.session_state.username}!**")

# ========================
# CHAT MANAGEMENT UI
# ========================

# Chat Management Section in Sidebar
st.sidebar.divider()
st.sidebar.markdown("### 💬 Chat Management")

# New Chat and My Chats buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🆕 New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

with col2:
    if st.button("📚 My Chats", use_container_width=True):
        st.session_state.show_chat_manager = not st.session_state.show_chat_manager

# Display current chat information
if st.session_state.current_chat_id:
    chat_info = st.session_state.chat_storage.get_chat_info(st.session_state.current_chat_id)
    if chat_info:
        st.sidebar.markdown(f"**Current:** {chat_info['title']}")

# Chat Manager Interface - shows recent chats with load/delete options
if st.session_state.show_chat_manager:
    st.sidebar.markdown("#### 📋 Your Recent Chats")
    user_chats = st.session_state.chat_storage.get_user_chats(st.session_state.username, limit=10)
    
    if user_chats:
        for chat in user_chats:
            chat_col1, chat_col2 = st.sidebar.columns([3, 1])
            
            with chat_col1:
                # Create a shorter display title for UI constraints
                display_title = chat['title']
                if len(display_title) > 25:
                    display_title = display_title[:22] + "..."
                
                if st.button(
                    f"📄 {display_title}", 
                    key=f"load_chat_{chat['id']}", 
                    use_container_width=True,
                    help=f"Messages: {chat['message_count']}, Files: {chat['file_count']}"
                ):
                    load_chat(chat['id'])
                    st.session_state.show_chat_manager = False
                    st.rerun()
            
            with chat_col2:
                if st.button("🗑️", key=f"delete_chat_{chat['id']}", help="Delete chat"):
                    delete_chat(chat['id'])
                    st.rerun()
    else:
        st.sidebar.markdown("*No saved chats yet*")

# ========================
# PDF EXPORT FUNCTIONALITY
# ========================

# Emphasized Export as PDF Button - only show when files are uploaded
if st.session_state.uploaded_files:
    st.sidebar.divider()
    st.sidebar.markdown("### 📄 Export Chat & Insights")
    st.sidebar.markdown("Download your chat and dataset info as a PDF report.")
    
    def get_pdf_buffer_for_export():
        """Generate PDF buffer for download with chat messages or dataset info."""
        if st.session_state.messages:
            return export_chat_to_pdf(st.session_state.messages)
        else:
            # Minimal PDF with just dataset info and file name if no chat messages
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("<b>InsightBot Dataset Report</b>", styles['Title']))
            story.append(Spacer(1, 18))
            for file_info in st.session_state.uploaded_files:
                story.append(Paragraph(f"<b>File:</b> {file_info['name']}", styles['Normal']))
                story.append(Paragraph(f"<b>Size:</b> {round(file_info['size']/1024, 1)} KB", styles['Normal']))
                story.append(Spacer(1, 8))
            info = st.session_state.code_executor.get_dataframe_info()
            story.append(Paragraph("<b>Dataset Information:</b>", styles['Heading2']))
            story.append(Paragraph(f"<pre>{info}</pre>", styles['Code']))
            doc.build(story)
            buffer.seek(0)
            return buffer
    
    if st.sidebar.button("⬇️ Download PDF Report", key="export_pdf_btn_main", use_container_width=True):
        pdf_buffer = get_pdf_buffer_for_export()
        st.sidebar.download_button(
            label="Download Chat & Insights PDF",
            data=pdf_buffer,
            file_name="InsightBot_Report.pdf",
            mime="application/pdf"
        )

st.sidebar.divider()

# ========================
# FILE UPLOAD INTERFACE
# ========================
st.sidebar.header("📁 File Management")
st.sidebar.markdown("Upload your data file to start analyzing with AI insights. Supported: CSV, Excel, JSON, XML, TXT, PDF.")
file_uploaded = st.sidebar.file_uploader(
    "Choose a data file", 
    type=["csv", "xls", "xlsx", "json", "xml", "txt", "pdf"],
    help="Upload a CSV, Excel, JSON, XML, TXT, or PDF file to begin your data analysis conversation",
    accept_multiple_files=False,
    key=None,
    label_visibility="visible",
    disabled=False,
    # Set max upload size to 1GB (Streamlit must also be configured via config.toml)
    # This parameter is for clarity/documentation; Streamlit's config must be set for true effect
    # See: https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options
    # st.file_uploader does not have a max_size param, but we document it here
)
if st.sidebar.button("\U0001F4E4 Upload File", use_container_width=True):
    if file_uploaded:
        if upload_file_locally(file_uploaded):
            st.sidebar.success(f"✅ File '{file_uploaded.name}' uploaded successfully!")
            st.session_state.start_chat = True
            
            # Create new chat for this file upload if we don't have one
            if not st.session_state.current_chat_id:
                create_new_chat()
            else:
                # Update existing chat with new file info
                st.session_state.chat_storage.save_chat_files(
                    st.session_state.current_chat_id,
                    [st.session_state.uploaded_files[-1]]  # Just the latest file
                )
            
            # --- Unified First Look Dashboard with LLM Analysis ---
            dataset_context = st.session_state.code_executor.get_dataframe_info()
            llm_first_look_prompt = (
                "You are a data analyst. Please provide an initial analysis of the uploaded dataset, "
                "highlighting any detected anomalies, trends, or correlations. "
                "Explain each finding in plain language for non-technical users. "
                "Then, suggest a list of relevant questions or analysis prompts that would help a user explore their data and discover insights. "
                "You must include at least one visualization code block (matplotlib or seaborn) in your response, using the columns you reference. "
                "Format your response as follows:\n"
                "### Initial Analysis\n<your analysis>\n\n### Suggested Questions\n- <question 1>\n- <question 2>\n..."
            )
            with st.spinner("🤔 InsightBot is thinking..."):
                llm_first_look_response = generate_llm_response(llm_first_look_prompt, dataset_context)
                first_look_segments = parse_response_with_inline_visualizations(llm_first_look_response)
            st.session_state.first_look_segments = first_look_segments
            
            # Save the first look analysis as an assistant message
            if st.session_state.current_chat_id:
                text_content = ""
                for segment in first_look_segments:
                    if segment["type"] == "text":
                        text_content += segment["content"] + "\n"
                
                st.session_state.chat_storage.save_message(
                    st.session_state.current_chat_id,
                    "assistant",
                    f"📊 First Look Analysis:\n{text_content.strip()}",
                    first_look_segments
                )
        else:
            st.sidebar.error("❌ Failed to upload file.")
    else:
        st.sidebar.error("⚠️ Please select a file to upload.")

# --- First Look Dashboard Display (always at the top, not in chat) ---
if "first_look_segments" in st.session_state and st.session_state.first_look_segments:
    st.markdown("## 📊 First Look Dashboard: Automated Data Insights")
    for segment in st.session_state.first_look_segments:
        if segment["type"] == "text":
            st.markdown(segment["content"])
        elif segment["type"] == "code_output":
            st.code(segment["content"], language="text")
        elif segment["type"] == "image":
            for img_b64 in segment["content"]:
                st.image(base64.b64decode(img_b64), use_container_width=True)

# --- Current Dataset Section ---
if st.session_state.uploaded_files:
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Current Dataset")
    for file_info in st.session_state.uploaded_files:
        file_size_kb = round(file_info['size'] / 1024, 1)
        st.sidebar.markdown(f"📄 **{file_info['name']}**  ")
        st.sidebar.markdown(f"<span style='font-size: 0.9em; color: #888;'>📏 {file_size_kb} KB</span>", unsafe_allow_html=True)
    st.sidebar.markdown("")
    if st.sidebar.button("📈 View Dataset Info", use_container_width=True):
        info = st.session_state.code_executor.get_dataframe_info()
        st.sidebar.text_area("Dataset Information", info, height=200)

# --- Advanced Actions Section ---
if st.session_state.uploaded_files or st.session_state.messages:
    st.sidebar.divider()
    with st.sidebar.expander("⚙️ Advanced Actions", expanded=False):
        if st.session_state.uploaded_files:
            if st.button("🗑️ Delete All Files", key="delete_files_btn", use_container_width=True):
                delete_all_files()
                st.session_state.start_chat = False
                st.session_state.messages = []
                st.session_state.conversation_history = []
                # Don't delete the chat, just clear the current session
                st.success("✅ All files deleted and chat reset.")
        if st.session_state.messages:
            if st.button("🔄 Reset Chat", key="reset_chat_btn", use_container_width=True):
                # Save current chat before resetting
                if st.session_state.current_chat_id:
                    save_current_chat_messages()
                st.session_state.messages = []
                st.session_state.conversation_history = []
                logger.info("Chat history cleared.")
                st.success("✅ Chat history cleared.")
        
        # Chat title update
        if st.session_state.current_chat_id:
            st.markdown("**📝 Update Chat Title:**")
            chat_info = st.session_state.chat_storage.get_chat_info(st.session_state.current_chat_id)
            if chat_info:
                new_title = st.text_input("Chat Title", value=chat_info['title'], key="title_input")
                if st.button("💾 Update Title", use_container_width=True):
                    st.session_state.chat_storage.update_chat_title(st.session_state.current_chat_id, new_title)
                    st.success("✅ Chat title updated!")

# Logout button
st.sidebar.divider()
if st.sidebar.button("🚪 Logout", use_container_width=True, type="secondary"):
    # Save current chat before logout
    if st.session_state.current_chat_id and st.session_state.messages:
        save_current_chat_messages()
    logout()

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
            # Display each content segment (text, code output, image)
            segments = message.get("content_segments", [])
            for segment in segments:
                if segment["type"] == "text":
                    st.markdown(segment["content"])
                elif segment["type"] == "image":
                    import base64
                    from io import BytesIO
                    # Support both single image (str) and multiple images (list)
                    images = segment["content"]
                    if isinstance(images, list):
                        for img_b64 in images:
                            image_data = base64.b64decode(img_b64)
                            st.image(BytesIO(image_data))
                    else:
                        image_data = base64.b64decode(images)
                        st.image(BytesIO(image_data))
                elif segment["type"] == "code_output":
                    st.code(segment["content"], language="python")

# User Input and Chat Process
if st.session_state.start_chat:
    if prompt := st.chat_input("\U0001F4AC Ask me anything about your data! What insights would you like to discover?"):
        # Create a new chat if we don't have one
        if not st.session_state.current_chat_id:
            create_new_chat()
        
        # Add User Message
        st.session_state.messages.append({"type": "user", "content": prompt})
        
        # Save user message to database immediately
        if st.session_state.current_chat_id:
            st.session_state.chat_storage.save_message(
                st.session_state.current_chat_id,
                "user",
                prompt
            )
        
        logger.info("User input received: %s", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        # Process the message with spinner
        with st.spinner("🤔 InsightBot is thinking..."):
            result = process_user_message(prompt)
        # Prepare assistant message with new content segments structure
        assistant_message = {
            "type": "assistant",
            "content_segments": result['content_segments'],
            "has_visualization": result['has_visualization']
        }
        # Add assistant message to session
        st.session_state.messages.append(assistant_message)
        
        # Save assistant message to database immediately
        if st.session_state.current_chat_id:
            text_content = ""
            for segment in result['content_segments']:
                if segment["type"] == "text":
                    text_content += segment["content"] + "\n"
            
            st.session_state.chat_storage.save_message(
                st.session_state.current_chat_id,
                "assistant",
                text_content.strip(),
                result['content_segments']
            )
        
        # Display assistant response with inline visualizations and code outputs
        with st.chat_message("assistant"):
            for segment in result['content_segments']:
                if segment["type"] == "text":
                    st.markdown(segment["content"])
                elif segment["type"] == "image":
                    import base64
                    from io import BytesIO
                    # Support both single image (str) and multiple images (list)
                    images = segment["content"]
                    if isinstance(images, list):
                        for img_b64 in images:
                            image_data = base64.b64decode(img_b64)
                            st.image(BytesIO(image_data))
                    else:
                        image_data = base64.b64decode(images)
                        st.image(BytesIO(image_data))
                elif segment["type"] == "code_output":
                    st.code(segment["content"], language="python")