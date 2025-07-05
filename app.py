import os
import base64
import io
import logging
import time
from typing import Optional, List, Dict, Any
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Configure matplotlib before other imports for Streamlit Cloud compatibility
import matplotlib
matplotlib.use('Agg')

from instruction import INSTRUCTIONS
from code_executor import SecureCodeExecutor
from chat_storage import SQLiteChatStorage

# ReportLab imports for PDF export
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from PIL import Image as PILImage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Authentication Configuration
VALID_USERS = {
    "ihsankhan": "ergb4809uibewf",  # Change these credentials in production
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
                        # Set query param for session persistence
                        st.query_params["auth_user"] = username
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
    # Clear query params for session persistence
    st.query_params.clear()
    # Clear all session data
    for key in list(st.session_state.keys()):
        if key not in ['authenticated', 'username']:
            del st.session_state[key]
    st.rerun()

def create_new_chat():
    """Create a new chat session"""
    # Save current chat if it exists and has messages
    if st.session_state.current_chat_id and st.session_state.messages:
        save_current_chat_messages()
    
    # Create new chat
    metadata = {
        "dataset_files": [f["name"] for f in st.session_state.uploaded_files],
        "created_from": "new_chat_button"
    }
    
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
    
    # Clear current messages and conversation history
    st.session_state.messages = []
    st.session_state.conversation_history = []
    
    logger.info(f"Created new chat {st.session_state.current_chat_id}")

def save_current_chat_messages():
    """Save current chat messages to database"""
    if not st.session_state.current_chat_id or not st.session_state.messages:
        return
    
    try:
        # Get existing message count from database
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
                    content_segments = message.get("content_segments", [])
                    # For storage, we'll save a text representation as the main content
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
    """Load a chat and its messages"""
    try:
        # Save current chat first
        if st.session_state.current_chat_id and st.session_state.messages:
            save_current_chat_messages()
        
        # Load chat info
        chat_info = st.session_state.chat_storage.get_chat_info(chat_id)
        if not chat_info:
            st.error("Chat not found")
            return
        
        # Load messages
        messages = st.session_state.chat_storage.load_chat_messages(chat_id)
        
        # Load files
        files = st.session_state.chat_storage.load_chat_files(chat_id)
        
        # Update session state
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
    """Delete a chat"""
    try:
        success = st.session_state.chat_storage.delete_chat(chat_id, st.session_state.username)
        if success:
            # If we're deleting the current chat, reset to new chat
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

# Check for existing authentication in browser session
if not st.session_state.authenticated:
    # Try to restore authentication from query params or session
    query_params = st.query_params
    if "auth_user" in query_params:
        stored_username = query_params["auth_user"]
        if stored_username in VALID_USERS:
            st.session_state.authenticated = True
            st.session_state.username = stored_username
            # Clean up the URL
            st.query_params.clear()
            st.rerun()

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

# Initialize chat storage
if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = SQLiteChatStorage()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "show_chat_manager" not in st.session_state:
    st.session_state.show_chat_manager = False

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
    """Parse response text and create inline content segments with executed code outputs (text and images)"""
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
            # Only add code_output if it's not just 'Code executed successfully.'
            if output and output.strip() and output.strip().lower() != 'code executed successfully.':
                content_segments.append({
                    'type': 'code_output',
                    'content': output.strip()
                })
            # Always add all images if present
            if image_b64:
                # If image_b64 is a list, add as is; else, wrap in a list
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

def export_chat_to_pdf(messages, filename="InsightBot_Report.pdf"):
    """Generate a PDF report from chat messages including text, code, and images."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
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
    # Add chat messages as usual
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
    doc.build(story)
    buffer.seek(0)
    return buffer

# Sidebar for File Upload and Management
st.sidebar.markdown("# 🔍 InsightBot")
st.sidebar.markdown("*AI-powered data analysis through natural conversation*")
st.sidebar.markdown(f"**👤 Welcome, {st.session_state.username}!**")

# Chat Management Section
st.sidebar.divider()
st.sidebar.markdown("### 💬 Chat Management")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🆕 New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

with col2:
    if st.button("📚 My Chats", use_container_width=True):
        st.session_state.show_chat_manager = not st.session_state.show_chat_manager

# Display current chat info
if st.session_state.current_chat_id:
    chat_info = st.session_state.chat_storage.get_chat_info(st.session_state.current_chat_id)
    if chat_info:
        st.sidebar.markdown(f"**Current:** {chat_info['title']}")

# Chat Manager
if st.session_state.show_chat_manager:
    st.sidebar.markdown("#### 📋 Your Recent Chats")
    user_chats = st.session_state.chat_storage.get_user_chats(st.session_state.username, limit=10)
    
    if user_chats:
        for chat in user_chats:
            chat_col1, chat_col2 = st.sidebar.columns([3, 1])
            
            with chat_col1:
                # Create a shorter display title
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

# --- Emphasized Export as PDF Button ---
if st.session_state.uploaded_files:
    st.sidebar.divider()
    st.sidebar.markdown("### 📄 Export Chat & Insights")
    st.sidebar.markdown("Download your chat and dataset info as a PDF report.")
    def get_pdf_buffer_for_export():
        if st.session_state.messages:
            return export_chat_to_pdf(st.session_state.messages)
        else:
            # Minimal PDF with just dataset info and file name
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

# --- File Upload Section ---
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