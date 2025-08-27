
# 📊 InsightBot: AI-Powered Data Analysis Chatbot

## Local Setup Instructions

Follow these step-by-step instructions to set up InsightBot on your local machine:

### **Step 1: Install Visual Studio Code**
1. Download VS Code from the official website: [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Run the installer and follow the installation wizard
3. Launch VS Code after installation
4. **Recommended Extensions** (install via Extensions panel in VS Code):
   - Python extension by Microsoft
   - Pylance (for Python IntelliSense)
   - Python Indent
   - GitLens (for Git integration)

### **Step 2: Install Python 3.11.9**
1. Download Python 3.11.9 from: [https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)
2. **Important**: During installation, check "Add Python to PATH"
3. Verify installation by opening Command Prompt/PowerShell and running:
   ```bash
   python --version
   ```
   You should see: `Python 3.11.9`

### **Step 3: Open Project Folder in VS Code**
1. Extract the InsightBot project zip file to your desired location (e.g., `C:\Projects\InsightBot`)
2. Open VS Code
3. Click **File > Open Folder** (or press `Ctrl+K, Ctrl+O`)
4. Navigate to and select the InsightBot project folder
5. Click **Select Folder** to open the project

### **Step 4: Create Virtual Environment (VENV)**
1. Open the Command Palette again: Ctrl+Shift+P
2. Type: Python: Create Environment
3. Choose Venv as the environment type.
4. Select the base Python interpreter you want to use.
5. Select `Python 3.11.9`
6. VS Code will automatically create and configure the virtual environment.

### **Step 5: Install Required Libraries**
1. Ensure your virtual environment is activated (you should see `(venv)` in terminal)
2. Install all dependencies from requirements.txt: (In Terminal)
   ```bash
   pip install -r requirements.txt
   ```
3. Wait for all packages to install (this may take a few minutes)
4. Verify installation by checking if key packages are installed:
   ```bash
   pip list | findstr streamlit
   pip list | findstr openai
   ```

### **Step 6: Configure API Key**
1. Create a `.env` file in the project root directory (same level as `app.py`)
2. Add your OpenAI API key to the `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. **To get an OpenAI API key**:
   - Visit [https://platform.openai.com/](https://platform.openai.com/)
   - Sign up or log in to your account
   - Go to **API Keys** section
   - Click **Create new secret key**
   - Copy the key and paste it in your `.env` file

   **⚠️ Important**: Never share your API key or commit it to version control!

### **Step 7: Run Streamlit Application**
1. Make sure your virtual environment is activated and you're in the project directory
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. The application will start and automatically open in your default web browser
4. You should see the URL: `http://localhost:8501`

### **Step 8: Login to InsightBot**
1. When the app opens in your browser, you'll see the login screen
2. Use the default credentials:
   - **Username**: `ihsankhan`
   - **Password**: `************************`
3. Click **Login** to access the main application

---

### 🔧 **Troubleshooting Common Issues**

**Issue**: "Python is not recognized as an internal or external command"
- **Solution**: Reinstall Python and ensure "Add Python to PATH" is checked

**Issue**: "streamlit: command not found"
- **Solution**: Ensure virtual environment is activated and requirements are installed

**Issue**: "OpenAI API key not found"
- **Solution**: Check that `.env` file exists and contains `OPENAI_API_KEY=your_key`

**Issue**: "Permission denied" when activating virtual environment
- **Solution**: Run PowerShell as Administrator and execute: `Set-ExecutionPolicy RemoteSigned`

**Issue**: Port already in use
- **Solution**: Change port with: `streamlit run app.py --server.port 8502`



## 🤖 Project Overview

InsightBot is a **next-generation AI-powered data analysis platform** that revolutionizes how people interact with their data. Built on GPT-4 technology and featuring a sophisticated web-based interface, InsightBot transforms complex data analysis into natural conversations. Whether you're a business analyst, researcher, student, or data enthusiast, InsightBot makes advanced analytics accessible without requiring programming expertise.

**🎯 Core Mission**: Democratize data science by enabling anyone to extract meaningful insights from their data through simple, conversational interactions.

> **InsightBot represents the future of data analysis—where artificial intelligence meets human intuition to unlock the stories hidden in your data.**

### **What Makes InsightBot Special:**

- **🧠 Conversational Intelligence**: Chat naturally with your data using everyday language
- **🔒 Enterprise-Grade Security**: Sandboxed code execution with multiple security layers
- **📊 Professional Analytics**: Statistical analysis, anomaly detection, and trend identification
- **🎨 Beautiful Visualizations**: Publication-ready charts and graphs generated instantly
- **📱 Modern Interface**: Streamlined, responsive design optimized for productivity
- **💾 Persistent Memory**: Full conversation and analysis history with SQLite storage
- **📄 Professional Reporting**: One-click PDF export with embedded visualizations

**It empowers users to:**

- 🔍 Uncover hidden patterns and trends in their data
- 📈 Generate professional-quality visualizations on demand
- 💡 Receive actionable insights in plain English
- 🗂️ Document and share findings effortlessly

---

## 🚀 Key Features & Capabilities

### **💬 Intelligent Conversational Analysis**
- **Natural Language Interface**: Ask questions in plain English—no SQL or Python knowledge required
- **Context-Aware Responses**: AI remembers your conversation history and builds upon previous insights
- **Guided Discovery**: Smart suggestions help you explore new dimensions of your data
- **Multi-Turn Conversations**: Deep-dive into specific aspects through follow-up questions

### **� Advanced Data Processing**
- **Multi-Format Support**: Seamlessly work with CSV, Excel (.xls/.xlsx), JSON, XML, TXT, and PDF files
- **Intelligent File Parsing**: Automatic detection and processing of different data structures
- **Text Content Analysis**: Extract and analyze text from PDFs and documents
- **Robust Error Handling**: Graceful handling of malformed or incomplete data files

### **🔍 AI-Powered Analytics Engine**
- **Automated First Look Dashboard**: Instant overview with key statistics and initial insights upon file upload
- **Smart Anomaly Detection**: Automatically identify outliers, unusual patterns, and data quality issues
- **Trend Analysis**: Discover temporal patterns, correlations, and hidden relationships
- **Statistical Summaries**: Comprehensive descriptive statistics presented in accessible language

### **📈 Professional Visualization Suite**
- **On-Demand Charts**: Generate histograms, scatter plots, bar charts, heatmaps, and more through conversation
- **Interactive Plotting**: Matplotlib and Seaborn integration for publication-quality graphics
- **Contextual Visualizations**: Charts automatically tailored to your data types and analysis goals
- **Embedded Display**: Visualizations appear inline within the chat for seamless analysis flow

### **🛡️ Enterprise-Grade Security**
- **Sandboxed Code Execution**: AI-generated code runs in a secure, isolated environment
- **Import Restrictions**: Only approved data science libraries (pandas, numpy, matplotlib, seaborn) allowed
- **AST-Based Validation**: Advanced security scanning prevents execution of dangerous operations
- **User Authentication**: Secure login system with session management and logout functionality

### **💾 Intelligent Session Management**
- **Persistent Chat History**: SQLite database stores all conversations and analysis results
- **Multi-Chat Support**: Create, manage, and switch between different analysis sessions
- **File Tracking**: Automatic association of uploaded files with specific chat sessions
- **Search & Retrieval**: Easily find and reload previous analyses and insights

### **📄 Professional Reporting**
- **One-Click PDF Export**: Generate comprehensive reports with embedded visualizations
- **Complete Chat Transcripts**: Include full conversation history with user questions and AI responses
- **Visual Integration**: Charts and graphs are embedded directly in the PDF output
- **Structured Format**: Professional layout suitable for presentations and documentation

### **🎯 User Experience Excellence**
- **Responsive Design**: Modern, intuitive interface optimized for both desktop and mobile
- **Real-Time Feedback**: Loading indicators and progress updates during analysis
- **Error Recovery**: Clear error messages with suggestions for resolution
- **Accessibility**: Clean typography and logical navigation for users of all technical levels

---

## 🆕 Latest Features & Enhancements

### **🔄 Version 2.0 - Major Platform Upgrade**

**Multi-Format File Intelligence**
InsightBot now provides comprehensive support for diverse data sources and document types:

- **📊 Structured Data**: CSV, Excel (.xls/.xlsx), JSON, XML with automatic pandas DataFrame conversion
- **📄 Document Analysis**: PDF text extraction and analysis using advanced parsing libraries
- **📝 Text Processing**: Smart TXT file handling with automatic CSV/TSV detection fallback
- **🔧 Unified Workflow**: All file types processed through a single, intelligent pipeline

**Enhanced User Experience**
- **💬 Persistent Chat System**: SQLite-powered conversation history with full session management
- **👥 Multi-User Support**: Secure user authentication with isolated data environments
- **🎨 Modern Interface**: Streamlined UI with responsive design and intuitive navigation
- **📱 Mobile Optimization**: Full functionality across desktop, tablet, and mobile devices

**Advanced AI Capabilities**
- **🤖 GPT-4 Integration**: Leveraging the latest AI model for superior analytical insights
- **🧠 Context Awareness**: AI maintains conversation context for progressive analysis
- **📊 Smart Visualization**: Automatic chart type selection based on data characteristics
- **🔍 Intelligent Suggestions**: AI-powered question recommendations tailored to your dataset

**Enterprise-Ready Features**
- **🔒 Enhanced Security**: Multiple security layers with AST-based code validation
- **📄 Professional Reporting**: Advanced PDF export with embedded visualizations
- **💾 Data Persistence**: Reliable session storage with automatic backup capabilities
- **🚀 Performance Optimization**: Faster processing and improved memory management

**What's Working Now:**
- ✅ Upload any supported file format and receive instant analysis
- ✅ Chat naturally with your data using everyday language
- ✅ Generate professional visualizations through conversation
- ✅ Export complete analysis sessions as PDF reports
- ✅ Manage multiple chat sessions with persistent history
- ✅ Secure user authentication with session isolation
- ✅ Automated anomaly detection and pattern recognition
- ✅ Real-time code execution with comprehensive security

> **Ready to Experience the Future of Data Analysis?** InsightBot makes advanced analytics accessible to everyone—no programming required!

---

## 💡 How to Use InsightBot

**InsightBot transforms data analysis into a natural conversation.** Here's your complete guide to unlocking insights from your data:

### **📤 Getting Started (3 Simple Steps)**

1. **Upload Your Data**: 
   - Use the sidebar file uploader to select your data file
   - Supported formats: CSV, Excel, JSON, XML, TXT, or PDF
   - Files are processed instantly with automatic format detection

2. **Meet Your AI Assistant**: 
   - Receive an automatic "First Look Dashboard" with initial insights
   - Review key statistics, patterns, and suggested analysis directions
   - Use suggested questions as conversation starters

3. **Start Exploring**: 
   - Ask questions in natural language—no technical jargon required
   - Request visualizations, comparisons, or deep-dive analyses
   - Build upon previous insights with follow-up questions

### **🎯 Conversation Examples**

**Exploratory Questions:**
- *"What does this dataset tell us about customer behavior?"*
- *"Show me the distribution of sales across different regions"*
- *"Are there any surprising patterns in this data?"*

**Analytical Requests:**
- *"Which factors correlate most strongly with revenue?"*
- *"Can you identify any outliers or anomalies?"*
- *"How has performance changed over time?"*

**Visualization Commands:**
- *"Create a scatter plot showing the relationship between age and income"*
- *"Generate a heatmap of correlations between all numerical variables"*
- *"Plot monthly trends for the past year"*

### **💼 Professional Workflow**

1. **Data Upload & Initial Review**: Let InsightBot automatically analyze your dataset structure and quality
2. **Guided Exploration**: Follow AI-generated suggestions or ask specific business questions
3. **Deep Analysis**: Dive deeper into interesting findings with targeted follow-up queries
4. **Visualization**: Request charts and graphs to support your findings
5. **Documentation**: Export your entire analysis session as a professional PDF report
6. **Session Management**: Save your work and return later to continue analysis

### **📈 Advanced Tips**

- **Layer Your Questions**: Start broad, then narrow down to specific insights
- **Request Context**: Ask InsightBot to explain what patterns mean for your business
- **Combine Insights**: Ask about relationships between different findings
- **Validate Assumptions**: Use InsightBot to test hypotheses about your data
- **Visual Verification**: Request charts to confirm statistical findings

---

## 🛡️ Advanced Security Architecture

InsightBot implements **multiple layers of security** to ensure safe AI-generated code execution while maintaining powerful analytical capabilities:

### **🏗️ Sandboxed Execution Environment**
- **Isolated Python Runtime**: All AI-generated code runs in a completely separate execution context
- **Memory Isolation**: Each analysis session operates with dedicated memory space and variable scoping
- **Automatic Cleanup**: Resources are automatically freed after each code execution to prevent memory leaks
- **Error Containment**: Failed operations are safely contained without affecting the main application

### **🚫 Import Restrictions & Whitelisting**
- **Approved Libraries Only**: Restricted to safe data science packages (pandas, numpy, matplotlib, seaborn, scipy)
- **Blocked Dangerous Modules**: File system access, network operations, and system commands are prohibited
- **AST-Based Validation**: Code is parsed and analyzed for security risks before execution
- **Runtime Monitoring**: Active scanning for attempts to access unauthorized functionality

### **🔍 Code Analysis & Validation**
- **Pattern Detection**: Identifies and blocks potentially malicious code patterns
- **Function Call Monitoring**: Validates all function calls against security policies
- **Variable Scope Control**: Ensures code cannot access sensitive application variables
- **Output Sanitization**: All execution results are cleaned before display to prevent injection attacks

### **🔐 User Authentication & Session Security**
- **Secure Login System**: SHA-256 password hashing with session management
- **Session Isolation**: Each user's data and conversations are completely separated
- **Automatic Logout**: Sessions expire after inactivity to protect sensitive data
- **Audit Trail**: All user actions and code executions are logged for security monitoring

---

## 🧠 Technical Architecture & AI Logic

InsightBot employs a sophisticated **multi-layered architecture** that seamlessly integrates AI reasoning with secure code execution and data visualization:

### **🔄 Intelligent Processing Pipeline**

**1. Query Understanding & Context Building**
- Natural language processing analyzes user intent and extracts analytical requirements
- Conversation history provides context for follow-up questions and progressive analysis
- Dataset metadata informs the AI about available columns, data types, and statistical properties

**2. AI Response Generation & Code Synthesis**
- GPT-4 generates human-readable insights combined with executable Python code
- Code generation is optimized for the specific dataset structure and user's analytical goals
- Response formatting includes explanatory text, code blocks, and visualization commands

**3. Secure Code Execution & Result Processing**
- AI-generated code undergoes security validation and sandboxed execution
- Results (statistical outputs, charts, tables) are captured and converted to displayable formats
- Visualizations are encoded as base64 images for seamless integration into the chat interface

**4. Response Assembly & Display**
- Text insights, code outputs, and visualizations are combined into structured response segments
- Content is rendered inline within the chat for an integrated analytical experience
- All interactions are stored in SQLite database for persistence and retrieval

### **🎯 AI-Driven Automation Features**

**Automated First Look Analysis**
- Upon file upload, AI automatically generates comprehensive dataset overview
- Includes data quality assessment, statistical summaries, and recommended exploration paths
- Provides visualization examples and suggests relevant analytical questions

**Smart Anomaly & Pattern Detection**
- IQR-based outlier detection for numerical variables
- Trend analysis using correlation coefficients and monotonicity testing
- Correlation discovery between variables with statistical significance testing
- Missing value analysis with data quality recommendations

**Contextual Visualization Intelligence**
- AI selects appropriate chart types based on data characteristics and user intent
- Automatic color schemes and styling for professional-quality outputs
- Dynamic axis labeling and legends that reflect actual data content

### **💾 Data Management & Persistence**

**Session State Architecture**
- Real-time chat state management with immediate database synchronization
- File metadata tracking with automatic association to chat sessions
- User authentication state persistence across browser sessions

**SQLite Database Schema**
- **Users Table**: Secure user account management with hashed credentials
- **Chats Table**: Chat session metadata with titles, timestamps, and user associations
- **Messages Table**: Complete conversation history with segmented content storage
- **Chat Files Table**: File upload tracking with size, type, and timestamp information

---

## ✨ Smart Conversation Starters

**Ready to explore your data?** Try these conversation starters to unlock insights from your dataset:

### **🔍 Discovery & Exploration**
- *"What story does this data tell?"*
- *"Give me a comprehensive overview of this dataset"*
- *"What are the most interesting patterns you can find?"*
- *"Are there any data quality issues I should know about?"*
- *"What makes this dataset unique or unusual?"*

### **📊 Statistical Analysis**
- *"Show me the key statistical summaries"*
- *"Which variables have the strongest relationships?"*
- *"Are there any significant correlations in the data?"*
- *"What does the distribution of [column name] look like?"*
- *"Can you detect any outliers or anomalies?"*

### **📈 Trend & Pattern Analysis**
- *"How have values changed over time?"*
- *"What seasonal patterns can you identify?"*
- *"Are there any emerging trends I should watch?"*
- *"Which factors seem to drive the biggest changes?"*
- *"Can you spot any cyclical patterns?"*

### **🎯 Business Intelligence**
- *"What insights would help with decision-making?"*
- *"Which segments or categories perform best?"*
- *"Where are the biggest opportunities for improvement?"*
- *"What factors contribute most to success metrics?"*
- *"Can you identify any concerning trends?"*

### **📋 Comparative Analysis**
- *"How do different groups compare?"*
- *"What are the key differences between categories?"*
- *"Which factors separate high performers from low performers?"*
- *"Show me the breakdown by [category/region/time period]"*
- *"How does performance vary across different segments?"*

### **🔮 Predictive Insights**
- *"Based on historical patterns, what trends might continue?"*
- *"What factors are most predictive of [outcome]?"*
- *"Are there early warning signs in the data?"*
- *"What conditions lead to the best results?"*
- *"Can you identify leading indicators?"*

---

## 🌟 Real-World Applications & Use Cases

**InsightBot empowers professionals across industries** to make data-driven decisions with confidence. Here's how different sectors leverage our platform:

### **💼 Business Analytics & Operations**
- **Sales Performance Analysis**: Track revenue trends, identify top-performing products, and analyze customer behavior patterns
- **Marketing Campaign Optimization**: Measure campaign effectiveness, segment audiences, and optimize conversion rates
- **Financial Reporting**: Generate executive dashboards, monitor KPIs, and detect financial anomalies
- **Supply Chain Management**: Analyze inventory trends, optimize procurement, and forecast demand patterns
- **Customer Success Analytics**: Track retention rates, identify churn risks, and improve customer satisfaction

### **🎓 Academic Research & Education**
- **Research Data Analysis**: Process survey results, experimental data, and longitudinal studies with statistical rigor
- **Student Performance Tracking**: Analyze academic outcomes, identify learning gaps, and optimize curriculum design
- **Grant Proposal Support**: Generate compelling visualizations and statistical evidence for funding applications
- **Collaborative Research**: Share reproducible analyses through PDF exports with embedded visualizations
- **Teaching Data Science**: Demonstrate analytical concepts interactively without requiring programming knowledge

### **🏥 Healthcare & Life Sciences**
- **Clinical Data Analysis**: Process patient outcomes, treatment effectiveness, and medical device performance data
- **Public Health Monitoring**: Track disease trends, vaccination rates, and health system performance
- **Research Study Analysis**: Analyze clinical trial data, biomarker studies, and epidemiological research
- **Quality Improvement**: Monitor healthcare quality metrics and identify areas for process optimization
- **Population Health Insights**: Analyze demographic health patterns and social determinants of health

### **🏭 Manufacturing & Quality Control**
- **Production Optimization**: Analyze manufacturing efficiency, defect rates, and quality control metrics
- **Predictive Maintenance**: Identify equipment failure patterns and optimize maintenance schedules
- **Supply Chain Analytics**: Track supplier performance, material costs, and delivery reliability
- **Process Improvement**: Use statistical process control to identify optimization opportunities
- **Safety Analytics**: Monitor workplace safety metrics and identify risk factors

### **💡 Consulting & Client Services**
- **Rapid Client Insights**: Quickly analyze client data and generate professional reports for meetings
- **Proposal Development**: Create compelling data visualizations to support business proposals
- **Benchmarking Studies**: Compare client performance against industry standards and best practices
- **Strategic Planning**: Use data to inform strategic recommendations and business transformation initiatives
- **Training & Workshops**: Demonstrate analytical concepts to client teams without technical barriers

### **📊 Data Journalism & Communications**
- **Story Development**: Discover newsworthy patterns and trends in public datasets
- **Fact-Checking**: Verify claims with statistical analysis and data validation
- **Visualization Creation**: Generate publication-ready charts for articles and reports
- **Public Data Analysis**: Make government and public datasets accessible to broader audiences
- **Interactive Reporting**: Create engaging data narratives that drive reader engagement

### **🚀 Startups & Innovation**
- **Market Research**: Analyze competitive landscapes, customer segments, and market opportunities
- **Product Analytics**: Track user behavior, feature adoption, and product performance metrics
- **Investor Relations**: Create compelling data visualizations for pitch decks and investor updates
- **A/B Testing**: Analyze experiment results and optimize product features
- **Growth Analytics**: Track user acquisition, retention, and revenue growth patterns

**Ready to transform your data analysis workflow? Upload your data and discover what insights await!**



