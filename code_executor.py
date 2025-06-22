import pandas as pd
import matplotlib
# Set backend before importing pyplot - crucial for Streamlit Cloud
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
import re
import ast
import sys
import contextlib
from typing import Dict, Any, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SecureCodeExecutor:
    """
    A secure code executor for data analysis tasks.
    Restricts imports and provides sandboxed execution environment.
    """
    
    ALLOWED_IMPORTS = {
        'pandas', 'pd', 'numpy', 'np', 'matplotlib.pyplot', 'plt', 
        'seaborn', 'sns', 'math', 'statistics', 'json', 're'
    }
    
    DANGEROUS_KEYWORDS = [
        'import os', 'import sys', 'import subprocess', 'import shutil',
        'import socket', 'import urllib', 'import requests', 'import http',
        'open(', 'file(', 'input(', 'raw_input(', 'exec(', 'eval(',
        'globals()', 'locals()', 'vars()', 'dir(',
        'getattr', 'setattr', 'delattr', 'hasattr'
    ]
    
    def __init__(self):
        self.dataframe: Optional[pd.DataFrame] = None
        # Configure matplotlib for Streamlit Cloud compatibility
        plt.style.use('default')
        sns.set_style("whitegrid")
        
        self.execution_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'df': None,  # Will be set when data is loaded
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'min': min, 'max': max, 'sum': sum, 'abs': abs,
                'round': round, 'sorted': sorted, 'enumerate': enumerate,
                'zip': zip, 'range': range, 'type': type, 'isinstance': isinstance,
                'print': print,
                '__import__': __import__
            }
        }
    
    def load_dataframe(self, file_content: bytes, filename: str) -> bool:
        """Load a CSV file into pandas DataFrame"""
        try:
            if filename.endswith('.csv'):
                self.dataframe = pd.read_csv(io.BytesIO(file_content))
                self.execution_globals['df'] = self.dataframe
                return True
            else:
                raise ValueError("Only CSV files are supported")
        except Exception as e:
            raise Exception(f"Failed to load file: {str(e)}")
    
    def validate_code(self, code: str) -> bool:
        """Validate code for security issues"""
        # Check for dangerous keywords
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword in code:
                raise ValueError(f"Dangerous operation detected: {keyword}")
        
        # Parse AST to check for dangerous operations
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Check imports
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name not in self.ALLOWED_IMPORTS:
                                raise ValueError(f"Import not allowed: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module not in self.ALLOWED_IMPORTS:
                            raise ValueError(f"Import not allowed: {node.module}")
        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {str(e)}")
        
        return True
    
    def execute_code(self, code: str) -> Tuple[str, Optional[str]]:
        """Execute code safely and return results"""
        if self.dataframe is None:
            return "Error: No data loaded. Please upload a CSV file first.", None
        
        # Validate code first
        try:
            self.validate_code(code)
        except ValueError as e:
            return f"Security validation failed: {str(e)}", None
        
        # Capture output
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        image_buffer = None
        
        try:
            # Clear any existing plots
            plt.close('all')
            
            # Configure matplotlib for Streamlit Cloud
            matplotlib.use('Agg')
            plt.ioff()  # Turn off interactive mode
            
            # Redirect stdout and stderr
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                # Execute the code
                exec(code, self.execution_globals)
                
                # Check if a plot was created
                if plt.get_fignums():
                    # Save plot to buffer with explicit settings for cloud compatibility
                    img_buffer = io.BytesIO()
                    plt.savefig(
                        img_buffer, 
                        format='png', 
                        dpi=100,  # Reduced DPI for better cloud performance
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                        pad_inches=0.1
                    )
                    img_buffer.seek(0)
                    image_buffer = base64.b64encode(img_buffer.getvalue()).decode()
                    img_buffer.close()
                    plt.close('all')  # Close all figures
            
            # Get output
            output = output_buffer.getvalue()
            error = error_buffer.getvalue()
            
            if error:
                return f"Error: {error}", None
            
            return output if output else "Code executed successfully.", image_buffer
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            return f"Execution error: {str(e)}", None
        finally:
            # Ensure cleanup
            output_buffer.close()
            error_buffer.close()
    
    def get_dataframe_info(self) -> str:
        """Get basic information about the loaded dataframe"""
        if self.dataframe is None:
            return "No data loaded."
        
        # Get basic stats for numerical columns
        numerical_cols = self.dataframe.select_dtypes(include=[np.number]).columns
        categorical_cols = self.dataframe.select_dtypes(include=['object']).columns
        
        info = f"""
Dataset Summary:
- Shape: {self.dataframe.shape[0]} rows, {self.dataframe.shape[1]} columns
- Numerical columns ({len(numerical_cols)}): {', '.join(numerical_cols[:5])}{'...' if len(numerical_cols) > 5 else ''}
- Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}
- Missing values: {self.dataframe.isnull().sum().sum()} total

Key Statistics:
{self.dataframe.describe().round(2).to_string() if len(numerical_cols) > 0 else 'No numerical columns for statistics'}

Sample Data:
{self.dataframe.head(3).to_string()}
        """
        return info
    
    def is_visualization_code(self, code: str) -> bool:
        """Check if code is primarily for visualization"""
        viz_keywords = ['plt.', 'sns.', 'matplotlib', 'seaborn', 'plot', 'hist', 'scatter', 'bar', 'fig', 'ax']
        analysis_keywords = ['describe()', 'mean()', 'sum()', 'count()', 'groupby', 'merge', 'join']
        
        viz_count = sum(1 for keyword in viz_keywords if keyword in code)
        analysis_count = sum(1 for keyword in analysis_keywords if keyword in code)
        
        return viz_count > analysis_count
    
    def extract_code_from_response(self, response: str) -> List[str]:
        """Extract Python code blocks from LLM response"""
        # Pattern to match code blocks
        pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if not matches:
            # Try to find code without markdown formatting
            lines = response.split('\n')
            code_lines = []
            for line in lines:
                # Look for lines that look like Python code
                stripped = line.strip()
                if (stripped.startswith(('df.', 'pd.', 'plt.', 'sns.', 'np.')) or
                    '=' in stripped or stripped.startswith('print(')):
                    code_lines.append(line)
            
            if code_lines:
                matches = ['\n'.join(code_lines)]
        
        return matches
    
    def extract_code_blocks_with_positions(self, response: str) -> List[Dict]:
        """Extract Python code blocks from LLM response with their positions"""
        import re
        
        # Pattern to match code blocks
        pattern = r'```(?:python)?\s*(.*?)```'
        
        code_blocks = []
        for match in re.finditer(pattern, response, re.DOTALL):
            code_blocks.append({
                'code': match.group(1).strip(),
                'start': match.start(),
                'end': match.end(),
                'full_match': match.group(0)
            })
        
        return code_blocks
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the loaded pandas DataFrame, or None if not loaded."""
        return self.dataframe

    def detect_anomalies_trends_correlations(self) -> list:
        """Detect anomalies, trends, and correlations in the loaded DataFrame. Returns a list of plain-language findings."""
        findings = []
        df = self.dataframe
        if df is None or df.empty:
            return ["No data loaded to analyze."]

        # Anomaly detection: Outliers (IQR method for numerical columns)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            if len(outliers) > 0:
                findings.append(f"Column '{col}' has {len(outliers)} outlier value(s) (possible anomalies). Example: {outliers.iloc[0]:.2f}")

        # Trend detection: Time or index-based trends
        if 'date' in df.columns or 'Date' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'Date'
            try:
                df_sorted = df.sort_values(by=date_col)
                for col in num_cols:
                    if col == date_col:
                        continue
                    series = df_sorted[col].dropna()
                    if len(series) > 1:
                        corr = np.corrcoef(np.arange(len(series)), series)[0, 1]
                        if abs(corr) > 0.5:
                            trend = 'increasing' if corr > 0 else 'decreasing'
                            findings.append(f"Column '{col}' shows a {trend} trend over time.")
            except Exception:
                pass
        # General trend: Monotonicity or strong change
        for col in num_cols:
            series = df[col].dropna()
            if len(series) > 1:
                diff = np.diff(series)
                if np.all(diff > 0):
                    findings.append(f"Column '{col}' is strictly increasing.")
                elif np.all(diff < 0):
                    findings.append(f"Column '{col}' is strictly decreasing.")

        # Correlation detection: Pearson correlation for numerical columns
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()
            for i, col1 in enumerate(num_cols):
                for col2 in num_cols[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.7:
                        findings.append(f"Columns '{col1}' and '{col2}' are strongly correlated (correlation: {corr:.2f}).")

        # Missing values
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                findings.append(f"Column '{col}' has {count} missing value(s). Consider addressing these for better analysis.")

        # If nothing found
        if not findings:
            findings.append("No significant anomalies, trends, or strong correlations detected in the dataset.")
        return findings

    def generate_suggested_questions(self, max_questions: int = 7) -> list:
        """Generate a list of suggested questions/prompts based on the dataset's content."""
        df = self.dataframe
        if df is None or df.empty:
            return ["No data loaded to suggest questions."]
        questions = []
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        # General questions
        questions.append("What does this dataset contain?")
        questions.append("What are the main columns and their types?")
        if len(num_cols) > 0:
            for col in num_cols[:2]:
                questions.append(f"What is the distribution of '{col}'?")
                questions.append(f"Are there any outliers in '{col}'?")
            if len(num_cols) > 1:
                questions.append(f"Are any numerical columns correlated?")
        if len(cat_cols) > 0:
            for col in cat_cols[:2]:
                questions.append(f"What are the most common values in '{col}'?")
                questions.append(f"How does '{col}' relate to other columns?")
        if 'date' in df.columns or 'Date' in df.columns:
            questions.append("Are there any trends over time?")
        questions.append("Are there any missing values or anomalies in the data?")
        # Limit to max_questions
        return questions[:max_questions]
