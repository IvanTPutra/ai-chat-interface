from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from werkzeug.utils import secure_filename
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import docx
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
import os
import base64

# Flask application setup
app = Flask(__name__)
app.secret_key = 'your_secure_secret_key_here'  # Change this to a secure key

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"]
)

# User authentication (replace with proper database in production)
users = {
    "admin": base64.b64encode("password123".encode()).decode()
}

# Define the login_required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class DocumentProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.documents = []
        self.is_initialized = False
        self.dataframe = None  # For CSV/Excel analysis
        
    def process_file(self, filepath: str) -> List[str]:
        """Extract content from various file types"""
        ext = filepath.split('.')[-1].lower()
        
        if ext == 'csv':
            try:
                self.dataframe = pd.read_csv(filepath)
                return self._generate_data_insights()
            except Exception as e:
                logging.error(f"Error processing CSV file: {str(e)}")
                raise
                
        elif ext == 'xlsx':
            try:
                self.dataframe = pd.read_excel(filepath)
                return self._generate_data_insights()
            except Exception as e:
                logging.error(f"Error processing Excel file: {str(e)}")
                raise
                
        elif ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                return self._smart_chunk_text(text)
        
        elif ext == 'pdf':
            try:
                with open(filepath, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    full_text = ""
                    for page in pdf.pages:
                        full_text += page.extract_text() + "\n"
                    return self._smart_chunk_text(full_text)
            except Exception as e:
                logging.error(f"Error processing PDF file: {str(e)}")
                raise
            
        elif ext == 'docx':
            try:
                doc = docx.Document(filepath)
                full_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                return self._smart_chunk_text(full_text)
            except Exception as e:
                logging.error(f"Error processing DOCX file: {str(e)}")
                raise
            
        return []

    def _smart_chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Intelligently chunk text by preserving paragraph and sentence boundaries"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph is too long, split by sentences
            if len(paragraph) > max_chunk_size:
                sentences = paragraph.replace('!', '.').replace('?', '.').split('.')
                for sentence in sentences:
                    if len(sentence.strip()) == 0:
                        continue
                    
                    if len(current_chunk) + len(sentence) > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                    else:
                        current_chunk += sentence + ". "
            else:
                if len(current_chunk) + len(paragraph) > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
                else:
                    current_chunk += paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return [chunk for chunk in chunks if chunk.strip()]

    def _generate_data_insights(self) -> List[str]:
        """Generate textual insights from DataFrame"""
        if self.dataframe is None:
            return []
            
        insights = []
        
        # Basic DataFrame information
        basic_info = (
            f"Dataset Overview:\n"
            f"Total rows: {len(self.dataframe)}\n"
            f"Total columns: {len(self.dataframe.columns)}\n"
            f"Columns: {', '.join(self.dataframe.columns)}"
        )
        insights.append(basic_info)
        
        # Statistical summary for numeric columns
        numeric_cols = self.dataframe.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = self.dataframe[numeric_cols].describe().to_string()
            insights.append(f"Statistical Summary:\n{stats}")
        
        # Categorical column summaries
        categorical_cols = self.dataframe.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if len(self.dataframe[col].unique()) < 50:  # Only for columns with reasonable number of categories
                value_counts = self.dataframe[col].value_counts().head(10).to_string()
                insights.append(f"Top values for {col}:\n{value_counts}")
        
        # Missing values summary
        missing_data = self.dataframe.isnull().sum().to_string()
        insights.append(f"Missing Values Summary:\n{missing_data}")
        
        return insights

    def add_document(self, filepath: str):
        """Process document and add its content to the index"""
        try:
            texts = self.process_file(filepath)
            if not texts:
                raise ValueError(f"No text content extracted from file: {filepath}")
                
            # Create embeddings for each text chunk
            embeddings = self.embedder.encode(texts)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store original text chunks
            self.documents.extend(texts)
            
            self.is_initialized = True
            logging.info(f"Successfully processed file {filepath}. Total chunks: {len(texts)}")
            
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {str(e)}")
            raise

    def search(self, query: str, k: int = 5) -> List[str]:
        """Enhanced search with data analysis capabilities"""
        if not self.is_initialized:
            return []
            
        # Check for specific analysis requests
        query_lower = query.lower()
        
        if self.dataframe is not None:
            if "correlation" in query_lower:
                return [self.get_correlation_analysis()]
                
            if "analyze column" in query_lower:
                # Extract column name from query
                for col in self.dataframe.columns:
                    if col.lower() in query_lower:
                        return [self.analyze_column(col)]
        
        # Default document search
        try:
            query_embedding = self.embedder.encode([query])
            D, I = self.index.search(query_embedding.astype('float32'), min(k, len(self.documents)))
            return [self.documents[i] for i in I[0] if 0 <= i < len(self.documents)]
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []

    def analyze_column(self, column_name: str) -> str:
        """Analyze a specific column in detail"""
        if self.dataframe is None or column_name not in self.dataframe.columns:
            return "No data available or column not found."
            
        analysis = []
        series = self.dataframe[column_name]
        
        # Numeric column analysis
        if np.issubdtype(series.dtype, np.number):
            analysis.extend([
                f"Column: {column_name}",
                f"Data type: {series.dtype}",
                f"Mean: {series.mean():.2f}",
                f"Median: {series.median():.2f}",
                f"Standard deviation: {series.std():.2f}",
                f"Min: {series.min():.2f}",
                f"Max: {series.max():.2f}",
                f"Number of unique values: {len(series.unique())}"
            ])
            
        # Categorical column analysis
        else:
            analysis.extend([
                f"Column: {column_name}",
                f"Data type: {series.dtype}",
                f"Number of unique values: {len(series.unique())}",
                f"Most common value: {series.mode().iloc[0]}",
                f"Value counts:\n{series.value_counts().head(10).to_string()}"
            ])
            
        return "\n".join(analysis)
        
    def get_correlation_analysis(self) -> str:
        """Generate correlation analysis for numeric columns"""
        if self.dataframe is None:
            return "No data available."
            
        numeric_df = self.dataframe.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return "No numeric columns found for correlation analysis."
            
        correlation_matrix = numeric_df.corr()
        return f"Correlation Matrix:\n{correlation_matrix.to_string()}"
        
class ChatBot:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            load_in_4bit=torch.cuda.is_available(),
            device_map="auto"
        )
        self.model.eval()
        self.conversations: Dict[str, list] = {}
        self.doc_processor = DocumentProcessor()

    def get_response(self, user_input: str, session_id: str) -> str:
        try:
            # Initialize conversation history for new sessions
            if session_id not in self.conversations:
                self.conversations[session_id] = []

            # Get relevant document chunks
            relevant_docs = self.doc_processor.search(user_input)
            context = "\n".join(relevant_docs) if relevant_docs else ""
            
            logging.info(f"Found {len(relevant_docs)} relevant document chunks")

            # Build the conversation history
            conversation = self.conversations[session_id]

            # Create the full prompt
            system_message = (
                "<|system|>You are a focused assistant. For questions about characters or main characters, "
                "give only the name as a direct one-word or two-word answer without explanation. "
                "For other questions, provide helpful but concise responses.</s>"
            )

            context_message = f"<|system|>Context from documents:\n{context}</s>" if context else ""

            history = ""
            for msg in conversation[-3:]:
                role, content = msg
                history += f"<|{role}|>{content}</s>"

            current_input = f"<|user|>{user_input}</s><|assistant|>"
            full_prompt = f"{system_message}{context_message}{history}{current_input}"

            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.7,
                    num_beams=1,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("<|assistant|>")[-1].strip()

            # For character questions, ensure very short response
            if "character" in user_input.lower():
                words = response.split()
                if len(words) > 2:
                    response = " ".join(words[:2])

            self.conversations[session_id].append(("user", user_input))
            self.conversations[session_id].append(("assistant", response))

            return response

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return "I apologize, but I ran out of memory. Could you try a shorter message or wait a moment?"
            logging.error(f"Runtime error: {str(e)}")
            raise

# Initialize chatbot
chatbot = ChatBot()

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == base64.b64encode(password.encode()).decode():
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('chat.html', username=session.get('username'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file-input' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file-input']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process and index the uploaded file
            chatbot.doc_processor.add_document(filepath)
            
            return jsonify({
                'success': True,
                'message': 'File successfully uploaded and processed',
                'filename': filename
            })
            
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/chat', methods=['POST'])
@login_required
@limiter.limit("20 per minute")
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    session_id = session.get('username', 'default')

    try:
        response = chatbot.get_response(data['message'], session_id)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)