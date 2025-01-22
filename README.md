# AI Agent

This is a Flask-based AI agent application that provides a chatbot with document processing capabilities. The system can upload and process various document types, analyze data, and respond to user queries.

## Features
- User authentication system
- File upload and document processing (TXT, PDF, DOCX, CSV, XLSX)
- Text extraction and chunking for better processing
- FAISS-based document search for enhanced query responses
- AI chatbot powered by a Transformer-based language model
- Rate limiting for chat requests

## Technologies Used
- Python
- Flask
- Transformers (Hugging Face)
- FAISS
- Sentence-Transformers
- PyTorch
- Pandas
- NumPy
- PyPDF2, python-docx

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-agent.git
   cd ai-agent
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Update `app.secret_key` in `ai_agent.py` with a secure key.
4. Run the application:
   ```bash
   python ai_agent.py
   ```

## API Endpoints
- `/login` (GET, POST) - User login
- `/logout` (GET) - User logout
- `/` (GET) - Main chat page (requires login)
- `/upload` (POST) - Uploads and processes a document
- `/chat` (POST) - Sends a message to the chatbot

## Usage
1. Start the Flask application.
2. Login with the predefined credentials (`admin` / `password123` by default, base64-encoded).
3. Upload documents for processing.
4. Interact with the chatbot via the `/chat` endpoint.

## Notes
- The application uses FAISS for efficient document search.
- Data insights are generated for CSV/XLSX files.
- The chatbot model can be modified by changing the `model_name` in `ChatBot` class.

## Security Considerations
- **Authentication**: Uses base64 encoding for credentials (should be replaced with a secure authentication mechanism in production).
- **File Upload**: Only allows specific file types (TXT, PDF, DOCX, CSV, XLSX).
- **Rate Limiting**: Limits chat requests to prevent abuse.

## Future Enhancements
- Integrate a proper database for user authentication.
- Implement JWT authentication for security.
- Expand chatbot capabilities with fine-tuned AI models.
- Enhance document analysis and visualization capabilities.
