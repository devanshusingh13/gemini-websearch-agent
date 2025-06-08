# Gemini Chat Agent

A conversational AI system powered by Google's Gemini model with long-term memory capabilities.

## 🏗️ Project Structure
```
gemini-chat-agent/
├── backend_fastapi/           # FastAPI Backend
│   ├── app/
│   │   ├── Agent/            # Gemini Agent Implementation
│   │   ├── api/
│   │   ├── core/             # Core Functionality
│   │   ├── utils/
│   │   ├── database/         # Database Operations
│   │   └── main.py          # FastAPI Entry Point
│   ├── logs/
├── requirements.txt      # Python Dependencies
│
│
└── frontend_streamlit/       # Streamlit Frontend
    └── application/
        ├── services/        # API Client Services
        └── app.py          # Streamlit App Entry Point
```

## ✨ Features

- 🤖 Powered by Google's Gemini AI model
- 💾 Long-term conversation memory
- 🔒 JWT-based authentication
- 📝 Conversation summarization
- 🔍 Vector-based memory search
- 🌐 Web-based chat interface

## 📋 Prerequisites

- Python 3.11+
- Docker (optional)
- Google API Key for Gemini


2. Set up virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## ⚙️ Configuration

1. Create a `.env` file in the directory:
```env
DATABASE_URL
JWT_SECRET_KEY
GEMINI_API_KEY
TAVILY_API_KEY
SERPAPI_API_KEY
DB_HOST
DB_PORT
DB_NAME
DB_USER
DB_PASSWORD
```


## 🏃‍♂️ Running the Application

### Backend (FastAPI)
```powershell
uvicorn backend_fastapi.app.main:app --reload
```

### Frontend (Streamlit)
```powershell
streamlit run frontend_streamlit/application/app.py
```


## 🔧 Key Components

- **FastAPI Backend**: Handles authentication, database operations, and Gemini AI integration
- **Streamlit Frontend**: Provides a user-friendly chat interface
- **Gemini Agent**: Manages conversation context and memory
- **Vector Database**: Enables semantic search of conversation history
- **JWT Authentication**: Secures API endpoints and user sessions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API
- FastAPI
- Streamlit
- SQLAlchemy
- Sentence Transformers

For more information or support, please open an issue in the GitHub repository.
