# Gemini Chat Agent

A conversational AI system powered by Google's Gemini model with long-term memory capabilities.

## 🏗️ Project Structure
```
gemini-chat-agent/
├── backend_fastapi/           # FastAPI Backend
│   ├── app/
│   │   ├── Agent/            # Gemini Agent Implementation
│   │   ├── core/             # Core Functionality
│   │   ├── database/         # Database Operations
│   │   └── main.py          # FastAPI Entry Point
│   ├── migrations/           # Database Migrations
│   ├── requirements.txt      # Python Dependencies
│   ├── alembic.ini          # Migration Config
│   └── Dockerfile           # Backend Container Config
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

## 🚀 Installation

1. Clone the repository:
```powershell
git clone https://github.com/yourusername/gemini-chat-agent.git
cd gemini-chat-agent
```

2. Set up virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. Install backend dependencies:
```powershell
cd backend_fastapi
pip install -r requirements.txt
```

4. Install frontend dependencies:
```powershell
cd ../frontend_streamlit
pip install -r requirements.txt
```

## ⚙️ Configuration

1. Create a `.env` file in the backend directory:
```env
GOOGLE_API_KEY=your_gemini_api_key
JWT_SECRET_KEY=your_jwt_secret
DATABASE_URL=sqlite:///./chat.db
```

2. Initialize the database:
```powershell
cd backend_fastapi
alembic upgrade head
```

## 🏃‍♂️ Running the Application

### Backend (FastAPI)
```powershell
cd backend_fastapi
uvicorn app.main:app --reload --port 8000
```

### Frontend (Streamlit)
```powershell
cd frontend_streamlit
streamlit run application/app.py
```

## 🐳 Docker Support

Build and run the backend:
```powershell
cd backend_fastapi
docker build -t gemini-chat-backend .
docker run -d -p 8000:8000 -v chat-data:/app/data gemini-chat-backend
```

## 📚 API Documentation

Once running, access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

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
