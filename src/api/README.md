# FastAPI Customer Support Chatbot

RESTful API for the TechGear Electronics customer support chatbot.

## 🚀 Quick Start

### 1. Start the API Server

```bash
cd src/api
python main.py
```

The API will be available at:
- **API Base:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 2. Test the API

In a new terminal:

```bash
python src/api/test_api.py
```

## 📚 API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-30T12:00:00.000000"
}
```

### Chat
```http
POST /chat
```

Request:
```json
{
  "query": "What is the price of SmartWatch Pro X?",
  "conversation_id": "conv_123"
}
```

Response:
```json
{
  "query": "What is the price of SmartWatch Pro X?",
  "response": "The SmartWatch Pro X is priced at ₹15,999.",
  "category": "product",
  "confidence": 1.0,
  "needs_escalation": false,
  "conversation_id": "conv_123",
  "timestamp": "2026-01-30T12:00:00.000000",
  "metadata": {
    "rag_used": true,
    "document_count": 3
  }
}
```

### Get Categories
```http
GET /categories
```

Returns list of supported query categories with examples.

### Get Products
```http
GET /products
```

Returns list of available products with pricing.

## 🧪 Testing with cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Chat Request
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the price of SmartWatch Pro X?"}'
```

### Get Categories
```bash
curl http://localhost:8000/categories
```

### Get Products
```bash
curl http://localhost:8000/products
```

## 🐍 Python Client Example

```python
import requests

# Send a chat message
response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "What is the price of SmartWatch?"}
)

data = response.json()
print(data['response'])
```

## 🌐 Production Deployment

### Using Uvicorn

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Using Docker

```bash
# Build image
docker build -t chatbot-api .

# Run container
docker run -p 8000:8000 chatbot-api
```

## 🔒 CORS Configuration

Currently configured to allow all origins (`*`). For production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📊 API Features

✅ RESTful design  
✅ Automatic interactive documentation (Swagger UI)  
✅ Request/response validation (Pydantic)  
✅ Error handling with proper HTTP status codes  
✅ CORS support  
✅ Health check endpoint  
✅ Conversation tracking  
✅ Metadata in responses  

## 🛠️ Technology Stack

- **FastAPI** - Modern web framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server
- **LangGraph** - Workflow orchestration
- **Google Gemini** - LLM & embeddings
- **ChromaDB** - Vector database

## 📝 Environment Variables

Create `.env` file:

```bash
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-2.0-flash
EMBEDDING_MODEL=models/embedding-001
```

## 🔧 Configuration

Edit `src/api/main.py`:

```python
# Server configuration
HOST = "0.0.0.0"
PORT = 8000

# API configuration
TITLE = "Customer Support Chatbot API"
VERSION = "1.0.0"
```

## 📈 Performance

- Average response time: 2-3 seconds
- Concurrent requests: Handled by Uvicorn workers
- Rate limiting: Not implemented (add in production)

## 🚨 Error Handling

All errors return proper HTTP status codes:

- `400` - Bad request (invalid input)
- `404` - Not found
- `500` - Internal server error
- `503` - Service unavailable

## 🎯 Next Steps

1. Add authentication (JWT tokens)
2. Implement rate limiting
3. Add caching (Redis)
4. Set up monitoring (Prometheus)
5. Add request logging
6. Deploy to cloud (AWS/GCP/Azure)

---

**Built with ❤️ for TechGear Electronics**
