"""
FastAPI REST API for Customer Support Chatbot
Exposes the LangGraph workflow as a web service
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from graph.workflow import get_workflow, ChatbotWorkflow
from graph.state import ChatbotState

# Initialize FastAPI app
app = FastAPI(
    title="TechGear Electronics - Customer Support Chatbot API",
    description="AI-powered customer support chatbot using RAG and LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(
        ...,
        description="User's question or message",
        min_length=1,
        max_length=500,
        example="What is the price of SmartWatch Pro X?"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for tracking",
        example="conv_123456"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the price of SmartWatch Pro X?",
                "conversation_id": "conv_123456"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    query: str = Field(..., description="The user's original query")
    response: str = Field(..., description="The chatbot's response")
    category: str = Field(..., description="Query category (product/returns/general)")
    confidence: float = Field(..., description="Classification confidence score (0-1)")
    needs_escalation: bool = Field(..., description="Whether query needs human support")
    conversation_id: str = Field(..., description="Conversation identifier")
    timestamp: str = Field(..., description="Response timestamp (ISO format)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the price of SmartWatch Pro X?",
                "response": "The SmartWatch Pro X is priced at ₹15,999.",
                "category": "product",
                "confidence": 1.0,
                "needs_escalation": False,
                "conversation_id": "conv_123456",
                "timestamp": "2026-01-30T12:00:00.000000",
                "metadata": {
                    "rag_used": True,
                    "document_count": 3
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")


# =============================================================================
# GLOBAL WORKFLOW INSTANCE
# =============================================================================

# Initialize workflow on startup (singleton)
chatbot_workflow: Optional[ChatbotWorkflow] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot workflow on startup"""
    global chatbot_workflow
    try:
        print("🚀 Starting FastAPI server...")
        print("🤖 Initializing chatbot workflow...")
        chatbot_workflow = get_workflow()
        print("✅ Chatbot workflow initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing chatbot workflow: {str(e)}")
        print("⚠️  API will start but chatbot may not function correctly")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down FastAPI server...")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information"
)
async def root():
    """Root endpoint - API information"""
    return {
        "message": "TechGear Electronics - Customer Support Chatbot API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API service is running"
)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the bot",
    description="Send a query to the customer support chatbot and get a response",
    responses={
        200: {
            "description": "Successful response",
            "model": ChatResponse
        },
        400: {
            "description": "Bad request - invalid input",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def chat(request: ChatRequest):
    """
    Chat endpoint - main chatbot interaction
    
    Process a user query through the LangGraph workflow and return a response.
    
    **Workflow:**
    1. Query is classified (product/returns/general)
    2. Routed to appropriate handler (RAG or Escalation)
    3. Response is generated and returned
    
    **Categories:**
    - **product**: Questions about products, prices, features
    - **returns**: Return, refund, exchange requests
    - **general**: Support hours, payment methods, shipping info
    """
    try:
        # Validate workflow is initialized
        if chatbot_workflow is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chatbot workflow not initialized. Please try again later."
            )
        
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Process query through workflow
        result = chatbot_workflow.run(
            user_query=request.query.strip(),
            verbose=False
        )
        
        # Extract response data
        response_data = ChatResponse(
            query=request.query,
            response=result.get("final_response", "I apologize, but I couldn't generate a response."),
            category=result.get("classified_category", "unknown"),
            confidence=result.get("confidence_score", 0.0),
            needs_escalation=result.get("needs_escalation", False),
            conversation_id=request.conversation_id or result.get("conversation_id", ""),
            timestamp=datetime.now().isoformat(),
            metadata=result.get("metadata", {})
        )
        
        return response_data
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"❌ Error processing chat request: {str(e)}")
        
        # Return error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


@app.get(
    "/categories",
    summary="Get supported categories",
    description="List all query categories supported by the chatbot"
)
async def get_categories():
    """Get list of supported query categories"""
    return {
        "categories": [
            {
                "name": "product",
                "description": "Product information, pricing, features, specifications",
                "examples": [
                    "What is the price of SmartWatch Pro X?",
                    "Tell me about Wireless Earbuds features",
                    "Does the power bank support fast charging?"
                ]
            },
            {
                "name": "returns",
                "description": "Returns, refunds, exchanges, defective products",
                "examples": [
                    "How do I return a product?",
                    "I want a refund for my order",
                    "Can I exchange my defective earbuds?"
                ]
            },
            {
                "name": "general",
                "description": "General support, payment, shipping, contact info",
                "examples": [
                    "What are your customer support hours?",
                    "Do you accept cash on delivery?",
                    "What payment methods do you accept?"
                ]
            }
        ]
    }


@app.get(
    "/products",
    summary="Get product list",
    description="List all products in the knowledge base"
)
async def get_products():
    """Get list of available products"""
    return {
        "products": [
            {
                "name": "SmartWatch Pro X",
                "price": "₹15,999",
                "description": "Fitness and lifestyle companion with heart rate monitoring, GPS tracking, and 7-day battery life"
            },
            {
                "name": "Wireless Earbuds Elite",
                "price": "₹4,999",
                "description": "Premium earbuds with Active Noise Cancellation, 24-hour battery, and IPX4 water resistance"
            },
            {
                "name": "Power Bank Ultra 20000mAh",
                "price": "₹2,499",
                "description": "High-capacity power bank with 22.5W fast charging and dual USB ports"
            }
        ]
    }


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# =============================================================================
# MAIN - FOR DEVELOPMENT
# =============================================================================

if __name__ == "__main__":
    """Run the API server"""
    print("="*70)
    print("🚀 Starting TechGear Electronics Customer Support Chatbot API")
    print("="*70)
    print(f"\n📍 API will be available at: http://localhost:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"📖 ReDoc: http://localhost:8000/redoc")
    print(f"\n{'='*70}\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
