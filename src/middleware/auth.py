"""
Authentication and Security Middleware for TechGear Electronics Chatbot API
Provides API key authentication, JWT tokens, and request validation
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, List
from functools import wraps

from fastapi import HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel


# =============================================================================
# Configuration
# =============================================================================

# API Key configuration
API_KEY_NAME = "X-API-Key"
API_KEY_AUTH_ENABLED = os.getenv("API_KEY_AUTH_ENABLED", "false").lower() == "true"
VALID_API_KEYS = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set()

# JWT configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(64))
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "60"))

# Security schemes
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =============================================================================
# Models
# =============================================================================

class TokenData(BaseModel):
    """JWT token data"""
    username: Optional[str] = None
    email: Optional[str] = None
    scopes: List[str] = []


class Token(BaseModel):
    """Token response"""
    access_token: str
    token_type: str
    expires_in: int


class User(BaseModel):
    """User model"""
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: List[str] = []


# =============================================================================
# API Key Authentication
# =============================================================================

async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """
    Verify API key from request header
    
    Args:
        api_key: API key from request header
        
    Returns:
        bool: True if valid
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not API_KEY_AUTH_ENABLED:
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return True


# =============================================================================
# JWT Authentication
# =============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Data to encode in token
        expires_delta: Token expiry duration
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """
    Verify and decode JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData: Decoded token data
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        email: str = payload.get("email")
        scopes: list = payload.get("scopes", [])
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return TokenData(username=username, email=email, scopes=scopes)
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> Optional[TokenData]:
    """
    Get current user from JWT token
    
    Args:
        credentials: Bearer token credentials
        
    Returns:
        TokenData: Current user data
        
    Raises:
        HTTPException: If token is invalid
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return verify_token(credentials.credentials)


# =============================================================================
# Password Utilities
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        bool: True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


# =============================================================================
# Combined Authentication
# =============================================================================

async def verify_authentication(
    api_key_valid: bool = Depends(verify_api_key),
    user: Optional[TokenData] = Depends(get_current_user)
) -> bool:
    """
    Verify authentication using either API key or JWT token
    
    Args:
        api_key_valid: API key validation result
        user: JWT user data
        
    Returns:
        bool: True if authenticated
    """
    # If API key authentication is enabled and valid, allow access
    if API_KEY_AUTH_ENABLED and api_key_valid:
        return True
    
    # If JWT user exists, allow access
    if user:
        return True
    
    # If no authentication method is enabled, allow access
    if not API_KEY_AUTH_ENABLED:
        return True
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


# =============================================================================
# Utility Functions
# =============================================================================

def generate_api_key() -> str:
    """
    Generate a secure API key
    
    Returns:
        str: New API key
    """
    return secrets.token_urlsafe(32)


def require_scopes(required_scopes: List[str]):
    """
    Decorator to require specific scopes for an endpoint
    
    Args:
        required_scopes: List of required scopes
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: TokenData = Depends(get_current_user), **kwargs):
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )
            
            for scope in required_scopes:
                if scope not in user.scopes:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Missing required scope: {scope}",
                    )
            
            return await func(*args, user=user, **kwargs)
        
        return wrapper
    
    return decorator


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Generate new API key
    print("New API Key:", generate_api_key())
    
    # Generate new JWT secret
    print("New JWT Secret:", secrets.token_urlsafe(64))
    
    # Create sample token
    token = create_access_token(
        data={"sub": "testuser", "email": "test@example.com", "scopes": ["read", "write"]}
    )
    print("Sample Token:", token)
    
    # Verify token
    try:
        token_data = verify_token(token)
        print("Token Data:", token_data)
    except Exception as e:
        print("Token verification failed:", e)
