"""Middleware package for TechGear Electronics Chatbot API"""

from .auth import (
    verify_api_key,
    create_access_token,
    verify_token,
    get_current_user,
    verify_authentication,
    generate_api_key,
    Token,
    TokenData,
    User,
)

from .rate_limit import (
    RateLimiter,
    RateLimitMiddleware,
    create_rate_limiter,
)

__all__ = [
    # Authentication
    "verify_api_key",
    "create_access_token",
    "verify_token",
    "get_current_user",
    "verify_authentication",
    "generate_api_key",
    "Token",
    "TokenData",
    "User",
    
    # Rate Limiting
    "RateLimiter",
    "RateLimitMiddleware",
    "create_rate_limiter",
]
