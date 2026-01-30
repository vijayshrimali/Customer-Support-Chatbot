"""Rate limiting middleware for TechGear Electronics Chatbot API"""

import time
from typing import Dict, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import os

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class RateLimiter:
    """
    Token bucket rate limiter
    Supports per-minute and per-hour limits
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Storage for request counts
        self.minute_requests: Dict[str, list] = defaultdict(list)
        self.hour_requests: Dict[str, list] = defaultdict(list)
        
        # Last cleanup time
        self.last_cleanup = datetime.now()
    
    def _cleanup_old_requests(self):
        """Remove old request timestamps to prevent memory growth"""
        now = datetime.now()
        
        # Cleanup every 5 minutes
        if (now - self.last_cleanup).total_seconds() < 300:
            return
        
        one_hour_ago = now - timedelta(hours=1)
        
        # Clean minute requests
        for key in list(self.minute_requests.keys()):
            self.minute_requests[key] = [
                ts for ts in self.minute_requests[key]
                if ts > one_hour_ago.timestamp()
            ]
            if not self.minute_requests[key]:
                del self.minute_requests[key]
        
        # Clean hour requests
        for key in list(self.hour_requests.keys()):
            self.hour_requests[key] = [
                ts for ts in self.hour_requests[key]
                if ts > one_hour_ago.timestamp()
            ]
            if not self.hour_requests[key]:
                del self.hour_requests[key]
        
        self.last_cleanup = now
    
    def is_allowed(self, identifier: str) -> Tuple[bool, str]:
        """
        Check if request is allowed
        
        Args:
            identifier: Unique identifier (e.g., IP address)
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600
        
        # Cleanup old requests periodically
        self._cleanup_old_requests()
        
        # Get recent requests
        recent_minute = [
            ts for ts in self.minute_requests[identifier]
            if ts > one_minute_ago
        ]
        recent_hour = [
            ts for ts in self.hour_requests[identifier]
            if ts > one_hour_ago
        ]
        
        # Check per-minute limit
        if len(recent_minute) >= self.requests_per_minute:
            retry_after = int(60 - (now - min(recent_minute)))
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute. Retry after {retry_after} seconds."
        
        # Check per-hour limit
        if len(recent_hour) >= self.requests_per_hour:
            retry_after = int(3600 - (now - min(recent_hour)))
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour. Retry after {retry_after} seconds."
        
        # Record this request
        self.minute_requests[identifier].append(now)
        self.hour_requests[identifier].append(now)
        
        return True, ""
    
    def get_usage(self, identifier: str) -> Dict[str, int]:
        """
        Get current usage stats for an identifier
        
        Args:
            identifier: Unique identifier
            
        Returns:
            Dict with usage stats
        """
        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600
        
        minute_count = len([
            ts for ts in self.minute_requests[identifier]
            if ts > one_minute_ago
        ])
        hour_count = len([
            ts for ts in self.hour_requests[identifier]
            if ts > one_hour_ago
        ])
        
        return {
            "requests_last_minute": minute_count,
            "requests_last_hour": hour_count,
            "limit_per_minute": self.requests_per_minute,
            "limit_per_hour": self.requests_per_hour,
            "remaining_minute": max(0, self.requests_per_minute - minute_count),
            "remaining_hour": max(0, self.requests_per_hour - hour_count),
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting
    """
    
    def __init__(self, app, rate_limiter: RateLimiter):
        """
        Initialize middleware
        
        Args:
            app: FastAPI application
            rate_limiter: RateLimiter instance
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for health check and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier (IP address)
        # In production with reverse proxy, use X-Forwarded-For
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        # Check rate limit
        allowed, reason = self.rate_limiter.is_allowed(client_ip)
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=reason,
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.rate_limiter.requests_per_minute),
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        usage = self.rate_limiter.get_usage(client_ip)
        
        response.headers["X-RateLimit-Limit-Minute"] = str(usage["limit_per_minute"])
        response.headers["X-RateLimit-Limit-Hour"] = str(usage["limit_per_hour"])
        response.headers["X-RateLimit-Remaining-Minute"] = str(usage["remaining_minute"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(usage["remaining_hour"])
        
        return response


# =============================================================================
# Factory function
# =============================================================================

def create_rate_limiter() -> RateLimiter:
    """
    Create rate limiter from environment configuration
    
    Returns:
        RateLimiter instance
    """
    requests_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    requests_per_hour = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    
    return RateLimiter(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test rate limiter
    limiter = RateLimiter(requests_per_minute=5, requests_per_hour=20)
    
    # Simulate requests
    for i in range(7):
        allowed, reason = limiter.is_allowed("192.168.1.1")
        print(f"Request {i+1}: {'Allowed' if allowed else 'Blocked'} - {reason}")
        
        if allowed:
            usage = limiter.get_usage("192.168.1.1")
            print(f"  Usage: {usage}")
