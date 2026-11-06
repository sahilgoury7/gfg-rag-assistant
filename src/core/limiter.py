"""Rate Limiting Configuration Module.

Enforces IP-based rate limiting across API endpoints using Slowapi,
safeguarding public LLM and Vector DB quotas from spam, automated bots,
and DDoS attempts.
"""

from slowapi import Limiter
from starlette.requests import Request


def get_real_client_ip(request: Request) -> str:
    """Extracts the true client IP address, handling reverse proxies correctly.

    Checks X-Forwarded-For header first (standard in Render, Cloudflare, AWS ALB)
    to avoid sharing rate limit quotas across all users behind the same proxy.
    """
    forwarded = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


# Global limiter instance keyed by client IP address
# Default fallback limit: 60 requests per minute across general routes
limiter = Limiter(
    key_func=get_real_client_ip,
    default_limits=["60/minute"],
)
