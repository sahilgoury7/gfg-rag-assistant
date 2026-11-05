"""Rate Limiting Configuration Module.

Enforces IP-based rate limiting across API endpoints using Slowapi,
safeguarding public LLM and Vector DB quotas from spam, automated bots,
and DDoS attempts.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Global limiter instance keyed by client IP address
# Default fallback limit: 60 requests per minute across general routes
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
)
