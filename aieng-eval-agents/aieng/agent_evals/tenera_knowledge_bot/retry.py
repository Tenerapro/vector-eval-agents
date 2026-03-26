"""Retry configuration and error handling for Gemini API calls."""

from google.genai.errors import ClientError


# Max retries for empty model responses
MAX_EMPTY_RESPONSE_RETRIES = 2

# API retry configuration for rate limit and quota exhaustion
API_RETRY_MAX_ATTEMPTS = 5
API_RETRY_INITIAL_WAIT = 1  # seconds
API_RETRY_MAX_WAIT = 60  # seconds
API_RETRY_JITTER = 5  # seconds


def is_retryable_api_error(exception: BaseException) -> bool:
    """Check if an exception is a retryable API error (rate limit/quota).

    Does NOT retry context overflow or cache expiration errors.
    """
    if isinstance(exception, ClientError):
        error_str = str(exception).lower()

        # Don't retry context overflow
        if "token count exceeds" in error_str or ("invalid_argument" in error_str and "token" in error_str):
            return False

        # Don't retry cache expiration
        if "cache" in error_str and "expired" in error_str:
            return False

        # Retry rate limit / quota errors
        if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
            return True
    return False


def is_context_overflow_error(exception: BaseException) -> bool:
    """Check if an exception is a context overflow error."""
    if isinstance(exception, ClientError):
        error_str = str(exception).lower()
        return "token count exceeds" in error_str or ("invalid_argument" in error_str and "token" in error_str)
    return False
