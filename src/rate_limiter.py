"""
Rate Limiting and Retry Utilities
"""

import time
import random
from functools import wraps

class RateLimiter:
    """Rate limiter with exponential backoff"""
    
    def __init__(self, min_delay=1, max_delay=30):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0
    
    def wait(self):
        """Wait with exponential backoff"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed + random.uniform(0, 0.5)
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

def with_retry(max_retries=3, backoff_factor=2):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    wait_time = backoff_factor ** retries + random.uniform(0, 1)
                    print(f"⏳ Retry {retries}/{max_retries} in {wait_time:.1f}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

# Global rate limiter instance
rate_limiter = RateLimiter()