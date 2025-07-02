"""
Simple timing utility for PAC computations.
"""
import time
import functools

# Global variable to store start times
_start_times = {}

def tic(label="default"):
    """Start timing a code block."""
    _start_times[label] = time.time()
    
def toc(label="default", print_time=True):
    """End timing a code block and optionally print the elapsed time."""
    if label not in _start_times:
        raise ValueError(f"No timing started for '{label}'")
    
    elapsed = time.time() - _start_times[label]
    if print_time:
        print(f"TIME [{label}]: {elapsed:.4f} seconds")
    
    return elapsed

def simple_timer(func):
    """Simple decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"TIME [{func.__name__}]: {elapsed:.4f} seconds")
        return result
    return wrapper
