"""
Torch preloading coordination module.
This module provides thread-safe coordination for torch module loading.
"""
import threading

# Global event to signal when preloading is complete
_preload_complete = threading.Event()

def set_preload_complete():
    """Signal that torch preloading is complete."""
    _preload_complete.set()

def ensure_torch_ready():
    """
    Call this function before any torch operations to ensure preloading is complete.
    This is non-blocking for GUI initialization but ensures torch is ready when needed.
    """
    if not _preload_complete.is_set():
        _preload_complete.wait()

def is_preload_complete():
    """Check if preloading is complete without blocking."""
    return _preload_complete.is_set()