class RequestError(Exception):
    """Custom exception for request errors."""
    
    def __init__(self, message):
        super().__init__(message)
        self.message = message