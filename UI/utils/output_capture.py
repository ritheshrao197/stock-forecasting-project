"""
Output capture utility for capturing stdout/stderr
"""

import sys
from io import StringIO

class OutputCapture:
    """Capture stdout and stderr for logging"""
    
    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        
    def __enter__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self
        
    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        
    def get_output(self):
        """Get combined output from stdout and stderr"""
        output = self.stdout.getvalue()
        error = self.stderr.getvalue()
        return output + error
        
    def get_stdout(self):
        """Get stdout only"""
        return self.stdout.getvalue()
        
    def get_stderr(self):
        """Get stderr only"""
        return self.stderr.getvalue()