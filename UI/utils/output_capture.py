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

    def get_all_output_as_list(self):
        """Get all output split into lines
        
        Returns:
        --------
        list: List of output lines
        """
        output = self.get_output()
        return [line for line in output.split('\n') if line.strip()]

    def has_errors(self):
        """Check if any errors were captured
        
        Returns:
        --------
        bool: True if errors were captured
        """
        return len(self.stderr.getvalue().strip()) > 0