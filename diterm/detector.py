"""
Core detector module for identifying dangerous terminal commands.
"""

import re


class DangerousCommandDetector:
    """Detects potentially dangerous terminal commands."""
    
    def __init__(self):
        # Patterns that indicate dangerous operations
        self.danger_patterns = [
            r'rm\s+-rf\s+/',
            r'rm\s+-rf\s+\*',
            r'dd\s+if=/dev/(zero|random|urandom)',
            r'mkfs\.',
            r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:',  # Fork bomb
            r'>\s*/dev/(sd[a-z]|nvme[0-9]n[0-9]|hd[a-z])',
            r'mv\s+/\s+',
            r'chmod\s+-R\s+777\s+/',
            r'DROP\s+DATABASE',
            r'DELETE\s+FROM.*WHERE\s+1=1',
            r'TRUNCATE\s+TABLE',
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.danger_patterns]
    
    def is_dangerous(self, command):
        """
        Check if a command matches dangerous patterns.
        
        Args:
            command (str): The command to check
            
        Returns:
            bool: True if command is dangerous, False otherwise
        """
        for pattern in self.compiled_patterns:
            if pattern.search(command):
                return True
        return False
    
    def get_risk_score(self, command):
        """
        Calculate a risk score for a command.
        
        Args:
            command (str): The command to check
            
        Returns:
            int: Risk score (0-10)
        """
        matches = sum(1 for p in self.compiled_patterns if p.search(command))
        return min(matches * 10, 10)
