#!/usr/bin/env python
"""Custom completers to fix prompt_toolkit issues."""

from prompt_toolkit.completion import PathCompleter, NestedCompleter
from prompt_toolkit.document import Document


class SafePathCompleter(PathCompleter):
    """A safer PathCompleter that handles None values gracefully."""
    
    def get_completions(self, document, complete_event):
        """Get path completions, catching any errors that might occur."""
        try:
            # Safely process document
            text = document.text
            if text is None:
                return
                
            yield from super().get_completions(document, complete_event)
        except (AttributeError, TypeError, ValueError):
            # Silently fail rather than crashing
            return


class SafeNestedCompleter(NestedCompleter):
    """NestedCompleter that handles errors gracefully."""
    
    @classmethod
    def from_nested_dict(cls, options):
        """Create a SafeNestedCompleter from a nested dictionary."""
        # First create with the regular method
        completer = super().from_nested_dict(options)
        # Convert to safe version
        completer.__class__ = SafeNestedCompleter
        return completer
    
    def get_completions(self, document, complete_event):
        """Get completions, catching any errors that might occur."""
        try:
            # Only process if we have valid input
            if document.text is None:
                return
                
            # Process first tokens
            text = document.text.strip()
            parts = text.split(" ", 1)
            first_term = parts[0]
            
            if first_term in self.options:
                option = self.options.get(first_term)
                
                # Process option
                if isinstance(option, dict):
                    # Create new document for the remainder
                    if len(parts) > 1:
                        remainder = parts[1]
                        remainder_doc = Document(remainder, cursor_position=len(remainder))
                        completer = SafeNestedCompleter.from_nested_dict(option)
                        yield from completer.get_completions(remainder_doc, complete_event)
                elif callable(option):
                    # If it's a callable completer
                    if len(parts) > 1:
                        remainder = parts[1]
                        remainder_doc = Document(remainder, cursor_position=len(remainder))
                        try:
                            yield from option.get_completions(remainder_doc, complete_event)
                        except (AttributeError, TypeError, ValueError):
                            # Silently handle errors
                            return
            else:
                # Return completions for first term
                first_term_lower = first_term.lower()
                for key in self.options:
                    try:
                        if key and isinstance(key, str) and key.lower().startswith(first_term_lower):
                            yield from self._yield_completion(key)
                    except (AttributeError, TypeError):
                        continue
        except Exception:
            # Catch any other errors and silently fail
            return
            
    def _yield_completion(self, key):
        """Safely yield a completion."""
        try:
            from prompt_toolkit.completion import Completion
            yield Completion(key, start_position=-len(key))
        except Exception:
            pass