#!/usr/bin/env python
"""Script to test and fix completion issues."""

from pathlib import Path
import os
import sys
import site

def main():
    """Test completion and apply fixes."""
    print("Fixing prompt_toolkit completion issues...")
    
    # Find site-packages directory
    site_packages = None
    
    # Try to find using site module first
    for path in site.getsitepackages():
        if "site-packages" in path:
            site_packages = Path(path)
            break
            
    # Fallback to venv path
    if not site_packages:
        venv_path = os.environ.get("VIRTUAL_ENV")
        if venv_path:
            lib_path = Path(venv_path) / "lib"
            # Find python3.x directory
            for item in lib_path.iterdir():
                if item.is_dir() and item.name.startswith("python3"):
                    site_packages = item / "site-packages"
                    break
    
    if not site_packages or not site_packages.exists():
        print("ERROR: Could not find site-packages directory")
        return
        
    print(f"Found site-packages at {site_packages}")
    
    # Find the prompt_toolkit word_completer.py file
    word_completer_path = site_packages / "prompt_toolkit" / "completion" / "word_completer.py"
    
    if not word_completer_path.exists():
        print(f"ERROR: Could not find {word_completer_path}")
        return
        
    print(f"Found word_completer.py at {word_completer_path}")
    
    # Read the file content
    content = word_completer_path.read_text()
    
    # Check if the fix has already been applied
    if "# PATCHED FOR NONE VALUES" in content:
        print("Fix already applied")
        return
        
    # Apply the fix - adding None check before calling lower()
    patched_content = content.replace(
        "def word_matches(a):",
        "def word_matches(a):\n            # PATCHED FOR NONE VALUES\n            if a is None:\n                return False"
    )
    
    patched_content = patched_content.replace(
        "word = word.lower()",
        "try:\n                word = word.lower()\n            except (AttributeError, TypeError):\n                return False"
    )
    
    # Write the fixed content
    word_completer_path.write_text(patched_content)
    print("Fix applied successfully!")
    print("Please restart the application for the fix to take effect.")

if __name__ == "__main__":
    main()