#!/usr/bin/env python
"""Installation script for claim-assistant."""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10+."""
    required_major = 3
    required_minor = 10
    
    if sys.version_info.major < required_major or \
       (sys.version_info.major == required_major and sys.version_info.minor < required_minor):
        print(f"Error: Python {required_major}.{required_minor}+ is required")
        print(f"Current Python version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")


def check_tesseract():
    """Check if Tesseract OCR is installed."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"], 
            capture_output=True, 
            text=True,
            check=False
        )
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            print(f"✅ {version}")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️ Tesseract OCR not found")
    print("Please install Tesseract OCR:")
    print("  - MacOS: brew install tesseract")
    print("  - Ubuntu/Debian: sudo apt-get install tesseract-ocr")
    print("  - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    return False


def setup_virtual_env():
    """Set up virtual environment and install dependencies."""
    venv_dir = Path("venv")
    
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Determine path to pip in virtual environment
    if sys.platform == "win32":
        pip_path = venv_dir / "Scripts" / "pip"
    else:
        pip_path = venv_dir / "bin" / "pip"
    
    # Install package in development mode
    print("Installing claim-assistant...")
    subprocess.run([str(pip_path), "install", "-e", "."], check=True)
    
    print("✅ Installation complete")


def setup_config():
    """Set up configuration file."""
    config_path = Path.home() / ".claimctl.ini"
    sample_path = Path("claimctl.ini.sample")
    
    if not config_path.exists() and sample_path.exists():
        print(f"Creating config file at {config_path}")
        with open(sample_path, "r") as src:
            content = src.read()
        
        with open(config_path, "w") as dst:
            dst.write(content)
        
        print("⚠️ Please edit ~/.claimctl.ini to set your OpenAI API key")


def main():
    """Main installation function."""
    print("Setting up Construction Claim Assistant...")
    
    check_python_version()
    check_tesseract()
    setup_virtual_env()
    setup_config()
    
    print("\nTo activate the environment:")
    if sys.platform == "win32":
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")
    
    print("\nTo get started:")
    print("  claimctl --help")


if __name__ == "__main__":
    main()