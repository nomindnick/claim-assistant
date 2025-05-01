#!/bin/bash
# Launcher for claim-assistant

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Run ./setup_env.sh first."
    exit 1
fi

# Check for arguments
if [ $# -eq 0 ]; then
    # No arguments, run interactive mode
    python claim_assistant.py
else
    # Arguments provided, run normal CLI
    claimctl "$@"
fi