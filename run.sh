#!/bin/bash
# Simple launcher for claim-assistant

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Run ./setup_env.sh first."
    exit 1
fi

# Run the CLI with provided arguments
claimctl "$@"