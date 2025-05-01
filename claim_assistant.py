#!/usr/bin/env python
"""Interactive shell for Construction Claim Assistant."""

from claimctl.interactive import ClaimAssistantShell


def main() -> None:
    """Main entry point."""
    # Import here to avoid circular imports
    from claimctl.database import init_database
    
    # Initialize database before starting the shell
    try:
        init_database()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        
    shell = ClaimAssistantShell()
    shell.run()


if __name__ == "__main__":
    main()
