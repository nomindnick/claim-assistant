#!/usr/bin/env python
"""Interactive shell for Construction Claim Assistant."""

from claimctl.interactive import ClaimAssistantShell


def main() -> None:
    """Main entry point."""
    shell = ClaimAssistantShell()
    shell.run()


if __name__ == "__main__":
    main()
