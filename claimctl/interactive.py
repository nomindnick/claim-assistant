"""Interactive shell for Construction Claim Assistant."""

from pathlib import Path
from typing import List, Tuple
import sys
import os

# Add parent directory to path so we can import custom_completer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import our custom completers, fall back to regular ones if not available
try:
    from custom_completer import SafePathCompleter, SafeNestedCompleter
except ImportError:
    print("Warning: custom_completer.py not found, using regular completers")
    from prompt_toolkit.completion import NestedCompleter, PathCompleter
    SafePathCompleter = PathCompleter
    SafeNestedCompleter = NestedCompleter

# Type checking imports - mypy will ignore missing stubs
from prompt_toolkit import PromptSession  # type: ignore
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory  # type: ignore
from prompt_toolkit.history import FileHistory  # type: ignore
from prompt_toolkit.styles import Style  # type: ignore
from rich.console import Console
from rich.panel import Panel

from .cli import app
from .config import get_config, get_current_matter
from .database import Matter, get_session, init_database

console = Console()


class ClaimAssistantShell:
    """Interactive shell for Construction Claim Assistant."""

    def __init__(self) -> None:
        """Initialize the shell."""
        self.running = True
        self.config = get_config()
        
        # Initialize database to ensure tables exist
        try:
            init_database()
            console.print("[green]Database initialized successfully[/green]")
        except Exception as e:
            console.print(f"[bold red]Error initializing database: {str(e)}[/bold red]")
            
        self.current_matter = get_current_matter()

        # Set up history file
        history_dir = Path.home() / ".claimctl"
        history_dir.mkdir(exist_ok=True)
        self.history_file = history_dir / "history"

        # Set up prompt style
        self.style = Style.from_dict(
            {
                "prompt": "ansigreen bold",
                "matter": "ansiyellow",
            }
        )

        # Set up command completer
        self._create_completer()

        # Set up prompt session
        self.session = PromptSession(
            history=FileHistory(str(self.history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            style=self.style,
            complete_while_typing=True,
        )

    def _create_completer(self) -> None:
        """Create nested completer for command auto-completion."""
        # Create path completer for file arguments - use our safe version
        path_completer = SafePathCompleter(
            expanduser=True, file_filter=lambda path: path and isinstance(path, str) and path.endswith(".pdf")
        )

        # Get matters for command completion - ensure they're all strings
        matters = self._get_matters_for_completion()
        
        # Build command hierarchy
        command_dict = {
            "ingest": {
                "--project": None,
                "-p": None,
                "--matter": None,
                "-m": None,
                "--recursive": None,
                "-r": None,
                "--batch-size": None,
                "-b": None,
                "--resume": None,
                "--no-resume": None,
                "--logging": None,
                "--no-logging": None,
                # Use explicit string paths - path_completer can sometimes cause issues
                "": path_completer,
            },
            "ask": None,  # Dynamic completion handled separately
            "matter": {
                "list": None,
                "create": None,
                "switch": {},  # Will be populated with valid matters
                "info": {},    # Will be populated with valid matters
                "delete": {},  # Will be populated with valid matters
            },
            "logs": {
                "list": {
                    "--matter": None,
                    "-m": None,
                    "--limit": None,
                    "-l": None,
                },
                "show": {
                    "--matter": None, 
                    "-m": None,
                },
            },
            "timeline": {
                "extract": {
                    "--matter": None,
                    "-m": None,
                },
                "show": {
                    "--matter": None,
                    "-m": None,
                    "--from": None,
                    "--to": None,
                    "--type": None,
                    "-t": None,
                    "--min-importance": None,
                    "--min-confidence": None,
                    "--max-events": None,
                    "--format": None,
                    "-f": None,
                },
                "export": {
                    "--matter": None,
                    "-m": None,
                    "--from": None,
                    "--to": None,
                    "--type": None,
                    "-t": None,
                    "--min-importance": None,
                    "--min-confidence": None,
                    "--max-events": None,
                    "--open": None,
                },
                "types": None,
            },
            "config": {
                "show": None,
                "init": None,
            },
            "clear": {
                "--database": None,
                "-d": None,
                "--embeddings": None,
                "-e": None,
                "--images": None,
                "-i": None,
                "--cache": None,
                "-c": None,
                "--exhibits": None,
                "-x": None,
                "--exports": None,
                "-p": None,
                "--resume-log": None,
                "-r": None,
                "--all": None,
                "-a": None,
            },
            "help": None,
            "exit": None,
            "quit": None,
        }
        
        # Safely add matters to the command dictionary
        for matter in matters:
            if matter is not None and isinstance(matter, str):
                command_dict["matter"]["switch"][matter] = None
                command_dict["matter"]["info"][matter] = None
                command_dict["matter"]["delete"][matter] = None
                
        # Create the completer using our safe version
        self.completer = SafeNestedCompleter.from_nested_dict(command_dict)

    def _update_prompt(self) -> List[Tuple[str, str]]:
        """Update the prompt with current matter information."""
        # Get fresh matter information from config
        self.current_matter = get_current_matter()
        
        # Verify matter actually exists in database
        if self.current_matter:
            try:
                with get_session() as session:
                    matter_exists = session.query(Matter).filter(Matter.name == self.current_matter).first() is not None
                    if not matter_exists:
                        # Matter doesn't exist but config thinks it does
                        from .config import set_current_matter
                        set_current_matter("")
                        self.current_matter = ""
            except Exception:
                # If there's a database error, just use what we have from config
                pass
                
        matter_display = (
            f"{self.current_matter}" if self.current_matter else "no matter"
        )
        return [
            ("class:prompt", "claim-assistant"),
            ("", " "),
            ("class:matter", f"[{matter_display}]"),
            ("", "> "),
        ]

    def _format_typer_args(self, args: List[str]) -> List[str]:
        """Format arguments for Typer app."""
        # Convert our simplified command format to Typer's format
        if not args:
            return []

        command = args[0]
        rest = args[1:]

        if command == "ask" and rest:
            # Join the rest as a single question string
            question = " ".join(rest)
            
            # For ask command, always include the question and additional parameters
            formatted_args = [command, question]
            
            # Debug: The top_k parameter will be added later
            return formatted_args

        return [command] + rest

    def _get_matters_for_completion(self) -> List[str]:
        """Get list of matters for command completion."""
        try:
            # Ensure database is initialized before querying
            try:
                init_database()
            except Exception as db_init_error:
                # Log but continue, as tables might already exist
                print(f"Note: Database initialization attempt: {db_init_error}")
                
            with get_session() as session:
                matters = session.query(Matter.name).all()
                # Extract matter names, ensure they're strings, and filter out None values
                result = []
                for m in matters:
                    if m[0] is not None:
                        try:
                            # Convert to string if it's not already and validate
                            matter_name = str(m[0])
                            if matter_name:  # Skip empty strings
                                result.append(matter_name)
                        except (TypeError, ValueError):
                            # Skip any value that can't be safely converted to string
                            pass
                return result
        except Exception as e:
            # Log the exception to help with debugging
            print(f"Error getting matters for completion: {e}")
            return []

    def show_welcome(self) -> None:
        """Show welcome message."""
        console.print(
            Panel.fit(
                f"[bold green]Construction Claim Assistant[/bold green]\n"
                f"Interactive CLI - Type 'help' for available commands\n"
                f"Current matter: [yellow]{self.current_matter or 'None'}[/yellow]",
                title="Welcome",
                subtitle="v0.1.0",
            )
        )

    def show_help(self) -> None:
        """Show help information."""
        console.print(
            Panel(
                "[bold]Available Commands:[/bold]\n\n"
                "ingest <pdf_path>      - Process PDF documents\n"
                "ask <question>         - Ask a question about your claim\n"
                "matter list            - List available matters\n"
                "matter create <name>   - Create a new matter\n"
                "matter switch <name>   - Switch to a different matter\n"
                "matter info [name]     - Show matter information\n"
                "matter delete <name>   - Delete a matter\n"
                "logs list              - List recent ingestion logs\n"
                "logs show [log_file]   - Show ingestion log summary\n"
                "timeline extract       - Extract timeline events from documents\n"
                "timeline show          - Display timeline of claim events\n"
                "timeline export        - Export timeline as PDF\n"
                "timeline types         - List valid timeline event types\n"
                "config show            - Show current configuration\n"
                "clear --all            - Clear all data\n"
                "help                   - Show this help message\n"
                "exit                   - Exit the application",
                title="Help",
            )
        )

    def handle_command(self, text: str) -> None:
        """Handle command input."""
        if not text.strip():
            return

        # Parse command
        parts = text.split()
        command = parts[0].lower()
        args = parts[1:]

        if command in ("exit", "quit"):
            self.running = False
            return

        elif command == "help":
            self.show_help()
            return

        # Format arguments for Typer
        typer_args = self._format_typer_args([command] + args)
        
        # Debug the command being run
        if command == "ask":
            console.log(f"Interactive: Running 'ask' command with args: {typer_args[1:]}")
            # Add --top-k 25 if not already specified
            if not any(arg.startswith("--top-k") or arg == "-k" for arg in typer_args):
                typer_args.append("--top-k")
                typer_args.append("25")
                console.log(f"Interactive: Added explicit --top-k 25 parameter: {typer_args}")

        # Run command through Typer
        try:
            app(typer_args)
        except SystemExit:
            # Catch SystemExit to prevent shell from exiting
            pass
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}")

    def run(self) -> None:
        """Run the interactive shell."""
        self.show_welcome()

        while self.running:
            try:
                # Get command
                text = self.session.prompt(self._update_prompt())

                # Handle command
                self.handle_command(text)

                # Refresh completer to get updated matter list
                self._create_completer()

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

        console.print("Goodbye!")


def main() -> None:
    """Main entry point for the interactive shell."""
    shell = ClaimAssistantShell()
    shell.run()


if __name__ == "__main__":
    main()
