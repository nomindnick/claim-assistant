"""Configuration module for the claim assistant."""

import configparser
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import typer


@dataclass
class PathsConfig:
    """Paths configuration for the application."""

    DATA_DIR: str
    INDEX_DIR: str


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""

    API_KEY: str
    MODEL: str
    EMBED_MODEL: str


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""

    TOP_K: int
    SCORE_THRESHOLD: float
    CONTEXT_SIZE: int  # Character limit per chunk for context window
    ANSWER_CONFIDENCE: bool  # Whether to include confidence indicators


@dataclass
class ChunkingConfig:
    """Text chunking configuration."""

    CHUNK_SIZE: int
    CHUNK_OVERLAP: int


@dataclass
class BM25Config:
    """BM25 configuration for keyword search."""

    K1: float = 1.5  # Term saturation parameter
    B: float = 0.75  # Length normalization parameter
    WEIGHT: float = 0.3  # Weight for combining with vector search


@dataclass
class ProjectConfig:
    """Project metadata configuration."""

    DEFAULT_PROJECT: Optional[str] = None


@dataclass
class Config:
    """Complete application configuration."""

    paths: PathsConfig
    openai: OpenAIConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    bm25: BM25Config
    project: ProjectConfig


def load_config() -> configparser.ConfigParser:
    """Load configuration from file."""
    # Default configuration file location
    config_path = os.path.expanduser("~/.claimctl.ini")

    # Create config parser
    config = configparser.ConfigParser()

    # Set default values
    config["paths"] = {
        "DATA_DIR": "./data",
        "INDEX_DIR": "./index",
    }
    config["openai"] = {
        "API_KEY": "",
        "MODEL": "gpt-4o-mini",
        "EMBED_MODEL": "text-embedding-3-large",
    }
    config["retrieval"] = {
        "TOP_K": "6",
        "SCORE_THRESHOLD": "0.6",
        "CONTEXT_SIZE": "2000",
        "ANSWER_CONFIDENCE": "True",
    }
    config["chunking"] = {
        "CHUNK_SIZE": "400",
        "CHUNK_OVERLAP": "100",
    }
    config["bm25"] = {
        "K1": "1.5",
        "B": "0.75",
        "WEIGHT": "0.3",
    }
    config["project"] = {
        "DEFAULT_PROJECT": "",
    }

    # Read configuration file
    if os.path.exists(config_path):
        config.read(config_path)

        # Add any missing sections from defaults
        for section in config.sections():
            if section in config.defaults():
                for key, value in config.defaults()[section].items():
                    if not config.has_option(section, key):
                        config.set(section, key, value)

    # Save config if it doesn't exist
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            config.write(f)

    return config


def get_config() -> Config:
    """Get configuration as a structured object."""
    config_parser = load_config()

    # Parse paths config
    paths_config = PathsConfig(
        DATA_DIR=config_parser.get("paths", "DATA_DIR"),
        INDEX_DIR=config_parser.get("paths", "INDEX_DIR"),
    )

    # Parse OpenAI config
    openai_config = OpenAIConfig(
        API_KEY=config_parser.get("openai", "API_KEY"),
        MODEL=config_parser.get("openai", "MODEL"),
        EMBED_MODEL=config_parser.get("openai", "EMBED_MODEL"),
    )

    # Parse retrieval config
    retrieval_config = RetrievalConfig(
        TOP_K=config_parser.getint("retrieval", "TOP_K"),
        SCORE_THRESHOLD=config_parser.getfloat("retrieval", "SCORE_THRESHOLD"),
        CONTEXT_SIZE=config_parser.getint("retrieval", "CONTEXT_SIZE", fallback=2000),
        ANSWER_CONFIDENCE=config_parser.getboolean(
            "retrieval", "ANSWER_CONFIDENCE", fallback=True
        ),
    )

    # Parse chunking config
    chunking_config = ChunkingConfig(
        CHUNK_SIZE=config_parser.getint("chunking", "CHUNK_SIZE"),
        CHUNK_OVERLAP=config_parser.getint("chunking", "CHUNK_OVERLAP"),
    )

    # Parse BM25 config with defaults if section missing
    if not config_parser.has_section("bm25"):
        config_parser.add_section("bm25")
        config_parser.set("bm25", "K1", "1.5")
        config_parser.set("bm25", "B", "0.75")
        config_parser.set("bm25", "WEIGHT", "0.3")

    bm25_config = BM25Config(
        K1=config_parser.getfloat("bm25", "K1"),
        B=config_parser.getfloat("bm25", "B"),
        WEIGHT=config_parser.getfloat("bm25", "WEIGHT"),
    )

    # Parse project config with defaults if section missing
    if not config_parser.has_section("project"):
        config_parser.add_section("project")
        config_parser.set("project", "DEFAULT_PROJECT", "")

    project_config = ProjectConfig(
        DEFAULT_PROJECT=config_parser.get("project", "DEFAULT_PROJECT") or None,
    )

    return Config(
        paths=paths_config,
        openai=openai_config,
        retrieval=retrieval_config,
        chunking=chunking_config,
        bm25=bm25_config,
        project=project_config,
    )


def ensure_dirs() -> None:
    """Ensure all required directories exist."""
    config = get_config()

    # Create data directory
    Path(config.paths.DATA_DIR).mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    (Path(config.paths.DATA_DIR) / "cache").mkdir(exist_ok=True)
    (Path(config.paths.DATA_DIR) / "pages").mkdir(exist_ok=True)
    (Path(config.paths.DATA_DIR) / "tmp").mkdir(exist_ok=True)

    # Create index directory
    Path(config.paths.INDEX_DIR).mkdir(exist_ok=True, parents=True)
    
    # Create ask exports directory
    Path("./Ask_Exports").mkdir(exist_ok=True, parents=True)


def show_config() -> Dict[str, Any]:
    """Return configuration as a dictionary for display."""
    config = get_config()

    return {
        "paths": {
            "DATA_DIR": config.paths.DATA_DIR,
            "INDEX_DIR": config.paths.INDEX_DIR,
        },
        "openai": {
            "MODEL": config.openai.MODEL,
            "EMBED_MODEL": config.openai.EMBED_MODEL,
            "API_KEY": (
                "****" + config.openai.API_KEY[-4:] if config.openai.API_KEY else ""
            ),
        },
        "retrieval": {
            "TOP_K": config.retrieval.TOP_K,
            "SCORE_THRESHOLD": config.retrieval.SCORE_THRESHOLD,
            "CONTEXT_SIZE": config.retrieval.CONTEXT_SIZE,
            "ANSWER_CONFIDENCE": config.retrieval.ANSWER_CONFIDENCE,
        },
        "chunking": {
            "CHUNK_SIZE": config.chunking.CHUNK_SIZE,
            "CHUNK_OVERLAP": config.chunking.CHUNK_OVERLAP,
        },
        "bm25": {
            "K1": config.bm25.K1,
            "B": config.bm25.B,
            "WEIGHT": config.bm25.WEIGHT,
        },
        "project": {
            "DEFAULT_PROJECT": config.project.DEFAULT_PROJECT or "None",
        },
    }


def save_config(config_dict: Dict[str, Dict[str, str]]) -> None:
    """Save configuration to file."""
    config_path = os.path.expanduser("~/.claimctl.ini")

    # Create config parser
    config = configparser.ConfigParser()

    # Load existing config
    if os.path.exists(config_path):
        config.read(config_path)

    # Update config with new values
    for section, values in config_dict.items():
        if not config.has_section(section):
            config.add_section(section)

        for key, value in values.items():
            config.set(section, key, str(value))

    # Save config
    with open(config_path, "w") as f:
        config.write(f)
