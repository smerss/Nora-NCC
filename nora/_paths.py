"""Repository and data path discovery for Nora."""
import os
from pathlib import Path


def get_repo_root() -> Path:
    try:
        start = Path(__file__).resolve()
        for parent in start.parents:
            if (parent / "pyproject.toml").exists() or (parent / "README.md").exists():
                return parent
    except (NameError, Exception):
        pass
    return Path.cwd()


REPO_ROOT = get_repo_root()
