"""Canonical filesystem paths for the repo.

Import these from CLIs so defaults don't depend on where scripts live.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "model"
