"""Where things live on disk. Everything personal stays under the data directory."""

import os
from pathlib import Path


def data_dir() -> Path:
    """Root for downloaded and derived data; override with RPA_DATA_DIR."""
    return Path(os.environ.get("RPA_DATA_DIR", "data"))


def graph_root() -> Path:
    return data_dir() / "graph"


def current_graph_dir() -> Path:
    """Symlink to the graph that commands use by default."""
    return graph_root() / "current"
