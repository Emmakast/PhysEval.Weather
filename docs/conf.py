"""Sphinx configuration for PhysMetrics.Weather documentation."""

import os
import sys
from pathlib import Path

# Add src/ directory to sys.path so autodoc can discover physmetrics_weather
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

project = "PhysMetrics.Weather"
copyright = "2026, Anonymous authors"
author = "Anonymous authors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

try:
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"

html_static_path = []
