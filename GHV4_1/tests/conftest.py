# tests conftest — add python/ source directory to sys.path

import sys
from pathlib import Path

# Ensure the python/ source directory is importable
_python_dir = str(Path(__file__).resolve().parent.parent / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)
