#!/usr/bin/env python
"""Thin wrapper that strips MSYSTEM env before invoking idf.py.

ESP-IDF's idf.py silently aborts on MSYSTEM presence. On Windows hosts
spawned from Git-Bash, MSYSTEM is inherited into every child process
from the Windows parent and cannot be dropped via `unset`/`env -u`.
This wrapper removes it from os.environ (affects THIS process only)
before importing/exec'ing idf.py's main.

Usage:  python tools/idf_wrapper.py [idf.py args...]
"""
from __future__ import annotations

import os
import runpy
import sys

os.environ.pop("MSYSTEM", None)
os.environ.pop("MINGW_PREFIX", None)
os.environ.pop("MINGW_CHOST", None)

idf_path = os.environ.get("IDF_PATH")
if not idf_path:
    sys.stderr.write("ERROR: IDF_PATH not set\n")
    sys.exit(2)

idf_py = os.path.join(idf_path, "tools", "idf.py")
if not os.path.exists(idf_py):
    sys.stderr.write(f"ERROR: {idf_py} not found\n")
    sys.exit(2)

sys.path.insert(0, os.path.join(idf_path, "tools"))
sys.argv = [idf_py] + sys.argv[1:]
runpy.run_path(idf_py, run_name="__main__")
