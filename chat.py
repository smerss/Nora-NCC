#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chat.py — Nora Interactive WebUI & Terminal Chat launcher."""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
venv_py = os.path.join(_root, ".venv", "bin", "python")
if os.path.isfile(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
    os.execv(venv_py, [venv_py] + sys.argv)

if _root not in sys.path:
    sys.path.insert(0, _root)

from nora.chat import main

if __name__ == "__main__":
    main()
