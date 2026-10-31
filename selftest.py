#!/usr/bin/env python3
"""Backward-compatibility forwarding wrapper for batteries.selftest."""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

if __name__ == "__main__":
    import importlib
    mod = importlib.import_module("batteries.selftest")
    if hasattr(mod, "main"):
        res = mod.main()
        if isinstance(res, int):
            sys.exit(res)
else:
    from batteries.selftest import *
