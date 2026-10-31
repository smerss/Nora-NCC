#!/usr/bin/env python3
"""Backward-compatibility forwarding wrapper for tools.benchmark_profile."""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

if __name__ == "__main__":
    import importlib
    mod = importlib.import_module("tools.benchmark_profile")
    if hasattr(mod, "main"):
        res = mod.main()
    elif hasattr(mod, "run_benchmark"):
        _use_torch = "--with-torch" in sys.argv or "--torch" in sys.argv
        res = mod.run_benchmark(with_torch=_use_torch)
    if isinstance(res, int):
        sys.exit(res)
else:
    from tools.benchmark_profile import *
