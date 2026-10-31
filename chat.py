#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chat.py — Nora Interactive Terminal Chat wrapper."""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from nora.chat import main

if __name__ == "__main__":
    main()
