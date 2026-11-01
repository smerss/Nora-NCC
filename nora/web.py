#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nora/web.py — Nora WebUI and Real-Time Event Server
Provides a modern, lightweight, zero-dependency browser interface for Nora AGI.
Automatically opens a browser tab and handles interactive chat, live vitals,
and the real-time stream of consciousness (inner thoughts).
"""

from __future__ import annotations

import html
import json
import os
import queue
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from nora._paths import REPO_ROOT
try:
    from nora.cognition.mind import I_E, I_W, I_I
except ImportError:
    I_E, I_W, I_I = 45, 46, 47

HTML_PAGE = """<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nora AGI</title>
  <link rel="icon" type="image/png" href="/logo.png">
  <style>
    :root {
      --bg-app: #262624;
      --bg-sidebar: #1f1f1d;
      --bg-card: #30302d;
      --bg-card-hover: #383834;
      --bg-input: #30302d;
      --border: #3d3d3a;
      --border-subtle: #2d2d2a;
      --border-focus: #52524e;
      --text: #ece9e2;
      --text-muted: #b0ada5;
      --text-dim: #737168;
      --accent: #d97757;
      --accent-hover: #e08365;
      --green-live: #10b981;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-serif: "Tiempos Headline", Georgia, Cambria, "Times New Roman", Times, serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--font-sans);
      background: var(--bg-app);
      color: var(--text);
      height: 100vh;
      width: 100vw;
      display: flex;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }

    /* Left Sidebar (Claude Style) */
    aside.sidebar {
      width: 260px;
      min-width: 260px;
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      height: 100%;
      user-select: none;
      transition: margin-left 0.25s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.2s ease;
      z-index: 30;
      flex-shrink: 0;
    }

    aside.sidebar.collapsed {
      margin-left: -260px;
      opacity: 0;
      pointer-events: none;
    }

    .sidebar-header {
      padding: 14px 16px 10px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .brand-title {
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
      text-decoration: none;
      font-family: var(--font-serif);
      letter-spacing: -0.3px;
    }

    .brand-title img {
      height: 22px;
      width: auto;
      border-radius: 4px;
    }

    .icon-btn {
      width: 28px;
      height: 28px;
      border-radius: 6px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s ease;
    }

    .icon-btn:hover {
      background: var(--bg-card);
      color: var(--text);
    }

    .btn-new-chat {
      margin: 6px 14px 12px 14px;
      padding: 8px 12px;
      border-radius: 8px;
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 0.88rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .btn-new-chat:hover {
      background: var(--bg-card);
      border-color: var(--text-dim);
    }

    .sidebar-nav {
      display: flex;
      flex-direction: column;
      gap: 2px;
      padding: 0 10px;
    }

    .nav-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 0.84rem;
      color: var(--text-muted);
      cursor: pointer;
      text-decoration: none;
      transition: all 0.15s ease;
    }

    .nav-item:hover {
      background: var(--bg-card);
      color: var(--text);
    }

    .nav-item-left {
      display: flex;
      align-items: center;
      gap: 9px;
    }

    .nav-badge {
      font-size: 0.7rem;
      padding: 1px 6px;
      border-radius: 10px;
      background: var(--bg-card);
      color: var(--accent);
      font-family: monospace;
    }

    .sidebar-section-title {
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      padding: 14px 16px 6px 16px;
      margin-top: 6px;
    }

    .chat-recents-list {
      flex: 1;
      overflow-y: auto;
      padding: 0 10px;
      display: flex;
      flex-direction: column;
      gap: 1px;
    }

    .recent-chat-item {
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 0.82rem;
      color: var(--text-muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: all 0.15s ease;
    }

    .recent-chat-item:hover, .recent-chat-item.active {
      background: var(--bg-card);
      color: var(--text);
    }

    .recent-del-btn {
      opacity: 0;
      color: var(--text-dim);
      background: transparent;
      border: none;
      cursor: pointer;
      font-size: 0.8rem;
      padding: 0 4px;
    }

    .recent-chat-item:hover .recent-del-btn {
      opacity: 1;
    }

    .recent-del-btn:hover {
      color: #ef4444;
    }

    .sidebar-footer {
      padding: 12px 14px;
      border-top: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .profile-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
      color: var(--text-muted);
      cursor: pointer;
    }

    .live-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--green-live);
      box-shadow: 0 0 6px var(--green-live);
    }

    /* Main Container */
    .main-viewport {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100%;
      position: relative;
      overflow: hidden;
      background: var(--bg-app);
    }

    /* Top bar */
    .main-topbar {
      height: 48px;
      padding: 0 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid transparent;
      z-index: 10;
      flex-shrink: 0;
    }

    .topbar-left {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .chat-title-text {
      font-size: 0.88rem;
      font-weight: 500;
      color: var(--text-muted);
    }

    .topbar-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Chat Scroll Area */
    .chat-scroll-area {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px 20px 140px 20px;
      position: relative;
    }

    /* Landing / Hero State (Claude Style) */
    .landing-hero {
      margin: auto 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      width: 100%;
      max-width: 680px;
      gap: 24px;
      animation: fadeIn 0.3s ease;
    }

    .hero-sparkle {
      color: var(--accent);
      font-size: 2.2rem;
      display: inline-block;
      line-height: 1;
    }

    .hero-title {
      font-family: var(--font-serif);
      font-size: 2.3rem;
      font-weight: 400;
      color: #ece9e2;
      letter-spacing: -0.5px;
      line-height: 1.2;
    }

    .hero-input-card {
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 16px 12px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .hero-input-card:focus-within {
      border-color: var(--border-focus);
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
    }

    .hero-textarea {
      width: 100%;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 0.98rem;
      line-height: 1.5;
      resize: none;
      min-height: 54px;
      max-height: 180px;
    }

    .hero-textarea::placeholder {
      color: var(--text-dim);
    }

    .hero-card-bottom {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-top: 4px;
    }

    .hero-tools-left {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .tool-btn {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 4px 8px;
      border-radius: 6px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.78rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .tool-btn:hover {
      background: var(--bg-card-hover);
      color: var(--text);
    }

    .btn-send-round {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: var(--accent);
      color: #ffffff;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.15s ease, opacity 0.15s ease;
    }

    .btn-send-round:hover {
      background: var(--accent-hover);
    }

    .btn-send-round:disabled {
      background: var(--border);
      color: var(--text-dim);
      cursor: not-allowed;
    }

    /* Artifacts / Suggestion Cards Row (Claude Style) */
    .hero-artifacts-section {
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-top: 10px;
      text-align: left;
    }

    .artifacts-header {
      font-size: 0.75rem;
      font-weight: 500;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
    }

    .artifacts-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }

    .artifact-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .artifact-card:hover {
      background: var(--bg-card-hover);
      border-color: var(--border-focus);
      transform: translateY(-1px);
    }

    .artifact-title {
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text);
      line-height: 1.4;
    }

    .artifact-sub {
      font-size: 0.72rem;
      color: var(--text-dim);
    }

    /* Active Conversation Flow */
    .chat-thread {
      width: 100%;
      max-width: 720px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .msg-item {
      display: flex;
      flex-direction: column;
      gap: 6px;
      animation: fadeIn 0.2s ease;
    }

    .msg-item.user {
      align-self: flex-end;
      max-width: 85%;
    }

    .msg-item.nora {
      align-self: flex-start;
      width: 100%;
    }

    .msg-bubble-user {
      background: #333330;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px 16px;
      border-radius: 16px;
      font-size: 0.93rem;
      line-height: 1.55;
      word-break: break-word;
      white-space: pre-wrap;
    }

    .msg-nora-head {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 2px;
    }

    .msg-nora-avatar {
      width: 20px;
      height: 20px;
      border-radius: 4px;
      background: var(--accent);
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.65rem;
      font-weight: 700;
    }

    .msg-body-nora {
      color: #ece9e2;
      font-size: 0.95rem;
      line-height: 1.65;
      word-break: break-word;
      padding-left: 28px;
    }

    .msg-body-nora p {
      margin: 0 0 10px 0;
    }

    .msg-body-nora p:last-child {
      margin-bottom: 0;
    }

    .msg-body-nora h1, .msg-body-nora h2, .msg-body-nora h3, .msg-body-nora h4 {
      color: var(--text);
      margin: 16px 0 8px 0;
      font-weight: 700;
    }

    .msg-body-nora h3 { font-size: 1.05rem; }
    .msg-body-nora h2 { font-size: 1.15rem; }
    .msg-body-nora h1 { font-size: 1.25rem; }

    .msg-body-nora ul.md-list {
      margin: 8px 0 12px 20px;
      padding: 0;
    }

    .msg-body-nora ul.md-list li {
      margin-bottom: 6px;
      line-height: 1.55;
    }

    .inline-code {
      background: #20201d;
      border: 1px solid var(--border);
      color: #d97757;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.85rem;
    }

    /* Modern Code Box (Kutulu Kod Görünümü) */
    .code-box {
      margin: 14px 0 18px 0;
      background: #171716;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }

    .code-box-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 14px;
      background: #20201d;
      border-bottom: 1px solid var(--border-subtle);
      user-select: none;
    }

    .code-box-lang-badge {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .code-box-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent);
      display: inline-block;
    }

    .code-box-lang {
      font-size: 0.72rem;
      font-family: ui-monospace, monospace;
      font-weight: 700;
      letter-spacing: 0.6px;
      color: var(--text-muted);
    }

    .code-box-copy {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 6px;
      padding: 4px 10px;
      font-size: 0.72rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }

    .code-box-copy:hover {
      background: var(--bg-card);
      color: var(--text);
      border-color: var(--text-dim);
    }

    .code-box-copy.copied {
      color: var(--green-live);
      border-color: var(--green-live);
    }

    .code-box pre {
      margin: 0;
      padding: 16px;
      overflow-x: auto;
      background: #141413;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.86rem;
      line-height: 1.55;
      color: #e6e4dd;
      white-space: pre;
    }

    .code-box code {
      font-family: inherit;
      background: transparent;
      padding: 0;
      color: inherit;
    }

    .msg-meta-row {
      font-size: 0.68rem;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 4px;
      padding-left: 28px;
    }

    .msg-item.user .msg-meta-row {
      padding-left: 0;
      justify-content: flex-end;
    }

    /* Typing dots */
    .typing-row {
      display: flex;
      align-items: center;
      gap: 4px;
      padding-left: 28px;
    }

    .typing-row span {
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: var(--text-dim);
      animation: blink 1.4s infinite both;
    }

    .typing-row span:nth-child(2) { animation-delay: 0.2s; }
    .typing-row span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes blink {
      0%, 80%, 100% { opacity: 0.2; transform: scale(0.9); }
      40% { opacity: 1; transform: scale(1.1); }
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(3px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Sticky Bottom Input Bar (Active Chat) */
    .active-input-container {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      padding: 16px 20px 20px 20px;
      background: linear-gradient(180deg, rgba(38, 38, 36, 0) 0%, rgba(38, 38, 36, 0.92) 20%, var(--bg-app) 100%);
      display: flex;
      flex-direction: column;
      align-items: center;
      z-index: 20;
    }

    .active-input-box {
      width: 100%;
      max-width: 720px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 10px 14px 8px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
      transition: border-color 0.15s ease;
    }

    .active-input-box:focus-within {
      border-color: var(--border-focus);
    }

    .active-textarea {
      width: 100%;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 0.92rem;
      line-height: 1.45;
      resize: none;
      min-height: 24px;
      max-height: 160px;
      padding: 2px 0;
    }

    .active-textarea::placeholder {
      color: var(--text-dim);
    }

    .active-bar-bottom {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Slide-Over Drawer */
    .drawer-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.55);
      backdrop-filter: blur(2px);
      z-index: 80;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
    }

    .drawer-backdrop.show {
      opacity: 1;
      pointer-events: auto;
    }

    .drawer-panel {
      position: fixed;
      top: 0;
      bottom: 0;
      right: 0;
      width: 380px;
      max-width: 90vw;
      background: var(--bg-sidebar);
      border-left: 1px solid var(--border);
      z-index: 90;
      display: flex;
      flex-direction: column;
      box-shadow: -10px 0 30px rgba(0, 0, 0, 0.4);
      transform: translateX(100%);
      transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .drawer-panel.show {
      transform: translateX(0);
    }

    .drawer-head {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.95rem;
      font-weight: 600;
    }

    .drawer-content {
      flex: 1;
      overflow-y: auto;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    /* Minimal Vitals Cards */
    .vitals-meter-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .meter-row {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }

    .meter-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      color: var(--text-muted);
    }

    .meter-track {
      height: 6px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 3px;
      overflow: hidden;
    }

    .meter-bar {
      height: 100%;
      border-radius: 3px;
      background: var(--accent);
      transition: width 0.3s ease;
      width: 0%;
    }

    /* Thoughts items in drawer */
    .thought-bubble {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 0.82rem;
      line-height: 1.5;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .thought-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.68rem;
      color: var(--text-dim);
    }

    /* Modals */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.65);
      backdrop-filter: blur(3px);
      z-index: 100;
      display: none;
      align-items: center;
      justify-content: center;
    }

    .modal-overlay.show {
      display: flex;
    }

    .modal-sheet {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      width: 90%;
      max-width: 480px;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5);
    }

    .modal-sheet h3 {
      font-size: 1.05rem;
      font-weight: 600;
    }

    .modal-sheet input {
      width: 100%;
      padding: 10px 12px;
      background: var(--bg-app);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-size: 0.9rem;
      outline: none;
    }

    .modal-sheet input:focus {
      border-color: var(--border-focus);
    }

    .modal-foot {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 6px;
    }

    .btn-action {
      padding: 7px 14px;
      border-radius: 6px;
      font-size: 0.82rem;
      cursor: pointer;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--text-muted);
    }

    .btn-action:hover {
      color: var(--text);
      background: var(--bg-card-hover);
    }

    .btn-accent {
      padding: 7px 16px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      border: none;
      background: var(--accent);
      color: #fff;
    }

    .btn-accent:hover {
      background: var(--accent-hover);
    }

    /* Toast */
    .toast-pill {
      position: fixed;
      bottom: 90px;
      left: 50%;
      transform: translateX(-50%);
      background: #333330;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 16px;
      border-radius: 20px;
      font-size: 0.82rem;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
      z-index: 200;
      animation: fadeIn 0.15s ease;
    }

    @media (max-width: 768px) {
      aside.sidebar {
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
      }
      .artifacts-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>

  <!-- Left Sidebar (Claude.ai Replica) -->
  <aside id="sidebar" class="sidebar">
    <div class="sidebar-header">
      <a href="#" class="brand-title" onclick="newChat(); return false;">
        <span style="color: var(--accent); font-size: 1.2rem;">✳</span>
        <span>Nora</span>
      </a>
      <button class="icon-btn" onclick="toggleSidebar()" title="Kenar Çubuğunu Daralt">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
          <line x1="9" y1="3" x2="9" y2="21"></line>
        </svg>
      </button>
    </div>

    <!-- New Chat Button -->
    <button class="btn-new-chat" onclick="newChat()">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="12" y1="5" x2="12" y2="19"></line>
        <line x1="5" y1="12" x2="19" y2="12"></line>
      </svg>
      <span>+ Yeni Sohbet</span>
    </button>

    <!-- Navigation Shortcuts -->
    <div class="sidebar-nav">
      <div class="nav-item" onclick="openDrawer('drawer-thoughts')">
        <div class="nav-item-left">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
          <span>Zihin Akışı</span>
        </div>
        <span id="side-thoughts-count" class="nav-badge">0</span>
      </div>

      <div class="nav-item" onclick="openDrawer('drawer-vitals')">
        <div class="nav-item-left">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
          </svg>
          <span>Biyolojik Homeostaz</span>
        </div>
        <span id="side-vitals-pill" class="nav-badge" style="color: var(--green-live);">%93</span>
      </div>

      <div class="nav-item" onclick="openTeachModal()">
        <div class="nav-item-left">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="16"></line>
            <line x1="8" y1="12" x2="16" y2="12"></line>
          </svg>
          <span>1-Shot Bilgi Öğret</span>
        </div>
      </div>

      <div class="nav-item" onclick="saveBrain()">
        <div class="nav-item-left">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
            <polyline points="17 21 17 13 7 13 7 21"></polyline>
            <polyline points="7 3 7 8 15 8"></polyline>
          </svg>
          <span>Beyni Kaydet</span>
        </div>
      </div>

      <div class="nav-item" onclick="triggerSleep()">
        <div class="nav-item-left">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
          </svg>
          <span>Bakım Uykusu</span>
        </div>
      </div>
    </div>

    <!-- Recents Section -->
    <div class="sidebar-section-title">Geçmiş Sohbetler</div>
    <div id="recents-list" class="chat-recents-list">
      <!-- Session items will populate dynamically -->
    </div>

    <!-- Sidebar Bottom Profile / Status -->
    <div class="sidebar-footer">
      <div class="profile-pill" onclick="openDrawer('drawer-vitals')" title="Durum ayrıntıları">
        <span class="live-dot"></span>
        <span id="side-profile-text">Nora · Yaş 7039</span>
      </div>
      <button class="icon-btn" onclick="openAbsorbModal()" title="Metin Dosyası Oku (/absorb)">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
      </button>
    </div>
  </aside>

  <!-- Main Viewport -->
  <div class="main-viewport">

    <!-- Topbar (Visible when sidebar collapsed or for title) -->
    <div class="main-topbar">
      <div class="topbar-left">
        <button id="btn-expand-sidebar" class="icon-btn" style="display: none;" onclick="toggleSidebar()" title="Kenar Çubuğunu Aç">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="9" y1="3" x2="9" y2="21"></line>
          </svg>
        </button>
        <span id="topbar-chat-title" class="chat-title-text">Nora</span>
      </div>

      <div class="topbar-actions">
        <button class="tool-btn" onclick="openDrawer('drawer-thoughts')" title="Zihin Akışı (İçses)">
          <span>💭 İçses</span>
          <span id="top-thought-badge" class="nav-badge">0</span>
        </button>
      </div>
    </div>

    <!-- Scroll Area: Landing Hero OR Active Thread -->
    <div id="chat-scroll" class="chat-scroll-area">

      <!-- 1. LANDING HERO (Empty / New Chat State like Claude.ai) -->
      <div id="landing-hero" class="landing-hero">
        <div class="hero-sparkle">✳</div>
        <h1 class="hero-title">Merhaba, ben Nora</h1>

        <!-- Centered Large Input Card (Claude Style) -->
        <div class="hero-input-card">
          <textarea
            id="hero-input"
            class="hero-textarea"
            placeholder="Nora'ya ne sormak istersin?"
            rows="2"
            onkeydown="onHeroKey(event)"
            oninput="resizeHero(this)"
          ></textarea>

          <div class="hero-card-bottom">
            <div class="hero-tools-left">
              <button class="tool-btn" onclick="openTeachModal()" title="1-Shot Bilgi Öğret">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
                <span>Öğret</span>
              </button>
              <button class="tool-btn" onclick="openDrawer('drawer-vitals')" title="Durum">
                <span>Nora 0-Epoch · Canlı</span>
              </button>
            </div>

            <button id="btn-hero-send" class="btn-send-round" onclick="sendFromHero()" title="Gönder">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <line x1="12" y1="19" x2="12" y2="5"></line>
                <polyline points="5 12 12 5 19 12"></polyline>
              </svg>
            </button>
          </div>
        </div>

        <!-- Suggestion Cards (Like Claude's Recent Artifacts / Prompts) -->
        <div class="hero-artifacts-section">
          <div class="artifacts-header">
            <span>Örnek Başlangıçlar</span>
          </div>
          <div class="artifacts-grid">
            <div class="artifact-card" onclick="quickAsk('Serçe bir kuş mudur?')">
              <div class="artifact-title">Serçe bir kuş mudur?</div>
              <div class="artifact-sub">Ontolojik akıl yürütme</div>
            </div>
            <div class="artifact-card" onclick="quickAsk('Bedensel durumun ve enerjin nasıl?')">
              <div class="artifact-title">Bedensel durumun nasıl?</div>
              <div class="artifact-sub">Biyolojik homeostaz ve dürtü</div>
            </div>
            <div class="artifact-card" onclick="quickAsk('12 elmam vardı, 5\\'ini verirsem kaç kalır?')">
              <div class="artifact-title">12 - 5 elma kaç kalır?</div>
              <div class="artifact-sub">Sistem-2 aritmetik</div>
            </div>
          </div>
        </div>
      </div>

      <!-- 2. ACTIVE CONVERSATION THREAD -->
      <div id="chat-thread" class="chat-thread" style="display: none;">
        <!-- Message bubbles populate here -->
      </div>

      <!-- Typing Indicator -->
      <div id="typing-indicator" class="typing-row" style="display: none; width: 100%; max-width: 720px; margin-top: 10px;">
        <span></span><span></span><span></span>
      </div>

    </div>

    <!-- Active Chat Sticky Bottom Input (Visible during active conversation) -->
    <div id="active-input-wrap" class="active-input-container" style="display: none;">
      <div class="active-input-box">
        <textarea
          id="active-input"
          class="active-textarea"
          placeholder="Nora'ya mesaj yazın..."
          rows="1"
          onkeydown="onActiveKey(event)"
          oninput="resizeActive(this)"
        ></textarea>

        <div class="active-bar-bottom">
          <div class="hero-tools-left">
            <button class="tool-btn" onclick="openTeachModal()" title="1-Shot Bilgi Öğret">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line>
              </svg>
              <span>Öğret</span>
            </button>
          </div>

          <button id="btn-active-send" class="btn-send-round" onclick="sendFromActive()" title="Gönder">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <line x1="12" y1="19" x2="12" y2="5"></line>
              <polyline points="5 12 12 5 19 12"></polyline>
            </svg>
          </button>
        </div>
      </div>
    </div>

  </div>

  <!-- Drawer: Zihin Akışı / İçses (Slides from Right) -->
  <div id="drawer-backdrop" class="drawer-backdrop" onclick="closeDrawer()"></div>

  <div id="drawer-thoughts" class="drawer-panel">
    <div class="drawer-head">
      <span>Zihin Akışı (İçses)</span>
      <button class="icon-btn" onclick="closeDrawer()">&times;</button>
    </div>
    <div id="thoughts-feed" class="drawer-content">
      <div id="thoughts-empty" style="color: var(--text-dim); font-size: 0.85rem; text-align: center; margin: auto;">
        Nora'nın arka plandaki yaşayan organizmasının iç sesleri burada gerçek zamanlı belirir.
      </div>
    </div>
  </div>

  <!-- Drawer: Biyolojik Homeostaz (Slides from Right) -->
  <div id="drawer-vitals" class="drawer-panel">
    <div class="drawer-head">
      <span>Biyolojik Homeostaz & Durum</span>
      <button class="icon-btn" onclick="closeDrawer()">&times;</button>
    </div>
    <div class="drawer-content">
      <div class="vitals-meter-card">
        <div class="meter-row">
          <div class="meter-labels">
            <span>Enerji</span>
            <span id="v-energy">-%</span>
          </div>
          <div class="meter-track"><div id="v-bar-energy" class="meter-bar"></div></div>
        </div>
        <div class="meter-row">
          <div class="meter-labels">
            <span>Su</span>
            <span id="v-water">-%</span>
          </div>
          <div class="meter-track"><div id="v-bar-water" class="meter-bar" style="background: #3b82f6;"></div></div>
        </div>
        <div class="meter-row">
          <div class="meter-labels">
            <span>Bütünlük</span>
            <span id="v-integrity">-%</span>
          </div>
          <div class="meter-track"><div id="v-bar-integrity" class="meter-bar" style="background: #10b981;"></div></div>
        </div>
      </div>

      <div class="vitals-meter-card" style="font-size: 0.84rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
          <span style="color: var(--text-dim);">Yaşam Adımı:</span>
          <strong id="v-age">-</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
          <span style="color: var(--text-dim);">Baskın Dürtü:</span>
          <strong id="v-mood">-</strong>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: var(--text-dim);">Bilinç İnancı:</span>
          <strong id="v-conscious" style="color: var(--green-live);">-</strong>
        </div>
      </div>
    </div>
  </div>

  <!-- Modal: 1-Shot Bilgi Öğret (/teach) -->
  <div id="modal-teach" class="modal-overlay">
    <div class="modal-sheet">
      <h3>1-Shot Bilgi Öğret</h3>
      <p style="font-size: 0.84rem; color: var(--text-muted); line-height: 1.5;">
        Nora tek bir cümleden ontolojik kavram ve kuralları 0-epoch ile anında öğrenir ve kalıcı olarak hafızaya kaydeder.
      </p>
      <input id="inp-teach" placeholder="Örn: Mars kızıl bir gezegendir." onkeydown="if(event.key==='Enter') submitTeach()">
      <div class="modal-foot">
        <button class="btn-action" onclick="closeModal('modal-teach')">İptal</button>
        <button class="btn-accent" onclick="submitTeach()">Hafızaya İşle</button>
      </div>
    </div>
  </div>

  <!-- Modal: Metin Dosyası Oku (/absorb) -->
  <div id="modal-absorb" class="modal-overlay">
    <div class="modal-sheet">
      <h3>Metin Dosyası Oku & Özümse</h3>
      <p style="font-size: 0.84rem; color: var(--text-muted); line-height: 1.5;">
        Belirtilen metin dosyasındaki tüm cümleler semantik hafızaya aktarılır.
      </p>
      <input id="inp-absorb" placeholder="Örn: data/bilgi.txt" onkeydown="if(event.key==='Enter') submitAbsorb()">
      <div class="modal-foot">
        <button class="btn-action" onclick="closeModal('modal-absorb')">İptal</button>
        <button class="btn-accent" onclick="submitAbsorb()">Dosyayı Oku</button>
      </div>
    </div>
  </div>

  <script>
    // State
    let isGenerating = false;
    let thoughtCount = 0;
    let currentSessionId = null;
    let sessions = []; // [{ id, title, messages: [] }]

    // Elements
    const sidebar = document.getElementById("sidebar");
    const btnExpandSidebar = document.getElementById("btn-expand-sidebar");
    const landingHero = document.getElementById("landing-hero");
    const chatThread = document.getElementById("chat-thread");
    const activeInputWrap = document.getElementById("active-input-wrap");
    const chatScroll = document.getElementById("chat-scroll");
    const heroInput = document.getElementById("hero-input");
    const activeInput = document.getElementById("active-input");
    const typingIndicator = document.getElementById("typing-indicator");
    const recentsList = document.getElementById("recents-list");
    const topbarChatTitle = document.getElementById("topbar-chat-title");

    function toast(txt) {
      const existing = document.querySelector(".toast-pill");
      if (existing) existing.remove();
      const el = document.createElement("div");
      el.className = "toast-pill";
      el.textContent = txt;
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 3200);
    }

    // Sidebar Toggle
    function toggleSidebar() {
      const isCollapsed = sidebar.classList.toggle("collapsed");
      btnExpandSidebar.style.display = isCollapsed ? "flex" : "none";
      localStorage.setItem("nora_sidebar_collapsed", isCollapsed ? "1" : "0");
    }

    if (localStorage.getItem("nora_sidebar_collapsed") === "1") {
      sidebar.classList.add("collapsed");
      btnExpandSidebar.style.display = "flex";
    }

    // Textarea Auto-Resize
    function resizeHero(el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 180) + "px";
    }
    function resizeActive(el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }

    function onHeroKey(e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendFromHero();
      }
    }
    function onActiveKey(e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendFromActive();
      }
    }

    function quickAsk(text) {
      heroInput.value = text;
      sendFromHero();
    }

    // Sessions & Storage
    function loadSessions() {
      try {
        const raw = localStorage.getItem("nora_sessions");
        sessions = raw ? JSON.parse(raw) : [];
      } catch (e) {
        sessions = [];
      }
      renderRecents();
    }

    function saveSessions() {
      localStorage.setItem("nora_sessions", JSON.stringify(sessions));
      renderRecents();
    }

    function renderRecents() {
      recentsList.innerHTML = "";
      sessions.forEach(sess => {
        const item = document.createElement("div");
        item.className = `recent-chat-item ${sess.id === currentSessionId ? 'active' : ''}`;
        item.onclick = (e) => {
          if (!e.target.classList.contains("recent-del-btn")) {
            selectSession(sess.id);
          }
        };

        const span = document.createElement("span");
        span.textContent = sess.title || "Yeni Sohbet";
        span.style.overflow = "hidden";
        span.style.textOverflow = "ellipsis";

        const del = document.createElement("button");
        del.className = "recent-del-btn";
        del.innerHTML = "&times;";
        del.title = "Sohbeti Sil";
        del.onclick = (e) => {
          e.stopPropagation();
          deleteSession(sess.id);
        };

        item.appendChild(span);
        item.appendChild(del);
        recentsList.appendChild(item);
      });
    }

    function newChat() {
      currentSessionId = null;
      chatThread.innerHTML = "";
      chatThread.style.display = "none";
      activeInputWrap.style.display = "none";
      landingHero.style.display = "flex";
      topbarChatTitle.textContent = "Nora";
      heroInput.value = "";
      heroInput.style.height = "54px";
      setTimeout(() => heroInput.focus(), 50);
      renderRecents();
    }

    function selectSession(id) {
      const sess = sessions.find(s => s.id === id);
      if (!sess) return;
      currentSessionId = id;
      topbarChatTitle.textContent = sess.title || "Sohbet";

      landingHero.style.display = "none";
      chatThread.style.display = "flex";
      activeInputWrap.style.display = "flex";
      chatThread.innerHTML = "";

      sess.messages.forEach(m => {
        renderMessageItem(m.sender, m.text, m.latency);
      });

      renderRecents();
      chatScroll.scrollTop = chatScroll.scrollHeight;
      setTimeout(() => activeInput.focus(), 50);
    }

    function deleteSession(id) {
      sessions = sessions.filter(s => s.id !== id);
      saveSessions();
      if (currentSessionId === id) {
        newChat();
      }
    }

    function escapeHtml(str) {
      if (!str) return "";
      return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function formatMarkdown(text) {
      if (!text) return "";

      // 1. Fenced Code Blocks (```lang ... ```)
      const codeBlocks = [];
      const codeRegex = new RegExp("```([a-zA-Z0-9_.-]*)[\\\\r\\\\n]+([\\\\s\\\\S]*?)```", "g");
      let processed = text.replace(codeRegex, function(match, lang, code) {
        const idx = codeBlocks.length;
        codeBlocks.push({
          lang: (lang || "code").trim(),
          code: code.replace(/[\\\\r\\\\n]+$/, "")
        });
        return "\\n\\n__CODE_BLOCK_" + idx + "__\\n\\n";
      });

      // 2. HTML Escape
      processed = escapeHtml(processed);

      // 3. Inline code (`code`)
      processed = processed.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

      // 4. Bold and Italic
      processed = processed.replace(/[*]{2}([^*]+)[*]{2}/g, '<strong>$1</strong>');
      processed = processed.replace(/(^|[^*])[*]([^*]+)[*]([^*]|$)/g, '$1<em>$2</em>$3');

      // 5. Headings (###, ##, #)
      const headingRegex = new RegExp("^(#{1,4})\\\\s+(.+)$", "gm");
      processed = processed.replace(headingRegex, function(m, hashes, title) {
        const level = hashes.length;
        return "<h" + level + ">" + title.trim() + "</h" + level + ">";
      });

      // 6. Lists
      const lines = processed.split("\\n");
      const outLines = [];
      let inUl = false;
      const bulletRegex = new RegExp("^\\\\s*-\\\\s+(.*)$");

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const bulletMatch = line.match(bulletRegex);
        if (bulletMatch) {
          if (!inUl) {
            outLines.push('<ul class="md-list">');
            inUl = true;
          }
          outLines.push("  <li>" + bulletMatch[1] + "</li>");
        } else {
          if (inUl) {
            outLines.push("</ul>");
            inUl = false;
          }
          outLines.push(line);
        }
      }
      if (inUl) {
        outLines.push("</ul>");
      }
      processed = outLines.join("\\n");

      // 7. Paragraphs
      const paraRegex = new RegExp("\\\\n\\\\s*\\\\n");
      const paragraphs = processed.split(paraRegex);
      const nlRegex = new RegExp("\\\\n", "g");
      const formattedParas = paragraphs.map(function(p) {
        const trimmed = p.trim();
        if (!trimmed) return "";
        if (
          trimmed.startsWith("<h") ||
          trimmed.startsWith("<ul") ||
          trimmed.startsWith("<ol") ||
          trimmed.startsWith("__CODE_BLOCK_")
        ) {
          return trimmed;
        }
        return "<p>" + trimmed.replace(nlRegex, "<br>") + "</p>";
      }).filter(Boolean);

      let result = formattedParas.join("\\n\\n");

      // 8. Restore Code Boxes
      codeBlocks.forEach(function(block, idx) {
        const escapedCode = escapeHtml(block.code);
        const langLabel = block.lang ? block.lang.toUpperCase() : "KOD";
        const box = '<div class="code-box">' +
          '<div class="code-box-header">' +
          '<div class="code-box-lang-badge">' +
          '<span class="code-box-dot"></span>' +
          '<span class="code-box-lang">' + escapeHtml(langLabel) + '</span>' +
          '</div>' +
          '<button type="button" class="code-box-copy" onclick="copyCodeBlock(this)" title="Kodu Kopyala">' +
          '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
          '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
          '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
          '</svg>' +
          '<span>Kopyala</span>' +
          '</button>' +
          '</div>' +
          '<pre><code class="language-' + escapeHtml(block.lang) + '">' + escapedCode + '</code></pre>' +
          '</div>';

        result = result.replace(new RegExp("<p>__CODE_BLOCK_" + idx + "__<\\\\/p>", "g"), box);
        result = result.replace(new RegExp("__CODE_BLOCK_" + idx + "__", "g"), box);
      });

      return result;
    }

    function copyCodeBlock(btn) {
      const box = btn.closest(".code-box");
      const code = box ? box.querySelector("pre code") : null;
      if (!code) return;
      const text = code.textContent;

      const finish = function() {
        const span = btn.querySelector("span");
        const prev = span ? span.textContent : "Kopyala";
        if (span) span.textContent = "Kopyalandı!";
        btn.classList.add("copied");
        setTimeout(function() {
          if (span) span.textContent = prev;
          btn.classList.remove("copied");
        }, 2000);
      };

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(finish).catch(function() { fallbackCopy(text, finish); });
      } else {
        fallbackCopy(text, finish);
      }
    }

    function fallbackCopy(text, callback) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
        if (callback) callback();
      } catch (e) {
        console.error("Kopyalama hatası:", e);
      }
      document.body.removeChild(ta);
    }

    // Message Rendering
    function renderMessageItem(sender, text, latency = null) {
      const item = document.createElement("div");
      item.className = `msg-item ${sender}`;

      if (sender === "user") {
        const bubble = document.createElement("div");
        bubble.className = "msg-bubble-user";
        bubble.textContent = text;
        item.appendChild(bubble);
      } else {
        const head = document.createElement("div");
        head.className = "msg-nora-head";
        head.innerHTML = `<div class="msg-nora-avatar">N</div><span>Nora</span>`;

        const body = document.createElement("div");
        body.className = "msg-body-nora";
        body.innerHTML = formatMarkdown(text);

        const meta = document.createElement("div");
        meta.className = "msg-meta-row";
        if (latency !== null) {
          meta.innerHTML = `<span style="font-family: monospace;">${latency.toFixed(2)}s</span>`;
        }

        item.appendChild(head);
        item.appendChild(body);
        if (latency !== null) item.appendChild(meta);
      }

      chatThread.appendChild(item);
      chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    // Send Handlers
    function sendFromHero() {
      const text = heroInput.value.trim();
      if (!text || isGenerating) return;
      heroInput.value = "";

      // Switch to active view
      landingHero.style.display = "none";
      chatThread.style.display = "flex";
      activeInputWrap.style.display = "flex";

      // Create new session if none
      if (!currentSessionId) {
        currentSessionId = "sess_" + Date.now();
        const title = text.slice(0, 30) + (text.length > 30 ? "..." : "");
        sessions.unshift({ id: currentSessionId, title, messages: [] });
        topbarChatTitle.textContent = title;
        saveSessions();
      }

      executeSend(text);
    }

    function sendFromActive() {
      const text = activeInput.value.trim();
      if (!text || isGenerating) return;
      activeInput.value = "";
      resizeActive(activeInput);
      executeSend(text);
    }

    async function executeSend(text) {
      renderMessageItem("user", text);

      // Save to session
      const sess = sessions.find(s => s.id === currentSessionId);
      if (sess) {
        sess.messages.push({ sender: "user", text });
        saveSessions();
      }

      isGenerating = true;
      typingIndicator.style.display = "flex";
      chatScroll.scrollTop = chatScroll.scrollHeight;

      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text })
        });
        const data = await resp.json();
        typingIndicator.style.display = "none";

        const reply = data.reply || (data.error ? "[Hata] " + data.error : "...");
        renderMessageItem("nora", reply, data.elapsed);

        if (sess) {
          sess.messages.push({ sender: "nora", text: reply, latency: data.elapsed });
          saveSessions();
        }

        if (data.status) updateVitals(data.status);
      } catch (err) {
        typingIndicator.style.display = "none";
        renderMessageItem("nora", "[Bağlantı Hatası] " + err.message);
      } finally {
        isGenerating = false;
        activeInput.focus();
      }
    }

    // Drawers
    function openDrawer(id) {
      document.getElementById("drawer-backdrop").classList.add("show");
      document.querySelectorAll(".drawer-panel").forEach(d => d.classList.remove("show"));
      document.getElementById(id).classList.add("show");
    }

    function closeDrawer() {
      document.getElementById("drawer-backdrop").classList.remove("show");
      document.querySelectorAll(".drawer-panel").forEach(d => d.classList.remove("show"));
    }

    // Modals
    function openTeachModal() {
      document.getElementById("modal-teach").classList.add("show");
      const inp = document.getElementById("inp-teach");
      inp.value = "";
      setTimeout(() => inp.focus(), 100);
    }
    function openAbsorbModal() {
      document.getElementById("modal-absorb").classList.add("show");
      const inp = document.getElementById("inp-absorb");
      inp.value = "";
      setTimeout(() => inp.focus(), 100);
    }
    function closeModal(id) {
      document.getElementById(id).classList.remove("show");
    }

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeDrawer();
        closeModal("modal-teach");
        closeModal("modal-absorb");
      }
    });

    // Thoughts (SSE)
    function appendThought(t) {
      const feed = document.getElementById("thoughts-feed");
      const empty = document.getElementById("thoughts-empty");
      if (empty && empty.parentNode) empty.style.display = "none";

      thoughtCount++;
      document.getElementById("side-thoughts-count").textContent = thoughtCount;
      document.getElementById("top-thought-badge").textContent = thoughtCount;

      const bubble = document.createElement("div");
      bubble.className = "thought-bubble";

      const head = document.createElement("div");
      head.className = "thought-header";
      head.innerHTML = `<span>yaş=${t.age || '?'}</span><span>${t.time || ''}</span>`;

      const txt = document.createElement("div");
      txt.textContent = t.text || "";

      bubble.appendChild(head);
      bubble.appendChild(txt);
      feed.insertBefore(bubble, feed.firstChild);

      if (feed.children.length > 80) feed.removeChild(feed.lastChild);
    }

    // Vitals
    function updateVitals(v) {
      if (!v) return;
      const energy = Math.round((v.energy || 0) * 100);
      const water = Math.round((v.water || 0) * 100);
      const integrity = Math.round((v.integrity || 0) * 100);
      const age = v.age ?? "-";

      document.getElementById("side-vitals-pill").textContent = "%" + energy;
      document.getElementById("side-profile-text").textContent = `Nora · Yaş ${age}`;

      document.getElementById("v-energy").textContent = energy + "%";
      document.getElementById("v-bar-energy").style.width = Math.min(100, Math.max(0, energy)) + "%";

      document.getElementById("v-water").textContent = water + "%";
      document.getElementById("v-bar-water").style.width = Math.min(100, Math.max(0, water)) + "%";

      document.getElementById("v-integrity").textContent = integrity + "%";
      document.getElementById("v-bar-integrity").style.width = Math.min(100, Math.max(0, integrity)) + "%";

      document.getElementById("v-age").textContent = age;
      document.getElementById("v-mood").textContent = v.mood || "-";
      document.getElementById("v-conscious").textContent = v.conscious_status || "bilinmiyor";
    }

    // Commands
    async function saveBrain() {
      try {
        const resp = await fetch("/api/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ command: "save" })
        });
        const data = await resp.json();
        toast("Beyin başarıyla kaydedildi.");
        if (data.status) updateVitals(data.status);
      } catch (e) {
        toast("Kayıt hatası: " + e.message);
      }
    }

    async function triggerSleep() {
      try {
        const resp = await fetch("/api/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ command: "sleep" })
        });
        const data = await resp.json();
        toast("Bakım uykusu tamamlandı.");
        if (data.reply && chatThread.style.display !== "none") {
          renderMessageItem("nora", data.reply);
        }
        if (data.status) updateVitals(data.status);
      } catch (e) {
        toast("Hata: " + e.message);
      }
    }

    async function submitTeach() {
      const val = document.getElementById("inp-teach").value.trim();
      if (!val) return;
      closeModal("modal-teach");
      try {
        const resp = await fetch("/api/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ command: "teach", arg: val })
        });
        const data = await resp.json();
        toast("Bilgi öğrenildi ve kaydedildi.");
        if (data.reply && chatThread.style.display !== "none") {
          renderMessageItem("nora", data.reply);
        }
        if (data.status) updateVitals(data.status);
      } catch (e) {
        toast("Öğretme hatası: " + e.message);
      }
    }

    async function submitAbsorb() {
      const val = document.getElementById("inp-absorb").value.trim();
      if (!val) return;
      closeModal("modal-absorb");
      try {
        const resp = await fetch("/api/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ command: "absorb", arg: val })
        });
        const data = await resp.json();
        toast("Dosya okundu ve hafızaya işlendi.");
        if (data.reply && chatThread.style.display !== "none") {
          renderMessageItem("nora", data.reply);
        }
        if (data.status) updateVitals(data.status);
      } catch (e) {
        toast("Okuma hatası: " + e.message);
      }
    }

    // SSE Connect
    function initSSE() {
      const evt = new EventSource("/api/events");
      evt.onmessage = (e) => {
        try {
          const payload = JSON.parse(e.data);
          if (payload.type === "thought" && payload.data) {
            appendThought(payload.data);
          } else if (payload.type === "status" && payload.data) {
            updateVitals(payload.data);
          }
        } catch (_) {}
      };
      evt.onerror = () => {
        evt.close();
        setTimeout(initSSE, 4000);
      };
    }

    // Init
    loadSessions();
    fetch("/api/status").then(r => r.json()).then(updateVitals).catch(() => {});
    initSSE();
    heroInput.focus();
  </script>
</body>
</html>
"""


def find_free_port(host: str, start_port: int = 7860, max_attempts: int = 50) -> int:
    """Finds an open TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return start_port


def open_browser_tab(url: str, delay: float = 0.5) -> None:
    """Spawns a daemon thread to open a browser tab after a brief delay."""
    def _worker() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    threading.Thread(target=_worker, name="nora-browser-open", daemon=True).start()


class NoraWebServer:
    """Zero-dependency HTTP + SSE Server for Nora AGI WebUI."""

    def __init__(
        self,
        agi: Any,
        lock: threading.RLock,
        stop: threading.Event,
        args: Any,
        save_checkpoint: Callable[[], None],
        status_summary: Callable[[], str],
        emit: Callable[[str], None],
        host: str = "127.0.0.1",
        port: int = 7860,
        open_browser: bool = True,
    ) -> None:
        self.agi = agi
        self.lock = lock
        self.stop = stop
        self.args = args
        self.save_checkpoint = save_checkpoint
        self.status_summary = status_summary
        self.emit = emit
        self.host = host
        self.preferred_port = port
        self.open_browser = open_browser
        self.do_train = bool(getattr(args, "train", False) or getattr(args, "boot_thinker", False))

        # SSE Subscribers
        self.subscribers: Set[queue.Queue] = set()
        self.subscribers_lock = threading.Lock()
        self.recent_thoughts: deque = deque(maxlen=200)

        # Actual port bound
        self.actual_port = find_free_port(self.host, self.preferred_port)
        self.server: Optional[ThreadingHTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

    def get_vitals(self) -> Dict[str, Any]:
        """Extracts current biological and cognitive status safely under lock."""
        with self.lock:
            obs = getattr(self.agi, "obs", None)
            if obs is not None and len(obs) > max(I_E, I_W, I_I):
                energy = float(obs[I_E])
                water = float(obs[I_W])
                integrity = float(obs[I_I])
            elif obs is not None and len(obs) >= 3:
                energy = float(obs[0])
                water = float(obs[1])
                integrity = float(obs[2])
            else:
                energy, water, integrity = 1.0, 1.0, 1.0
            beliefs = getattr(self.agi.self_model, "self_belief", {})
            conscious = beliefs.get("conscious", {})
            mood = (
                self.agi.chem.dominant_state_tr()
                if hasattr(self.agi, "chem") and hasattr(self.agi.chem, "dominant_state_tr")
                else "n/a"
            )
            age = getattr(getattr(self.agi, "mind", None), "age", 0)
            is_alive = getattr(getattr(self.agi, "mortality", None), "is_alive", True)

            return {
                "age": int(age),
                "is_alive": bool(is_alive),
                "energy": round(energy, 3),
                "water": round(water, 3),
                "integrity": round(integrity, 3),
                "mood": mood,
                "conscious_status": conscious.get("status", "bilinmiyor"),
            }

    def broadcast_event(self, event_type: str, data: Any) -> None:
        """Pushes an event payload to all connected SSE clients."""
        payload = json.dumps({"type": event_type, "data": data})
        with self.subscribers_lock:
            dead: List[queue.Queue] = []
            for q in self.subscribers:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.subscribers.discard(q)

    def on_thought(self, msg: Dict[str, Any]) -> None:
        """Called when Nora's autonomous inner thought is produced in background."""
        thought_data = {
            "text": str(msg.get("text") or "").strip(),
            "age": msg.get("age", "?"),
            "time": time.strftime("%H:%M:%S"),
        }
        if thought_data["text"]:
            self.recent_thoughts.append(thought_data)
            self.broadcast_event("thought", thought_data)

    def handle_command(self, cmd_name: str, arg: str = "") -> Dict[str, Any]:
        """Executes special slash commands and actions."""
        cmd = cmd_name.strip().casefold()
        if not cmd.startswith("/"):
            cmd = "/" + cmd

        if cmd in {"/status"}:
            return {
                "reply": self.status_summary(),
                "status": self.get_vitals(),
            }

        if cmd in {"/sleep"}:
            with self.lock:
                result = self.agi.sleep()
            return {
                "reply": f"[Uyku] Mod: {result.get('mode')} | Replay adımları: {result.get('replay')}",
                "status": self.get_vitals(),
            }

        if cmd in {"/save"}:
            self.save_checkpoint()
            return {
                "reply": "[Nora] Bütün beyin birleşik dosyaya kalıcı olarak kaydedildi.",
                "status": self.get_vitals(),
            }

        if cmd in {"/teach", "/öğret"}:
            fact = arg.strip()
            if not fact:
                return {"error": "Öğretilecek olgu/tanım cümlesi belirtilmedi."}
            with self.lock:
                try:
                    edges = self.agi.lang.teach_sentence(fact, speaker="insan")
                    self.agi._modified_in_session = True
                    self.save_checkpoint()
                    reply = f"[Öğrenildi ve Kaydedildi] '{fact}' -> {len(edges)} bağlantı hafızaya işlendi."
                except Exception as e:
                    return {"error": f"Öğrenilemedi: {e}"}
            return {"reply": reply, "status": self.get_vitals()}

        if cmd in {"/absorb", "/oku"}:
            target_file = arg.strip()
            if not target_file:
                return {"error": "Okunacak dosya yolu belirtilmedi."}
            p = Path(target_file).expanduser().resolve()
            if not p.is_file():
                return {"error": f"Dosya bulunamadı: {target_file}"}
            with self.lock:
                try:
                    stats = self.agi.read_corpus(str(p))
                    self.save_checkpoint()
                    reply = f"[Hafıza] Tamamlandı ve kalıcı kaydedildi: {stats}"
                except Exception as e:
                    return {"error": f"Okuma sırasında hata: {e}"}
            return {"reply": reply, "status": self.get_vitals()}

        return {"error": f"Bilinmeyen komut: {cmd_name}"}

    def converse(self, message: str) -> Dict[str, Any]:
        """Processes an incoming chat message or slash command."""
        text = message.strip()
        if not text:
            return {"reply": "..."}

        # Check for commands
        if text.startswith("/"):
            parts = text.split(" ", 1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""
            res = self.handle_command(cmd, arg)
            if "reply" in res:
                return {"reply": res["reply"], "elapsed": 0.01, "status": res.get("status")}
            if "error" in res:
                return {"error": res["error"], "status": self.get_vitals()}

        with self.lock:
            if not getattr(self.agi.mortality, "is_alive", True):
                return {
                    "reply": "[Nora] Yaşam fonksiyonları durdu, yanıt veremiyor.",
                    "elapsed": 0.0,
                    "status": self.get_vitals(),
                }

            # Synchronize real time timestamp so life ticks don't double count
            self.agi.last_real_time = time.time()
            t0 = time.perf_counter()
            answer = self.agi.converse(text, speaker="insan")
            if self.do_train:
                self.agi._modified_in_session = True
            elapsed = time.perf_counter() - t0

        reply_text = answer if answer else "..."
        self.emit(f"WebUI Sohbet > İnsan: {text}\nNora ({elapsed:.2f}s) > {reply_text}\n")

        return {
            "reply": reply_text,
            "elapsed": round(elapsed, 2),
            "status": self.get_vitals(),
        }

    def start(self) -> None:
        """Starts the HTTP server and opens browser tab."""
        web_server_self = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                # Suppress noisy standard HTTP access logs
                pass

            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                # 1. Main UI
                if path in {"", "/"}:
                    content = HTML_PAGE.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return

                # 2. Logo
                if path in {"/logo.png", "/favicon.ico"}:
                    logo_path = REPO_ROOT / "logo.png"
                    if logo_path.is_file():
                        data = logo_path.read_bytes()
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("Cache-Control", "public, max-age=86400")
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return

                # 3. Status API
                if path == "/api/status":
                    vitals = web_server_self.get_vitals()
                    data = json.dumps(vitals).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                # 4. SSE Events Stream
                if path == "/api/events":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    client_q: queue.Queue = queue.Queue(maxsize=100)
                    with web_server_self.subscribers_lock:
                        web_server_self.subscribers.add(client_q)

                    try:
                        # Send initial status right away
                        init_status = json.dumps({"type": "status", "data": web_server_self.get_vitals()})
                        self.wfile.write(f"data: {init_status}\n\n".encode("utf-8"))
                        self.wfile.flush()

                        while not web_server_self.stop.is_set():
                            try:
                                msg = client_q.get(timeout=2.0)
                                self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                                self.wfile.flush()
                            except queue.Empty:
                                # Keepalive ping
                                self.wfile.write(b": ping\n\n")
                                self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, socket.error):
                        pass
                    finally:
                        with web_server_self.subscribers_lock:
                            web_server_self.subscribers.discard(client_q)
                    return

                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                # Read body
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

                try:
                    payload = json.loads(body)
                except Exception:
                    payload = {}

                # 1. Chat
                if path == "/api/chat":
                    msg = str(payload.get("message") or "")
                    res = web_server_self.converse(msg)
                    data = json.dumps(res).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                # 2. Command / Action
                if path == "/api/command":
                    cmd = str(payload.get("command") or "")
                    arg = str(payload.get("arg") or "")
                    res = web_server_self.handle_command(cmd, arg)
                    data = json.dumps(res).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                self.send_error(HTTPStatus.NOT_FOUND)

        self.server = ThreadingHTTPServer((self.host, self.actual_port), RequestHandler)
        url = f"http://{self.host}:{self.actual_port}/"

        banner = f"""
===================================================================
                   Nora WebUI Başlatıldı
===================================================================
  Arayüz Adresi: {url}
  Tarayıcıda otomatik sekme açılıyor...
  Durdurmak için terminalde Ctrl+C tuşlarına basabilirsiniz.
  (Terminal modu için: python chat.py --cli)
===================================================================
"""
        self.emit(banner)

        if self.open_browser:
            open_browser_tab(url)

        # Serve forever in current thread or caller's loop
        try:
            self.server.serve_forever()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            if self.server:
                self.server.server_close()
