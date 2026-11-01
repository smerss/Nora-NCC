#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test suite for Nora WebUI server."""

import json
import threading
import time
import urllib.request
from typing import Any

from nora.web import NoraWebServer


class DummyMortality:
    is_alive = True


class DummyMind:
    age = 42


class DummySelfModel:
    self_belief = {"conscious": {"status": "farkinda"}}


class DummyChem:
    def dominant_state_tr(self) -> str:
        return "Merak"


class DummyLang:
    def teach_sentence(self, fact: str, speaker: str = "insan") -> list:
        return [("dummy", "is_a", "dummy")]


class DummyAGI:
    def __init__(self):
        self.obs = [0.85, 0.90, 0.99]
        self.mortality = DummyMortality()
        self.mind = DummyMind()
        self.self_model = DummySelfModel()
        self.chem = DummyChem()
        self.lang = DummyLang()
        self.last_real_time = 0.0

    def converse(self, msg: str, speaker: str = "insan") -> str:
        return f"Nora yanıtı: {msg}"

    def sleep(self) -> dict:
        return {"mode": "rem", "replay": 5}


class DummyArgs:
    train = False
    boot_thinker = False


def test_webui_server_and_endpoints():
    agi = DummyAGI()
    lock = threading.RLock()
    stop = threading.Event()
    args = DummyArgs()
    saved = []

    def mock_save():
        saved.append(True)

    def mock_status():
        return "mock status"

    emitted = []

    def mock_emit(msg):
        emitted.append(msg)

    # Install opener that ignores proxies for localhost testing
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urllib.request.install_opener(opener)

    server = NoraWebServer(
        agi=agi,
        lock=lock,
        stop=stop,
        args=args,
        save_checkpoint=mock_save,
        status_summary=mock_status,
        emit=mock_emit,
        host="127.0.0.1",
        port=8970,
        open_browser=False,
    )

    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    time.sleep(0.3)

    base_url = f"http://127.0.0.1:{server.actual_port}"

    try:
        # 1. Test GET /
        with urllib.request.urlopen(f"{base_url}/") as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "Nora AGI" in html
            assert "Biyolojik Homeostaz" in html
            assert "Zihin Akışı" in html

        # 2. Test GET /logo.png
        with urllib.request.urlopen(f"{base_url}/logo.png") as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "image/png"
            assert len(resp.read()) > 0

        # 3. Test GET /api/status
        with urllib.request.urlopen(f"{base_url}/api/status") as resp:
            assert resp.status == 200
            status_data = json.loads(resp.read().decode("utf-8"))
            assert status_data["age"] == 42
            assert status_data["is_alive"] is True
            assert status_data["energy"] == 0.85
            assert status_data["water"] == 0.90
            assert status_data["integrity"] == 0.99
            assert status_data["mood"] == "Merak"
            assert status_data["conscious_status"] == "farkinda"

        # 4. Test POST /api/chat
        req_data = json.dumps({"message": "Merhaba Nora"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            chat_data = json.loads(resp.read().decode("utf-8"))
            assert "Nora yanıtı: Merhaba Nora" in chat_data["reply"]
            assert chat_data["status"]["age"] == 42

        # 5. Test POST /api/command (/sleep)
        req_data = json.dumps({"command": "sleep"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/command",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            sleep_data = json.loads(resp.read().decode("utf-8"))
            assert "rem" in sleep_data["reply"]

        # 6. Test POST /api/command (/save)
        req_data = json.dumps({"command": "save"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/command",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            save_data = json.loads(resp.read().decode("utf-8"))
            assert "kaydedildi" in save_data["reply"]
            assert len(saved) == 1

        # 7. Test POST /api/command (/teach)
        req_data = json.dumps({"command": "teach", "arg": "Mars bir gezegendir."}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/command",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            teach_data = json.loads(resp.read().decode("utf-8"))
            assert "Öğrenildi ve Kaydedildi" in teach_data["reply"]

        # 8. Test Thought broadcasting
        server.on_thought({"text": "Kendi kendime düşünüyorum", "age": 43})
        assert len(server.recent_thoughts) == 1
        assert server.recent_thoughts[0]["text"] == "Kendi kendime düşünüyorum"

    finally:
        stop.set()
        if server.server:
            server.server.server_close()


if __name__ == "__main__":
    test_webui_server_and_endpoints()
    print("ALL WEBUI TESTS PASSED!")
