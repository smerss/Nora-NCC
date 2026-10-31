#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chat.py — Nora Interactive Terminal Chat
Allows direct, interactive conversation with Nora without pre-scripted prompt sequences.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

try:
    import readline  # Enable arrow keys, history, and line editing in terminal
except ImportError:
    pass

from nora._paths import REPO_ROOT

TICK_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nora: Interactive Real-Time Living Terminal Chat"
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("NORA_REPO", str(REPO_ROOT)),
        help="Nora-NCC directory (default: detected repository root)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for Nora")
    parser.add_argument(
        "--think-iters",
        type=int,
        default=800,
        help="Initial boot/cognition budget (default: 800)",
    )
    parser.add_argument(
        "--boot-thinker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run initial boot thinker training (default: True, use --no-boot-thinker for fast start)",
    )
    parser.add_argument(
        "--corpus",
        nargs="*",
        default=[],
        help="Optional corpus text files to absorb at startup",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional save/load prefix path (e.g. state/nora_chat)",
    )
    parser.add_argument(
        "--journal",
        default="",
        help="Optional append-only journal file path",
    )
    parser.add_argument(
        "--single-life",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Single mortal biological life cycle (default: True)",
    )
    parser.add_argument(
        "--background-life",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run background biological life loop ticks (default: True)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()

    if not (repo / "nora" / "cognition" / "agi.py").is_file() and not (repo / "nora_agi.py").is_file():
        raise SystemExit(
            f"Nora repository not found at: {repo}\n"
            "Please specify --repo /path/to/Nora-NCC"
        )

    sys.path.insert(0, str(repo))

    from nora.cognition.agi import NoraAGI
    from nora.cognition.mind import I_E, I_I, I_W

    checkpoint = (
        Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    )
    journal = Path(args.journal).expanduser().resolve() if args.journal else None

    lock = threading.RLock()
    output_lock = threading.Lock()
    stop = threading.Event()

    def emit(message: str) -> None:
        with output_lock:
            print(message, flush=True)

    def status_summary() -> str:
        obs = agi.obs
        energy = float(obs[I_E])
        water = float(obs[I_W])
        integrity = float(obs[I_I])

        beliefs = getattr(agi.self_model, "self_belief", {})
        conscious = beliefs.get("conscious", {})
        mood = agi.chem.dominant_state_tr() if hasattr(agi, "chem") and hasattr(agi.chem, "dominant_state_tr") else "n/a"

        return (
            f"[Durum] Yaş: {agi.mind.age} | "
            f"Canlı: {agi.mortality.is_alive} | "
            f"Enerji: {energy:.2f} | Su: {water:.2f} | Bütünlük: {integrity:.2f} | "
            f"Dürtü: {mood} | Bilinç İnancı: {conscious.get('status', 'bilinmiyor')}"
        )

    # Initialize Nora
    if checkpoint and Path(str(checkpoint) + ".agi.json").is_file():
        emit(f"[Nora] Checkpoint yükleniyor: {checkpoint}")
        agi = NoraAGI.load(str(checkpoint))
    else:
        emit("[Nora] Bilişsel ve biyolojik sistemler başlatılıyor...")
        broca_path = repo / "data" / "broca_tr.pt"
        broca_str = None
        if broca_path.is_file():
            try:
                import torch  # noqa: F401
                broca_str = str(broca_path)
            except ImportError:
                broca_str = None

        agi = NoraAGI(
            name="Nora",
            seed=args.seed,
            boot_thinker=args.boot_thinker,
            single_life=args.single_life,
            thinker_iters=args.think_iters,
            corpus=[str(Path(c).expanduser().resolve()) for c in args.corpus],
            broca=broca_str,
            code_pretrain=True,
        )

    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if journal:
        journal.parent.mkdir(parents=True, exist_ok=True)
        agi.open_journal(str(journal))

    def save_checkpoint() -> None:
        if not checkpoint:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            agi.save(str(checkpoint))
        emit(f"[Nora] Checkpoint kaydedildi: {checkpoint}")

    def life_loop() -> None:
        next_tick = time.monotonic() + TICK_SECONDS
        while not stop.is_set():
            wait_seconds = max(0.0, next_tick - time.monotonic())
            if stop.wait(wait_seconds):
                break

            try:
                with lock:
                    if not agi.mortality.is_alive:
                        stop.set()
                        break

                    state = agi.step_life(1)
                    messages = agi.drain_outbox()

                for msg in messages:
                    text = str(msg.get("text") or "").strip()
                    if text:
                        emit(f"\n[Nora içses / yaş={msg.get('age', '?')}] {text}")

                if not state.get("alive", False):
                    emit("\n[Nora] Yaşam döngüsü sona erdi.")
                    stop.set()
                    break

            except Exception as exc:
                emit(f"\n[Yaşam döngüsü hatası] {type(exc).__name__}: {exc}")
                stop.set()
                break

            next_tick += TICK_SECONDS

    worker = None
    if args.background_life:
        worker = threading.Thread(target=life_loop, name="nora-life", daemon=True)
        worker.start()

    banner = """
===================================================================
                     Nora İnteraktif Sohbet
===================================================================
 Herhangi bir hazır komut dosyası olmadan doğrudan sohbet edebilirsiniz.
 Komutlar:
   /status  - Beden, enerji, su ve zihin durumunu gösterir
   /sleep   - Bakım uykusuna geçirir
   /save    - Durumu checkpoint'e kaydeder
   /help    - Bu yardım iletisini gösterir
   /exit    - Çıkış yapar
===================================================================
"""
    emit(banner)

    try:
        while not stop.is_set():
            try:
                line = input("Sen > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            command = line.casefold()

            if command in {"/exit", "/quit", "exit", "quit", "çıkış", "q"}:
                break

            if command in {"/help", "/yardım", "/yardim"}:
                emit(
                    "Komutlar:\n"
                    "  /status: yaş, enerji, su, organlar ve bilinç durumu\n"
                    "  /sleep: bakım uykusu (konsolidasyon)\n"
                    "  /save: checkpoint kaydet\n"
                    "  /exit: güvenli çıkış\n"
                )
                continue

            with lock:
                if not agi.mortality.is_alive:
                    emit("[Nora] Yaşam fonksiyonları durdu, yanıt veremiyor.")
                    continue

                if command == "/status":
                    emit(status_summary())
                    continue

                if command == "/sleep":
                    result = agi.sleep()
                    emit(
                        f"[Uyku] Mod: {result.get('mode')} | "
                        f"Replay adımları: {result.get('replay')}"
                    )
                    continue

                if command == "/save":
                    if not checkpoint:
                        emit(
                            "Checkpoint için başlatırken --checkpoint <dosya_yolu> parametresini belirtin."
                        )
                    else:
                        agi.save(str(checkpoint))
                        emit(f"[Nora] Checkpoint kaydedildi: {checkpoint}")
                    continue

                # Synchronize real time timestamp so life ticks don't double count
                agi.last_real_time = time.time()
                t0 = time.perf_counter()
                answer = agi.converse(line, speaker="insan")
                elapsed = time.perf_counter() - t0

            if answer:
                emit(f"Nora ({elapsed:.2f}s) > {answer}\n")
            else:
                emit("Nora > ...\n")

    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=3.0)

        try:
            save_checkpoint()
        except Exception as exc:
            emit(f"[Kayıt hatası] {type(exc).__name__}: {exc}")

        with lock:
            if getattr(agi, "journal", None) is not None:
                try:
                    agi.journal.flush()
                except Exception:
                    pass

        emit("\n[Nora] Görüşmek üzere. Program sonlandırıldı.")


if __name__ == "__main__":
    main()
