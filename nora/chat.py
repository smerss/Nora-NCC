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
        "--train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model training and persistent learning during chat (default: False)",
    )
    parser.add_argument(
        "--boot-thinker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run initial boot thinker training (default: False, use --boot-thinker or --train to enable)",
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
        help="Optional unified brain path (default: state/nora_brain.nora)",
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
    parser.add_argument(
        "--web",
        "--webui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="WebUI arayüzünü başlat ve tarayıcıda sekme aç (varsayılan: True)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="WebUI yerine doğrudan terminal konsolunda çalıştır",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="WebUI sunucu portu (varsayılan: 7860)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="WebUI sunucu adresi (varsayılan: 127.0.0.1)",
    )
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tarayıcıda otomatik yeni sekme aç (varsayılan: True)",
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
        mood = (agi.chem.dominant_state_tr()
                if hasattr(agi, "chem") and hasattr(agi.chem, "dominant_state_tr") else "n/a")
        return (
            f"[Durum] Yaş: {agi.mind.age} | "
            f"Canlı: {agi.mortality.is_alive} | "
            f"Enerji: {energy:.2f} | Su: {water:.2f} | Bütünlük: {integrity:.2f} | "
            f"Dürtü: {mood} | Bilinç İnancı: {conscious.get('status', 'bilinmiyor')}"
        )

    # Normal kullanımda tek dosya yeterlidir. Eski beş-parça ``nora_sozcu``
    # kaydı yalnız ilk açılışta okunur ve hemen birleşik beyne taşınır.
    default_brain = repo / "state" / "nora_brain.nora"
    requested = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else default_brain
    checkpoint = Path(NoraAGI.unified_path(requested))
    legacy_default = repo / "state" / "nora_sozcu"
    do_train = bool(args.train or args.boot_thinker)

    if checkpoint.is_file():
        emit(f"[Nora] Birleşik beyin yükleniyor: {checkpoint.name}")
        agi = NoraAGI.load(str(checkpoint), attach_broca_async=False)
        if do_train and getattr(agi, "cognition", None) is None:
            agi._thinker_iters = int(args.think_iters)
            agi._ensure_cognition()
    elif ((checkpoint == default_brain and Path(str(legacy_default) + ".agi.json").is_file())
          or Path(str(requested) + ".agi.json").is_file()):
        legacy = legacy_default if checkpoint == default_brain and Path(str(legacy_default) + ".agi.json").is_file() else requested
        emit("[Nora] Eski parçalı beyin bulundu; tek dosyaya taşınıyor...")
        agi = NoraAGI.load(str(legacy), attach_broca_async=False)
        if do_train and getattr(agi, "cognition", None) is None:
            agi._thinker_iters = int(args.think_iters)
            agi._ensure_cognition()
        agi.save(str(checkpoint))
        emit(f"[Nora] Taşıma tamamlandı: {checkpoint}")
    else:
        emit("[Nora] Bilişsel ve biyolojik sistemler başlatılıyor...")
        agi = NoraAGI(
            name="Nora",
            seed=args.seed,
            boot_thinker=do_train,
            single_life=args.single_life,
            thinker_iters=args.think_iters,
            corpus=[str(Path(c).expanduser().resolve()) for c in args.corpus],
            broca=None,
            code_pretrain=do_train,
        )
        agi.brain_path = str(checkpoint)

    # Her oturumun söz zarı kendi donanım entropisiyle başlar; kayıtlı tohum aynı sözleri geri getirmez.
    agi.refresh_chance_streams()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if journal:
        journal.parent.mkdir(parents=True, exist_ok=True)
        agi.open_journal(str(journal))

    def save_checkpoint() -> None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            target = agi.save(str(checkpoint))
        emit(f"[Nora] Bütün beyin kaydedildi: {target}")

    use_web = bool(args.web and not args.cli)
    web_server = None
    if use_web:
        from nora.web import NoraWebServer
        web_server = NoraWebServer(
            agi=agi,
            lock=lock,
            stop=stop,
            args=args,
            save_checkpoint=save_checkpoint,
            status_summary=status_summary,
            emit=emit,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )

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
                        if web_server is not None:
                            web_server.on_thought(msg)

                if web_server is not None:
                    web_server.broadcast_event("status", web_server.get_vitals())

                if not state.get("alive", False):
                    emit("\n[Nora] Yaşam döngüsü sona erdi.")
                    if web_server is not None:
                        web_server.broadcast_event("status", web_server.get_vitals())
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
   /teach <cümle> - Canlı hafızaya anında 1-shot olgu/tanım ekler
   /absorb <dosya>- Dosyadaki metinleri doğrudan semantik hafızaya okur
   /status  - Beden, enerji, su ve zihin durumunu gösterir
   /sleep   - Bakım uykusuna geçirir
   /save    - Bütün beyni tek dosyaya kaydeder
   /help    - Bu yardım iletisini gösterir
   /exit    - Çıkış yapar
===================================================================
"""

    try:
        if use_web:
            assert web_server is not None
            web_server.start()
        else:
            emit(banner)
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
                        "  /save: bütün beyni tek dosyaya kaydet\n"
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

                    if command.startswith("/teach ") or command.startswith("/öğret "):
                        fact = line.split(" ", 1)[1].strip()
                        try:
                            edges = agi.lang.teach_sentence(fact, speaker="insan")
                            agi._modified_in_session = True
                            save_checkpoint()
                            emit(f"[Öğrenildi ve Kaydedildi] '{fact}' -> {len(edges)} bağlantı hafızaya işlendi.")
                        except Exception as e:
                            emit(f"[Hata] Öğrenilemedi: {e}")
                        continue

                    if command.startswith("/absorb ") or command.startswith("/oku "):
                        target_file = line.split(" ", 1)[1].strip()
                        p = Path(target_file).expanduser().resolve()
                        if not p.is_file():
                            emit(f"[Hata] Dosya bulunamadı: {target_file}")
                        else:
                            emit(f"[Nora] Dosya okunuyor ve semantik hafızaya işleniyor: {p}")
                            try:
                                stats = agi.read_corpus(str(p))
                                save_checkpoint()
                                emit(f"[Hafıza] Tamamlandı ve kalıcı kaydedildi: {stats}")
                            except Exception as e:
                                emit(f"[Hata] Okuma sırasında hata: {e}")
                        continue

                    if command == "/save":
                        save_checkpoint()
                        continue

                    # Synchronize real time timestamp so life ticks don't double count
                    agi.last_real_time = time.time()
                    t0 = time.perf_counter()
                    answer = agi.converse(line, speaker="insan")
                    if do_train:
                        agi._modified_in_session = True
                    elapsed = time.perf_counter() - t0

                if answer:
                    emit(f"Nora ({elapsed:.2f}s) > {answer}\n")
                else:
                    emit("Nora > ...\n")

    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if web_server is not None and web_server.server is not None:
            try:
                web_server.server.server_close()
            except Exception:
                pass
        if worker is not None:
            worker.join(timeout=3.0)

        if do_train or getattr(agi, "_modified_in_session", False):
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
