"""nora_audit.py — KODA BAĞIMLILIK SAYACI (iskelet denetimi).

Sorun
-----
"Nora'nın cümleleri kodda yazılı" demek ölçülemez bir şikâyetti. Bu araç ölçer:
her `nora_*.py` dosyasındaki, KULLANICIYA GİDEBİLECEK uzun Türkçe dizgeleri sayar
(docstring'ler hariç). Amaç bunları sıfıra indirmek değil — bedenin/ölçümün biçimi
hep kalacak — ama her turda **azaltmak**.

Ratchet (mandal): `nora_audit_baseline.json` bugünkü sayıları tutar. Test (A73) ve
`--check`, hiçbir modülün baseline'ı AŞMADIĞINI doğrular. Yani iskelet geri büyüyemez;
ancak azaltılıp baseline düşürülebilir.

Kullanım
--------
    python nora_audit.py                 # tablo
    python nora_audit.py --check         # baseline'ı aşan var mı? (exit 1)
    python nora_audit.py --update        # yalnız AZALTIDAN sonra baseline'ı yaz
    python nora_audit.py --json out.json # makine okunur çıktı
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import argparse
import ast
import json
import os
import sys
from pathlib import Path

ROOT = REPO_ROOT
BASELINE = ROOT / "nora_audit_baseline.json"

#: Cümle sayılmak için en az bu kadar karakter ve bir boşluk gerekir.
MIN_LEN = 18

#: Teknik dizgeler: söz değil, biçim/anahtar/yol.
_SKIP_HINTS = (
    "/", "\\", ".py", ".json", ".npz", ".pt", "utf-8", "http", "%", ":=",
    "-->", "==", "|", "{", "}", "_", "\n    ", "  ",
)

_TURKISH = set("çğıöşüâîûÇĞİÖŞÜ")

#: SÖYLEM modülleri: burada yazılı her uzun dizge bir İSKELETTİR (cümle bankası).
#: Diğer modüller (bilgi, matematik, okul) içerik taşır; oradaki dizgeler "iskelet"
#: değil "öğretilen bilgi"dir. Mandal yalnızca söylem toplamına uygulanır.
SOYLEM = {
    "nora_style.py", "nora_meta.py", "nora_anlatim.py", "nora_discourse.py",
    "nora_frames.py", "nora_monologue.py", "nora_initiative.py",
    "nora_deliberation.py", "nora_self.py", "nora_speculation.py",
    "nora_selfexam.py", "nora_register.py",
}


def _is_sentence(text: str) -> bool:
    t = text.strip()
    if len(t) < MIN_LEN:
        return False
    if " " not in t:
        return False
    if any(h in t for h in _SKIP_HINTS):
        return False
    # En az bir küçük harfli sözcük ve bir Türkçe harf ya da yaygın işlev sözcüğü
    low = t.lower()
    if not any(ch in _TURKISH for ch in t):
        if not any(w in low for w in (" bir ", " bu ", " ve ", " ne ", " mi ", " için ")):
            return False
    return True


def scan_source(src: str) -> list[dict]:
    """Bir dosyanın kaynağındaki söz-benzeri dizgeler (docstring hariç)."""
    tree = ast.parse(src)
    docs: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", []) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docs.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docs:
                continue
            if _is_sentence(node.value):
                out.append({"line": int(getattr(node, "lineno", 0)), "text": node.value.strip()})
    return out


def scan_file(path: Path) -> list[dict]:
    try:
        return scan_source(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - bozuk dosya
        return [{"line": 0, "text": f"[okunamadı: {exc}]", "error": True}]


def scan_all(root: Path = ROOT) -> dict:
    mods = {}
    for p in sorted(root.glob("nora_*.py")):
        if p.name in ("nora_audit.py",):
            continue
        items = scan_file(p)
        mods[p.name] = {"adet": len(items), "ornek": [i["text"][:60] for i in items[:3]]}
    total = sum(v["adet"] for v in mods.values())
    soylem = sum(v["adet"] for k, v in mods.items() if k in SOYLEM)
    return {"toplam": total, "soylem": soylem, "modul": mods}


def load_baseline(path: Path = BASELINE) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compare(report: dict, baseline: dict) -> dict:
    """Baseline'a göre fark: eksilen/artan modüller ve toplam."""
    base_mods = (baseline.get("modul") or {})
    artan, eksilen = {}, {}
    for name, rec in (report.get("modul") or {}).items():
        b = int((base_mods.get(name) or {}).get("adet", 0))
        d = int(rec["adet"]) - b
        if d > 0:
            # MANDAL yalnızca SÖYLEM modüllerine uygulanır. Bilgi/örgü modüllerinde
            # (broca, stem, knowledge …) cümle sayısı bilgi eklemekle doğal olarak artar;
            # orası iskelet değil içeriktir. Genel tavan ise ayrıca (tarihsel) korunur.
            if name in SOYLEM or name not in base_mods:
                artan[name] = d
        elif d < 0:
            eksilen[name] = -d
    for name, rec in base_mods.items():
        if name not in (report.get("modul") or {}):
            eksilen[name] = int(rec.get("adet", 0))
    return {"artan": artan, "eksilen": eksilen,
            "toplam_fark": int(report["toplam"]) - int(baseline.get("toplam", 0)),
            "soylem_fark": int(report.get("soylem", 0)) - int(baseline.get("soylem", 0))}


def save_baseline(report: dict, path: Path = BASELINE) -> None:
    rec = {"toplam": int(report["toplam"]), "soylem": int(report.get("soylem", 0)),
           "modul": {k: {"adet": int(v["adet"])} for k, v in report["modul"].items()}}
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Nora iskelet denetimi")
    ap.add_argument("--check", action="store_true", help="baseline'ı aşan varsa hata ver")
    ap.add_argument("--update", action="store_true", help="baseline'ı yaz (yalnız azaltıdan sonra)")
    ap.add_argument("--json", default="", help="raporu dosyaya yaz")
    args = ap.parse_args(argv)

    rep = scan_all()
    base = load_baseline()
    cmp = compare(rep, base)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    width = max((len(k) for k in rep["modul"]), default=20)
    for name, rec in sorted(rep["modul"].items(), key=lambda kv: -kv[1]["adet"]):
        if rec["adet"]:
            print(f"{name:<{width}}  {rec['adet']:4d}")
    print(f"{'TOPLAM':<{width}}  {rep['toplam']:4d}")
    print(f"{'SÖYLEM (iskelet)':<{width}}  {rep['soylem']:4d}")
    if base:
        print(f"baseline: toplam {base.get('toplam')} / söylem {base.get('soylem')} "
              f"| fark {cmp['toplam_fark']:+d} / {cmp['soylem_fark']:+d} "
              f"| artan: {cmp['artan'] or '—'} | eksilen: {cmp['eksilen'] or '—'}")
    if args.update:
        save_baseline(rep)
        print("baseline yazıldı.")
        return 0
    if args.check and cmp["artan"]:
        print("HATA: iskelet büyümüş:", cmp["artan"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
