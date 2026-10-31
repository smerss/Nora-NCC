"""Türkçe Vikipedi'den düzyazı toplar (Nora'nın okuyarak dil öğrenmesi için ham veri; kod hiçbir cümle içermez).
    python make_corpus_tr.py --pages 3000 --out data/text/tr_wiki.txt
"""

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)
import argparse, json, re, threading, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://tr.wikipedia.org/w/api.php"
Q = {"action": "query", "format": "json", "generator": "random", "grnnamespace": "0", "grnlimit": "20",
     "prop": "extracts", "exintro": "1", "explaintext": "1", "exlimit": "20"}


def fetch_batch(_):
    url = API + "?" + urllib.parse.urlencode(Q)
    wait = 2.0
    for _try in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NoraAGI-Corpus/1.0 (research; polite)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8"))
            return [(p["title"], p.get("extract", "")) for p in (d.get("query", {}).get("pages", {}) or {}).values()]
        except Exception:
            time.sleep(wait)          # 429 vb.: üstel geri çekilme (siteye saygı)
            wait = min(60.0, wait * 2)
    return []


def clean(t: str) -> str:
    t = re.sub(r"\([^)]*\)", "", t)               # parantez içi (doğum yılı, telaffuz vb.)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3000)
    ap.add_argument("--out", default="data/text/tr_wiki.txt")
    a = ap.parse_args()
    seen, lines = set(), []
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.pages // 20):
            for title, text in fetch_batch(i):
                text = clean(text)
                if title in seen or len(text) < 200:
                    continue
                seen.add(title)
                lines.append(text)
                f.write(text + "\n")
            f.flush()
            time.sleep(1.5)
    print(f"{len(lines)} madde, {sum(len(l) for l in lines)} karakter -> {a.out}")


if __name__ == "__main__":
    main()
