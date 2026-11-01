"""OPUS OpenSubtitles Türkçe tek-dilli dökümünü (satır başına bir altyazı cümlesi) temizler.

Altyazılar kısa ve konuşma dilidir; bu yüzden düzyazı temizleyicisinden (clean_tr_corpus.py)
daha hafif bir süzgeç kullanır: çeviri/senkron kredisi, müzik notası, sahne etiketi, tamamen
büyük harf satırlar, çok kısa/çok uzun satırlar ve yapay-zeka kalıpları elenir. Tekrarlar tek
seferde atılır. --drop-profanity verilirse en sert küfür kalıpları da elenir.

    python tools/clean_tr_subtitles.py --src data/corpus/raw/opensubtitles_tr_v2018.txt.gz \
        --out data/corpus/clean/tr_subtitles.txt --limit 20000000
"""

import argparse, gzip, hashlib, json, os, re, sys

from clean_tr_corpus import AI_RE, TR_LETTERS, normalize

CREDIT_RE = re.compile(
    r"çeviri|altyaz|senkron|www\.|http|@|opensubtitles|subtitle|subs|♪|♫|\[|\]|<|>|"
    r"^\s*[-–•]|\bdiscord\b|tercüme|iyi seyirler",
    re.IGNORECASE,
)
# Yalnızca en sert kalıplar; isteğe bağlı olarak elenir.
HARD_PROFANITY_RE = re.compile(r"\b(orospu|piç|amcık|sik|sikti|siktir|göt)\w*\b", re.IGNORECASE)


def keep(line, drop_profanity):
    if not 12 <= len(line) <= 200:
        return False
    words = line.split()
    if len(words) < 2:
        return False
    if line.isupper():  # başlık ve sahne yazıları
        return False
    if CREDIT_RE.search(line) or AI_RE.search(line):
        return False
    letters = [c for c in line if c.isalpha()]
    if not letters or sum(c in TR_LETTERS for c in letters) / len(letters) < 0.97:
        return False
    if drop_profanity and HARD_PROFANITY_RE.search(line):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="opensubtitles tr .txt.gz ya da .txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20_000_000, help="işlenecek en çok satır")
    ap.add_argument("--drop-profanity", action="store_true")
    args = ap.parse_args()

    opener = gzip.open if args.src.endswith(".gz") else open
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    seen = set()
    stats = {"read": 0, "kept": 0, "dup": 0}
    chars = 0
    with opener(args.src, "rt", encoding="utf-8", errors="replace") as src, \
            open(args.out, "w", encoding="utf-8") as out:
        for raw in src:
            if stats["read"] >= args.limit:
                break
            stats["read"] += 1
            line = normalize(raw)
            if not keep(line, args.drop_profanity):
                continue
            digest = hashlib.sha1(re.sub(r"\W+", "", line.lower()).encode("utf-8")).digest()[:8]
            if digest in seen:
                stats["dup"] += 1
                continue
            seen.add(digest)
            out.write(line + "\n")
            chars += len(line) + 1
            stats["kept"] += 1
    stats["chars"] = chars
    stats["approx_tokens"] = int(chars / 3.5)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
