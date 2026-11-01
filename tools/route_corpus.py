"""Düz metin korpusunu Nora'nın üç öğrenme yoluna böler: kavram ağı (%70), HDC vektör uzayı (%20), Broca sözdizimi (%10).

Bölüşüm cümle başına, içeriğe bağlı ve tekrarlanabilir: cümlenin MD5 değeri 100'e göre bölünür.
Kavram ağı yolunda nedensel cümleler (X yol açar / neden olur / sebep olur / doğurur) (A, CAUSES, B)
üçlülerine çevrilir; eşleşmeyen cümleler graph_unmatched.txt dosyasına yazılır, kaybolmaz.

    python tools/route_corpus.py --src data/corpus/clean/tr_combined.txt data/corpus/raw/trwiki_plain.txt --out data/routed
"""

import argparse, hashlib, os, re, sys

CAUSAL_RE = re.compile(
    r"^(?P<a>.+?)[,]?\s+(?:yol açar|neden olur|sebep olur|sebebiyet verir|doğurur|yol açmaktadır)[.!?]?$",
    re.IGNORECASE,
)
SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
MAX_TERM_WORDS = 5


def sentences(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            for sent in SPLIT_RE.split(line.strip()):
                words = sent.split()
                if 4 <= len(words) <= 40:
                    yield sent.strip()


def bucket_of(sent):
    value = int(hashlib.md5(sent.encode("utf-8")).hexdigest(), 16) % 100
    if value < 70:
        return "graph"
    if value < 90:
        return "hdc"
    return "broca"


def causal_triple(sent):
    """'Kuraklık kıtlığa yol açar' -> ('Kuraklık', 'CAUSES', 'kıtlığa'); eşleşmezse None.

    Sezgisel kural: fiilden hemen önceki kelime sonuçtur, ondan önceki en çok üç kelime nedendir.
    Türkçe hal ekleri ayrıştırılmaz; sonuç 'kıtlığa' gibi yüzey biçimiyle kalır.
    """
    if not CAUSAL_RE.match(sent):
        return None
    pre = CAUSAL_RE.match(sent).group("a").split()
    if len(pre) < 2:
        return None
    cause = " ".join(pre[:-1][-3:]).strip(" ,")
    effect = pre[-1].strip(" ,")
    if not cause or not effect:
        return None
    return (cause, "CAUSES", effect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = {name: open(os.path.join(args.out, name + ".txt"), "w", encoding="utf-8")
             for name in ("graph_triples", "graph_unmatched", "hdc_sentences", "broca_sentences")}
    counts = {"sentences": 0, "graph": 0, "triples": 0, "unmatched": 0, "hdc": 0, "broca": 0}
    try:
        for path in args.src:
            for sent in sentences(path):
                counts["sentences"] += 1
                bucket = bucket_of(sent)
                counts[bucket] += 1
                if bucket == "graph":
                    triple = causal_triple(sent)
                    if triple:
                        files["graph_triples"].write("\t".join(triple) + "\n")
                        counts["triples"] += 1
                    else:
                        files["graph_unmatched"].write(sent + "\n")
                        counts["unmatched"] += 1
                elif bucket == "hdc":
                    files["hdc_sentences"].write(sent + "\n")
                else:
                    files["broca_sentences"].write(sent + "\n")
    finally:
        for handle in files.values():
            handle.close()
    print(counts)


if __name__ == "__main__":
    sys.exit(main())
