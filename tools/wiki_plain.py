"""Vikipedi ham wikitext'ini düz metne çevirir: yalnızca biçim işaretlerini söker, içerik süzmez.

    python tools/wiki_plain.py --src data/corpus/raw/trwiki_raw.txt --out data/corpus/raw/trwiki_plain.txt
"""

import argparse, os, sys

from clean_tr_corpus import strip_wiki


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pages = 0
    with open(args.src, encoding="utf-8", errors="replace") as src, \
            open(args.out, "w", encoding="utf-8") as out:
        buf = []
        for line in src:
            if line.strip() == "":
                if buf:
                    text = strip_wiki("\n".join(buf))
                    if text:
                        out.write(text + "\n")
                    pages += 1
                    buf = []
            else:
                buf.append(line.rstrip("\n"))
        if buf:
            out.write(strip_wiki("\n".join(buf)) + "\n")
            pages += 1
    print(f"pages={pages} bytes={os.path.getsize(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
