"""Türkçe Vikipedi dökümünden ana ad alanı sayfalarının ham wikitext'ini tek dosyaya yazar.

Temizleme yapılmaz; yalnızca XML sarmalayıcısı açılır ve sayfalar boş satırla ayrılır.

    python tools/extract_tr_wiki_raw.py --src data/corpus/raw/trwiki-latest-pages-articles.xml.bz2 \
        --out data/corpus/raw/trwiki_raw.txt
"""

import argparse, bz2, os, sys
import xml.etree.ElementTree as ET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pages = 0
    with bz2.open(args.src, "rb") as src, open(args.out, "w", encoding="utf-8") as out:
        for _, elem in ET.iterparse(src, events=("end",)):
            if elem.tag.endswith("}page"):
                if elem.findtext("{*}ns") == "0":
                    body = elem.findtext("{*}revision/{*}text") or ""
                    if body:
                        out.write(body.replace("\r", "") + "\n\n")
                        pages += 1
                elem.clear()
    print(f"pages={pages} bytes={os.path.getsize(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
