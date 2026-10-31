import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

"""Türkçe Vikipedi döküm dosyasını AKIŞLA okuyup düzyazı çıkarır (Nora'nın okuyarak dil öğrenmesi için ham veri).
Tüm dosyayı indirmez: hedef karakter sayısına ulaşınca durur.
    python make_corpus_dump.py --chars 60000000 --out data/text/tr_dump.txt
"""
import argparse
import bz2
import os
import re
import urllib.request

URL = "https://dumps.wikimedia.org/trwiki/latest/trwiki-latest-pages-articles.xml.bz2"
RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_REF = re.compile(r"<ref[^>/]*?>.*?</ref>|<ref[^>]*/>", re.S)
RE_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
RE_TMPL = re.compile(r"\{\{[^{}]*\}\}")
RE_FILE = re.compile(r"\[\[(?:Dosya|Resim|File|Image|Kategori|Category)[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
RE_LINK2 = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
RE_LINK1 = re.compile(r"\[\[([^\]]*)\]\]")
RE_EXT = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
RE_BOLD = re.compile(r"'{2,}")
RE_PAREN = re.compile(r"\([^)]*\)")


def clean_wikitext(t: str) -> list[str]:
    t = RE_COMMENT.sub("", t)
    t = RE_REF.sub("", t)
    for _ in range(6):
        n = RE_TMPL.sub("", t)
        if n == t:
            break
        t = n
    t = RE_FILE.sub("", t)
    t = RE_LINK2.sub(r"\1", t)
    t = RE_LINK1.sub(r"\1", t)
    t = RE_EXT.sub(r"\1", t)
    t = RE_TAG.sub("", t)
    t = RE_BOLD.sub("", t)
    out = []
    for ln in t.split("\n"):
        ln = ln.strip()
        if len(ln) < 80 or ln[0] in "*#:;|!{=<[" or "http" in ln or "{{" in ln or "}}" in ln:
            continue
        ln = RE_PAREN.sub("", ln)
        ln = re.sub(r"\s+", " ", ln).strip()
        letters = sum(c.isalpha() for c in ln)
        if letters >= 0.75 * len(ln):
            out.append(ln)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=60_000_000)
    ap.add_argument("--out", default="data/text/tr_dump.txt")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "NoraAGI-Corpus/1.0 (research; streaming)"})
    total = pages = 0
    in_text, buf, ns0, redirect = False, [], True, False
    with urllib.request.urlopen(req, timeout=60) as resp, bz2.BZ2File(resp) as f, open(a.out, "w", encoding="utf-8") as out:
        for raw in f:
            ln = raw.decode("utf-8", errors="ignore")
            if "<page>" in ln:
                ns0, redirect = True, False
            elif "<ns>" in ln:
                ns0 = "<ns>0</ns>" in ln
            elif "<redirect" in ln:
                redirect = True
            if "<text" in ln:
                in_text, buf = True, []
                ln = ln.split(">", 1)[1] if ">" in ln else ""
            if in_text:
                if "</text>" in ln:
                    buf.append(ln.split("</text>")[0])
                    in_text = False
                    if ns0 and not redirect:
                        txt = "\n".join(buf).replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")
                        paras = clean_wikitext(txt)
                        if paras:
                            s = "\n".join(paras) + "\n"
                            out.write(s)
                            total += len(s)
                            pages += 1
                            if pages % 2000 == 0:
                                print(pages, total, flush=True)
                    if total >= a.chars:
                        break
                else:
                    buf.append(ln)
    print(f"{pages} madde, {total} karakter -> {a.out}")


if __name__ == "__main__":
    main()
