"""Türkçe ham metinleri (Vikipedi dökümü, OSCAR jsonl) temizleyip tek bir düzyazı dosyasına yazar.

Çıktı: her satır bir paragraf. Yapay-zeka kalıbı içeren, tekrar eden, bozuk kodlamalı,
Türkçe olmayan ya da markup kalıntısı taşıyan paragraflar elenir.

    python tools/clean_tr_corpus.py --wiki data/corpus/raw/trwiki-latest-pages-articles.xml.bz2 \
        --web data/corpus/raw/oscar2301_part.jsonl --out data/corpus/clean/tr_prose.txt
"""

import argparse, bz2, hashlib, json, os, re, sys, unicodedata
import xml.etree.ElementTree as ET

TR_LETTERS = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")

# Yapay-zeka asistanı kalıpları: insan metninde nadir, model çıktısında sık.
AI_PATTERNS = [
    r"elbette[!,.]", r"tabii ki[!,.]", r"harika bir soru", r"umarım bu (yardımcı|faydalı)",
    r"umarım (yardımcı|faydalı) olmuştur", r"bir yapay zeka (olarak|asistan|modeli)",
    r"yapay zeka asistanı", r"yardımcı olabileceğim", r"size nasıl yardımcı", r"başka bir sorunuz",
    r"^işte (bir|size|sana|birkaç)", r"aşağıda (bir|size|sana|birkaç|verilen)", r"sorunuz için teşekkür",
    r"as an ai", r"i hope this helps", r"certainly[!,.]", r"great question", r"let me know if",
]
AI_RE = re.compile("|".join(AI_PATTERNS), re.IGNORECASE | re.MULTILINE)

# Web sayfası çöplüğü: çerez, menü, alışveriş, yorum formu kalıntıları.
BOILERPLATE_RE = re.compile(
    r"https?://|www\.|@\w+\.\w+|çerez|cookie|tıklayın|sepete ekle|üye ol|giriş yap|"
    r"paylaş(ın|mak için)|kaydol|abone ol|yorum yap|javascript|\{|\}|<[a-z/][^>]*>|\|",
    re.IGNORECASE,
)


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def tr_ok(par):
    """Paragraf gerçekten Türkçe düzyazı mı? Harf oranı, Türkçe harf payı, cümle yapısı."""
    letters = [c for c in par if c.isalpha()]
    if len(letters) < 0.7 * len(par.replace(" ", "")):
        return False
    tr_share = sum(c in TR_LETTERS for c in letters) / max(1, len(letters))
    if tr_share < 0.97:  # Türkçe dışı yazı (Latin dışı ya da İngilizce ağırlıklı) elenir
        return False
    words = par.split()
    if len(words) < 12:
        return False
    avg = sum(len(w) for w in words) / len(words)
    if not 3.5 <= avg <= 9.5:
        return False
    if sum(w.isupper() and len(w) > 2 for w in words) > 0.15 * len(words):
        return False
    return True


def strip_wiki(text):
    prev = None
    while prev != text:  # iç içe şablonları sök
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)  # tablolar
    text = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>|<!--.*?-->|<[^>]+>", "", text, flags=re.S)
    text = re.sub(r"\[\[(?:Dosya|File|Resim|Image|Kategori|Category):[^\]]*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)  # bağlantılar -> görünen metin
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"'{2,}", "", text)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("=", "*", "#", ";", ":", "|", "!", "{", "}")):
            continue
        lines.append(s)
    return "\n".join(lines)


def wiki_pages(path):
    """Vikipedi XML dökümünden yalnızca ana ad alanı (ns=0) sayfalarını verir."""
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag.endswith("}page") or elem.tag == "page":
                ns = elem.findtext("{*}ns") or elem.findtext("ns")
                redirect = elem.find("{*}redirect") is not None or elem.find("redirect") is not None
                body = elem.findtext("{*}revision/{*}text") or elem.findtext("revision/text") or ""
                if ns == "0" and not redirect and not body.startswith("#"):
                    yield body
                elem.clear()


def paragraphs_from_wiki(path):
    for body in wiki_pages(path):
        for par in strip_wiki(body).split("\n"):
            yield normalize(par)


def paragraphs_from_web(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            for par in text.split("\n"):
                yield normalize(par)


def keep(par):
    if not (300 <= len(par) <= 1500):  # çok kısa parçalar bağlam taşımaz, çok uzunlar gürültü
        return False
    if par[-1] not in ".!?…\"”)":
        return False
    if BOILERPLATE_RE.search(par) or AI_RE.search(par):
        return False
    return tr_ok(par)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", help="trwiki-*-pages-articles.xml.bz2 yolu")
    ap.add_argument("--web", action="append", default=[], help="OSCAR-benzeri jsonl (birden çok verilebilir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-web-share", type=float, default=0.5,
                    help="çıktıdaki web paragraflarının en çok payı (Vikipedi ağırlığı korunsun)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    seen = set()
    stats = {"seen": 0, "kept": 0, "dup": 0, "wiki": 0, "web": 0}
    buckets = {"wiki": [], "web": []}

    def admit(source, pars):
        for par in pars:
            stats["seen"] += 1
            if not keep(par):
                continue
            digest = hashlib.sha1(re.sub(r"\W+", "", par.lower()).encode("utf-8")).hexdigest()
            if digest in seen:
                stats["dup"] += 1
                continue
            seen.add(digest)
            buckets[source].append(par)

    if args.wiki:
        admit("wiki", paragraphs_from_wiki(args.wiki))
    for path in args.web:
        admit("web", paragraphs_from_web(path))

    web = buckets["web"]
    cap = int(len(buckets["wiki"]) * args.max_web_share / max(1e-9, 1 - args.max_web_share))
    if args.wiki and len(web) > cap:
        web = web[:cap]  # sıra korunur; istenirse dışarıda örnekleme yapılabilir

    total_chars = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for source, pars in (("wiki", buckets["wiki"]), ("web", web)):
            for par in pars:
                out.write(par + "\n")
                total_chars += len(par)
                stats["kept"] += 1
                stats[source] += 1

    approx_tokens = total_chars / 3.5  # Türkçe için kaba bir alt-sözcük oranı
    print(json.dumps({**stats, "chars": total_chars, "approx_tokens": int(approx_tokens)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
