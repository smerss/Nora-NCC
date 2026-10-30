#!/usr/bin/env python3
"""Ham Türkçe kaynaklardan sabit train/confirmation akışlarını yeniden kur.

Film confirmation'ı train'e karışmaz. XQuAD yalnız bağımsız onam içindir. Bu betik
model seçmez ve development cümlelerini haricî akışa eklemez.
"""
import hashlib
import json
import os
import random
import re
import subprocess

from nora_besle import DIALOGUE, SENTENCES

RAW = ".cache/corpora/raw"
OUT = ".cache/corpora/processed"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def commit(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    target_rows = SENTENCES + DIALOGUE
    write_jsonl(f"{OUT}/target_development.jsonl",
                [{"text": x, "source": "target_every5_development"} for x in target_rows[::5]])
    wiki_path = f"{RAW}/turkish-sentences-dataset/wiki.tr.txt"
    chat_path = f"{RAW}/Turkish_Chatbot_Dataset/movie_lines.txt"
    wiki = [x.strip() for x in open(wiki_path, encoding="utf8") if x.strip()]
    chat = []
    for line in open(chat_path, encoding="utf8"):
        parts = re.split(r"\s+\+\+\+\$\+\+\++\s*", line.rstrip("\n"))
        if len(parts) >= 5:
            chat.append({"film": parts[2], "text": parts[4].strip()})
    films = sorted({x["film"] for x in chat})
    held_films = set(random.Random(51).sample(films, 24))
    held_chat = [x for x in chat if x["film"] in held_films]
    train_chat = [x for x in chat if x["film"] not in held_films and len(x["text"]) >= 4]
    train = ([{"text": x, "source": "wiki_cc0"} for x in wiki if len(x) >= 4]
             + [{"text": x["text"], "source": "chatbot_gpl3", "film": x["film"]}
                for x in train_chat])
    random.Random(140).shuffle(train)
    write_jsonl(f"{OUT}/external_train.jsonl", train)
    write_jsonl(f"{OUT}/chatbot_confirmation.jsonl", held_chat)

    xquad_path = f"{RAW}/xquad/xquad.tr.json"
    data = json.load(open(xquad_path, encoding="utf8"))
    contexts = [p["context"] for article in data["data"] for p in article["paragraphs"]]
    raw_sentences = [x.strip() for context in contexts
                     for x in re.split(r"(?<=[.!?…])\s+", context) if x.strip()]
    cleaned = [re.sub(r"[()]", " ", x) for x in raw_sentences]
    sentences = [x for x in cleaned if len(re.findall(r"[^\W\d_]+", x)) >= 4]
    write_jsonl(f"{OUT}/xquad_rebuilt_v2.jsonl",
                [{"text": x, "source": "xquad_cc_by_sa_4"} for x in sentences])

    # Bu kaynak bu turda yalnız nihai onam için açıldı; hiçbir eğitim/router akışına girmez.
    unseen = None
    unseen_path = f"{RAW}/unseen_faq/dataset_v1.txt"
    if os.path.exists(unseen_path):
        faq = []
        for line in open(unseen_path, encoding="utf8"):
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) != 3:
                continue
            page, question, answer = parts
            for kind, text in (("question", question), ("answer", answer)):
                for sentence in re.split(r"(?<=[.!?…])\s+", text):
                    sentence = re.sub(r"[()]", " ", sentence).strip()
                    if len(re.findall(r"[^\W\d_]+", sentence)) >= 4:
                        faq.append({"text": sentence, "source": "unseen_faq_mit",
                                    "page": page, "kind": kind})
        deduped, seen = [], set()
        for row in faq:
            if row["text"] not in seen:
                seen.add(row["text"])
                deduped.append(row)
        faq = deduped
        base_raw_texts = set(wiki) | {x["text"] for x in chat}
        train_texts = {row["text"] for row in train}
        faq = [row for row in faq if row["text"] not in base_raw_texts and row["text"] not in train_texts]
        write_jsonl(f"{OUT}/unseen_faq_confirmation.jsonl", faq)
        unseen = {"sentences": len(faq), "commit": commit(f"{RAW}/unseen_faq"),
                  "sha256": sha256(unseen_path), "license": "MIT",
                  "exact_overlap_raw_wiki_or_all_chat": sum(x["text"] in base_raw_texts for x in faq),
                  "exact_overlap_external_train": sum(x["text"] in train_texts for x in faq)}
    metadata = {
        "train_sentences": len(train), "chatbot_confirmation_lines": len(held_chat),
        "chatbot_confirmation_films": len(held_films), "chatbot_split_seed": 51,
        "train_shuffle_seed": 140, "xquad_rebuilt_v2_sentences": len(sentences),
        "xquad_preprocessing": "split .!?… whitespace; parentheses -> space; >=4 alphabetic words",
        "sources": {
            "wiki": {"commit": commit(f"{RAW}/turkish-sentences-dataset"), "sha256": sha256(wiki_path)},
            "chatbot": {"commit": commit(f"{RAW}/Turkish_Chatbot_Dataset"), "sha256": sha256(chat_path)},
            "xquad": {"commit": commit(f"{RAW}/xquad"), "sha256": sha256(xquad_path)},
        },
        "note": "XQuAD v2 eski 37086 tokenlık kayıp işlenmiş dosya değildir; taban bu dosyada yeniden ölçülür.",
        "unseen_confirmation": unseen,
    }
    with open(f"{OUT}/provenance.json", "w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
