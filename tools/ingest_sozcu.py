import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DATASET_REPO = "ctoraman/sozcu-news-2014"
OUT_DIR = os.path.join(_REPO_ROOT, "data", "text")
OUT_FILE = os.path.join(OUT_DIR, "sozcu_2014.txt")


def clean_text(text: str) -> str:
    # Boşluk ve satır sonlarını normalize et
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def download_single(filename: str):
    try:
        path = hf_hub_download(repo_id=DATASET_REPO, filename=filename, repo_type="dataset")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return clean_text(content)
    except Exception as e:
        return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Hugging Face API üzerinden '{DATASET_REPO}' dosya listesi alınıyor...")
    api = HfApi()
    info = api.dataset_info(DATASET_REPO)
    txt_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".txt")]
    print(f"Toplam {len(txt_files)} adet haber dosyası bulundu.")

    articles = []
    print("Haberler paralel olarak indiriliyor ve işleniyor...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(download_single, fn): fn for fn in txt_files}
        done_count = 0
        for fut in as_completed(futures):
            res = fut.result()
            if res and len(res) > 80:
                articles.append(res)
            done_count += 1
            if done_count % 200 == 0 or done_count == len(txt_files):
                print(f"İlerleme: {done_count}/{len(txt_files)} ({len(articles)} geçerli haber)")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for art in articles:
            # Her haberi tek veya temiz paragraflar halinde yaz
            f.write(art + "\n\n")

    total_chars = sum(len(a) for a in articles)
    print(f"\nİşlem tamamlandı!")
    print(f"Toplam haber: {len(articles)}")
    print(f"Toplam karakter: {total_chars:,}")
    print(f"Kayıt konumu: {OUT_FILE}")


if __name__ == "__main__":
    main()
