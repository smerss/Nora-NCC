"""NORA — BEYİN DEPOSU: sinapslar diskte, RAM yalnızca dokunulan sayfa kadar.

Neden?
------
Nora'nın hedefi "her PC'de çalışsın" (GPU'suz, LLM'siz). Ama beyin büyüdükçe bütün
ağırlıkları RAM'e kopyalamak hem yavaş hem gereksizdir: bir insan beyni de bütün
bağlantılarını aynı anda "RAM'de" tutmaz; erişilen izler aktive olur.

Çözüm üç parçadır:

1) `BrainStore` — sinaps matrisleri `numpy.memmap` ile doğrudan diskte tutulur.
   Dosyayı açmak O(1)'dir (tüm dosya okunmaz); bir sinapsa yazmak yalnızca o
   sayfayı (4 KB) RAM'e çeker. Bellek büyümesi = dokunulan sayfa sayısı, parametre
   sayısı değil. İşletim sisteminin sayfa önbelleği sayesinde sıcak sayfalar
   RAM hızında okunur.

2) `Journal` — her konuşma/yaşam adımı **append-only** günlüğe O(1) eklenir.
   Tüm beyni her turda yeniden yazmak yok (klasik JSON dump = O(boyut)).
   Çökme durumunda yalnızca son satır kaybolur; yarım dosya oluşmaz.

3) `snapshot_atomic` — tam anlık görüntü geçici dosyaya yazılıp `os.replace` ile
   atomik olarak yerine konur; kesinti sırasında eski sağlam görüntü kalır.

Bu modül Nora'nın çalışması için ZORUNLU değildir; isteğe bağlı depolama katmanıdır.
Canlı beyne bağlanabilir (`attach_live`): öğrenme sırasında yazılan sinapslar
doğrudan eşlenmiş sayfalara gider.
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterable, Iterator

import numpy as np

_HEADER = "_brain.json"


class BrainStore:
    """Bir dizinde memmap tabanlı sinaps deposu. Açma/yazma maliyeti O(1) sayfa."""

    def __init__(self, root: str):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)
        self._maps: dict[str, np.memmap] = {}
        self._header_path = os.path.join(self.root, _HEADER)

    # ------------------------------------------------------------------ meta
    def _read_header(self) -> dict:
        try:
            with open(self._header_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_header(self, h: dict) -> None:
        tmp = self._header_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(h, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self._header_path)

    # ---------------------------------------------------------------- diziler
    def array(self, name: str, shape: tuple[int, ...] | None = None,
              dtype=np.float32, create: bool = False) -> np.memmap:
        """Adlandırılmış sinaps matrisini eşle. `create=True` ise sıfırdan açar."""
        key = str(name)
        if key in self._maps:
            return self._maps[key]
        h = self._read_header()
        path = os.path.join(self.root, f"{key}.bin")
        if create and shape is not None:
            shape = tuple(int(s) for s in shape)
            need = int(np.prod(shape)) * np.dtype(dtype).itemsize
            # truncate: dosyayı fiziksel olarak yazmaz, boyut verir (seyrek dosya)
            with open(path, "wb") as f:
                f.truncate(need)
            h[key] = {"shape": list(shape), "dtype": np.dtype(dtype).name, "bytes": need}
            self._write_header(h)
        info = h.get(key)
        if info is None:
            raise KeyError(f"bilinmeyen dizi: {key}")
        dt = np.dtype(info.get("dtype") or dtype)
        arr = np.memmap(path, dtype=dt, mode="r+", shape=tuple(info["shape"]))
        self._maps[key] = arr
        return arr

    def flush(self) -> None:
        for arr in self._maps.values():
            try:
                arr.flush()
            except Exception:
                pass

    def close(self) -> None:
        self.flush()
        self._maps.clear()

    def bytes_on_disk(self) -> int:
        total = 0
        for dirpath, _dirs, files in os.walk(self.root):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        return total


class Journal:
    """Append-only olay günlüğü: her tur O(1) ekleme, O(1) disk.

    Satır biçimi: tek satır JSON. Yazma tamponludur; `flush()` ile diske iner.
    """

    def __init__(self, path: str, buffer_bytes: int = 1 << 16):
        self.path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._buf = buffer_bytes
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)

    def append(self, record: dict) -> int:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._f.write(line)
        return len(line.encode("utf-8"))

    def flush(self) -> None:
        try:
            self._f.flush()
            os.fsync(self._f.fileno())
        except Exception:
            pass

    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def replay(self, limit: int | None = None) -> Iterator[dict]:
        n = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
                    n += 1
                    if limit is not None and n >= limit:
                        return
        except FileNotFoundError:
            return

    def last(self, n: int = 1) -> list[dict]:
        try:
            lines = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    lines.append(line)
                    if len(lines) > 4 * max(1, n):
                        lines = lines[-max(1, n):]
            out = []
            for line in lines[-max(1, n):]:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
            return out
        except FileNotFoundError:
            return []

    def rotate(self, keep_last: int = 0) -> None:
        """Günlüğü döndür: istenirse son kayıtlar yeni dosyaya taşınır (O(k))."""
        keep = self.last(keep_last) if keep_last > 0 else []
        self.close()
        os.replace(self.path, self.path + ".1")
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)
        for rec in keep:
            self.append(rec)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def snapshot_atomic(path: str, arrays: dict[str, np.ndarray], meta: dict | None = None) -> str:
    """Anlık görüntüyü atomik yaz: `.tmp` + `os.replace`. Yarım dosya kalmaz."""
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)
    if meta is not None:
        mp = path + ".meta.json"
        mtmp = mp + ".tmp"
        with open(mtmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        os.replace(mtmp, mp)
    return path


def attach_live(agi, root: str) -> dict:
    """Canlı Nora'nın büyük sinaps matrislerini diske eşle (isteğe bağlı mod).

    Dönüş: hangi dizilerin eşlendiği. Öğrenme sırasında yazılan sinaps doğrudan
    dosyaya gider; RAM'de bütün matris tutulmaz.
    """
    store = BrainStore(root)
    attached = {}
    targets = [
        ("fast_mb_w_fast", lambda: agi.fast.mb.W_fast),
        ("fast_mb_w_slow", lambda: agi.fast.mb.W_slow),
        ("mind_wq", lambda: agi.mind.Wq),
    ]
    for name, getter in targets:
        try:
            cur = getter()
            shape = tuple(cur.shape)
            arr = store.array(name, shape=shape, dtype=cur.dtype, create=True)
            if hasattr(cur, "shape"):
                arr[...] = cur[...]
            attached[name] = shape
        except Exception:
            continue
    # canlı bağ: alanları eşlenmiş diziye çevir
    try:
        agi.fast.mb.W_fast = store.array("fast_mb_w_fast")
        agi.fast.mb.W_slow = store.array("fast_mb_w_slow")
    except Exception:
        pass
    try:
        agi.mind.ctxs[agi.mind.cur]["Wq"] = store.array("mind_wq")
    except Exception:
        pass
    agi._brain_store = store
    return attached


def brain_rss_kb() -> int:
    """Sürecin yerleşik bellek boyu (KB) — Linux /proc; yoksa -1."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return -1


def timed_open_ms(store_factory) -> float:
    t = time.perf_counter()
    obj = store_factory()
    _ = obj
    return (time.perf_counter() - t) * 1000.0
