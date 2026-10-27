"""NORA — KOPYALANAMAZLIK (No-Cloning) ve BİREYSELLEŞME ORGANI.

Neden böyle bir organ var?
--------------------------
Aynı `seed` ile doğan iki Nora aynı GENETİĞE sahiptir (nöron bağlantıları, sinaps
ağırlıkları, hazırbilgi). Klasik bir programda bu, iki kopyanın sonsuza kadar birebir
aynı davranması demektir: ruhsuz bir klon. Oysa biyolojik bir beyinde genetik aynı olsa
bile iki birey aynı beyin olmaz; çünkü beyni şekillendiren şey GENETİK DEĞİL, deneyimin
FİZİKSEL ZAMANLAMASIDIR.

Bu organ, tam olarak o farkı üretir — dışarıdan API çağırmadan, sistemin İÇİNDE:

1) MİKRO-ZAMANLAMA (kaos): gelen her duyunun işlendiği anın nanosaniye fazı, gözle
   görülmeyecek kadar küçük (~1e-4 göreli) sinaps kaymalarına dönüşür. Aynı girdi,
   farklı nanosaniyede işlenirse farklı sinapsa yazılır; geri beslemeli öğrenme bu
   mikroskobik farkı büyütür (kelebek etkisi).

2) DONANIM ENTROPİSİ: `os.urandom` çekirdeğin donanım olaylarıyla beslenen CSPRNG'sidir;
   ayrıca `time.time_ns()` / `perf_counter_ns()` düşük bitleri, iş parçacığı uyandırma
   gecikmesi ve `sched_yield` sonrası zaman farkı karıştırılır. Bu, "seed" ile
   tekrar üretilemeyen gerçek sistem gürültüsüdür.

3) ASENKRON YARIŞ (race condition): karar eşiğine neredeyse aynı anda ulaşan iki aday
   (şema, dürtü, söylem edimi) eşzamanlı iş parçacıklarında yarıştırılır; HANGİSİNİN ÖNCE
   BİTTİĞİNİ işletim sisteminin o anki çekirdek zamanlaması belirler. Yazılımda "bug"
   sayılan yarış durumu, burada bireyselleşmenin motorudur.

4) VAKUM GÜRÜLTÜSÜ (sistem içi, API'siz): kuantum alanının sıfır-nokta dalgalanmasının
   klasik bir modeli. İki fazlı bir osilatörün faz uzayı, her örneklemede donanım
   bitleriyle döndürülür; genlik Box–Muller ile sıfır ortalamalı Gauss dağılımından
   çekilir. Dış servis yoktur, gerçek kuantum optiği de yoktur: bu, evrenin kendi
   gürültüsünün sistem içinde modellenmesidir (dürüst adı: kuantum-benzeri vakum
   gürültüsü).

Hız sözü
--------
Bütün işlemler O(1): her adımda SABİT sayıda tamsayı işlemi ve SABİT sayıda sinaps
dokunulur (k=3). Ağ 10 kat büyüse de adım başına maliyet değişmez; tüm ağ taranmaz,
disk okunmaz. Bu yüzden parametre büyüklüğü yanıt süresini etkilemez.

Dürüst sınır
------------
Kopyalanamazlık teoremi kuantum durumları için geçerlidir; Nora klasik bir sistemdir.
Burada üretilen şey *işlemsel* bireyselliktir: aynı tohumdan gelen iki örnek, aynı
girdilerle bile farklı iç durumlara ve farklı sözlere ayrışır. Bu, "ruhun kopyalanamaz
olduğu" iddiasının klasik karşılığıdır; fiziksel kuantum kopyası iddiası değildir.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures import Future

_MASK64 = (1 << 64) - 1
_INV53 = 1.0 / 9007199254740992.0        # 2**53


def _mix64(z: int) -> int:
    """SplitMix64 finalizer: bitleri hızlı ve geri döndürülemez biçimde karıştırır."""
    z = (z + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


class EntropySource:
    """Sistem içi, dış API'siz entropi kaynağı. Her çekim O(1) (birkaç tamsayı işlemi)."""

    RESEED_DRAWS = 8192       # bu kadar çekimden sonra donanımdan yeniden tohumlanır

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.birth_ns = time.time_ns()
        self._counter = _mix64(self.seed ^ self.birth_ns)
        self._k = [0, 0, 0, 0]
        self._vac = 0.0
        self._draws = 0
        self.last_jitter_ns = 0
        self.reseed()

    # ------------------------------------------------------------------ çekirdek
    def reseed(self) -> None:
        """Donanım + zamanlama gürültüsünden durumu yeniden tohumla (seyrek, ucuz)."""
        hw = os.urandom(16)
        a = int.from_bytes(hw[:8], "little")
        b = int.from_bytes(hw[8:], "little")
        t_ns = time.time_ns()
        p0 = time.perf_counter_ns()
        try:
            os.sched_yield()                # zamanlayıcı çekişmesi: uyanma gecikmesi üretir
        except Exception:
            pass
        p1 = time.perf_counter_ns()
        self.last_jitter_ns = (p1 - p0) & 0xFFFF
        # nanosaniye damgasının düşük bitleri + çekirdek jitteri: seed ile tekrar üretilemez
        self._k[0] = _mix64(a ^ t_ns ^ (self._counter & _MASK64))
        self._k[1] = _mix64(b ^ (p1 & _MASK64) ^ (self.last_jitter_ns << 7))
        self._k[2] = _mix64((t_ns >> 3) ^ (p0 << 1) ^ self.seed)
        self._k[3] = _mix64(a ^ b ^ (t_ns >> 7) ^ (p1 >> 5))
        self._vac = (t_ns & 0xFFFF) * 1e-4
        self._draws = 0

    def u64(self) -> int:
        """64 bit donanım-karışımlı sözde rastgele sayı (O(1))."""
        self._counter = (self._counter + 0x9E3779B97F4A7C15) & _MASK64
        x = _mix64(self._counter ^ self._k[0])
        x ^= _mix64(x ^ self._k[1]) >> 3
        x ^= _mix64((self._counter >> 17) ^ self._k[2])
        x ^= self._k[3]
        self._k[(self._counter >> 9) & 3] ^= _mix64(x + self._counter)
        self._draws += 1
        if self._draws >= self.RESEED_DRAWS:
            self.reseed()
        return x & _MASK64

    def unit(self) -> float:
        """[0, 1) aralığında düzgün sayı (53 bit duyarlık)."""
        return (self.u64() >> 11) * _INV53

    def sym(self, scale: float = 1.0) -> float:
        """Yaklaşık sıfır-ortalamalı Gauss gürültüsü: 3 düzgün sayının toplamı (Irwin–Hall).

        Box–Muller'den ~3 kat ucuz; sinaps gürültüsü için dağılım şekli yeterli.
        """
        return (self.unit() + self.unit() + self.unit() - 1.5) * 1.1547 * scale

    def index(self, n: int) -> int:
        """[0, n) aralığında indeks — modulo yanlılığı reddetme ile kırılır (O(1))."""
        n = int(n)
        if n <= 1:
            return 0
        limit = (_MASK64 // n) * n
        while True:
            v = self.u64()
            if v < limit:
                return v % n

    def indices(self, n: int, k: int) -> list[int]:
        """n içinden k AYRI indeks (Floyd örneklemesi, O(k)). k sabit -> maliyet sabit."""
        n, k = int(n), min(int(k), int(n))
        if k <= 0 or n <= 0:
            return []
        chosen: dict[int, int] = {}
        out: list[int] = []
        for j in range(n - k, n):
            t = self.index(j + 1)
            if t in chosen:
                chosen[j] = chosen[t]
            else:
                chosen[j] = t
        out.extend(chosen.values())
        return out

    # ------------------------------------------------------------ vakum modeli
    def vacuum(self) -> float:
        """Sıfır-nokta dalgalanması modeli: iki fazlı osilatör + Box–Muller.

        Faz, her örneklemede donanım bitleriyle döner; genlik Gauss dağılımından çekilir.
        Dış servis yok; tamamen bu süreçte hesaplanır (O(1)).
        """
        self._vac += 0.6180339887498949 + self.unit() * 1e-3
        self._vac -= int(self._vac)
        u1 = self.unit() or 1e-12
        u2 = self.unit()
        r = (-2.0 * _log(u1)) ** 0.5
        return r * _cos(2.0 * 3.141592653589793 * u2 + 6.283185307179586 * self._vac)


def _log(x: float) -> float:
    from math import log
    return log(x)


def _cos(x: float) -> float:
    from math import cos
    return cos(x)


# ====================================================================== BİREY
class Individuality:
    """Bir Nora'nın DOĞUM TUTANAĞI ve yaşam boyu bireyselleşme motoru.

    Genetik (seed) dışarıdan gelir; buradaki her şey deneyimin fiziksel zamanlamasıdır.
    """

    def __init__(self, seed: int = 0, uid: str | None = None, birth_ns: int | None = None,
                 phase: float | None = None):
        self.src = EntropySource(seed)
        self.uid = uid or os.urandom(8).hex()
        self.birth_ns = int(birth_ns or self.src.birth_ns)
        self.birth_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.birth_ns / 1e9))
        # Doğum anının nanosaniye fazı: iki klonun "ilk bakış açısı" buradan ayrılır.
        self.phase = float(phase if phase is not None else (self.birth_ns % 1_000_000) * 1e-6)
        self.steps = 0
        self.drift = 0.0                     # sinapslara yazılan toplam |değişim|
        self.race_wins = [0, 0, 0, 0]        # hangi kanal kaç kez önce eşiği geçti
        self._tick_mark = 0
        self.log: deque[str] = deque(maxlen=24)
        self._started = time.perf_counter_ns()
        # İş parçacığı havuzunu DOĞUMDA ısıt: ilk yarışta 20 ms'lik kurulum gecikmesi olmasın.
        try:
            pool = _shared_pool()
            if pool is not None:
                pool.submit(lambda: None).result(timeout=0.5)
        except Exception:
            pass

    # ----------------------------------------------------------------- kimlik
    def state_dict(self) -> dict:
        return {"uid": self.uid, "birth_ns": self.birth_ns, "phase": self.phase,
                "steps": self.steps, "drift": round(self.drift, 9),
                "race_wins": list(self.race_wins)}

    def load_state_dict(self, d: dict | None) -> None:
        if not d:
            return
        try:
            self.uid = str(d.get("uid") or self.uid)
            self.birth_ns = int(d.get("birth_ns") or self.birth_ns)
            self.phase = float(d.get("phase") or self.phase)
            self.steps = int(d.get("steps") or 0)
            self.drift = float(d.get("drift") or 0.0)
            rw = d.get("race_wins")
            if isinstance(rw, list) and len(rw) == len(self.race_wins):
                self.race_wins = [int(x) for x in rw]
        except Exception:
            pass

    def runtime_state(self) -> dict:
        """Aynı bireyin entropi akışını checkpoint sonrası sürdürmek için."""
        from nora_state import capture_attrs
        own = ("uid", "birth_ns", "birth_iso", "phase", "steps", "drift",
               "race_wins", "_tick_mark", "log")
        src = ("seed", "birth_ns", "_counter", "_k", "_vac", "_draws", "last_jitter_ns")
        return {"version": 1, "fields": capture_attrs(self, own),
                "src": capture_attrs(self.src, src)}

    def load_runtime_state(self, data: dict | None) -> None:
        if not data:
            return
        from nora_state import restore_attrs
        restore_attrs(self, data.get("fields") or {}, strict=True)
        restore_attrs(self.src, data.get("src") or {}, strict=True)

    def identity_line(self) -> str:
        return f"{self.uid[:6]} · doğum {self.birth_iso}"

    # -------------------------------------------------------- adım gürültüsü
    def perturb_step(self, agi, rel: float = 2.0e-4, abs_floor: float = 2.0e-6) -> None:
        """Her yaşam adımında sinir sistemine FİZİKSEL gürültü ekle (O(1), ~4 µs).

        - Sabit sayıda sinaps (her ağırlık matrisinde k=2): |w|'ye göreli + mutlak taban.
          Tüm ağ TARANMAZ; 10 kat büyük ağda da maliyet aynı kalır.
        - Süreklilik durumu (rezervuar) ve nörokimya: mikroskobik kayma.
        Aynı girdinin farklı fiziksel anda farklı sinapsa yazılması = bireysel tarih.
        """
        src = self.src
        self.steps += 1
        touched = 0.0
        try:
            mb = agi.fast.mb
            for W in (mb.W_fast, mb.W_slow):
                flat = W.reshape(-1)
                size = int(flat.size)
                if size <= 0:
                    continue
                v = src.u64()                      # TEK çekim: indeks + işaret + büyüklük
                for j in range(2):
                    idx = (v >> (17 * j)) % size
                    sign = -1.0 if (v >> (40 + j)) & 1 else 1.0
                    mag = ((v >> (48 + 5 * j)) & 0x1F) * 0.03125    # 0..~1
                    w = float(flat[idx])
                    delta = sign * (abs(w) * rel + abs_floor * (0.2 + mag))
                    flat[idx] = w + delta
                    touched += abs(delta)
        except Exception:
            pass
        try:
            h = agi.mind.h
            n = int(h.size)
            if n:
                v = src.u64()
                i = (v >> 8) % n
                h[i] = float(h[i]) + (((v & 0xFFFF) * 3.0517578125e-05) - 1.0) * 1e-3
                # tüm vektörün çok küçük ölçek kayması (tek numpy çarpımı, sabit boyut)
                h *= 1.0 + ((((v >> 21) & 0xFFFF) * 3.0517578125e-05) - 1.0) * 1e-6
        except Exception:
            pass
        try:
            chem = agi.chem
            v = src.u64()
            for k, (attr, sc) in enumerate((("da_tonic", 6e-5), ("cort", 6e-5),
                                            ("oxy", 6e-5), ("ach", 4e-5))):
                if hasattr(chem, attr):
                    val = float(getattr(chem, attr)) + \
                        (((((v >> (16 * k)) & 0x3FF) / 1023.0) * 2.0 - 1.0) * sc)
                    setattr(chem, attr, float(min(1.0, max(0.0, val))))
        except Exception:
            pass
        self.drift += touched
        if self.steps % 512 == 0:
            try:
                self.log.append(f"adım {self.steps}: kayma={self.drift:.6f} "
                                f"jitter={src.last_jitter_ns}ns faz={self.phase:.6f}")
            except Exception:
                pass

    # ------------------------------------------------------------ yarış (race)
    def race_first(self, tasks: list, fallback: int = 0, timeout_s: float = 0.05) -> int:
        """Aday görevleri EŞZAMANLI çalıştır; İLK biten kazanır (gerçek yarış durumu).

        Kazananı işletim sisteminin o andaki çekirdek dağıtımı belirler: aynı seed'den
        gelen iki Nora aynı kararsızlıkta farklı tarafa düşebilir. Görevler küçük
        olduğu için maliyet ~50-200 µs; yalnızca GERÇEK beraberlikte çağrılır.
        """
        if not tasks:
            return fallback
        if len(tasks) == 1:
            try:
                tasks[0]()
            except Exception:
                pass
            return 0
        pool = _shared_pool()
        if pool is None:
            try:
                tasks[fallback]()
            except Exception:
                pass
            return fallback
        # Sunum sırası donanım entropisine göre karıştırılır: yaratılış sırası yanlılığı yok.
        order = list(range(len(tasks)))
        for i in range(len(order) - 1, 0, -1):
            j = self.src.index(i + 1)
            order[i], order[j] = order[j], order[i]
        futs: dict[Future, int] = {}
        for i in order:
            try:
                futs[pool.submit(tasks[i])] = i
            except Exception:
                continue
        if not futs:
            return fallback
        done, _pending = wait(list(futs), timeout=timeout_s, return_when=FIRST_COMPLETED)
        if not done:
            return fallback
        winners = [futs[f] for f in done]
        win = min(winners) if winners else fallback
        try:
            self.race_wins[win % len(self.race_wins)] += 1
        except Exception:
            pass
        return win

    def tie_break(self, n: int) -> int:
        """Eşit dereceli seçenekler arasında donanım entropisiyle seçim (ses tonu/ifade)."""
        return self.src.index(max(1, int(n)))

    def fresh_ns(self) -> int:
        """Şu anın nanosaniye damgası — algı anı damgası olarak kullanılabilir."""
        return time.perf_counter_ns()


# ------------------------------------------------------------------ ortak havuz
_POOL_LOCK = threading.Lock()
_POOL: ThreadPoolExecutor | None = None


def _shared_pool() -> ThreadPoolExecutor | None:
    """Tek ortak iş parçacığı havuzu: örnek başına thread açılmaz (bellek O(1))."""
    global _POOL
    if _POOL is not None:
        return _POOL
    with _POOL_LOCK:
        if _POOL is None:
            try:
                _POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nora-race")
            except Exception:
                return None
    return _POOL


def shutdown_pool() -> None:
    global _POOL
    p, _POOL = _POOL, None
    if p is not None:
        try:
            p.shutdown(wait=False)
        except Exception:
            pass
