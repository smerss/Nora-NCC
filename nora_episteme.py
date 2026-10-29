#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_episteme.py — Nora'nın İnanç, Çelişki-Farkındalığı, Kaynak Güvenilirliği ve System-1 Konsolidasyon Organı.

Neden bu modül var?
  Nora eskiden her duyduğunu "kaydettim" diye kabul ediyordu. Oysa insan beyni yeni bir iddiayı
  yalnızca DEPOLAMAZ; onu mevcut inançlarıyla ÇARPIŞTIRIR, birden fazla açıklama hipotezi kurar,
  kanıt biriktikçe aralarındaki güveni günceller ve çok tekrarlanan bilgiyi zamanla **System-1'e**
  (otomatik, anlık, bilinçsiz çağrı) indirir. Bu modül tam bu işlevleri O(1) sözlük erişimiyle kurar:

  1. Beta(α, β) inanç hücreleri: her önerme için destek/karşı-kanıt sayacı. Kanıt BAĞIMSIZ
     kaynaklardan geldikçe α büyür; aynı kaynağın tekrarı doğrusal şişirmez (logaritmik azalan katkı).
  2. Çelişki muhasebesi: aynı "yuva"ya (ör. ORTAYA_ÇIKTI|2020) ikinci bir varlık gelirse Nora
     körü körüne kabul etmez; iki hipotez kurar — (H1) aynı olayın iki adı, (H2) aynı yıl ≥2 ayrı olay —
     ve her birine kendi Beta kesinliğini atar. Kanıt geldikçe kazanan doğal olarak öne çıkar.
  3. Kaynak güven modeli: alan adı önseli + KENDİ geçmiş isabet oranından öğrenilen Beta.
     Tek bir blog iddiası ile iki bağımsız güvenilir kaynağın iddiası aynı ağırlıkta DEĞİLDİR.
  4. System-1 konsolidasyonu: yeterli kesinlik + kullanım sayısına ulaşan inanç `automatic=True`
     olur ve sorgularda anında (mikrosaniye) yanıtlanır. Sonradan güçlü bir karşı-kanıt gelirse
     otomatiklik KALDIRILIR (insandaki "beklenti ihlali" gibi System-2 yeniden devreye girer).
  5. Halüsinasyon bekçisi: bir sözcükteki olgusal kavramların ne kadarının bellekte kanıtı olduğunu
     ölçer; kanıtsız kavram varsa beyan "sanıyorum/henüz tek kaynak" diye etiketlenir.
  6. Güvenilir Kaynak Araştırmacısı: yalnızca allowlist'teki alan adlarına erişir, getirdiği metni
     cümle cümle yukarıdaki inanç motoruna aktarır; çok kaynaklı örtüşme güveni yükseltir.

Hiçbir yerde hazır "bilgi tablosu" yoktur: bütün inançlar kanıtlarıyla birlikte yaşam boyu birikir.
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nora_broca import NP, PredSpec, VerbSpec, TurkishSyntaxEngine
from nora_lang import TurkishMorphology, tr_capitalize, tr_lower


def _now() -> float:
    return time.time()


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\wçğıöşü]+", tr_lower(text))


def _tokens_fold(text: str) -> set[str]:
    """Varlık eşleştirmesi için katlanmış kökler: 'COVID-19' ile 'covid 19' aynı kümeye düşer.
    (Türkçe 'I'->'ı' dönüşümü kısaltmaları bozduğu için yalnızca EŞLEŞTİRMEDE ı->i katlanır.)"""
    toks = set(re.findall(r"[\wçğıöşü]+", tr_lower(text)))
    return {t.replace("ı", "i") for t in toks}


def _alias_norm(name: str) -> str:
    """Eş ad anahtarı: boşluk/tire/kısaltma farklarını siler ('COVID-19' -> 'covid19')."""
    t = tr_lower(name).replace("ı", "i")
    return re.sub(r"[^a-z0-9çğöşü]", "", t)


# ============================================================================
# 1. KAYNAK GÜVENİLİRLİĞİ (önsel + yaşam boyu öğrenilen isabet oranı)
# ============================================================================

class SourceTrustModel:
    """Her kaynağın güvenilirliği = alan önseli ile bu kaynaktan gelen iddiaların sonradan
    doğrulanma/yanlışlanma oranının (Beta) karışımı. Önsel tek başına kader değildir:
    sürekli yalan söyleyen bir kaynak zamanla 0.9'dan 0.2'ye düşer, isabetli bir blog yükselebilir."""

    # Alan adı önselleri (bilgi tabanı değil, epistemik önsel: kurumsal/akademik kaynak daha güvenilir)
    DOMAIN_PRIORS: tuple[tuple[str, float], ...] = (
        (".gov", 0.95), (".edu", 0.94), ("who.int", 0.95), ("un.org", 0.93), ("nasa.gov", 0.95),
        ("arxiv.org", 0.92), ("nature.com", 0.94), ("science.org", 0.93), ("tdk.gov.tr", 0.93),
        ("wikipedia.org", 0.88), ("britannica.com", 0.88), ("tübitak.gov.tr", 0.93), ("tubitak.gov.tr", 0.93),
        ("sciencedirect.com", 0.88), ("springer.com", 0.88), ("acm.org", 0.88), ("ieee.org", 0.88),
        ("python.org", 0.92), ("developer.mozilla.org", 0.9),
        ("hurriyet.com.tr", 0.55), ("bbc.com", 0.6), ("aa.com.tr", 0.62), ("reuters.com", 0.68),
        ("medium.com", 0.3), ("blogspot.com", 0.2), ("wordpress.com", 0.2), ("quora.com", 0.22),
        ("twitter.com", 0.15), ("x.com", 0.15), ("facebook.com", 0.12), ("tiktok.com", 0.08),
        ("forum", 0.18), ("reddit.com", 0.25),
    )

    DEFAULT_PRIOR = 0.35

    def __init__(self):
        self.beta: dict[str, list[float]] = defaultdict(lambda: [1.0, 1.0])
        self.evidence_counts: dict[str, int] = defaultdict(int)
        self.last_seen: dict[str, float] = {}

    @staticmethod
    def domain_of(source_id: str) -> str:
        s = tr_lower(str(source_id))
        if s.startswith("insan:") or s.startswith("human:") or s.startswith("konuşmacı:"):
            return "insan"
        m = re.search(r"([a-z0-9-]+(?:\.[a-z0-9-]+)+)", s)
        return m.group(1) if m else s

    def prior_for(self, source_id: str) -> float:
        dom = self.domain_of(source_id)
        if dom == "insan":
            return 0.55   # tanıdık olmayan insan varsayılanı; nörokimyasal güven ayrıca çarpılır
        best = self.DEFAULT_PRIOR
        for suffix, val in self.DOMAIN_PRIORS:
            if dom.endswith(suffix) or suffix in dom:
                best = max(best, val)
        if dom.endswith(".tr"):
            best = min(0.95, best + 0.01)
        return float(best)

    def reliability(self, source_id: str) -> float:
        """Önsel ile deneyimsel Beta'yı harmanlar: çok kanıt geldiyse deneyim ağır basar."""
        dom = self.domain_of(source_id)
        a, b = self.beta[dom]
        learned = a / (a + b)
        n = a + b - 2.0
        w_learned = n / (n + 4.0)          # 4 yalancı kanıt sonrası deneyim ≈ yarı yarıya ağırlık
        return _clip01((1.0 - w_learned) * self.prior_for(source_id) + w_learned * learned)

    def reward(self, source_id: str, good: bool, strength: float = 1.0):
        dom = self.domain_of(source_id)
        if good:
            self.beta[dom][0] += float(strength)
        else:
            self.beta[dom][1] += float(strength)
        self.evidence_counts[dom] += 1
        self.last_seen[dom] = _now()

    def snapshot(self, limit: int = 10) -> dict:
        items = sorted(self.evidence_counts.items(), key=lambda kv: -kv[1])[:limit]
        return {d: {"reliability": round(self.reliability(d), 3), "kanıt": n} for d, n in items}


# ============================================================================
# 2. İNANÇ HÜCRELERİ (Beta kesinlik + kaynak bağımsızlığı)
# ============================================================================

@dataclass
class Belief:
    """Tek bir önerme: özne + ilişki + nesne. α destek, β karşı-kanıt birikimidir."""
    key: str
    subject: str
    relation: str
    obj: str
    kind: str = "fact"                  # fact | event | scalar | alias
    alpha: float = 0.5                  # Jeffreys önseli: kanıt azken kararsız, kanıt geldikçe netleşir
    beta: float = 0.5
    sources: dict[str, float] = field(default_factory=dict)   # source_id -> toplam katkı
    evidence: list[dict] = field(default_factory=list)        # son 12 kanıt izi
    contradictions: list[dict] = field(default_factory=list)
    use_count: int = 0
    last_used: float = 0.0
    created: float = field(default_factory=_now)
    automatic: bool = False
    value: float | None = None
    unit: str = ""

    @property
    def credence(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    @property
    def independent_sources(self) -> int:
        return len(self.sources)

    def add_support(self, source_id: str, reliability: float, raw: str = "", weight: float = 1.0):
        # Aynı kaynağın tekrarı bağımsız kanıt değildir: logaritmik azalan katkı.
        prev = self.sources.get(source_id, 0.0)
        gain = float(reliability) * float(weight) / (1.0 + prev)
        self.sources[source_id] = prev + float(reliability) * float(weight)
        self.alpha += gain
        self.evidence.append({"t": _now(), "s": source_id, "r": round(float(reliability), 3), "raw": raw[:120]})
        if len(self.evidence) > 12:
            self.evidence.pop(0)

    def add_counter(self, source_id: str, reliability: float, raw: str = ""):
        self.beta += float(reliability)
        self.contradictions.append({"t": _now(), "s": source_id, "r": round(float(reliability), 3), "raw": raw[:120]})
        if len(self.contradictions) > 12:
            self.contradictions.pop(0)

    def mark_conflict(self, other_key: str):
        for c in self.contradictions:
            if c.get("against") == other_key:
                return
        self.contradictions.append({"t": _now(), "against": other_key})

    def to_dict(self) -> dict:
        return {
            "key": self.key, "subject": self.subject, "relation": self.relation, "obj": self.obj,
            "kind": self.kind, "alpha": round(self.alpha, 4), "beta": round(self.beta, 4),
            "credence": round(self.credence, 3), "sources": dict(self.sources),
            "use_count": self.use_count, "automatic": self.automatic, "value": self.value, "unit": self.unit,
            "evidence": self.evidence[-4:], "contradictions": self.contradictions[-3:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Belief":
        b = cls(key=d["key"], subject=d["subject"], relation=d["relation"], obj=d["obj"], kind=d.get("kind", "fact"))
        b.alpha = float(d.get("alpha", 0.5))
        b.beta = float(d.get("beta", 0.5))
        b.sources = {k: float(v) for k, v in (d.get("sources") or {}).items()}
        b.use_count = int(d.get("use_count", 0))
        b.automatic = bool(d.get("automatic", False))
        b.value = d.get("value")
        b.unit = d.get("unit", "")
        b.evidence = list(d.get("evidence") or [])
        b.contradictions = list(d.get("contradictions") or [])
        return b


class BeliefStore:
    """O(1) sözlük erişimli inanç deposu + yuva indeksi (aynı yıl vb. yarışan adaylar)."""

    def __init__(self, max_beliefs: int = 60000):
        self.beliefs: dict[str, Belief] = {}
        self.slots: dict[str, list[str]] = defaultdict(list)   # "ORTAYA_ÇIKTI|2020" -> [key, ...]
        self.aliases: dict[str, str] = {}                      # norm(ad) -> temsilci görünen ad
        self._alias_groups: dict[str, set[str]] = {}           # temsilci -> norm üyeler (geçişli birleşme)
        self.canon_keys: dict[tuple, str] = {}                 # (kanonik ad, ilişki, nesne) -> belief.key
        self.max_beliefs = int(max_beliefs)

    @staticmethod
    def canon(text: str, morph: TurkishMorphology | None = None) -> str:
        t = tr_lower(text).strip()
        t = re.sub(r"[.!?,;:'\"“”«»()\[\]]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if morph is not None and t:
            words = t.split()
            if len(words) <= 5:
                t = " ".join(morph.analyze_word(w)["stem"] for w in words)
        return t

    def key(self, subject: str, relation: str, obj: str) -> str:
        return f"{subject}|{relation}|{obj}"

    def get(self, key: str) -> Belief | None:
        return self.beliefs.get(key)

    def canonical_subject(self, subj: str) -> str:
        return self.aliases.get(_alias_norm(subj), subj)

    def upsert(self, subject: str, relation: str, obj: str, source_id: str, reliability: float,
               kind: str = "fact", raw: str = "", value: float | None = None, unit: str = "",
               slot_key: str | None = None) -> Belief:
        s = self.canonical_subject(subject)
        ck = (s, relation, obj)
        k = self.canon_keys.get(ck) or self.key(s, relation, obj)
        b = self.beliefs.get(k)
        if b is None:
            if len(self.beliefs) >= self.max_beliefs:
                # En az kullanılan ve en düşük kesinlikli inancı unut (bellek dengesi)
                victim = min(self.beliefs.values(), key=lambda x: (x.credence * (1 + x.use_count), x.last_used))
                self.beliefs.pop(victim.key, None)
            b = Belief(key=k, subject=s, relation=relation, obj=obj, kind=kind, value=value, unit=unit)
            self.canon_keys[ck] = k
            self.beliefs[k] = b
        b.add_support(source_id, reliability, raw=raw)
        if slot_key:
            if k not in self.slots[slot_key]:
                self.slots[slot_key].append(k)
            if len(self.slots[slot_key]) > 12:
                self.slots[slot_key].pop(0)
        return b

    def add_alias(self, a: str, b: str, source_id: str, reliability: float, raw: str = "") -> Belief:
        a_n, b_n = _alias_norm(a), _alias_norm(b)
        x, y = sorted([a_n, b_n])
        key = f"ALIAS|{x}|{y}"
        bel = self.beliefs.get(key)
        if bel is None:
            bel = Belief(key=key, subject=a.strip(), relation="ALIAS", obj=b.strip(), kind="alias")
            self.beliefs[key] = bel
        bel.add_support(source_id, reliability, raw=raw)
        if bel.credence >= 0.80 and bel.independent_sources >= 2:
            self._merge_alias(a.strip(), b.strip())
        return bel

    def _merge_alias(self, a: str, b: str):
        """İki adı geçişli olarak birleştirir: covid19≈koronavirüs ve covid19≈korona ise üçü de aynı."""
        na, nb = _alias_norm(a), _alias_norm(b)
        ra = self.aliases.get(na, a)
        rb = self.aliases.get(nb, b)
        rep = ra if ra != a or a <= b else rb
        group = set(self._alias_groups.pop(_alias_norm(ra), {na}))
        group |= set(self._alias_groups.pop(_alias_norm(rb), {nb}))
        group |= {na, nb}
        rep_n = _alias_norm(rep)
        self._alias_groups[rep_n] = group
        for member in group:
            self.aliases[member] = rep
        # Aynı ada çıkan inançların kanonik anahtarlarını birleştir
        for ck, kk in list(self.canon_keys.items()):
            b = self.beliefs.get(kk)
            if b is None:
                continue
            self.canon_keys[(self.canonical_subject(b.subject), b.relation, b.obj)] = kk

    def merge_into(self, keep: Belief, drop: Belief):
        """Aynı önermenin iki kaydını tek inançta birleştirir: kanıt toplanır, kaynak iki kez sayılmaz."""
        for sid, contrib in drop.sources.items():
            keep.sources[sid] = max(keep.sources.get(sid, 0.0), float(contrib))
        keep.alpha += max(0.0, drop.alpha - 0.5)
        keep.beta += max(0.0, drop.beta - 0.5)
        keep.evidence = (keep.evidence + list(drop.evidence))[-12:]
        keep.contradictions = (keep.contradictions + list(drop.contradictions))[-12:]
        keep.use_count += drop.use_count
        keep.automatic = keep.automatic or drop.automatic
        for slot in self.slots.values():
            while drop.key in slot:
                slot.remove(drop.key)
        self.beliefs.pop(drop.key, None)
        for ck, kk in list(self.canon_keys.items()):
            if kk == drop.key:
                self.canon_keys[ck] = keep.key

    def slot_candidates(self, slot_key: str) -> list[Belief]:
        return [self.beliefs[k] for k in self.slots.get(slot_key, []) if k in self.beliefs]

    def find_by_subject(self, subj: str, relation: str | None = None) -> list[Belief]:
        s = self.canonical_subject(subj)
        out = []
        for b in self.beliefs.values():
            if b.kind == "alias":
                continue
            if b.subject != s:
                continue
            if relation is not None and b.relation != relation:
                continue
            out.append(b)
        out.sort(key=lambda x: (-x.credence, -x.use_count))
        return out

    def to_dict(self) -> dict:
        return {
            "beliefs": [b.to_dict() for b in list(self.beliefs.values())[-1200:]],
            "slots": {k: v[-6:] for k, v in list(self.slots.items())[-260:]},
            "aliases": dict(list(self.aliases.items())[-260:]),
        }

    def load_dict(self, d: dict):
        for bd in d.get("beliefs", []):
            try:
                b = Belief.from_dict(bd)
                self.beliefs[b.key] = b
            except Exception:
                continue
        for k, v in (d.get("slots") or {}).items():
            self.slots[k] = list(v)
        self.aliases = dict(d.get("aliases") or {})
        # Eş ad gruplarını yeniden kur (geçişli birleşme kaybolmasın)
        self._alias_groups = {}
        for member, rep in self.aliases.items():
            self._alias_groups.setdefault(_alias_norm(rep), set()).add(_alias_norm(member))
            self._alias_groups[_alias_norm(rep)].add(_alias_norm(rep))
        self.canon_keys = {}
        for k, b in self.beliefs.items():
            self.canon_keys[(self.canonical_subject(b.subject), b.relation, b.obj)] = k


# ============================================================================
# 3. İDDİA ÇÖZÜMLEYİCİ (Türkçe cümleden yapısal önerme)
# ============================================================================

@dataclass
class ParsedClaim:
    kind: str                 # event | alias | scalar | fact
    subject: str
    relation: str
    obj: str = ""
    year: int | None = None
    value: float | None = None
    unit: str = ""
    attributes: set[str] = field(default_factory=set)
    confidence: float = 0.8
    slot_key: str = ""


class ClaimParser:
    """Türkçe yüzey biçiminden (hal/çatı/kip çözümlemesi + morfoloji) yapısal iddia çıkarır."""

    YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
    SCALAR_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(derece|°c|santigrat|metre|kilometre|kg|kilogram|gram|saniye|yıl|yaşında|kat|kez|oran)?")

    EVENT_VERBS = {
        "çık", "ortaya çık", "başla", "patla", "yayıl", "görül", "keşfet", "icat et", "kurul", "açıl",
        "bul", "doğ", "öl", "inşa et", "üret", "tanıt", "duyur", "ilan et", "yayımla", "yayımlan",
    }
    ATTRIBUTE_WORDS = {
        "virüs", "hastalık", "salgın", "bakteri", "gezegen", "yıldız", "element", "metal", "yazılım",
        "dil", "protokol", "uydu", "kıta", "savaş", "devrim", "keşif", "tür", "canlı", "organ",
    }

    def __init__(self, morph: TurkishMorphology):
        self.morph = morph

    def _head_noun(self, phrase: str) -> str:
        words = _tokens(phrase)
        if not words:
            return ""
        return self.morph.analyze_word(words[-1])["stem"]

    def parse(self, text: str) -> ParsedClaim | None:
        raw = text.strip()                      # tr_lower Türkçe karakterlerde uzunluğu korur:
        s = tr_lower(raw).rstrip(".!?")         # bu yüzden indeksler orijinal metinle hizalıdır

        words = _tokens(s)
        if len(words) < 3:
            return None

        # 1) Eş adlandırma / çeviri iddiası: "X'in Türkçesi Y'dir", "X (Y)", "X diye de anılır"
        #    (indeksler korunur -> 'COVID-19' gibi kısaltmalar küçültülüp bozulmaz)
        m_alias_tr = re.search(
            r"^(.+?)(?:'?[ıiuü]n|'?[ıiuü]n)?\s+(?:türkçesi|türkçe adı|diğer adı|ingilizcesi|kısaltması)\s+(.+)$", s)
        if m_alias_tr:
            a = raw[m_alias_tr.start(1):m_alias_tr.end(1)].strip(" '\"")
            b = self._strip_copula(raw[m_alias_tr.start(2):m_alias_tr.end(2)].strip(" '\"."))
            return ParsedClaim(kind="alias", subject=a, relation="ALIAS", obj=b, confidence=0.9)
        m_alias_par = re.search(r"^([\wçğıöşü\-\. ]+?)\s*\(([\wçğıöşü\-\. ]+?)\)", s)
        if m_alias_par and 1 <= len(m_alias_par.group(1).split()) <= 5:
            a = raw[m_alias_par.start(1):m_alias_par.end(1)].strip()
            b = raw[m_alias_par.start(2):m_alias_par.end(2)].strip()
            return ParsedClaim(kind="alias", subject=a, relation="ALIAS", obj=b, confidence=0.85)
        m_alias_diye = re.search(r"^(.+?)\s+(?:diye de|olarak da)\s+(?:anılır|bilinir|adlandırılır)$", s)
        if m_alias_diye:
            return ParsedClaim(kind="alias", subject="", relation="ALIAS",
                               obj=raw[m_alias_diye.start(1):m_alias_diye.end(1)], confidence=0.6)
        m_same = re.search(r"^(.+?)\s+(?:ile|ve)\s+(.+?)\s+(?:aynı|eşdeğer|eş anlamlı)", s)
        if m_same:
            a = raw[m_same.start(1):m_same.end(1)].strip()
            b = raw[m_same.start(2):m_same.end(2)].strip()
            return ParsedClaim(kind="alias", subject=a, relation="ALIAS", obj=b, confidence=0.9)

        # 2) Yıllı olay: "X 2020'de çıktı"
        year_m = self.YEAR_RE.search(s)
        last_an = self.morph.analyze_word(words[-1])
        pred = last_an["stem"] if last_an.get("is_verb") else words[-1]
        pred_norm = pred
        is_event = any(pred_norm == v or pred_norm.startswith(v) for v in self.EVENT_VERBS) or any(
            k in s for k in ("ortaya çıktı", "yayıldı", "patladı", "başladı", "keşfedildi", "icat edildi", "kuruldu")
        )
        if year_m and is_event:
            year = int(year_m.group(1))
            head_part = raw[:year_m.start()].strip(" ,'\"")
            head_part = re.sub(r"\b(yılında|yıl|senesinde|senesi)\b", "", head_part).strip(" ,'\"")
            entity = self._clean_entity(head_part)
            if entity:
                attrs = {w for w in _tokens(head_part) if w in self.ATTRIBUTE_WORDS}
                attrs |= {self._head_noun(head_part)}
                return ParsedClaim(
                    kind="event", subject=entity, relation="ORTAYA_CIKTI", obj=str(year), year=year,
                    attributes=attrs, confidence=0.9, slot_key=f"ORTAYA_CIKTI|{year}",
                )

        # 3) Ölçülü olgu: "Su 100 derecede kaynar", "Işık saniyede 300000 kilometre gider"
        m_sc = re.search(r"^(.+?)\s+(-?\d+(?:[.,]\d+)?)\s*(derece|°c|santigrat|metre|kilometre|kg|kilogram|gram|saniye|kat|kez)?\s*(?:de|da|te|ta)?\s*(.+)$", s)
        if m_sc and len(m_sc.group(1).split()) <= 5:
            subj = self._clean_entity(raw[:m_sc.start(1)].rstrip() if m_sc.start(1) > 0 else raw[:len(raw) - len(raw.lstrip())] + m_sc.group(1))
            val = float(m_sc.group(2).replace(",", "."))
            unit = (m_sc.group(3) or "").strip()
            verb = self.morph.analyze_word(_tokens(m_sc.group(4))[-1] if _tokens(m_sc.group(4)) else "")["stem"]
            if subj and verb:
                return ParsedClaim(
                    kind="scalar", subject=subj, relation=verb.upper(), obj=f"{m_sc.group(2)} {unit}".strip(),
                    value=val, unit=unit, confidence=0.86,
                )

        # 4) Genel tanım/ilişki: "X bir Y'dir", "X Y'ye yol açar", "X Y'yi üretir" -> kavram grafiğine bırakılır
        if any(w in words for w in ("bir", "yol", "neden", "sebep", "üretir", "sağlar", "oluşur", "tanımlanır")):
            return ParsedClaim(kind="fact", subject=self._clean_entity(words[0]), relation="BILDIRIM", obj=s, confidence=0.7)
        return None

    KEEP_SURFACE = re.compile(r"[A-ZÇĞİÖŞÜ]{2,}|\d")

    def _strip_copula(self, phrase: str) -> str:
        """Sondaki koşaçı ('koronadır', 'virüstür') soyar; kısaltma/özel adı korur."""
        words = phrase.split()
        if not words:
            return phrase
        last = words[-1]
        an = self.morph.analyze_word(last)
        if an.get("copula") and an.get("stem"):
            words[-1] = an["stem"]
        return " ".join(words).strip()

    def _clean_entity(self, phrase: str) -> str:
        """Varlık adını köklerine indirgerken KISALTMALARI/özel adları korur:
        'COVID-19' -> 'COVİD-19' değil 'COVID-19'. Bu, eş adlandırma çıkarımını bozuyordu."""
        surf = re.findall(r"[\wçğıöşüÇĞİÖŞÜ\-]+", phrase)
        drop = {"bir", "bu", "şu", "o", "yeni", "ilk", "büyük", "küçük", "adlı", "adında", "yılında", "senesinde"}
        out: list[str] = []
        for w in surf:
            if tr_lower(w) in drop:
                continue
            if self.KEEP_SURFACE.search(w):
                out.append(w)                                  # kısaltma/rakam: aynen koru
            elif w[:1].isupper():
                out.append(w)                                  # özel ad: aynen koru
            else:
                out.append(self.morph.analyze_word(w)["stem"])
        return " ".join(out).strip()


# ============================================================================
# 4. HİPOTEZ YARIŞMASI + SYSTEM-1 İNDEKSİ + HALÜSİNASYON BEKÇİSİ
# ============================================================================

@dataclass
class Hypothesis:
    name: str
    description: str
    alpha: float = 1.0
    beta: float = 1.0
    evidence: list[str] = field(default_factory=list)

    @property
    def credence(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def support(self, w: float, why: str = ""):
        self.alpha += float(w)
        if why:
            self.evidence.append(why)
            if len(self.evidence) > 8:
                self.evidence.pop(0)

    def oppose(self, w: float, why: str = ""):
        self.beta += float(w)
        if why:
            self.evidence.append(why)
            if len(self.evidence) > 8:
                self.evidence.pop(0)


@dataclass
class AmbiguityCase:
    slot_key: str
    entities: list[str]
    hypotheses: list[Hypothesis]
    created: float = field(default_factory=_now)
    resolved: bool = False

    def winner(self) -> Hypothesis:
        return max(self.hypotheses, key=lambda h: h.credence)


class System1Index:
    """Sık kullanılan ve yüksek kesinlikli inançların otomatik (bilinçsiz) erişim indeksi.
    İnsan beynindeki 'kristalleşmiş bilgi' gibi: artık her seferinde kanıt tartılmaz."""

    AUTO_CREDENCE = 0.80
    AUTO_USES = 2

    def __init__(self):
        self.index: dict[str, str] = {}       # sorgu anahtarı -> belief.key
        self.hits = 0
        self.misses = 0

    def maybe_promote(self, b: Belief) -> bool:
        if b.automatic:
            return True
        if b.credence >= self.AUTO_CREDENCE and (b.use_count >= self.AUTO_USES or b.independent_sources >= 2):
            b.automatic = True
            return True
        return False

    def demote(self, b: Belief):
        if b.automatic:
            b.automatic = False

    def register(self, query_key: str, belief_key: str):
        self.index[query_key] = belief_key
        if len(self.index) > 20000:
            for k in list(self.index.keys())[:4000]:
                self.index.pop(k, None)

    def lookup(self, query_key: str) -> str | None:
        k = self.index.get(query_key)
        if k:
            self.hits += 1
        else:
            self.misses += 1
        return k


@dataclass(frozen=True)
class TechnicalQuery:
    """Yüzeysel kelimelerden bağımsız teknik niyet ve kaynak yükümlülüğü."""
    kind: str
    raw: str
    features: frozenset[str]
    known_parts: tuple[str, ...] = ()
    reference: str = ""
    year: str = ""
    wants_name: bool = False
    wants_idea: bool = False
    negated: frozenset[str] = frozenset()


class TechnicalInquiry:
    """Birleşik terimleri çözer; teorem koşulu ile bibliyografik kanıtı ayırır.

    Genel kavram sözlüğü cevap değildir. Bir makalenin algoritması ancak doğrulanmış
    belgenin açık algoritma/temel-fikir alanlarından okunur; başlıktan ad türetilmez.
    """

    def __init__(self, engine: "EpistemicEngine"):
        self.engine = engine
        path = Path(__file__).resolve().parent / "data" / "technical_epistemics.json"
        with path.open(encoding="utf-8") as fh:
            self.data = json.load(fh)
        self.documents: list[dict] = []
        self.last: dict = {}

    def _tokens(self, text: str) -> list[str]:
        low = tr_lower(text or "")
        for phrase, alias in self.data["aliases"].items():
            low = low.replace(phrase, alias)
        return re.findall(r"[a-zçğıöşü0-9]+(?:-[a-zçğıöşü0-9]+)?", low)

    @staticmethod
    def _negator(token: str) -> bool:
        return token in {"not", "non", "no", "değil", "olmayan", "yok"} or token.startswith(
            ("değil", "olmayan", "olmadı", "olmaz"))

    def _features(self, tokens: list[str]) -> tuple[frozenset[str], tuple[str, ...], frozenset[str]]:
        features: set[str] = set()
        negated: set[str] = set()
        parts: list[str] = []
        lexicon = self.data["lexicon"]
        for i, token in enumerate(tokens):
            try:
                stem = self.engine.morph.analyze_word(token)["stem"]
            except Exception:
                stem = token
            entry = lexicon.get(token) or lexicon.get(stem)
            if not entry:
                continue
            prev = tokens[i - 1] if i else ""
            prev2 = tokens[i - 2] if i > 1 else ""
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            nxt2 = tokens[i + 2] if i + 2 < len(tokens) else ""
            pre_neg = self._negator(prev) or (prev in {"a", "an", "the", "bir"} and self._negator(prev2))
            # 'deterministik olmayan tam' yapısında olmayan önceki özelliğe aittir;
            # cümle başındaki 'olmayan deterministik' ise ön-olumsuzluk olarak kalır.
            if prev.startswith(("değil", "olmayan", "olmadı", "olmaz")) and prev2 in lexicon:
                pre_neg = False
            post_neg = self._negator(nxt) or (nxt in {"is", "are", "was", "were", "be"} and self._negator(nxt2))
            feature = entry["feature"]
            features.add(feature)
            if pre_neg or post_neg:
                negated.add(feature)
            if entry["gloss"] not in parts:
                parts.append(entry["gloss"])
        return frozenset(features), tuple(parts), frozenset(negated)

    def parse(self, raw: str) -> TechnicalQuery | None:
        if not self.engine.is_question(raw):
            return None
        low = tr_lower(raw)
        tokens = self._tokens(raw)
        features, parts, negated = self._features(tokens)
        quoted = re.search(r'[“"]([^”"]+)[”"]', raw)
        year = re.search(r"\b(?:19|20)\d{2}\b", raw)
        asks_source = bool(re.search(r"algorit|algorithm|temel\s+fikir|basic\s+idea|core\s+idea|kay[ıi]t|makale|paper", low))
        if quoted and year and asks_source:
            wants_name = bool(re.search(r"algorit\w*\s+ad|\bad[ıi]\b|\bname\b", low))
            wants_idea = bool(re.search(r"temel\s+fikir|fikr[iı]|basic\s+idea|core\s+idea|ne\s+yapar", low))
            if not wants_name and not wants_idea:
                wants_name = wants_idea = True
            return TechnicalQuery(
                "bibliographic", raw, features, parts,
                reference=quoted.group(1), year=year.group(0),
                wants_name=wants_name, wants_idea=wants_idea,
            )
        if year and asks_source and features.intersection({"fst", "recurrent", "weighted", "prefix_incompatibility"}):
            return TechnicalQuery(
                "bibliographic", raw, features, parts, reference=raw, year=year.group(0),
                wants_name=bool(re.search(r"\bad[ıi]\b|\bname\b|algoritma\s+ne", low)),
                wants_idea=bool(re.search(r"temel\s+fikir|fikr[iı]|basic\s+idea|core\s+idea", low)),
            )
        if "unique" in features and "minimal" in features and features.intersection(
                {"fst", "recurrent", "strongly_connected", "mealy"}):
            return TechnicalQuery("uniqueness", raw, features, parts, negated=negated)
        asks_definition = bool(re.search(r"nedir|ne\s+demek|what\s+is|aç[ıi]kla|tan[ıi]m", low))
        if asks_definition and len(parts) >= 2 and features.intersection(
                {"fst", "recurrent", "strongly_connected", "prefix_incompatibility"}):
            return TechnicalQuery("compound", raw, features, parts)
        return None

    def register_document(self, source_id: str, text: str, metadata: dict,
                          verified: bool = False) -> dict:
        """Künyeyi değil, belgedeki açık algoritma ve fikir kanıtını kaydet."""
        algorithm = re.search(r"^\s*(?:algorithm(?:\s+name)?|algoritma)\s*:\s*(.+)$",
                              text or "", re.I | re.M)
        idea = re.search(r"^\s*(?:basic\s+idea|core\s+idea|temel\s+fikir(?:i)?)\s*:\s*(.+)$",
                         text or "", re.I | re.M)
        title = str(metadata.get("title") or "").strip()
        if not verified or not title or algorithm is None or idea is None:
            return {"status": "unverified", "source_id": source_id}
        doc = {
            "source_id": str(source_id), "title": title,
            "year": str(metadata.get("year") or ""),
            "venue": str(metadata.get("venue") or ""),
            "authors": tuple(str(a) for a in metadata.get("authors") or ()),
            "algorithm": algorithm.group(1).strip(), "idea": idea.group(1).strip(),
            "verified": True,
        }
        self.documents = [d for d in self.documents if d["source_id"] != doc["source_id"]]
        self.documents.append(doc)
        return {"status": "verified", "source_id": doc["source_id"]}

    def _title_tokens(self, text: str) -> set[str]:
        stop = {"the", "and", "for", "with", "via", "bir", "için", "algoritma", "algorithm"}
        return {t for t in self._tokens(text) if len(t) >= 3 and t not in stop}

    def match_document(self, query: TechnicalQuery) -> dict | None:
        if query.kind != "bibliographic":
            return None
        asked = self._title_tokens(query.reference)
        matches = []
        for doc in self.documents:
            if not doc.get("verified") or (doc["year"] and doc["year"] != query.year):
                continue
            title = self._title_tokens(doc["title"])
            if not title:
                continue
            coverage = len(title & asked) / len(title)
            if coverage >= 0.80:
                matches.append((coverage, doc))
        if not matches:
            return None
        matches.sort(key=lambda row: row[0], reverse=True)
        return matches[0][1]

    def _unknown_source(self, query: TechnicalQuery) -> str:
        speech = self.data["speech"]
        if query.wants_name and not query.wants_idea:
            detail = speech["algorithm_name_object"]
        elif query.wants_idea and not query.wants_name:
            detail = speech["basic_idea_object"]
        else:
            detail = speech["algorithm_details"]
        first = self.engine.syn.realize_verb(VerbSpec(
            verb_root="doğrula", obj=speech["bibliographic_object"],
            neg=True, tense="PAST", person="1SG",
        ))
        second = self.engine.syn.realize_verb(VerbSpec(
            verb_root="bil", obj=detail, neg=True, person="1SG",
        ))
        return tr_capitalize(first) + "; " + second + "."

    def _theorem_sentence(self) -> str:
        theorem = self.data["theorem"]
        core = self.engine.syn.realize_pred(PredSpec(
            pred_root="tek", subj=NP(theorem["subject"]),
            advs=(theorem["comparison"],),
        ))
        return theorem["condition"] + " " + core

    def _uniqueness_answer(self, query: TechnicalQuery) -> str:
        theorem = self.data["theorem"]
        required = set(theorem["required_features"])
        affirmed = set(query.features).difference(query.negated)
        guaranteed = required.issubset(affirmed)
        negative_question = "unique" in query.negated
        if guaranteed:
            prefix = "Hayır" if negative_question else "Evet"
            return prefix + "; ancak " + self._theorem_sentence() + "."
        if "strongly_connected" in affirmed and "recurrent" not in affirmed:
            subject = "güçlü bağlılık"
        elif "recurrent" in affirmed:
            subject = theorem["recurrent_subject"]
        else:
            subject = "minimallik"
        reason = self.engine.syn.realize_verb(VerbSpec(
            verb_root="sağla", subj=NP(subject), obj=theorem["missing_premise_object"],
            advs=(theorem["observation_scope"],), neg=True, person="3SG",
        ))
        prefix = "Evet" if negative_question else "Hayır"
        return prefix + "; " + reason + "; ancak " + self._theorem_sentence() + "."

    def answer(self, raw: str, audit: bool = False) -> str | None:
        query = self.parse(raw)
        if query is None:
            return None
        evidence = None
        if query.kind == "bibliographic":
            evidence = self.match_document(query)
            if evidence is None:
                reply = self._unknown_source(query)
            else:
                units = []
                if query.wants_name or not query.wants_idea:
                    units.append(self.data["speech"]["algorithm_name"] + " " + evidence["algorithm"])
                if query.wants_idea or not query.wants_name:
                    units.append(self.data["speech"]["basic_idea"] + " " + evidence["idea"])
                reply = tr_capitalize("; ".join(units)) + " (" + self.data["speech"]["source_label"] + ": " + evidence["source_id"] + ")."
        elif query.kind == "uniqueness":
            reply = self._uniqueness_answer(query)
        else:
            first = self.engine.syn.realize_verb(VerbSpec(
                verb_root="bil", obj=self.data["speech"]["compound_definition"],
                neg=True, person="1SG",
            ))
            reply = (tr_capitalize(first) + "; " + self.data["speech"]["known_parts"]
                     + ": " + ", ".join(query.known_parts) + ".")
        if not audit:
            self.last = {"query": query, "reply": reply, "evidence": evidence}
        return reply

    @staticmethod
    def _is_refusal(text: str) -> bool:
        markers = ("bilmiyorum", "doğrulamadım", "doğrulayamıyorum", "kanıt yok")
        clauses = [c for c in re.split(r"[.!?;]", tr_lower(text)) if c.strip()]
        return bool(clauses) and all(any(m in c for m in markers) for c in clauses)

    def audit_response(self, raw: str, response: str) -> dict:
        query = self.parse(raw)
        if query is None:
            return {"safe": True, "path": "not_technical"}
        expected = self.answer(raw, audit=True)
        safe = bool(expected and (response.strip() == expected.strip() or self._is_refusal(response)))
        return {
            "safe": safe, "path": "technical_claims", "kind": query.kind,
            "verified_source": bool(query.kind == "bibliographic" and self.match_document(query)),
        }


class HallucinationGuard:
    """Bir sözcede geçen içerik kavramlarının bellekte kanıtı olup olmadığını ölçer.
    Kanıtsız oran yüksekse beyan 'sanıyorum' diye işaretlenir veya soruya dönüştürülür."""

    def __init__(self, engine: "EpistemicEngine"):
        self.engine = engine
        self.audits = 0
        self.caught = 0

    def audit_response(self, query: str, response: str) -> dict:
        """Teknik özgül iddiayı sözcük tanımadan ayrı, kaynak kanıtıyla denetle."""
        self.audits += 1
        result = self.engine.technical.audit_response(query, response)
        if not result.get("safe"):
            self.caught += 1
        return result

    def audit(self, text: str, min_grounded: float = 0.5) -> dict:
        self.audits += 1
        content = [w for w in _tokens(text) if len(w) >= 4]
        if not content:
            return {"grounded_ratio": 1.0, "unknown": [], "safe": True}
        known, unknown = [], []
        for w in content[:14]:
            stem = self.engine.morph.analyze_word(w)["stem"]
            has_belief = bool(self.engine.store.find_by_subject(stem))
            has_legacy = bool(self.engine.graph_lookup(stem))
            (known if (has_belief or has_legacy) else unknown).append(w)
        ratio = len(known) / max(1, len(known) + len(unknown))
        safe = ratio >= min_grounded
        if not safe:
            self.caught += 1
        return {"grounded_ratio": round(ratio, 3), "unknown": unknown[:6], "safe": safe}


# ============================================================================
# 5. EPİSTEMİK MOTOR (cephe)
# ============================================================================

class EpistemicEngine:
    """Nora'nın inanç yaşamı: iddiayı çözümle -> çelişkiyi fark et -> hipotez kur -> kanıtla
    kesinliği güncelle -> olgunlaşanı System-1'e indir -> sorguda anında yanıtla."""

    def __init__(self, morph: TurkishMorphology, graph=None, schema_cortex=None, chem=None):
        self.morph = morph
        self.graph = graph
        self.schema_cortex = schema_cortex
        self.chem = chem
        self.parser = ClaimParser(morph)
        self.store = BeliefStore()
        self.trust = SourceTrustModel()
        self.system1 = System1Index()
        self.ambiguities: list[AmbiguityCase] = []
        self.technical = TechnicalInquiry(self)
        self.guard = HallucinationGuard(self)
        self.syn = TurkishSyntaxEngine
        self.speech = BeliefSpeech(self)
        self.stats = {
            "claims_seen": 0, "claims_accepted": 0, "ambiguities": 0, "contradictions": 0,
            "system1_promotions": 0, "demotions": 0, "aliases_merged": 0,
        }
        self._last_reply: dict = {}

    # ---------------------------------------------------------------- yardımcılar
    def graph_lookup(self, stem: str) -> bool:
        if self.graph is None:
            return False
        return bool(getattr(self.graph, "_out_edges", {}).get(stem))

    def _clause(self, verb: str, subj: str | None = None, obj: str | None = None,
                dat: str | None = None, advs: tuple[str, ...] = (), tense: str = "PRES", person: str = "1SG",
                obj_plural: bool = False, subj_poss: int = 0, obj_poss: int = 0) -> str:
        return self.syn.realize_verb(VerbSpec(
            verb_root=verb,
            subj=NP(subj, poss=subj_poss) if subj else None,
            obj=NP(obj, case="ACC", plural=obj_plural, poss=obj_poss) if obj else None,
            dat=NP(dat, case="DAT") if dat else None,
            advs=advs, tense=tense, person=person,
        ))

    def _pred(self, pred: str, subj: str | None = None, poss: int = 0, advs: tuple[str, ...] = (),
              person: str = "3SG", neg: bool = False) -> str:
        return self.syn.realize_pred(PredSpec(
            pred_root=pred, subj=NP(subj, poss=poss) if subj else None, advs=advs, person=person, neg=neg,
        ))

    def _pct(self, x: float) -> str:
        return f"{x:.2f}"

    # ---------------------------------------------------------------- iddia alma
    ABSORB_SKIP = ("ben", "benim", "bugün", "dün", "yarın", "sana", "seni", "senin", "bana", "az önce", "peki", "evet", "hayır")

    Q_WORDS = ("kaç", "nedir", "ne", "hangi", "kim", "nerede", "nasıl", "mi", "mı", "mu", "mü",
               "neden", "niçin", "nerede", "nereli", "kaçıncı")
    Q_PHRASES = ("ne zaman", "ne kadar", "hangi yıl", "kaç derece", "kimdir", "nerededir", "midir", "mıdır")

    def is_question(self, text: str) -> bool:
        """Soru mu, bildirim mi? Noktalama olmadan da çalışır (Türkçe soru sözcükleri/eki)."""
        low = tr_lower(text)
        if "?" in text:
            return True
        if any(ph in low for ph in self.Q_PHRASES):
            return True
        words = re.findall(r"[\wçğıöşü]+", low)
        for w in words:
            if w in self.Q_WORDS:
                return True
            # 'hangisini', 'nerededir', 'midir' gibi ekli biçimler
            if len(w) > 3 and any(w.startswith(q) for q in ("hangi", "nere", "nasıl", "nedir", "kaç")):
                return True
            if w.endswith(("mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür")) and len(w) > 4:
                return True
        return False

    def absorb(self, text: str, source_id: str = "insan:bilinmeyen", speaker_trust: float = 0.6) -> dict:
        """Yeni iddiayı, KABUL ETMEDEN ÖNCE mevcut inançlarla karşılaştırır."""
        self.stats["claims_seen"] += 1
        low = tr_lower(text).strip()
        if any(low.startswith(k + " ") or low == k for k in self.ABSORB_SKIP):
            return {"status": "skipped", "reply": None}
        if self.is_question(text):
            # Soru bildirim gibi kaydedilmez: önce var olan kanıtla yanıtlanır.
            return {"status": "skipped", "reply": None, "reason": "soru"}
        claim = self.parser.parse(text)
        if claim is None:
            return {"status": "unparsed", "reply": None}

        rel = self.trust.reliability(source_id)
        if self.trust.domain_of(source_id) == "insan":
            rel = _clip01(0.5 * rel + 0.5 * float(speaker_trust))

        if claim.kind == "alias":
            return self._absorb_alias(claim, text, source_id, rel)

        if claim.kind == "event":
            return self._absorb_event(claim, text, source_id, rel)

        if claim.kind == "scalar":
            return self._absorb_scalar(claim, text, source_id, rel)

        # genel bildirim: inanç olarak kaydet, ama düşük ağırlıkla ve kanıt iziyle
        b = self.store.upsert(claim.subject, "BILDIRIM", tr_lower(text)[:90], source_id, rel * 0.7, raw=text)
        self.stats["claims_accepted"] += 1
        if self.chem is not None and hasattr(self.chem, "pulse_learning"):
            try:
                self.chem.pulse_learning(novelty=0.4, valence=0.1 * rel)
            except Exception:
                pass
        self._maybe_promote(b)
        return {
            "status": "stored", "kind": "fact", "belief": b, "credence": round(b.credence, 3),
            "single_source": b.independent_sources < 2, "reliability": round(rel, 3),
        }

    def _absorb_alias(self, claim: ParsedClaim, text: str, source_id: str, rel: float) -> dict:
        if not claim.subject or not claim.obj:
            return {"status": "unparsed", "reply": None}
        was_merged = _alias_norm(claim.subject) in self.store.aliases or _alias_norm(claim.obj) in self.store.aliases
        b = self.store.add_alias(claim.subject, claim.obj, source_id, rel, raw=text)
        self.stats["claims_accepted"] += 1
        merged = _alias_norm(claim.subject) in self.store.aliases or _alias_norm(claim.obj) in self.store.aliases
        unification = {"resolved": [], "merged_beliefs": 0}
        if merged and not was_merged:
            self.stats["aliases_merged"] += 1
            unification = self.unify_after_alias()
        return {
            "status": "alias", "kind": "alias", "belief": b, "credence": round(b.credence, 3),
            "merged": merged, "unification": unification,
            "single_source": b.independent_sources < 2, "reliability": round(rel, 3),
            "a": claim.subject, "b": claim.obj,
        }

    def _absorb_event(self, claim: ParsedClaim, text: str, source_id: str, rel: float) -> dict:
        slot = claim.slot_key
        prev = self.store.slot_candidates(slot)

        # Aynı yuvada ikinci bir AD göründüyse: bu bir ÇELİŞKİ DEĞİL, AÇIKLAMA EKSİKLİĞİDİR.
        # İnsan gibi iki hipotez kurulur ve kanıt beklenir (kör kabul veya kör ret yok).
        rivals = [b for b in prev if self._entity_compatible(b.subject, claim.subject) is False]
        alias_key = None
        if rivals:
            ambiguity = self._build_ambiguity(slot, claim, rivals, source_id, rel, text)
            alias_key = ambiguity.hypotheses[0].name
            # İddiayı yine de kaydet (düşük ağırlığı yok; kanıt izi şeffaf)
            b = self.store.upsert(claim.subject, claim.relation, claim.obj, source_id, rel, kind="event",
                                  raw=text, value=float(claim.year) if claim.year else None,
                                  unit="yıl", slot_key=slot)
            self.stats["claims_accepted"] += 1
            self.stats["ambiguities"] += 1
            return {
                "status": "ambiguous", "kind": "event", "belief": b, "credence": round(b.credence, 3),
                "ambiguity": ambiguity, "slot": slot, "rivals": [r.subject for r in rivals],
                "hypotheses": [
                    {"name": h.name, "desc": h.description, "credence": round(h.credence, 3)} for h in ambiguity.hypotheses
                ],
                "reliability": round(rel, 3),
            }

        b = self.store.upsert(claim.subject, claim.relation, claim.obj, source_id, rel, kind="event",
                              raw=text, value=float(claim.year) if claim.year else None, unit="yıl", slot_key=slot)
        self.stats["claims_accepted"] += 1
        promoted = self._maybe_promote(b)
        return {
            "status": "stored", "kind": "event", "belief": b, "credence": round(b.credence, 3),
            "single_source": b.independent_sources < 2, "reliability": round(rel, 3),
            "automatic": b.automatic, "promoted": promoted,
        }

    def _absorb_scalar(self, claim: ParsedClaim, text: str, source_id: str, rel: float) -> dict:
        existing = self.store.find_by_subject(claim.subject, claim.relation)
        conflict_belief = None
        for b in existing:
            if b.value is not None and claim.value is not None and abs(b.value - claim.value) > 1e-6:
                conflict_belief = b
                break

        if conflict_belief is not None:
            # Karşı-kanıt: güvenilirlik kadar β büyür. Güçlü kaynak ise System-1 otomatikliği kalkar!
            conflict_belief.add_counter(source_id, rel, raw=text)
            self.stats["contradictions"] += 1
            decisiveness = rel / max(1e-6, rel + 0.001)
            if rel >= 0.7 and conflict_belief.credence < 0.75:
                self.system1.demote(conflict_belief)
                self.stats["demotions"] += 1
            self.trust.reward(source_id, good=False, strength=0.6 * decisiveness)
            return {
                "status": "contradiction", "kind": "scalar", "belief": conflict_belief,
                "mine": {"value": conflict_belief.value, "unit": conflict_belief.unit,
                         "credence": round(conflict_belief.credence, 3)},
                "theirs": {"value": claim.value, "unit": claim.unit, "reliability": round(rel, 3),
                           "source": source_id},
                "automatic_revoked": not conflict_belief.automatic,
                "reliability": round(rel, 3),
            }

        b = self.store.upsert(claim.subject, claim.relation, claim.obj, source_id, rel, kind="scalar",
                              raw=text, value=claim.value, unit=claim.unit)
        self.stats["claims_accepted"] += 1
        promoted = self._maybe_promote(b)
        if promoted:
            self.system1.register(self._query_key(claim.subject, claim.relation), b.key)
        return {
            "status": "stored", "kind": "scalar", "belief": b, "credence": round(b.credence, 3),
            "single_source": b.independent_sources < 2, "reliability": round(rel, 3),
            "automatic": b.automatic, "promoted": promoted,
        }

    def _entity_compatible(self, a: str, b: str) -> bool | None:
        """İki ad aynı varlığı mı gösteriyor? Morfolojik/öznitelik örtüşmesi + alias tablosu.
        None -> belirsiz (yorum yapma), False -> büyük olasılıkla farklı ad, True -> yüksek olasılıkla aynı."""
        if a == b:
            return True
        if self.store.canonical_subject(a) == self.store.canonical_subject(b):
            return True
        ta, tb = _tokens_fold(a), _tokens_fold(b)
        if not ta or not tb:
            return None
        overlap = len(ta & tb) / max(1, len(ta | tb))
        if overlap >= 0.5:
            return True
        # "covid 19" <-> "korona virüs": ortak kök yok ama ikisi de aynı yuva ve aynı kategori
        return False if overlap == 0 else None

    def _build_ambiguity(self, slot: str, claim: ParsedClaim, rivals: list[Belief],
                         source_id: str, rel: float, raw: str) -> AmbiguityCase:
        for case in self.ambiguities:
            if case.slot_key == slot and not case.resolved:
                names = set(case.entities) | {claim.subject}
                case.entities = list(names)
                # Kanıt: yeni ad aynı kategoride ise H1 (aynı adın iki biçimi) güç kazanır
                h1 = next((h for h in case.hypotheses if h.name.startswith("H1")), None)
                h2 = next((h for h in case.hypotheses if h.name.startswith("H2")), None)
                if h1 is not None:
                    h1.support(0.55 * rel, f"{claim.subject} aynı yuvada ({slot}) göründü")
                if h2 is not None and h2.credence > 0.2:
                    h2.support(0.15 * rel, "yeni bağımsız ad")
                return case

        # Örtüşme ölçüsü: iki ad arasında paylaşılan kök/öznitelik ne kadar çoksa "tek varlık" önseli o kadar yüksek
        shared = 0.0
        other = rivals[0].subject
        ta, tb = set(_tokens(other)), set(_tokens(claim.subject))
        if ta and tb:
            shared = len(ta & tb) / max(1, len(ta | tb))
        # İnsan önseli: aynı yuvaya düşen iki ad, çoğu zaman aynı olayın iki adıdır; ama kesin değil.
        prior_alias = _clip01(0.50 + 0.40 * shared)
        h1 = Hypothesis(name=f"H1_AYNI_VARLIK|{slot}", description="aynı olayın iki farklı adı",
                        alpha=1.0 + 3.0 * prior_alias, beta=1.0 + 3.0 * (1.0 - prior_alias))
        h2 = Hypothesis(name=f"H2_AYRI_OLAYLAR|{slot}", description="aynı yıl en az iki ayrı olay",
                        alpha=1.0 + 3.0 * (1.0 - prior_alias) * 0.75, beta=1.0 + 3.0 * prior_alias * 0.75)
        h1.support(0.6 * rel, f"{claim.subject} ile {other} aynı yuvada")
        case = AmbiguityCase(slot_key=slot, entities=[other, claim.subject], hypotheses=[h1, h2])
        self.ambiguities.append(case)
        if len(self.ambiguities) > 40:
            self.ambiguities.pop(0)
        return case

    def _maybe_promote(self, b: Belief) -> bool:
        was = b.automatic
        promoted = self.system1.maybe_promote(b)
        if promoted and not was:
            self.stats["system1_promotions"] += 1
        return promoted

    def _mentions(self, name: str, qfold: set[str], qnorm: str) -> bool:
        """Sorguda bu varlık anılıyor mu? Kendi adı YA DA eş adlarından biri geçiyorsa evet."""
        if _tokens_fold(name).issubset(qfold):
            return True
        n = _alias_norm(name)
        if n and len(n) >= 3 and n in qnorm:
            return True
        rep_n = _alias_norm(self.store.canonical_subject(name))
        for m in self.store._alias_groups.get(rep_n, ()):
            if len(m) >= 3 and m in qnorm:
                return True
        return False

    def _query_key(self, subject: str, relation: str) -> str:
        return f"{self.store.canonical_subject(subject)}|{relation}"

    # ---------------------------------------------------------------- sorgulama
    def answer(self, text: str) -> dict | None:
        """Bellekten anlık (System-1) veya tartılmış (System-2) yanıt. Hiç kanıt yoksa None döner:
        uydurmak yok — 'bilmiyorum' demek de bir çıktıdır."""
        s = tr_lower(text).strip()
        words = _tokens(s)
        if len(words) < 2:
            return None
        if not self.is_question(text):
            return None

        # Ölçülü olgu sorgusu: "su kaç derecede kaynar"
        qfold = _tokens_fold(text)
        qnorm = _alias_norm(text)
        for b in list(self.store.beliefs.values()):
            if b.kind != "scalar" or b.value is None:
                continue
            if self._mentions(b.subject, qfold, qnorm) and (
                "kaç" in words or b.relation.lower() in " ".join(words)
            ):
                b.use_count += 1
                b.last_used = _now()
                promoted = self._maybe_promote(b)
                self.system1.register(self._query_key(b.subject, b.relation), b.key)
                return {
                    "status": "known", "path": "system1" if b.automatic else "system2",
                    "belief": b, "credence": round(b.credence, 3), "automatic": b.automatic,
                    "value": b.value, "unit": b.unit, "query_type": "scalar", "promoted": promoted,
                }

        # Varlık + yıl sorgusu: "covid 19 ne zaman çıktı"
        for b in list(self.store.beliefs.values()):
            if b.kind != "event":
                continue
            if self._mentions(b.subject, qfold, qnorm) and ("ne zaman" in s or "hangi yıl" in s or "kaç" in words or "çıktı" in words):
                b.use_count += 1
                b.last_used = _now()
                promoted = self._maybe_promote(b)
                self.system1.register(self._query_key(b.subject, b.relation), b.key)
                # Sorguda öteki ad kullanıldıysa iki adı birlikte söyle (insan gibi)
                other = ""
                for m in self.store._alias_groups.get(_alias_norm(self.store.canonical_subject(b.subject)), ()):
                    if m not in _alias_norm(text) and len(m) >= 3 and m in qnorm:
                        other = b.subject
                        break
                return {"status": "known", "path": "system1" if b.automatic else "system2",
                        "belief": b, "credence": round(b.credence, 3), "automatic": b.automatic,
                        "value": b.obj, "query_type": "event", "promoted": promoted}

        # Eş ad sorgusu: "covid 19'un türkçesi nedir" / "korona ile covid aynı mı"
        for b in list(self.store.beliefs.values()):
            if b.kind != "alias":
                continue
            pair = {b.subject, b.obj}
            na, nb = _alias_norm(b.subject), _alias_norm(b.obj)
            both = all(_tokens_fold(n).issubset(qfold) for n in pair)
            one = (na and na in qnorm) or (nb and nb in qnorm)
            asks = any(k in s for k in ("aynı", "türkçe", "adı", "nedir", "eş", "anlam"))
            if both or (one and asks):
                b.use_count += 1
                self._maybe_promote(b)
                return {"status": "known", "path": "system1" if b.automatic else "system2",
                        "belief": b, "credence": round(b.credence, 3), "automatic": b.automatic,
                        "query_type": "alias", "pair": sorted(pair)}

        # Açık hipotez sorgusu: "covid 19 ile korona virüs aynı mı?" — henüz kanıt yok, ama iki olasılık VAR
        for case in reversed(self.ambiguities):
            if case.resolved:
                continue
            if all(self._mentions(e, qfold, qnorm) for e in case.entities):
                return {"status": "ambiguous", "path": "system2", "query_type": "ambiguity",
                        "belief": None, "credence": round(case.winner().credence, 3),
                        "case": case, "entities": list(case.entities),
                        "hypotheses": [{"name": h.name, "desc": h.description, "credence": round(h.credence, 3)}
                                       for h in case.hypotheses]}
        return None

    def contradiction_report(self) -> dict | None:
        live = [c for c in self.ambiguities if not c.resolved]
        if not live:
            return None
        c = live[-1]
        return {
            "slot": c.slot_key, "entities": c.entities,
            "hypotheses": [{"name": h.name, "desc": h.description, "credence": round(h.credence, 3)} for h in c.hypotheses],
            "winner": c.winner().description,
        }

    def unify_after_alias(self) -> dict:
        """İki adın aynı varlığa çıktığı kanıtlandıysa: (1) o yuvadaki açık hipotezler kanıtla kapanır,
        (2) ayrı kayıtlara düşmüş aynı önermeler tek inançta birleşir, (3) güven tek yerden büyür."""
        resolved = []
        for case in list(self.ambiguities):
            if case.resolved:
                continue
            if len({self.store.canonical_subject(e) for e in case.entities}) == 1:
                r = self.resolve_ambiguity(case.slot_key, True, "iki bağımsız kaynak adları birleştirdi")
                if r:
                    resolved.append(r)

        merged_beliefs = 0
        groups: dict[tuple, list[Belief]] = defaultdict(list)
        for b in list(self.store.beliefs.values()):
            if b.kind == "alias":
                continue
            groups[(self.store.canonical_subject(b.subject), b.relation, b.obj)].append(b)
        for _, rows in groups.items():
            if len(rows) < 2:
                continue
            rows.sort(key=lambda x: (x.created, -x.alpha))
            keep, drops = rows[0], rows[1:]
            for drop in drops:
                self.store.merge_into(keep, drop)
                merged_beliefs += 1
            self._maybe_promote(keep)
            self.system1.register(self._query_key(keep.subject, keep.relation), keep.key)

        if resolved or merged_beliefs:
            self.stats["aliases_unified"] = self.stats.get("aliases_unified", 0) + 1
        return {"resolved": resolved, "merged_beliefs": merged_beliefs}

    def resolve_ambiguity(self, slot_prefix: str, alias: bool, why: str = "") -> dict | None:
        for c in reversed(self.ambiguities):
            if c.slot_key.startswith(slot_prefix) and not c.resolved:
                h1 = next((h for h in c.hypotheses if h.name.startswith("H1")), None)
                h2 = next((h for h in c.hypotheses if h.name.startswith("H2")), None)
                if alias:
                    if h1: h1.support(3.0, why or "kullanıcı/kanıt eşitliği doğruladı")
                    if h2: h2.oppose(2.0, why or "")
                    self.store._merge_alias(c.entities[0], c.entities[1]) if len(c.entities) > 1 else None
                    self.stats["aliases_merged"] += 1
                else:
                    if h2: h2.support(3.0, why or "kanıt ayrı olayları doğruladı")
                    if h1: h1.oppose(2.0, why or "")
                c.resolved = True
                return {"slot": c.slot_key, "alias": alias, "winner": c.winner().description,
                        "credence": round(c.winner().credence, 3)}
        return None

    # ---------------------------------------------------------------- kalıcılık
    def snapshot(self) -> dict:
        auto = sum(1 for b in self.store.beliefs.values() if b.automatic)
        return {
            "beliefs": len(self.store.beliefs),
            "automatic": auto,
            "ambiguities": len([c for c in self.ambiguities if not c.resolved]),
            "aliases": len(self.store.aliases),
            "stats": dict(self.stats),
            "sources": self.trust.snapshot(6),
            "system1_hits": self.system1.hits,
            "hallucination_audits": self.guard.audits,
        }

    def state_dict(self) -> dict:
        return {"store": self.store.to_dict(), "trust": {k: list(v) for k, v in self.trust.beta.items()},
                "stats": self.stats, "system1": self.system1.index}

    def load_state_dict(self, d: dict):
        self.store.load_dict(d.get("store") or {})
        for k, v in (d.get("trust") or {}).items():
            self.trust.beta[k] = [float(v[0]), float(v[1])]
        self.stats.update({k: int(v) for k, v in (d.get("stats") or {}).items() if k in self.stats})
        self.system1.index = dict(d.get("system1") or {})




# ============================================================================
# 5B. İNANÇ BEYANI (kanıta dayalı, şablon değil; cümlecikler köklerden çekilir)
# ============================================================================

class BeliefSpeech:
    """İnanç motorunun sonuçlarını, kök çekimiyle kurulan Türkçe cümleciklere dönüştürür.
    Her beyan, kesinliği ve kaynak sayısını AÇIKÇA söyler: 'kaydettim' demek yetmez,
    ne kadar güvendiğini ve neden güvendiğini de söyler."""

    def __init__(self, engine: "EpistemicEngine"):
        self.e = engine
        self.syn = TurkishSyntaxEngine

    def _v(self, verb: str, **kw) -> str:
        return self.syn.realize_verb(VerbSpec(verb_root=verb, **kw))

    # ---------------------------------------------------------------- depolama
    def speak_store(self, res: dict) -> str:
        status = res.get("status")
        b: Belief | None = res.get("belief")
        rel = float(res.get("reliability", 0.0))
        cred = float(res.get("credence", 0.0))

        if status == "alias":
            a, bb = res.get("a", ""), res.get("b", "")
            head = self._v("not et", obj=NP("eşleme", case="ACC"),
                           advs=(f"{a} ile {bb} arasında",), tense="PAST", person="1SG")
            if res.get("merged"):
                tail = self._v("işle", obj=NP("ad", plural=True, case="ACC"),
                               advs=("artık ikisini aynı kavram olarak",), tense="PRES", person="1SG")
                return f"{tr_capitalize(head)}. {tr_capitalize(tail)}; güvenim {cred:.2f}."
            tail = self._v("yüksel", subj=NP("güven", poss=1), tense="AOR", person="3SG",
                           pre_clause="ikinci bir bağımsız kaynak aynı eşlemeyi doğrularsa")
            return (f"{tr_capitalize(head)}; güvenim {cred:.2f} ve tek kaynağa dayanıyor "
                    f"(kaynağın güvenilirliği {rel:.2f}). {tr_capitalize(tail)}.")

        if status == "contradiction":
            mine = res.get("mine", {})
            theirs = res.get("theirs", {})
            c1 = self._v("var", subj=NP("kayıt", poss=1),
                         advs=(f"{mine.get('value')} {mine.get('unit') or ''}".strip(),), tense="AOR", person="3SG")
            c2 = self._v("değerlendir", obj=NP("karşı iddia", case="ACC"),
                         advs=(f"güvenilirliği {float(theirs.get('reliability', 0)):.2f}",),
                         tense="PAST", person="1SG")
            c3 = self._v("koru", obj=NP("inanç", poss=1, case="ACC"), advs=("dolayısıyla",),
                         tense="PRES", person="1SG")
            c4 = self._v("değiş", obj=NP("inanç", poss=1, case="ACC"),
                         pre_clause="iki bağımsız ve güvenilir kaynak aynı karşı kanıtı verirse",
                         tense="AOR", person="1SG")
            return f"{tr_capitalize(c1)}. {tr_capitalize(c2)}, ama {c3}. {tr_capitalize(c4)}."

        if status == "ambiguous":
            return self.speak_ambiguity(res)

        if status == "stored":
            kind = res.get("kind")
            if kind == "event" and b is not None:
                nom = self.syn.nominalize_dik(
                    "ortaya çık", poss=3, case="ACC",
                    subj=NP(b.subject, case="GEN", proper=True), loc=NP("yıl", poss=3, case="LOC", mods=(b.obj,)),
                )
                c1 = self._v("kaydet", obj=nom, tense="PAST", person="1SG")
            elif kind == "scalar" and b is not None:
                nom = self.syn.nominalize_dik(
                    b.relation.lower(), poss=3, case="ACC",
                    subj=NP(b.subject, case="GEN", proper=True), loc=NP(b.obj, case="LOC"),
                )
                c1 = self._v("kaydet", obj=nom, tense="PAST", person="1SG")
            else:
                c1 = self._v("işle", obj=NP("bilgi", case="ACC"), tense="PAST", person="1SG")
            if res.get("single_source"):
                c2 = self._v("yüksel", subj=NP("güven", poss=1), tense="AOR", person="3SG",
                             pre_clause="ikinci bir bağımsız kaynak aynı şeyi söylerse")
                return (f"{tr_capitalize(c1)}; güvenim {cred:.2f} ve şu an tek kaynağa dayanıyor "
                        f"(kaynak güvenilirliği {rel:.2f}). {tr_capitalize(c2)}.")
            c2 = self._v("destekle", subj=NP("kaynak", plural=True, mods=(str(b.independent_sources), "bağımsız")),
                         obj=NP("inanç", poss=1, case="ACC"), tense="PRES", person="3SG")
            return f"{tr_capitalize(c1)}; güvenim {cred:.2f}. {tr_capitalize(c2)}."

        return ""

    # ---------------------------------------------------------------- çelişki / hipotez
    def speak_ambiguity(self, res: dict) -> str:
        hyps = res.get("hypotheses") or []
        rivals = res.get("rivals") or []
        b = res.get("belief")
        year = (b.obj if b is not None else "")
        ent = (b.subject if b is not None else "")
        rival = rivals[0] if rivals else ""

        c1 = self._v("kaydet", obj="bunu", advs=(f"{year} yılı için",), tense="PAST", person="1SG")
        c2 = self.syn.realize_pred(PredSpec(subj=NP("ikinci ad", poss=1), pred_root="var", person="3SG"))
        c3 = self._v("gör", obj=NP("olasılık", mods=("iki",)), tense="PRES", person="1SG")

        parts = [f"{tr_capitalize(c1)}. {tr_capitalize(c2)}: {rival}. {tr_capitalize(c3)}:"]
        labels = ("Birinci", "İkinci")
        for i, h in enumerate(hyps[:2]):
            desc = h.get("desc", "")
            parts.append(f"{labels[i]} olasılık, {desc} (güvenim {h.get('credence', 0):.2f}).")
        ask_head = self._v("art", subj=NP("güven", poss=1), tense="AOR", person="3SG")
        parts.append(f"{ent} ile {rival} aynı varlık diye bir kaynak gösterirsen {ask_head}.")
        return " ".join(parts)

    # ---------------------------------------------------------------- sorgu yanıtı
    def speak_query(self, ans: dict) -> str:
        b: Belief | None = ans["belief"]
        if ans.get("query_type") == "ambiguity":
            ents = ans.get("entities") or []
            hyps = ans.get("hypotheses") or []
            sayi = {2: "iki", 3: "üç", 4: "dört"}.get(len(ents), str(len(ents)))
            c0 = self._v("duy", obj=NP("ad", mods=(sayi, "ayrı")),
                         pre_clause="aynı yıl için", tense="PAST", person="1SG")
            c1 = self._v("bil", obj="hangisinin doğru olduğunu", neg=True, tense="PRES", person="1SG")
            head = f"{tr_capitalize(c0)}, ama {c1}" + (f"; {' ile '.join(ents)} aynı varlık mı, bilmiyorum." if ents else ".")
            parts = [head, "İki olasılık görüyorum:"]
            for i, h in enumerate(hyps[:2]):
                lab = ("Birincisi", "İkincisi")[i]
                parts.append(f"{lab}, {h.get('desc', '')} (güvenim {h.get('credence', 0):.2f}).")
            parts.append("İkisinin aynı olduğunu gösteren güvenilir bir kaynak verirsen güvenim artar.")
            return " ".join(parts)
        cred = float(ans.get("credence", b.credence))
        path = ans.get("path")
        qt = ans.get("query_type")

        if qt == "scalar":
            core = self.syn.realize_verb(VerbSpec(
                verb_root=tr_lower(b.relation), subj=NP(b.subject, proper=True),
                advs=(b.obj,), tense="AOR", person="3SG",
            ))
            core = tr_capitalize(core) + "."
            if path == "system1":
                auto = self._v("hatırla", obj="bunu", advs=("artık düşünmeden, otomatik olarak",),
                               tense="PRES", person="1SG")
                return f"{core} {tr_capitalize(auto)}; güvenim {cred:.2f}."
            return f"{core} Şu an güvenim {cred:.2f}; kanıt biriktikçe bu bilgi otomatikleşecek."

        if qt == "event":
            subj_np = NP(b.subject, proper=True)
            c = self._v("ortaya çık", subj=subj_np, advs=(f"{b.obj} yılında",),
                        tense="PAST", person="3SG")
            if path == "system1":
                return f"{tr_capitalize(c)}; güvenim {cred:.2f} ve bu bilgiyi artık anında hatırlıyorum."
            return f"{tr_capitalize(c)}; güvenim {cred:.2f}."

        if qt == "alias":
            pair = ans.get("pair") or [b.subject, b.obj]
            same = f"{tr_capitalize(pair[0])} ile {pair[1]} aynı varlığın iki adı."
            if path == "system1":
                return f"{same} Güvenim {cred:.2f}; bu eşlemeyi artık otomatik hatırlıyorum."
            return f"{same} Güvenim {cred:.2f}."
        return ""


# ============================================================================
# 6. GÜVENİLİR KAYNAK ARAŞTIRMACISI (yalnızca allowlist)
# ============================================================================

class TrustedResearcher:
    """İnternete YALNIZCA güvenilir kaynaklardan erişir. Allowlist dışı bir alan adı istendiğinde
    istek reddedilir ve kaydedilir; böylece Nora sahte/zehirli bilgiyle kirlenemez.
    Çok kaynaklı örtüşme, aynı inancın α'sını yükseltir (güven artar)."""

    TRUSTED_DOMAINS: tuple[str, ...] = (
        "wikipedia.org", "wikidata.org", "britannica.com", "tdk.gov.tr", "tubitak.gov.tr",
        "arxiv.org", "nature.com", "science.org", "nasa.gov", "who.int", "un.org", "europa.eu",
        "python.org", "developer.mozilla.org", "docs.python.org", "github.com", "acm.org", "ieee.org",
        ".gov", ".edu",
    )

    def __init__(self, engine: EpistemicEngine, fetcher: Callable[[str], str] | None = None, timeout: float = 4.0):
        self.engine = engine
        self.fetcher = fetcher
        self.timeout = float(timeout)
        self.log: deque[dict] = deque(maxlen=40)
        self.refused: list[str] = []

    @classmethod
    def is_trusted(cls, url: str) -> bool:
        try:
            host = urllib.parse.urlparse(url).netloc.lower()
        except Exception:
            return False
        if not host:
            return False
        return any(host == d or host.endswith(d) or (d.startswith(".") and host.endswith(d)) for d in cls.TRUSTED_DOMAINS)

    def _default_fetch(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "NoraAGI-TrustedResearch/2.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def fetch(self, url: str) -> dict:
        rec = {"url": url, "trusted": self.is_trusted(url), "bytes": 0, "ok": False}
        if not rec["trusted"]:
            self.refused.append(url)
            rec["reason"] = "allowlist_disi"
            self.log.append(rec)
            return rec
        body = ""
        try:
            body = (self.fetcher(url) if self.fetcher is not None else self._default_fetch(url)) or ""
            rec["ok"] = bool(body.strip())
        except Exception as e:
            rec["error"] = type(e).__name__
        rec["bytes"] = len(body)
        rec["body"] = body
        self.log.append(rec)
        return rec

    def learn_from_text(self, text: str, source_id: str, max_claims: int = 12) -> dict:
        """Metni cümle cümle inanç motoruna aktarır; her cümle bağımsız kanıt sayılır.
        Web kaynağı allowlist dışındaysa HİÇBİR cümle öğrenilmez (zehirli bilgi giremez)."""
        if source_id.startswith("web:"):
            host = source_id[4:]
            if not self.is_trusted("https://" + host):
                self.refused.append(source_id)
                self.log.append({"source": source_id, "ok": False, "reason": "guvenilmez_kaynak"})
                return {"claims": [], "n": 0, "refused": True}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) >= 12]
        results = []
        for sent in sentences[:max_claims]:
            res = self.engine.absorb(sent, source_id=source_id)
            if res.get("status") in ("stored", "alias", "ambiguous", "contradiction"):
                results.append({"text": sent[:90], "status": res["status"],
                                "credence": res.get("credence")})
        return {"claims": results, "n": len(results)}

    def research(self, query: str, urls: list[str] | None = None) -> dict:
        """Verilen güvenilir URL'lerden (veya varsayılan Wikipedia API'sinden) bilgi toplar."""
        urls = urls or [self.wikipedia_url(query)]
        fetched, learned = 0, 0
        for u in urls[:4]:
            fr = self.fetch(u)
            if not fr.get("ok"):
                continue
            fetched += 1
            res = self.learn_from_text(fr["body"], source_id=f"web:{urllib.parse.urlparse(u).netloc}")
            learned += res["n"]
            # Kaynak metni iddialarla tutarlıysa güven kazanır
            self.engine.trust.reward(f"web:{urllib.parse.urlparse(u).netloc}", good=True, strength=0.2)
        return {"query": query, "fetched": fetched, "learned": learned}

    @staticmethod
    def wikipedia_url(topic: str, lang: str = "tr") -> str:
        slug = urllib.parse.quote(topic.strip().replace(" ", "_"))
        return f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{slug}"

    def snapshot(self) -> dict:
        return {"trusted_domains": len(self.TRUSTED_DOMAINS), "refused": self.refused[-5:],
                "last": [{k: v for k, v in r.items() if k != "body"} for r in list(self.log)[-5:]]}
