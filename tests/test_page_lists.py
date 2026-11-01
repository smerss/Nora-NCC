"""Sayfa korteksi, metin ve liste durumu: kutuya yazılanı ekle / göster, listeyi boşalt.

Öğretim `data/ogretim/liste_temel.md` (createElement, dizi+innerHTML, textContent tarzları).
İstekler o örneklerde görülmemiş bileşimlerdir; denetim organın kendi doğrulamasından bağımsızdır.
"""
import shutil
from pathlib import Path

import pytest

import nora.synthesis.code  # noqa: F401  (nora_* takma adlarını kurar)
from nora.synthesis import sayfa
from nora.synthesis.sayfa import PageCortex, parse_page

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="DOM kabuğu için node gerekli")

DATA = Path(__file__).resolve().parent.parent / "data" / "ogretim"
LISTS = (DATA / "liste_temel.md").read_text(encoding="utf-8")
COUNTERS = (DATA / "sayfa_temel.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def taught():
    pc = PageCortex()
    for doc in (COUNTERS, LISTS):
        res = pc.teach_document(doc)
        assert res["ok"] and not res["failed"]
    return pc


def _run(markup: str, actions_for) -> tuple[sayfa.Page, dict]:
    page = parse_page(markup)
    controls = {page.text(i): i for i, n in enumerate(page.nodes) if n["attrs"].get("onclick")}
    box = next(i for i, n in enumerate(page.nodes) if n["tag"] == "input")
    obs = sayfa.DomHarness().run(page, [actions_for(controls, box)], strict=True)
    assert not obs.get("load_errors") and not obs["runs"][0]["errors"], obs
    return page, obs["runs"][0]


def test_text_effects_are_observed_not_declared(taught):
    seen = {f.label: (f.meaning, f.role) for f in taught.fragments if f.role != "sayı"}
    assert seen == {"ekle": ("append", "liste"), "al": ("append", "liste"),
                    "temizle": ("clear", "liste"), "göster": ("show", "metin")}


def test_append_and_clear_from_two_styles(taught):
    res = taught.handle("kutuya yazılanı listeye ekleyen ve listeyi temizleyen butonları olan bir sayfa yap")
    assert res["ok"] and len(res["sources"]) == 2
    page, run = _run(res["program"], lambda c, box: [
        ["type", box, "elma"], c["ekle"], ["type", box, "armut"], c["ekle"], c["temizle"]])
    lists = [i for i, n in enumerate(page.nodes) if n["tag"] in ("ul", "ol")]
    assert len(lists) == 1
    counts = [s[lists[0]][3] for s in run["snaps"]]
    assert counts == [0, 0, 1, 1, 2, 0]
    assert "elma" in run["snaps"][4][lists[0]][1] and "armut" in run["snaps"][4][lists[0]][1]


def test_missing_target_is_borrowed_and_bound_to_one_box(taught):
    res = taught.handle("yazılanı ekranda gösteren ve listeye ekleyen butonları olan bir sayfa yap")
    assert res["ok"]
    page, run = _run(res["program"], lambda c, box: [["type", box, "Ayşe"], c["göster"]])
    assert any("Ayşe" in s[0] for s in run["snaps"][-1]), "yazılan ad ekranda görünmedi"
    assert sum(n["tag"] == "input" for n in page.nodes) == 1


def test_counter_and_list_in_one_page(taught):
    res = taught.handle("yazılanı listeye ekleyen, sayıyı 2 artıran ve listeyi temizleyen butonları olan bir sayfa yap")
    assert res["ok"] and [c["op"] for c in res["checks"]] == ["append", "add", "clear"]
    page, run = _run(res["program"], lambda c, box: [c["artır"], c["artır"]])
    display = page.find_id("sayi")
    assert [sayfa._first_number(s[display][0]) for s in run["snaps"]] == [0, 2, 4]


def test_untestable_clear_is_not_claimed(taught):
    res = taught.handle("listeyi temizleyen butonu olan bir sayfa yap")
    assert not res["ok"] and res["path"] == "page_unverified" and res["program"] == ""


def test_generated_pages_are_checked_in_strict_mode():
    page = parse_page("<b id=\"s\">1</b><button onclick=\"zz = 5\">x</button>")
    harness = sayfa.DomHarness()
    assert harness.run(page, [[1]])["runs"][0]["errors"] == []
    assert harness.run(page, [[1]], strict=True)["runs"][0]["errors"]
