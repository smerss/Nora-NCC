"""Sayfa korteksi: örnekten öğren, görülmemiş isteği birleştir, tıklayarak doğrula.

Öğretim verisi `data/ogretim/sayfa_temel.md` içindeki üç örnektir (+1 satır içi kod, -1 fonksiyonla,
sıfırla/+1 dinleyiciyle). Buradaki istekler o örneklerde hiç görülmemiş büyüklük ve bileşimlerdir.
"""
import inspect
import re
import shutil
from pathlib import Path

import pytest

import nora.synthesis.code  # noqa: F401  (nora_* takma adlarını kurar)
from nora.synthesis import sayfa
from nora.synthesis.code import CodeSynthesisCortex
from nora.synthesis.sayfa import PageCortex, parse_page

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="DOM kabuğu için node gerekli")

CURRICULUM = (Path(__file__).resolve().parent.parent / "data" / "ogretim" / "sayfa_temel.md").read_text(
    encoding="utf-8")
KATLA = ("Öğren: Çarpan oyunu: 'katla' butonu sayıyı 2 ile katlar.\n```html\n"
         "<html><body><b id=\"s\">3</b><button onclick=\"s = s * 2; "
         "document.getElementById('s').textContent = s\">katla</button>"
         "<script>let s = 3;</script></body></html>\n```")


@pytest.fixture(scope="module")
def taught():
    pc = PageCortex()
    res = pc.teach_document(CURRICULUM)
    assert len(res["learned"]) == 3 and not res["failed"]
    return pc


def _independent_check(markup: str, expected: list[tuple[str, float]]) -> None:
    """Organın kendi doğrulamasına güvenmeden sayfayı yeniden koştur ve sayıyı oku."""
    page = parse_page(markup)
    controls = [i for i, n in enumerate(page.nodes) if n["attrs"].get("onclick")]
    assert len(controls) == len(expected)
    obs = sayfa.DomHarness().run(page, [[c, c] for c in controls])
    assert not obs.get("load_errors")
    for (op, k), run in zip(expected, obs["runs"]):
        assert not run["errors"]
        changed = [d for d in range(len(page.nodes)) if run["snaps"][0][d][0] != run["snaps"][1][d][0]]
        assert changed, "düğme hiçbir şeyi değiştirmedi"
        n0, n1, n2 = (sayfa._first_number(s[changed[0]][0]) for s in run["snaps"])
        assert sayfa.effect_holds(op, k, n0, n1, n2), (op, k, n0, n1, n2)


def test_organ_holds_no_page_skeleton():
    # DOM kabuğu (ölçüm aleti) dışında organın kaynağında hiçbir sayfa/düğme/betik parçası yok.
    src = re.sub(r'_SHIM = r""".*?"""', "", inspect.getsource(sayfa), flags=re.DOTALL)
    for literal in ("<button", "<!DOCTYPE", "<html", "<body", "textContent ="):
        assert literal not in src
    assert not PageCortex().is_requested("Sayıyı 5 artıran butonu olan bir sayaç sayfası yap")


def test_meaning_comes_from_observed_effect(taught):
    keys = {f.label: (f.meaning, f.slot) for f in taught.fragments}
    assert keys["artır"] == ("add+", True)
    assert keys["azalt"] == ("add-", True)
    assert keys["gol"][0] == "add+"
    assert keys["sıfırla"][0] in ("set", "set=0")
    # Her örnekte geçen sözcükler hiçbir etkiye bağlanmaz (çapraz-durumsal tutarlılık).
    for word in ("sayıyı", "butonu", "sayfası"):
        assert taught._meaning_of(word) is None


def test_unseen_magnitudes_are_composed_and_verified(taught):
    res = taught.handle("Sayıyı 5 artıran ve 2 azaltan butonları olan bir sayaç sayfası yap")
    assert res["ok"] and res["verified"]
    assert [(c["op"], c["k"]) for c in res["checks"]] == [("add", 5.0), ("add", -2.0)]
    _independent_check(res["program"], [("add", 5.0), ("add", -2.0)])
    examples = [m for _, m in PageCortex.split_document(CURRICULUM)]
    assert all(res["program"].strip() != ex.strip() for ex in examples)


def test_three_styles_are_unified_on_one_state(taught):
    res = taught.handle("3 artıran, 3 azaltan ve sıfırlayan butonlu bir sayaç sayfası yap")
    assert res["ok"]
    assert len(res["sources"]) == 3, "parçalar üç ayrı örnekten gelmeli"
    assert [c["label"] for c in res["checks"]] == ["artır", "azalt", "sıfırla"]
    _independent_check(res["program"], [("add", 3.0), ("add", -3.0), ("set", 0.0)])


def test_unknown_effect_is_admitted_not_invented():
    pc = PageCortex()
    pc.teach_document(CURRICULUM)
    res = pc.handle("sayıyı ikiye katlayan butonu olan bir sayaç sayfası yap")
    assert not res["ok"] and res["path"] == "page_gap"
    assert any("katla" in u for u in res["unknown"])
    assert res["program"] == ""
    # Tek bir örnek yeter: öğrendikten sonra görülmemiş büyüklükle ve başka parçalarla birleşir.
    assert pc.teach_document(KATLA)["ok"]
    res = pc.handle("sayıyı 3 ile katlayan ve 1 azaltan butonları olan bir sayaç sayfası yap")
    assert res["ok"]
    _independent_check(res["program"], [("mul", 3.0), ("add", -1.0)])


def test_learning_survives_save_and_load(taught):
    fresh = PageCortex()
    fresh.load_state_dict(taught.state_dict())
    res = fresh.handle("7 artıran butonu olan bir sayaç sayfası yap")
    assert res["ok"]
    _independent_check(res["program"], [("add", 7.0)])


def test_code_cortex_routes_teaching_and_requests():
    cortex = CodeSynthesisCortex(seed=3, pretrain=False)
    for block in CURRICULUM.split("Öğren:")[1:]:
        assert cortex.is_document("Öğren:" + block)
        assert cortex.handle_document("Öğren:" + block)["ok"]
    request = "4 artıran ve sıfırlayan butonları olan bir sayaç sayfası yap"
    assert cortex.is_requested(request)
    res = cortex.handle(request)
    assert res["ok"] and res["language"] == "html"
    reply = cortex.speak(res)
    assert "```html" in reply and "✓" in reply and "✗" not in reply


def test_page_learning_survives_whole_brain_save(tmp_path):
    from nora.cognition.agi import NoraAGI

    agi = NoraAGI(seed=741, boot_thinker=False)
    for block in CURRICULUM.split("Öğren:")[1:]:
        agi.converse("Öğren:" + block, speaker="insan")
    path = agi.save(str(tmp_path / "beyin"))
    again = NoraAGI.load(path)
    reply = again.converse("6 artıran ve sıfırlayan butonları olan bir sayaç sayfası yap", speaker="insan")
    assert "```html" in reply and "+6 ✓" in reply and "=0 ✓" in reply
