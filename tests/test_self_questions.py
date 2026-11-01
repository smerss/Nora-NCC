"""Nora'nın kendisine sorulan sorular: tekrar algısı oturuma bağlıdır, tercih kanıttan gelir."""
import nora.synthesis.code  # noqa: F401  (nora_* takma adlarını kurar)
from nora.cognition.agi import NoraAGI
from nora_register import normalize


def test_question_from_an_earlier_session_is_not_a_repeat(tmp_path):
    agi = NoraAGI(seed=21, boot_thinker=False)
    agi.converse("sen kimsin?")
    again = NoraAGI.load(agi.save(str(tmp_path / "beyin")))
    reply = again.converse("sen kimsin?")
    assert "ikinci kez" not in reply, reply
    # Aynı oturumda gerçekten tekrar edilince fark eder.
    assert "ikinci kez" in again.converse("sen kimsin?")


def test_favorite_of_unknown_category_is_not_answered_from_world_knowledge():
    agi = NoraAGI(seed=22, boot_thinker=False)
    reply = agi.converse("en sevdiğin renk ne?")
    assert reply.startswith("Renkleri tanımıyorum"), reply


def test_favorite_without_evidence_is_not_invented_and_with_evidence_is_explained():
    agi = NoraAGI(seed=23, boot_thinker=False)
    for fact in ("Kedi bir hayvandır.", "Köpek bir hayvandır."):
        agi.converse(fact, speaker="insan")
    reply = agi.converse("en sevdiğin hayvan hangisi?")
    assert "uydurmuyorum" in reply and "kedi" in reply.casefold(), reply
    agi.converse("Kedi güzeldir.", speaker="insan")
    reply = agi.converse("en sevdiğin hayvan hangisi?")
    assert reply.startswith("En sevdiğim hayvan kedi") and "güzel" in reply, reply


def test_request_tail_does_not_leave_period_question_mark_mid_text():
    out = normalize("Bana kendini anlatır mısın? Bunu az önce de söylemiştin?")
    assert ".?" not in out and "anlatsan iyi olur." in out
