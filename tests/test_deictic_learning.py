"""Zamir özneli cümleler kavram ağına dünya bilgisi olarak yazılmaz.

Okunan metindeki 'Benim annem evhamlıdır' cümlesi yazarına aittir; 'benim' düğümüne
bağlanırsa sonraki her 'benim' sözünde Nora o yabancı hayatı kendi hayatı gibi anlatır.
"""
from nora.language.lang import NoraLanguage
from nora_chem import NeuroChemistry


def _lang():
    return NoraLanguage(chem=NeuroChemistry())


def test_pronoun_subject_is_not_learned_as_world_fact():
    lang = _lang()
    for sentence in ("Benim annem çok evhamlıdır.", "Sen bir robotsun.", "Bizim evimiz büyüktür."):
        assert lang.teach_sentence(sentence, speaker="kaynak") == []
    assert not lang.graph._out_edges.get("benim")


def test_ordinary_subject_is_still_learned():
    lang = _lang()
    learned = [e for e in lang.teach_sentence("Zorvath bir alet türüdür.") if e]
    assert ("zorvath", "ISA", "alet türü") in {(e.subj, e.rel, e.obj) for e in learned}


def test_retract_edges_removes_from_queries_but_keeps_slots():
    lang = _lang()
    graph = lang.graph
    graph.add_edge("benim", "HAS_PROP", "evhamlı", source="duyum", speaker="kaynak")
    kept = graph.add_edge("kedi", "ISA", "hayvan", source="duyum")
    n_before = len(graph.edges)
    assert graph.retract_edges(lambda e: NoraLanguage.is_deictic(e.subj)) == 1
    assert len(graph.edges) == n_before
    assert not graph._out_edges.get("benim")
    assert ("benim", "HAS_PROP", "evhamlı") not in graph._edge_map
    assert graph._edge_map[("kedi", "ISA", "hayvan")] is kept
