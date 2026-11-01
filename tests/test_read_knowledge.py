"""Okunan metinden gelen bilgi: hemen kullanılır, ama kaynağıyla çekinerek söylenir.

Tek-sunum ilkesi korunur (okunan ilişki aynı tur içinde sorgulanabilir). Tek tanıklı okuma
'provisional' işaretlidir: konuşmada kesin olgu gibi sunulmaz, çok adımlı çıkarımın ara basamağı
olmaz ve ancak BAĞIMSIZ ikinci bir kaynak (ya da muhatap) aynı ilişkiyi verince doğrulanır.
"""
import pytest

from nora.cognition.agi import NoraAGI

TEXT = "Zorvath bir alettir.\nZorvath kayaları keser.\n"


@pytest.fixture
def reader(tmp_path):
    agi = NoraAGI(seed=7, boot_thinker=False)
    path = tmp_path / "ansiklopedi.txt"
    path.write_text(TEXT, encoding="utf-8")
    agi.read_corpus(str(path))
    return agi, path


def _edges(agi, subj):
    return {(e.rel, e.obj): e for e in agi.lang.graph.edges if e.subj == subj}


def test_read_knowledge_is_usable_at_once_but_marked(reader):
    agi, _ = reader
    edges = _edges(agi, "zorvath")
    assert {e.status for e in edges.values()} == {"provisional"}
    assert {e.speaker for e in edges.values()} == {"okuma:ansiklopedi.txt"}
    assert agi.lang.graph.check_relation("zorvath", "alet")["answer"] is True


def test_answers_resting_on_reading_name_their_source(reader):
    agi, _ = reader
    for question in ("Zorvath nedir?", "Zorvath kayaları keser mi?"):
        reply = agi.converse(question)
        assert reply.startswith("Okuduğum bir metne göre (ansiklopedi.txt):"), reply


def test_same_source_again_is_not_a_second_witness(reader):
    agi, path = reader
    agi.read_corpus(str(path))
    assert _edges(agi, "zorvath")[("ISA", "alet")].status == "provisional"


def test_independent_witness_corroborates_and_drops_the_hedge(reader):
    agi, _ = reader
    agi.converse("Zorvath bir alettir.", speaker="insan")
    assert _edges(agi, "zorvath")[("ISA", "alet")].status == "corroborated"
    # Cevap hâlâ doğrulanmamış 'kaya keser'e dayanıyorsa çekince kalır; o da doğrulanınca kalkar.
    agi.converse("Zorvath kayaları keser.", speaker="insan")
    assert {e.status for e in _edges(agi, "zorvath").values()} == {"corroborated"}
    reply = agi.converse("Zorvath nedir?")
    assert "alet" in reply.casefold() and not reply.startswith("Okuduğum"), reply


def test_provisional_isa_is_not_a_chain_step(reader):
    agi, _ = reader
    agi.lang.teach_sentence("Alet bir nesnedir.", speaker="insan")
    chained = [(e.subj, e.rel, e.obj) for e, path in agi.lang.graph.query("zorvath") if path]
    assert chained == []


def test_old_brains_mark_text_sources_by_who_actually_spoke(tmp_path):
    agi = NoraAGI(seed=8, boot_thinker=False)
    agi.lang.teach_sentence("Velkor bir kuştur.", speaker="hf_corpus")
    agi.converse("Mirtan bir balıktır.", speaker="ses-9")
    again = NoraAGI.load(agi.save(str(tmp_path / "beyin")))
    assert _edges(again, "velkor")[("ISA", "kuş")].status == "provisional"
    assert _edges(again, "mirtan")[("ISA", "balık")].status == "accepted"
