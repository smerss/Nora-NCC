"""İskelet mandalı: koddaki kullanıcıya gidebilecek Türkçe cümle sayısı geri büyüyemez."""
from nora.synthesis import audit


def test_audit_sees_the_package():
    # Paket taşınınca tarama boş dönüp mandal sessizce kör olmuştu; bir daha olmasın.
    rep = audit.scan_all()
    assert len(rep["modul"]) >= 50
    assert rep["toplam"] > 0


def test_no_module_exceeds_baseline():
    rep = audit.scan_all()
    base = audit.load_baseline()
    assert base, "nora_audit_baseline.json okunamadı"
    cmp = audit.compare(rep, base)
    assert not cmp["artan"], f"iskelet büyümüş: {cmp['artan']}"
    assert cmp["toplam_fark"] <= 0, f"toplam iskelet büyümüş: {cmp['toplam_fark']:+d}"
