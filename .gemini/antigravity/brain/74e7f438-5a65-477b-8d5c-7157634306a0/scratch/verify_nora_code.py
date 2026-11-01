from nora.synthesis.code import CodeSynthesisCortex

cortex = CodeSynthesisCortex()
queries = [
    "Bana bir register sayfası yap",
    "register yap",
    "bana register yap",
    "hesap makinesi yap",
    "tic-tac-toe oyunu yap",
    "todo uygulaması kodla",
    "dashboard tasarla",
    "kahve dükkanı için dinamik bir website oluştur",
    "merhaba dünya yazan bir html sayfası",
    "sayaç yap"
]

for q in queries:
    is_req = cortex.is_requested(q)
    assert is_req, f"Failed is_requested on: {q}"
    res = cortex.handle(q)
    assert res.get("ok"), f"Failed handle on {q}: {res.get('errors')}"
    spoken = cortex.speak(res)
    assert "```html" in spoken, f"Markup missing in output for {q}"
    print(f"PASSED: {q} -> cat={res.get('category')} tags={res.get('total_tags')}")

print("\nALL 10 DYNAMIC WEB QUERIES VERIFIED SUCCESSFULLY IN CODE SYNTHESIS CORTEX!")
