import sys
sys.path.insert(0, "/home/offihito/Nora-NCC/.gemini/antigravity/brain/74e7f438-5a65-477b-8d5c-7157634306a0/scratch")
from test_full_synthesis import CreativeHtmlSynthesizer

synth = CreativeHtmlSynthesizer()
tests = [
    ("bana bir html sayfası yap", True),
    ("login sayfası hazırla", True),
    ("kullanıcı giriş formu tasarla", True),
    ("bana bir portföy web sitesi yap", True),
    ("hava durumu uygulaması arayüzü yap", True),
    ("basit bir web sitesi kodla", True),
    ("html nedir?", False),
    ("web sayfası ne demek?", False),
    ("JavaScript ne işe yarar?", False),
]

for q, expected in tests:
    res = synth.is_requested(q)
    assert res == expected, f"Failed on {q}: got {res}, expected {expected}"
    if res:
        synth_res = synth.synthesize(q)
        assert synth_res["ok"], f"Failed to synthesize {q}: {synth_res.get('errors')}"
        print(f"PASSED: {q} -> cat={synth_res['category']}")

print("ALL EDGE CASES PASSED PERFECTLY!")
