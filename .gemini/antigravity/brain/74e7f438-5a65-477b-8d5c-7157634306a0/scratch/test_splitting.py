import re

raw = "Bana bir web sayfası yap Portfolyo olsun Offihito yazsın 16 yaşında web developer yazsın renk olarak siyah ve beyaz kullan."

# Markers that separate instructions
markers = ["olsun", "yazsın", "yazan", "yazmalı", "yaz", "kullan", "yap", "tasarla", "kodla", "hazırla"]
pattern = r"\b(" + "|".join(markers) + r")\b|[,\.;]"

tokens = []
last_end = 0
for match in re.finditer(pattern, raw, re.IGNORECASE):
    chunk = raw[last_end:match.start()].strip()
    marker = match.group(0).lower()
    if chunk:
        tokens.append((chunk, marker))
    last_end = match.end()

print("Parsed instruction chunks:")
for chunk, marker in tokens:
    print(f"  Chunk: '{chunk}' -> Action/Marker: '{marker}'")
