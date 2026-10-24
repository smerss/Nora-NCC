from nora_agi import NoraAGI
from nora_besle import besle
g = NoraAGI(seed=1, boot_thinker=False); besle(g, save=None, verbose=False)
qs = ["Merhaba.", "Kediyi seviyor musun?", "Kediyi seviyor musun?", "Kediyi seviyor musun?",
      "Kedileri sever misin?", "Sen kedi sever misin?", "Köpekleri sever misin?",
      "Neden kediyi sevdiğini söylüyorsun?", "Bugün biraz yorgunum.", "Bugün biraz yorgunum.",
      "Sıkıldım.", "Ne yapmak istersin?", "Kediler neden mırlar?", "Kediler neden mırlar?",
      "Bilim insanları uzayı nasıl inceliyor?", "Bana bir şaka yap.", "Kendini nasıl hissediyorsun?",
      "Sen kimsin?", "Ne düşünüyorsun?", "Hava yağmurlu olunca ne yapıyorsun?"]
for q in qs:
    print("S>", q); print("N>", g.converse(q)); print()
