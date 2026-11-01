from nora.cognition.agi import NoraAGI

agi = NoraAGI()

test_conversations = [
    ("Bana bir register sayfası yap", "Kayıt Ol & Giriş Yap"),
    ("Hesap makinesi yap", "Hesap Makinesi"),
    ("Tic-tac-toe oyunu yap", "Tic-Tac-Toe"),
    ("Kahve dükkanı için dinamik bir website oluştur", "Kahve"),
    ("Merhaba Nora", "Merhaba"),
]

for prompt, expected_keyword in test_conversations:
    reply = agi.converse(prompt)
    assert expected_keyword.lower() in reply.lower(), f"Prompt '{prompt}' reply did not contain '{expected_keyword}'. Reply: {reply[:150]}"
    print(f"PASSED converse: '{prompt}' -> contains '{expected_keyword}'")

print("\nALL CONVERSE TESTS PASSED 100%!")
