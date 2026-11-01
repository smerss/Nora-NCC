Öğren: Bir sayaç sayfası: 'artır' butonu sayıyı 1 artırır.

```html
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>Sayaç</title>
  <style>
    body { font-family: sans-serif; text-align: center; margin-top: 3rem; }
    #sayi { font-size: 3rem; margin: 1rem; }
    button { font-size: 1.2rem; margin: 0.3rem; padding: 0.4rem 1rem; }
  </style>
</head>
<body>
  <h1>Sayaç</h1>
  <div id="sayi">0</div>
  <button onclick="n = n + 1; document.getElementById('sayi').textContent = n">artır</button>
  <script>
    let n = 0;
  </script>
</body>
</html>
```

Öğren: Geri sayım: 'azalt' düğmesi değeri 1 azaltır.

```html
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>Geri sayım</title>
</head>
<body>
  <p>Kalan: <span id="deger">10</span></p>
  <button onclick="deger = deger - 1; guncelle()">azalt</button>
  <script>
    let deger = 10;
    function guncelle() {
      document.getElementById('deger').textContent = deger;
    }
  </script>
</body>
</html>
```

Öğren: Puan tablosu: 'sıfırla' butonu puanı sıfırlar, 'gol' butonu puanı 1 artırır.

```html
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>Puan</title>
</head>
<body>
  <strong id="puan">7</strong>
  <button id="gol">gol</button>
  <button id="sifir">sıfırla</button>
  <script>
    let puan = 7;
    const ekran = document.getElementById('puan');
    document.getElementById('gol').addEventListener('click', function () {
      puan = puan + 1;
      ekran.textContent = puan;
    });
    document.getElementById('sifir').addEventListener('click', () => {
      puan = 0;
      ekran.textContent = puan;
    });
  </script>
</body>
</html>
```
