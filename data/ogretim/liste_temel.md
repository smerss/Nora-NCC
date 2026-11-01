Öğren: Yapılacaklar sayfası: 'ekle' butonu kutuya yazılanı listeye ekler.

```html
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>Yapılacaklar</title>
  <style>
    body { font-family: sans-serif; max-width: 28rem; margin: 2rem auto; }
    input { padding: 0.4rem; width: 60%; }
    button { padding: 0.4rem 0.8rem; margin-left: 0.3rem; }
    li { padding: 0.2rem 0; }
  </style>
</head>
<body>
  <h1>Yapılacaklar</h1>
  <input id="is" placeholder="Yeni iş">
  <button onclick="ekle()">ekle</button>
  <ul id="liste"></ul>
  <script>
    function ekle() {
      const kutu = document.getElementById('is');
      const madde = document.createElement('li');
      madde.textContent = kutu.value;
      document.getElementById('liste').appendChild(madde);
      kutu.value = '';
    }
  </script>
</body>
</html>
```

Öğren: Alışveriş listesi: 'al' düğmesi yazılan ürünü listeye ekler, 'temizle' düğmesi listeyi temizler.

```html
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>Alışveriş</title>
</head>
<body>
  <input id="urun">
  <button id="al">al</button>
  <button id="sil">temizle</button>
  <ol id="sepet"></ol>
  <script>
    let urunler = [];
    const sepet = document.getElementById('sepet');
    function ciz() {
      sepet.innerHTML = urunler.map(u => '<li>' + u + '</li>').join('');
    }
    document.getElementById('al').addEventListener('click', () => {
      urunler.push(document.getElementById('urun').value);
      ciz();
    });
    document.getElementById('sil').addEventListener('click', () => {
      urunler = [];
      ciz();
    });
  </script>
</body>
</html>
```

Öğren: Selamlama: 'göster' butonu kutuya yazılan adı ekranda gösterir.

```html
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>Selam</title>
</head>
<body>
  <input id="ad" placeholder="Adın">
  <button onclick="document.getElementById('selam').textContent = 'Merhaba ' + document.getElementById('ad').value">göster</button>
  <p id="selam"></p>
</body>
</html>
```
