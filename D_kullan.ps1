# D-disiplini: C'ye (0.7GB boş) yazmayı engelle. Her PowerShell oturumunda çalıştır:
#   . D:\NoRa_AGI\D_kullan.ps1
$env:TMP = "D:\newtemps"
$env:TEMP = "D:\newtemps"
$env:PIP_CACHE_DIR = "D:\NoRa_AGI\.cache\pip"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:HF_HUB_CACHE = "D:\NoRa_AGI\.cache\hf"
$env:TORCH_HOME = "D:\NoRa_AGI\.cache\torch"
$env:PYTHONUTF8 = "1"
$DPY = "D:\NoRa_AGI\.venv\Scripts\python.exe"
Write-Host "D-kilidi aktif. Python: $DPY"
