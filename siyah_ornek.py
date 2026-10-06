#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIYAH NOKTA ORNEKLEMESI (altin kume incelemesini hizlandirmak icin)

gunluk.py ciktisindaki siyah nokta klasorunden RASTGELE 1/4'u yerinde birakir,
geri kalanini '_incelenmeyen_siyah_nokta' klasorune tasir. Sen sadece yerinde
kalan dortte bire bakarsin. altin_olcum.py bu dortte biri 4 kat agirlikla
sayar; rastgele secildigi icin olcum tarafsiz kalir.

gunluk.py'den HEMEN SONRA, incelemeye baslamadan once bir kez calistirin.
  python siyah_ornek.py cikti_altin_925
"""

import sys
import json
import shutil
from pathlib import Path

import numpy as np

ORAN = 4
ANAHTAR = "黑斑"


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python siyah_ornek.py <cikti_klasoru>"); sys.exit(1)
    kok = Path(sys.argv[1])
    if (kok / "_ornekleme.json").exists():
        print("Bu klasorde orneklem zaten yapilmis, tekrar yapilmadi."); sys.exit(0)
    kl = next((d for d in kok.iterdir() if d.is_dir() and ANAHTAR in d.name), None)
    if kl is None:
        print(f"HATA: {kok} icinde siyah nokta klasoru yok."); sys.exit(1)
    dosyalar = sorted(f for f in kl.iterdir() if f.is_file())
    n = len(dosyalar)
    kalan = max(1, round(n / ORAN))
    rng = np.random.RandomState(25)
    kalacak = set(rng.choice(n, kalan, replace=False).tolist())
    hedef = kok / "_incelenmeyen_siyah_nokta"
    hedef.mkdir(exist_ok=True)
    for i, f in enumerate(dosyalar):
        if i not in kalacak:
            shutil.move(str(f), hedef / f.name)
    (kok / "_ornekleme.json").write_text(json.dumps(
        {"klasor": kl.name, "toplam": n, "kalan": kalan, "agirlik": n / kalan},
        ensure_ascii=False), encoding="utf-8")
    print(f"'{kl.name}': {n} gorsel -> {kalan} tanesi incelemek icin yerinde,")
    print(f"  {n - kalan} tanesi '{hedef.name}' klasorune tasindi (bunlara bakmayin).")


if __name__ == "__main__":
    main()
