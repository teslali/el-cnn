#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KALAN GORSELLERDEN YENI ALTIN ORNEKLEM (9.25'in dokunulmamis kismi)

9.25'in altin kumeye girmeyen gorsellerinden (envanterde 'altin_gun_haric')
vardiya ve hat dengeli yeni bir orneklem secer; altin_kume7 klasorune kopyalar.
Kullanilan modellerin hicbiri 9.25 ile egitilmedi, simdiki esiklerin hicbiri
9.25'e bakilarak secilmedi.

Not: 9.25'in operator C'lerinin hepsi ilk altin kumeye alinmisti; bu orneklemde
C neredeyse yok. Kategori ve A/B tarafini olcer.

  python kalan_ornek.py
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np

ENVANTER = Path("yeni_envanter_925.csv")
ESKI_ALTIN = Path("altin_kume4")
HEDEF = Path("altin_kume7")
HEDEF_CSV = Path("altin_kume7.csv")
ADET = 2000


def main():
    if not ENVANTER.exists():
        print(f"HATA: {ENVANTER} yok"); return
    if HEDEF.exists():
        print(f"{HEDEF} zaten var, uzerine yazilmadi."); return
    eski = {f.name for f in ESKI_ALTIN.iterdir()} if ESKI_ALTIN.is_dir() else set()
    R = [r for r in csv.DictReader(open(ENVANTER, encoding="utf-8-sig"))
         if Path(r["yol"]).name not in eski and Path(r["yol"]).exists()
         and "OverTime" not in r["yol"]]
    print(f"9.25'in dokunulmamis gorselleri: {len(R)}")

    gruplar = defaultdict(list)
    for r in R:
        gruplar[(r["vardiya"], r["hat"])].append(r)
    rng = np.random.RandomState(725)
    secilen = []
    for k, g in gruplar.items():
        n = round(ADET * len(g) / len(R))
        secilen += [g[i] for i in rng.permutation(len(g))[:n]]

    HEDEF.mkdir()
    for r in secilen:
        shutil.copy2(r["yol"], HEDEF / Path(r["yol"]).name)
    with open(HEDEF_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(secilen[0].keys()))
        w.writeheader(); w.writerows(secilen)
    print(f"{len(secilen)} gorsel -> {HEDEF.resolve()}")
    print(f"liste: {HEDEF_CSV}")


if __name__ == "__main__":
    main()
