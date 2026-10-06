#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SINIR CIFTLERI INCELEMESI

Kategori modelinin karistirdigi ciftlerden ornek cikarir. Model sinirlari
ancak sinirdan etiketli ornek gorerek ogrenir; bu ornekler bir sonraki
kategori egitiminin en degerli verisi.

Kaynak: ayrim_gruplar.csv, operator ile modelin FARKLI B kategorisi dedigi
hucreler. Her ciftten iki yonde toplam en fazla ORNEK gorsel.

Ciftler (altin kume olcumlerinden):
  siyah nokta  <-> siyah kenar     siyah noktanin tastigi yer
  siyah nokta  <-> hava akisi
  siyah nokta  <-> kirik grid
  duzensiz cizik <-> baski kaymasi cizik klasorunu kirleten
  duzensiz cizik <-> siyah kenar
  duzensiz cizik <-> kirik grid
  duzensiz cizik <-> hava akisi

INCELEME: onceki gibi. Her gorsele bak, gercek kategorisine gore
~/Desktop/yeni_duzeltmeler/<kategori> klasorune TASI. Emin olmadiklarin
"emin degilim" klasorune. Daha once inceledigin gorseller tekrar gelmez.

  python cift_inceleme.py
"""

import csv
import re
import shutil
from pathlib import Path
from collections import Counter

import numpy as np

GRUPLAR = Path("ayrim_gruplar.csv")
DUZELTMELER = Path.home() / "Desktop" / "yeni_duzeltmeler"
CIKTI = Path("cift_inceleme")
ORNEK = 90
ONEK = re.compile(r"^op-.*?_m-.*?_a\d\.\d+_")

AD = {"黑斑": "siyahnokta", "黑边": "siyahkenar", "气流": "havaakisi",
      "断栅": "kirikgrid", "不规则": "cizik", "偏移": "baskikaymasi"}
CIFTLER = [("黑斑", "黑边"), ("黑斑", "气流"), ("黑斑", "断栅"),
           ("不规则", "偏移"), ("不规则", "黑边"), ("不规则", "断栅"), ("不规则", "气流")]


def main():
    if not GRUPLAR.exists():
        print(f"HATA: {GRUPLAR} yok."); return
    gorulmus = set()
    if DUZELTMELER.is_dir():
        for f in DUZELTMELER.rglob("*"):
            if f.is_file():
                gorulmus.add(ONEK.sub("", f.name))
    R = [r for r in csv.DictReader(open(GRUPLAR, encoding="utf-8-sig"))
         if r["grup"] == "kat_anlasmazlik"
         and "OverTime" not in r["yol"]
         and Path(r["yol"]).name not in gorulmus]
    if CIKTI.exists():
        shutil.rmtree(CIKTI)
    rng = np.random.RandomState(1)
    toplam = 0
    print(f"{'cift':<32} {'havuz':>7} {'secilen':>8}")
    for a, b in CIFTLER:
        havuz = [r for r in R if {r["operator"], r["model_kat"]} == {a, b}]
        sec = [havuz[i] for i in rng.permutation(len(havuz))[:ORNEK]]
        kl = CIKTI / f"{AD[a]}_{AD[b]}"
        kl.mkdir(parents=True)
        for r in sec:
            p = Path(r["yol"])
            ad = (f"op-{AD.get(r['operator'], r['operator'])}_"
                  f"m-{AD.get(r['model_kat'], r['model_kat'])}_"
                  f"a{float(r['A_olas']):.2f}_{p.name}")
            try:
                shutil.copy2(p, kl / ad)
            except Exception:
                pass
        print(f"  {AD[a]+' <-> '+AD[b]:<30} {len(havuz):>7} {len(sec):>8}")
        toplam += len(sec)
    print(f"\nToplam {toplam} gorsel -> {CIKTI.resolve()}")
    print("Her gorseli gercek kategorisine gore ~/Desktop/yeni_duzeltmeler/<kategori>")
    print("klasorune TASI (onceki incelemeyle ayni klasorler).")
    print(f"\n  open {CIKTI.resolve()}")


if __name__ == "__main__":
    main()
