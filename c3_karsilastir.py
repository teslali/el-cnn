#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C v2 mi v3 mu? - ADIL KARSILASTIRMA (model calistirmadan, saniyeler)

Ayni 9.27 sinav hucrelerinde, AYNI SAYIDA C KACIRIRKEN hangi modelin C klasoru
daha kucuk? Iki model icin de genis bir esik izgarasi taranir; her "kacan C"
duzeyinde en kucuk C klasorunu veren esik cifti secilir.

v3 olasiliklari: egitimin kaydettigi model_c3/sinav_tur<N>.npy
v2 olasiliklari: cikti_altin_927/_kayit.csv (C_dedektor)

c_dedektor3.py ile ayni klasorde calistirin.
  python c3_karsilastir.py
"""

from pathlib import Path

import numpy as np

import c_dedektor3 as c3

TA = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
TG = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
KACAN_DUZEY = [0, 1, 2, 3, 4, 5, 7, 10]


def tara(p, g, aday, olcek):
    """her kacan-C duzeyi icin: (en kucuk klasor, esikler, kacan A adayi)"""
    en = {}
    for ta in TA:
        for tg in TG:
            if ta > tg:
                continue
            sec = np.where(aday, p >= ta, p >= tg)
            kacan = int((~sec & g).sum())
            kacan_aday = int((~sec & g & aday).sum())
            boyut = int((sec & g).sum() + (sec & ~g).sum() * olcek)
            for k in KACAN_DUZEY:
                if kacan <= k and (k not in en or boyut < en[k][0]):
                    en[k] = (boyut, ta, tg, kacan_aday, kacan)
    return en


def main():
    yollar, g, aday, p2, olcek = c3.sinav_hazirla()
    print(f"9.27 sinav: {len(g)} hucre, {int(g.sum())} gercek C\n")
    modeller = {"v2": p2}
    for t in (1, 2, 3):
        f = c3.CIKTI / f"sinav_tur{t}.npy"
        if f.exists():
            modeller[f"v3 tur{t}"] = np.load(f)
    sonuc = {ad: tara(p, g, aday, olcek) for ad, p in modeller.items()}

    print("En fazla bu kadar C kacirirken ulasilabilecek EN KUCUK C klasoru (tam gun tahmini):")
    print(f"  {'kacan C <=':>10} | " + " | ".join(f"{ad:^24}" for ad in modeller))
    for k in KACAN_DUZEY:
        satir = f"  {k:>10} | "
        parca = []
        for ad in modeller:
            s = sonuc[ad].get(k)
            if s:
                uyari = "!" if s[3] else " "
                parca.append(f"{s[0]:>5} ({s[1]:.2f}/{s[2]:.2f}) A:{s[3]}{uyari}")
            else:
                parca.append(f"{'-':^24}")
        print(satir + " | ".join(parca))
    print("\n  (esikler: A adaylarinda / digerlerinde)  A: kacan C'lerden kaci A adayi")
    print("  Ayni satirda daha kucuk sayi = o kacak duzeyinde daha az yanlis alarm.")
    print("  Not: tam gun tahmini gercek C klasorunden ~%25 kucuk cikiyor (kategori")
    print("  modeli yolu ve orneklem); iki model icin ayni oldugu icin karsilastirma adil.")


if __name__ == "__main__":
    main()
