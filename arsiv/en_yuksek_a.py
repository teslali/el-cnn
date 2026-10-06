#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B SIDDET MODELI - EN YUKSEK A OLASILIKLI GORSELLERI INCELE

SORUN
  ab_model.py uygula, 0.90 esigiyle 5368 gorselden sadece 3 tanesini
  A sinifi olarak isaretledi. Ortalama A olasiligi %0.2 - model neredeyse
  her seye "kesinlikle B" diyor.

  Ama gozle bakildiginda siyah nokta / siyah kenar klasorlerinde kayda deger
  sayida A sinifi urun var.

CEVABINI ARADIGIMIZ SORU
  Model bu hucreleri hic mi goremiyor, yoksa sadece esik mi yuksek?

NASIL
  ab_duzeltme.csv'yi A olasiligina gore siralar ve EN YUKSEK N tanesini
  ayri bir klasore kopyalar - esigi gecsin gecmesin.

  Bu gorsellere bakip su soruyu sorun:
    "Bunlarin icinde gercekten A sinifi olanlar var mi?"

  VARSA  -> esik sorunu. Esigi dusurerek cozulur.
  YOKSA  -> model bu ayrimi yapamiyor. Farkli yaklasim gerekir.

Kullanim:
    python en_yuksek_a.py /otomasyon_cikti/klasoru
    python en_yuksek_a.py /klasor --adet 100
"""

import sys
import csv
import shutil
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

CIKTI_AD = "_EN_YUKSEK_A"
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
C = "=" * 70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor", help="otomasyon_cikti klasoru")
    ap.add_argument("--adet", type=int, default=50)
    ap.add_argument("--csv", default=None,
                    help="ab_duzeltme.csv yolu (varsayilan: klasor icinde)")
    a = ap.parse_args()

    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    csv_yolu = Path(a.csv) if a.csv else kok / "ab_duzeltme.csv"
    if not csv_yolu.exists():
        print(f"HATA: {csv_yolu} yok.")
        print("Once su komutu calistirin:")
        print(f"  python ab_model.py uygula \"{kok}\"")
        sys.exit(1)

    # kayitlari oku
    kayitlar = []
    with open(csv_yolu, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                r["A_olasiligi"] = float(r["A_olasiligi"])
            except (ValueError, KeyError):
                continue
            kayitlar.append(r)

    if not kayitlar:
        print("HATA: csv bos veya okunamadi.")
        sys.exit(1)

    olaslar = np.array([r["A_olasiligi"] for r in kayitlar])
    print(f"{C}\nA OLASILIGI DAGILIMI\n{C}")
    print(f"Toplam gorsel : {len(kayitlar)}")
    print(f"Ortalama      : {olaslar.mean():.4f}")
    print(f"Medyan        : {np.median(olaslar):.4f}")
    print(f"En yuksek     : {olaslar.max():.4f}")
    print(f"\nEsik ustunde kac gorsel var:")
    for e in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
        n = int((olaslar >= e).sum())
        print(f"  >= {e:>4.2f} : {n:>6}  (%{100*n/len(kayitlar):.2f})")

    # dosyalari bul: klasor icinde ara
    print(f"\n{C}\nDOSYALAR ARANIYOR\n{C}")
    harita = {}
    for f in kok.rglob("*"):
        if f.is_file() and f.suffix.lower() in UZANTILAR:
            harita.setdefault(f.name, f)
    print(f"Klasorde bulunan gorsel: {len(harita)}")

    # sirala, en yuksek N
    kayitlar.sort(key=lambda r: -r["A_olasiligi"])
    secilen = kayitlar[:a.adet]

    hedef = kok / CIKTI_AD
    if hedef.exists():
        shutil.rmtree(hedef)
    hedef.mkdir()

    kopyalanan, bulunamayan = 0, 0
    for sira, r in enumerate(secilen, 1):
        ad = r["dosya"]
        kaynak = harita.get(ad)
        if kaynak is None:
            bulunamayan += 1
            continue
        yeni = hedef / f"{sira:03d}_a{r['A_olasiligi']:.4f}_{ad}"
        try:
            shutil.copy2(kaynak, yeni)
            kopyalanan += 1
        except Exception as e:
            print(f"  kopyalanamadi: {ad} ({e})")

    # talimat
    with open(hedef / "OKU.txt", "w", encoding="utf-8") as fh:
        fh.write("EN YUKSEK A OLASILIKLI GORSELLER\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("Dosyalar A olasiligina gore SIRALI.\n")
        fh.write("001 = modelin A sinifi olma ihtimalini EN YUKSEK gordugu.\n\n")
        fh.write("Dosya adindaki 'a0.4213' modelin A olasiligidir.\n\n")
        fh.write("SORU:\n")
        fh.write("  Bu gorsellerin icinde gercekten A sinifi olanlar var mi?\n\n")
        fh.write("SONUC:\n")
        fh.write("  Ilk 10-20'de cok sayida gercek A varsa\n")
        fh.write("    -> ESIK sorunu. Esigi dusurerek cozulur.\n\n")
        fh.write("  Hepsi gercekten B ise\n")
        fh.write("    -> Model bu ayrimi yapamiyor.\n")
        fh.write("       Farkli bir yaklasim gerekir.\n\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"{'sira':>5}  {'A_olasiligi':>12}  {'eski klasor':<30}\n")
        fh.write("-" * 56 + "\n")
        for sira, r in enumerate(secilen, 1):
            ek = r.get("eski_klasor", "")
            fh.write(f"{sira:>5}  {r['A_olasiligi']:>12.4f}  {ek[:30]:<30}\n")

    print(f"\n{C}\nHAZIR\n{C}")
    print(f"Kopyalanan : {kopyalanan}")
    if bulunamayan:
        print(f"Bulunamayan: {bulunamayan}  (dosya adi eslesmedi)")
    print(f"\nEn yuksek 10 A olasiligi:")
    for sira, r in enumerate(secilen[:10], 1):
        print(f"  {sira:>3}. {r['A_olasiligi']:.4f}  {r['dosya'][:52]}")

    kaynak_say = Counter(r.get("eski_klasor", "?") for r in secilen)
    print(f"\nBu {len(secilen)} gorsel hangi klasorden geliyor:")
    for k, n in kaynak_say.most_common():
        print(f"  {n:>4}  {k}")

    print(f"\nKlasor: {hedef}")
    print(f"\n  open \"{hedef}\"")
    print(f"\n{C}")
    print("OKU.txt dosyasini okuyun. Sirayla bakin: 001, 002, 003...")
    print("Gercek A sinifi olanlar nerede bitiyor, onu tespit edin.")


if __name__ == "__main__":
    main()
