#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - Adim 7: ESIK ANALIZI (ince ayarli model)

ince_ayar.py'nin urettigi model/tahminler.csv dosyasini okur ve asil soruyu
cevaplar: hangi kategoriler, hangi guven esiginde, kullanilabilir precision'a
ulasiyor ve bu kac gorseli otomatiklestiriyor?

Karsilastirma: dondurulmus oznitelikle %38.2 otomatik karar aliniyordu.

Kullanim:
    python esik.py
    python esik.py 0.90        (hedef precision'i degistir)
"""

import sys
import csv
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

MODEL_DIZIN = Path("model3")
TAHMIN_CSV = MODEL_DIZIN / "tahminler.csv"
RAPOR_DIZIN = Path("raporlar")

HEDEF_PRECISION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
ESIKLER = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]

C = "=" * 74


def kisa(ad, n=30):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def main():
    if not TAHMIN_CSV.exists():
        print(f"HATA: {TAHMIN_CSV} yok. Once ince_ayar.py calistirin.")
        sys.exit(1)

    yol, gercek, tahmin, guven = [], [], [], []
    with open(TAHMIN_CSV, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            yol.append(r["yol"])
            gercek.append(r["gercek"])
            tahmin.append(r["tahmin"])
            guven.append(float(r["guven"]))
    gercek = np.array(gercek)
    tahmin = np.array(tahmin)
    guven = np.array(guven)
    yol = np.array(yol)
    N = len(gercek)

    print(f"{C}\nESIK ANALIZI - INCE AYARLI MODEL\n{C}")
    print(f"Test gorseli   : {N}")
    print(f"Genel dogruluk : {100*(gercek==tahmin).mean():.1f}%")
    print(f"Hedef precision: {HEDEF_PRECISION:.2f}")

    kategoriler = sorted(set(gercek) | set(tahmin))

    # ------------------------------------------------------------------
    # Kategori bazli esik tablosu
    # ------------------------------------------------------------------
    print(f"\n{C}\nKATEGORI BAZLI ESIK TABLOSU\n{C}")
    print("precision = model o kategoriyi dediginde ne kadar hakli")
    print("kapsama   = o kategorinin kac ornegini yakaladi\n")

    en_iyi_esik = {}
    for kat in kategoriler:
        toplam = int((gercek == kat).sum())
        if toplam == 0:
            continue
        print(f"  {kisa(kat, 42)}   (test kumesinde {toplam} ornek)")
        print(f"    {'esik':>6} {'karar':>7} {'dogru':>7} "
              f"{'precision':>11} {'kapsama':>9}")
        print("    " + "-" * 46)
        bulundu = False
        for e in ESIKLER:
            m = (tahmin == kat) & (guven >= e)
            n = int(m.sum())
            if n == 0:
                print(f"    {e:>6.2f} {0:>7} {0:>7} {'-':>11} {'-':>9}")
                continue
            dogru = int((gercek[m] == kat).sum())
            prec = dogru / n
            kaps = dogru / toplam
            isaret = ""
            if prec >= HEDEF_PRECISION and n >= 10 and not bulundu:
                isaret = "  <<<"
                en_iyi_esik[kat] = (e, n, dogru, prec, kaps)
                bulundu = True
            print(f"    {e:>6.2f} {n:>7} {dogru:>7} {prec:>11.3f} "
                  f"{kaps:>9.2f}{isaret}")
        print()

    # ------------------------------------------------------------------
    # Otomatiklestirilebilir kategoriler
    # ------------------------------------------------------------------
    print(f"{C}\nSONUC: OTOMATIKLESTIRILEBILIR KATEGORILER\n{C}")
    if not en_iyi_esik:
        print(f"  Hicbir kategori {HEDEF_PRECISION:.2f} precision'a ulasmadi.")
        print(f"  Daha dusuk hedefle deneyin:  python esik.py 0.90")
    else:
        print(f"  {'kategori':<32} {'esik':>5} {'karar':>7} {'prec':>6} "
              f"{'kapsama':>9}")
        print("  " + "-" * 64)
        toplam_karar = 0
        toplam_dogru = 0
        for kat, (e, n, d, p, k) in sorted(
                en_iyi_esik.items(), key=lambda kv: -kv[1][1]):
            print(f"  {kisa(kat):<32} {e:>5.2f} {n:>7} {p:>6.3f} "
                  f"{d}/{int((gercek==kat).sum())}")
            toplam_karar += n
            toplam_dogru += d
        print("  " + "-" * 64)
        print(f"  TOPLAM otomatik karar : {toplam_karar} / {N} "
              f"(%{100*toplam_karar/N:.1f})")
        print(f"  Bunlarin dogrulugu    : %{100*toplam_dogru/max(toplam_karar,1):.1f}")
        print(f"  Operatore giden       : {N - toplam_karar} "
              f"(%{100*(N-toplam_karar)/N:.1f})")
        print(f"\n  Karsilastirma: dondurulmus oznitelikle %38.2 idi.")

    # ------------------------------------------------------------------
    # Gunluk olcege cevir
    # ------------------------------------------------------------------
    if en_iyi_esik:
        print(f"\n{C}\nGUNLUK OLCEK (8500 gorsel/gun varsayimi)\n{C}")
        oran = toplam_karar / N
        gunluk_oto = int(8500 * oran)
        gunluk_hata = int(gunluk_oto * (1 - toplam_dogru/max(toplam_karar,1)))
        print(f"  Otomatik sinifllanan : ~{gunluk_oto} gorsel/gun")
        print(f"  Bunlarda hata        : ~{gunluk_hata} gorsel/gun")
        print(f"  Operatore kalan      : ~{8500-gunluk_oto} gorsel/gun")

    # ------------------------------------------------------------------
    # Karisiklik: en sik hangi cift
    # ------------------------------------------------------------------
    print(f"\n{C}\nEN SIK KARISTIRILAN CIFTLER\n{C}")
    ciftler = Counter()
    for g, t in zip(gercek, tahmin):
        if g != t:
            ciftler[(g, t)] += 1
    for (g, t), n in ciftler.most_common(12):
        print(f"  {n:>5}x  {kisa(g,26):<28} -> {kisa(t,26)}")

    # ------------------------------------------------------------------
    # Kategori bazinda kacirilanlar (dusuk guvenli dogrular)
    # ------------------------------------------------------------------
    RAPOR_DIZIN.mkdir(exist_ok=True)
    d = RAPOR_DIZIN / "esik_hatalari.csv"
    with open(d, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["yol", "gercek", "tahmin", "guven", "durum"])
        for i in range(N):
            if gercek[i] != tahmin[i]:
                w.writerow([yol[i], gercek[i], tahmin[i],
                            f"{guven[i]:.4f}", "YANLIS"])
    print(f"\n  Yanlis tahminler: {d}")
    print(C)


if __name__ == "__main__":
    main()
