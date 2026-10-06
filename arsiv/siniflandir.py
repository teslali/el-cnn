#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - Adim 3: Siniflandirma Denemesi

Onceden cikarilmis oznitelikler uzerinde iki deney yapar:
  ASAMA 1 - Ikili   : Diger (iyi) vs Kusurlu
  ASAMA 2 - Cok sinifli : 300+ ornegi olan kategoriler

Bolme ZAMANA gore yapilir (rastgele DEGIL): ilk gunler egitim, son gunler test.
Sebep: ayni partiden saniyeler arayla cekilmis gorseller birbirine cok benzer;
rastgele bolme sonucu yaniltici derecede iyi gosterir.

Kullanim:
    python siniflandir.py
"""

import sys
import json
import csv
from pathlib import Path
from collections import Counter

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
except ImportError:
    print("HATA: scikit-learn kurulu degil.  pip install scikit-learn")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
GIRDI_BOYUT = 224
OZN_DIZIN = Path("ozellikler")
RAPOR_DIZIN = Path("raporlar")

IYI_KATEGORI = "zz其他Diger"        # iyi hucrelerin atildigi klasor
TEST_GUNLERI = {"20260907", "20260908"}   # son 2 gun test
MIN_ORNEK = 300                     # cok sinifli asamada alt sinir
GUVEN_ESIKLERI = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

C = "=" * 68


def kisa(ad, n=34):
    """Kategori adini okunur kisalt (Cince onek + Turkce)."""
    t = ad[1:] if ad[:1].isalpha() else ad
    return (t[:n]).strip()


def veri_yukle():
    ozn_yol = OZN_DIZIN / f"ozellikler_{GIRDI_BOYUT}.npy"
    meta_yol = OZN_DIZIN / f"meta_{GIRDI_BOYUT}.json"
    if not ozn_yol.exists():
        print(f"HATA: {ozn_yol} yok. Once oznitelik.py calistirin.")
        sys.exit(1)

    X = np.load(ozn_yol)
    meta = json.loads(meta_yol.read_text(encoding="utf-8"))
    if meta.get("tamamlanan", 0) < meta.get("toplam", 0):
        print("HATA: oznitelik cikarma yarim kalmis. oznitelik.py'yi tekrar calistirin.")
        sys.exit(1)

    kayitlar = meta["kayitlar"]
    y = np.array([k["kategori"] for k in kayitlar])
    gun = np.array([k["zaman"][:8] if k["zaman"] else "" for k in kayitlar])
    yol = np.array([k["yol"] for k in kayitlar])
    return X, y, gun, yol


def esik_analizi(model, Xte, yte, etiketler):
    """Guven esigi yukseldikce precision/kapsama nasil degisiyor."""
    olasilik = model.predict_proba(Xte)
    tahmin = model.classes_[olasilik.argmax(axis=1)]
    guven = olasilik.max(axis=1)

    print(f"\n  {'esik':>6} {'kapsama':>9} {'karar':>8} {'dogruluk':>10}")
    print("  " + "-" * 38)
    for e in GUVEN_ESIKLERI:
        m = guven >= e
        n = int(m.sum())
        if n == 0:
            print(f"  {e:>6.2f} {0:>8.1f}% {0:>8} {'-':>10}")
            continue
        dog = float((tahmin[m] == yte[m]).mean())
        print(f"  {e:>6.2f} {100*n/len(yte):>8.1f}% {n:>8} {100*dog:>9.1f}%")
    return tahmin, guven


def sinif_raporu(yte, tahmin, baslik):
    print(f"\n  {baslik}")
    print(f"  {'kategori':<36} {'prec':>6} {'recall':>7} {'f1':>6} {'adet':>6}")
    print("  " + "-" * 64)
    etk = sorted(set(yte) | set(tahmin))
    p, r, f1, s = precision_recall_fscore_support(
        yte, tahmin, labels=etk, zero_division=0)
    for i, e in enumerate(etk):
        if s[i] == 0 and p[i] == 0:
            continue
        print(f"  {kisa(e):<36} {p[i]:>6.2f} {r[i]:>7.2f} {f1[i]:>6.2f} {int(s[i]):>6}")
    mp, mr, mf, _ = precision_recall_fscore_support(
        yte, tahmin, labels=etk, average="macro", zero_division=0)
    print("  " + "-" * 64)
    print(f"  {'MAKRO ORTALAMA':<36} {mp:>6.2f} {mr:>7.2f} {mf:>6.2f}")
    return mf


def karisiklik_matrisi(yte, tahmin, etiketler):
    cm = confusion_matrix(yte, tahmin, labels=etiketler)
    print(f"\n  Karisiklik matrisi (satir=gercek, sutun=tahmin):")
    kis = [kisa(e, 14) for e in etiketler]
    print("       " + " ".join(f"{k[:6]:>6}" for k in kis))
    for i, e in enumerate(etiketler):
        satir = " ".join(f"{v:>6}" for v in cm[i])
        print(f"  {kis[i][:5]:<5} {satir}")
    # en sik karistirilan ciftler
    print("\n  En sik karistirilan ciftler:")
    ciftler = []
    for i in range(len(etiketler)):
        for j in range(len(etiketler)):
            if i != j and cm[i, j] > 0:
                ciftler.append((cm[i, j], etiketler[i], etiketler[j]))
    ciftler.sort(reverse=True)
    for n, g, t in ciftler[:8]:
        print(f"    {n:>5}x  {kisa(g,26):<28} -> {kisa(t,26)}")


def deney(X, y, gun, yol, ad, secili=None, ikili=False):
    print(f"\n\n{C}\n{ad}\n{C}")

    if secili is not None:
        m = np.isin(y, list(secili))
        X, y, gun, yol = X[m], y[m], gun[m], yol[m]

    if ikili:
        y = np.where(y == IYI_KATEGORI, "IYI", "KUSURLU")

    test_m = np.isin(gun, list(TEST_GUNLERI))
    Xtr, ytr = X[~test_m], y[~test_m]
    Xte, yte = X[test_m], y[test_m]

    if len(Xtr) == 0 or len(Xte) == 0:
        print("  HATA: egitim veya test kumesi bos.")
        return

    print(f"  Egitim : {len(Xtr):>6}   Test : {len(Xte):>6}")
    print(f"  Sinif  : {len(set(y))}")
    print(f"\n  Egitim dagilimi:")
    for k, n in Counter(ytr).most_common():
        print(f"    {kisa(k):<36} {n:>6}")
    print(f"  Test dagilimi:")
    for k, n in Counter(yte).most_common():
        print(f"    {kisa(k):<36} {n:>6}")

    olcek = StandardScaler().fit(Xtr)
    model = LogisticRegression(
        max_iter=2000, class_weight="balanced", n_jobs=-1, C=1.0)
    model.fit(olcek.transform(Xtr), ytr)

    Xte_s = olcek.transform(Xte)
    tahmin, guven = esik_analizi(model, Xte_s, yte, sorted(set(y)))

    sinif_raporu(yte, tahmin, "Sinif basina sonuclar (esik yok):")

    etiketler = sorted(set(yte))
    if len(etiketler) <= 10:
        karisiklik_matrisi(yte, tahmin, etiketler)

    # hatali ornekleri kaydet
    RAPOR_DIZIN.mkdir(exist_ok=True)
    dosya = RAPOR_DIZIN / f"hatalar_{ad.split()[0].lower()}.csv"
    with open(dosya, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["yol", "gercek", "tahmin", "guven"])
        for i in range(len(yte)):
            if tahmin[i] != yte[i]:
                w.writerow([yol[test_m][i], yte[i], tahmin[i], f"{guven[i]:.3f}"])
    print(f"\n  Hatali ornekler: {dosya}")


def main():
    X, y, gun, yol = veri_yukle()
    print(f"{C}\nEL KUSUR SINIFLANDIRMA - FIZIBILITE\n{C}")
    print(f"Oznitelik   : {X.shape}  (ResNet18, {GIRDI_BOYUT}x{GIRDI_BOYUT})")
    print(f"Bolme       : zamana gore")
    print(f"  egitim    : {sorted(set(gun) - TEST_GUNLERI)}")
    print(f"  test      : {sorted(TEST_GUNLERI)}")

    if IYI_KATEGORI not in set(y):
        print(f"\nUYARI: '{IYI_KATEGORI}' klasoru bulunamadi.")
        print("Mevcut kategoriler:")
        for k in sorted(set(y)):
            print(f"   {k}")
        print("\nsiniflandir.py icindeki IYI_KATEGORI degerini duzeltin.")
        sys.exit(1)

    # ---- ASAMA 1: ikili
    deney(X, y, gun, yol, "ASAMA 1 - IKILI (iyi vs kusurlu)", ikili=True)

    # ---- ASAMA 2: cok sinifli, 300+ ornek
    sayim = Counter(y)
    buyuk = {k for k, n in sayim.items() if n >= MIN_ORNEK}
    print(f"\n\n{C}")
    print(f"{MIN_ORNEK}+ ornegi olan kategoriler: {len(buyuk)}")
    for k in sorted(buyuk, key=lambda z: -sayim[z]):
        print(f"   {sayim[k]:>6}  {kisa(k)}")
    print(C)

    deney(X, y, gun, yol,
          f"ASAMA 2 - COK SINIFLI ({len(buyuk)} kategori)", secili=buyuk)

    print(f"\n\n{C}\nBITTI\n{C}")
    print("Yorumlama:")
    print("  - Onemli olan MAKRO ORTALAMA ve sinif basina precision.")
    print("  - Genel dogruluk yaniltici: model her seye en buyuk sinifi dese")
    print("    ikili asamada %84 alir.")
    print("  - Esik tablosu: yuksek esikte precision artiyorsa cekingen bir")
    print("    sistem kurulabilir demektir.")


if __name__ == "__main__":
    main()
