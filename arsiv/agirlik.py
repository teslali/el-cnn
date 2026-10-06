#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - Adim 4: Agirlik Karsilastirmasi + Sinif Bazli Esik

Uc farkli sinif agirliklandirmasi denenir:
  1) agirliksiz  - dogal dagilim
  2) yumusak     - dengelemenin karekoku
  3) dengeli     - class_weight='balanced' (onceki denemedeki hali)

Ardindan SINIF BASINA esik analizi yapilir: her kategori icin, guven esigi
yukseldikce precision nasil degisiyor ve kac gorsele karar verilebiliyor.

Amac: "hangi kategoriler, hangi esikte, kullanilabilir precision'a ulasiyor?"
Cunku hedef 19 kategoriyi birden cozmek degil; birkacini guvenle otomatiklestirip
gerisini operatore birakmak.

Kullanim:
    python agirlik.py
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
    from sklearn.metrics import precision_recall_fscore_support
except ImportError:
    print("HATA: scikit-learn kurulu degil.  pip install scikit-learn")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
GIRDI_BOYUT = 224
OZN_DIZIN = Path("ozellikler")
RAPOR_DIZIN = Path("raporlar")

IYI_KATEGORI = "zz其他Diger"
TEST_GUNLERI = {"20260908", "20260909"}
MIN_ORNEK = 300
ESIKLER = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
HEDEF_PRECISION = 0.95      # kullanilabilir kabul ettigimiz esik

C = "=" * 74


def kisa(ad, n=30):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def veri_yukle():
    ozn = OZN_DIZIN / f"ozellikler_{GIRDI_BOYUT}.npy"
    meta = OZN_DIZIN / f"meta_{GIRDI_BOYUT}.json"
    if not ozn.exists():
        print(f"HATA: {ozn} yok. Once oznitelik.py calistirin.")
        sys.exit(1)
    X = np.load(ozn)
    m = json.loads(meta.read_text(encoding="utf-8"))
    kay = m["kayitlar"]
    y = np.array([k["kategori"] for k in kay])
    gun = np.array([k["zaman"][:8] if k["zaman"] else "" for k in kay])
    yol = np.array([k["yol"] for k in kay])
    return X, y, gun, yol, Path(m["kok"])


def agirliklar(ytr, mod):
    """mod: 'yok' | 'yumusak' | 'dengeli'"""
    if mod == "yok":
        return None
    say = Counter(ytr)
    n, k = len(ytr), len(say)
    if mod == "dengeli":
        return {c: n / (k * say[c]) for c in say}
    if mod == "yumusak":
        return {c: (n / (k * say[c])) ** 0.5 for c in say}
    raise ValueError(mod)


def egit(Xtr, ytr, Xte, mod):
    olcek = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, C=1.0,
                           class_weight=agirliklar(ytr, mod))
    m.fit(olcek.transform(Xtr), ytr)
    P = m.predict_proba(olcek.transform(Xte))
    tah = m.classes_[P.argmax(axis=1)]
    guv = P.max(axis=1)
    return tah, guv


def ozet_satir(ad, yte, tah, guv):
    dog = float((tah == yte).mean())
    etk = sorted(set(yte) | set(tah))
    mp, mr, mf, _ = precision_recall_fscore_support(
        yte, tah, labels=etk, average="macro", zero_division=0)
    wp, _, _, _ = precision_recall_fscore_support(
        yte, tah, labels=etk, average="weighted", zero_division=0)
    print(f"  {ad:<12} {100*dog:>8.1f}% {mp:>10.3f} {mr:>9.3f} "
          f"{mf:>8.3f} {wp:>11.3f}")
    return mf


def sinif_detay(yte, tah, baslik):
    print(f"\n  {baslik}")
    print(f"  {'kategori':<32} {'prec':>6} {'recall':>7} {'f1':>6} {'adet':>6}")
    print("  " + "-" * 62)
    etk = sorted(set(yte) | set(tah))
    p, r, f1, s = precision_recall_fscore_support(
        yte, tah, labels=etk, zero_division=0)
    sirali = sorted(range(len(etk)), key=lambda i: -p[i])
    for i in sirali:
        if s[i] == 0:
            continue
        isaret = "  <-- iyi" if p[i] >= 0.85 else ""
        print(f"  {kisa(etk[i]):<32} {p[i]:>6.2f} {r[i]:>7.2f} "
              f"{f1[i]:>6.2f} {int(s[i]):>6}{isaret}")


def sinif_esik(yte, tah, guv, baslik):
    """Her kategori icin: esik yukseldikce precision ve kapsama."""
    print(f"\n{C}\n{baslik}\n{C}")
    print("Her satir: model o kategoriyi dediginde, o esikte ne kadar hakli?")
    print("'kapsama' = test kumesindeki kac gorsele o kategori atandi\n")

    kats = sorted(set(tah))
    for kat in kats:
        print(f"  {kisa(kat, 40)}")
        print(f"    {'esik':>6} {'karar':>7} {'precision':>11} {'yakalanan':>11}")
        print("    " + "-" * 40)
        gercek_toplam = int((yte == kat).sum())
        for e in ESIKLER:
            m = (tah == kat) & (guv >= e)
            n = int(m.sum())
            if n == 0:
                print(f"    {e:>6.2f} {0:>7} {'-':>11} {'-':>11}")
                continue
            prec = float((yte[m] == kat).mean())
            yakalanan = int((yte[m] == kat).sum())
            rec = yakalanan / max(gercek_toplam, 1)
            isaret = "  <<<" if prec >= HEDEF_PRECISION else ""
            print(f"    {e:>6.2f} {n:>7} {prec:>10.2f}  "
                  f"{rec:>10.2f}{isaret}")
        print()


def deney(X, y, gun, yol, ad, secili=None, ikili=False):
    print(f"\n\n{C}\n{ad}\n{C}")

    if secili is not None:
        m = np.isin(y, list(secili))
        X, y, gun, yol = X[m], y[m], gun[m], yol[m]
    if ikili:
        y = np.where(y == IYI_KATEGORI, "IYI", "KUSURLU")

    te = np.isin(gun, list(TEST_GUNLERI))
    Xtr, ytr, Xte, yte = X[~te], y[~te], X[te], y[te]
    print(f"  Egitim {len(Xtr)}   Test {len(Xte)}   Sinif {len(set(y))}")

    print(f"\n  {'agirlik':<12} {'dogruluk':>8} {'makro-prec':>10} "
          f"{'makro-rec':>9} {'makro-f1':>8} {'agirlikli-p':>11}")
    print("  " + "-" * 62)

    sonuc = {}
    for mod in ["yok", "yumusak", "dengeli"]:
        tah, guv = egit(Xtr, ytr, Xte, mod)
        mf = ozet_satir(mod, yte, tah, guv)
        sonuc[mod] = (tah, guv, mf)

    # en yuksek makro-precision veren modu sec
    en_iyi = max(sonuc, key=lambda k: precision_recall_fscore_support(
        yte, sonuc[k][0], labels=sorted(set(yte) | set(sonuc[k][0])),
        average="macro", zero_division=0)[0])
    print(f"\n  En yuksek makro-precision: '{en_iyi}'")

    tah, guv, _ = sonuc[en_iyi]
    sinif_detay(yte, tah, f"Sinif basina ({en_iyi} agirlik, esik yok):")
    return yte, tah, guv, yol[te], en_iyi


def main():
    X, y, gun, yol, kok = veri_yukle()
    RAPOR_DIZIN.mkdir(exist_ok=True)

    print(f"{C}\nAGIRLIK KARSILASTIRMASI + SINIF BAZLI ESIK\n{C}")
    print(f"Oznitelik : {X.shape}   Bolme: zamana gore")
    print(f"  egitim  : {sorted(set(gun) - TEST_GUNLERI)}")
    print(f"  test    : {sorted(TEST_GUNLERI)}")

    # ---------- ikili
    deney(X, y, gun, yol, "IKILI (iyi vs kusurlu)", ikili=True)

    # ---------- cok sinifli
    say = Counter(y)
    buyuk = {k for k, n in say.items() if n >= MIN_ORNEK}
    yte, tah, guv, yol_te, mod = deney(
        X, y, gun, yol, f"COK SINIFLI ({len(buyuk)} kategori)", secili=buyuk)

    sinif_esik(yte, tah, guv, f"SINIF BAZLI ESIK ANALIZI ({mod} agirlik)")

    # ---------- kullanilabilir kategoriler
    print(f"{C}\nSONUC: HANGI KATEGORILER OTOMATIKLESTIRILEBILIR?\n{C}")
    print(f"(hedef precision >= {HEDEF_PRECISION})\n")
    bulundu = []
    for kat in sorted(set(tah)):
        gercek_toplam = int((yte == kat).sum())
        for e in ESIKLER:
            m = (tah == kat) & (guv >= e)
            n = int(m.sum())
            if n < 10:
                continue
            prec = float((yte[m] == kat).mean())
            if prec >= HEDEF_PRECISION:
                yak = int((yte[m] == kat).sum())
                bulundu.append((kat, e, n, prec, yak, gercek_toplam))
                break
    if not bulundu:
        print("  Hicbir kategori hedefe ulasmadi.")
        print("  -> Dondurulmus oznitelik yetersiz. Ince ayar (fine-tuning)")
        print("     veya daha yuksek cozunurluk (384) denenmeli.")
    else:
        print(f"  {'kategori':<32} {'esik':>5} {'karar':>7} {'prec':>6} {'kapsam':>7}")
        print("  " + "-" * 62)
        for kat, e, n, p, yak, tot in bulundu:
            print(f"  {kisa(kat):<32} {e:>5.2f} {n:>7} {p:>6.2f} "
                  f"{yak}/{tot}")
        top = sum(b[2] for b in bulundu)
        print(f"\n  Toplam otomatik karar: {top} / {len(yte)} "
              f"(%{100*top/len(yte):.1f})")
        print(f"  Kalan {len(yte)-top} gorsel operatore gider.")

    # hatalari kaydet
    d = RAPOR_DIZIN / "hatalar_agirlik.csv"
    with open(d, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["yol", "gercek", "tahmin", "guven"])
        for i in range(len(yte)):
            if tah[i] != yte[i]:
                w.writerow([yol_te[i], yte[i], tah[i], f"{guv[i]:.3f}"])
    print(f"\n  Hatalar: {d}")
    print(C)


if __name__ == "__main__":
    main()
