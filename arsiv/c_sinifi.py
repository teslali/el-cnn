#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - Adim 5: C SINIFI TESPITI

C sinifi kusurlar (gizli catlak, baski plakasi izi) digerlerinden farkli
degerlendiriliyor. Bunlari KACIRMAK pahali; yanlis alarm ucuz.

Bu yuzden burada oncelik PRECISION DEGIL, RECALL.
  - Kacirilan gizli catlak  -> hucre A/B'ye gider, module girer, sahada ariza
  - Yanlis alarm            -> operator bakar, "degilmis" der, gecer

Sistem "supheli olan her seyi isaretle, insan elesin" mantiginda calismali.

Iki deney:
  DENEY 1 - Ikili : C sinifi vs geri kalan hepsi
  DENEY 2 - Tekil : her C kusuru ayri ayri (gizli catlak, baski plakasi)

Ayrica 224 ve 384 oznitelikleri ve bunlarin BIRLESIMI karsilastirilir.

Kullanim:
    python c_sinifi.py
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
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
except ImportError:
    print("HATA: scikit-learn kurulu degil.  pip install scikit-learn")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
OZN_DIZIN = Path("ozellikler")
RAPOR_DIZIN = Path("raporlar")
BOYUTLAR = [224, 384]

# C sinifi kusurlar - klasor adinin ICINDE gecen anahtar kelimeler
C_ANAHTAR = {
    "gizli_catlak": "隐裂",      # w隐裂Gizli catlak
    "baski_plakasi": "压板",     # d压板Baski plakasi izi
}

TEST_GUNLERI = {"20260908", "20260909"}
# Recall oncelikli: dusuk esiklere bakiyoruz
ESIKLER = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
HEDEF_RECALL = 0.90          # kacirmamayi hedefledigimiz oran

C = "=" * 76


def kisa(ad, n=32):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def yukle(boyut):
    ozn = OZN_DIZIN / f"ozellikler_{boyut}.npy"
    meta = OZN_DIZIN / f"meta_{boyut}.json"
    if not ozn.exists() or not meta.exists():
        return None
    m = json.loads(meta.read_text(encoding="utf-8"))
    if m.get("tamamlanan", 0) < m.get("toplam", 0):
        print(f"UYARI: {boyut} oznitelikleri yarim kalmis, atlaniyor.")
        return None
    X = np.load(ozn)
    kay = m["kayitlar"]
    y = np.array([k["kategori"] for k in kay])
    gun = np.array([k["zaman"][:8] if k["zaman"] else "" for k in kay])
    yol = np.array([k["yol"] for k in kay])
    return X, y, gun, yol


def oznitelik_setleri():
    """Mevcut oznitelikleri ve birlesimlerini dondur."""
    veriler = {}
    for b in BOYUTLAR:
        v = yukle(b)
        if v is not None:
            veriler[b] = v
            print(f"  yuklendi: {b}x{b}  ->  {v[0].shape}")
    if not veriler:
        print("HATA: hic oznitelik dosyasi yok. Once oznitelik.py calistirin.")
        sys.exit(1)

    setler = {}
    for b, (X, y, gun, yol) in veriler.items():
        setler[f"{b}"] = (X, y, gun, yol)

    # birlesim
    if len(veriler) >= 2:
        bs = sorted(veriler.keys())
        X0, y0, g0, p0 = veriler[bs[0]]
        X1, y1, g1, p1 = veriler[bs[1]]
        if len(X0) == len(X1) and (y0 == y1).all():
            Xb = np.hstack([X0, X1])
            setler["birlesik"] = (Xb, y0, g0, p0)
            print(f"  birlesik ({'+'.join(map(str,bs))})  ->  {Xb.shape}")
        else:
            print("  UYARI: iki set eslesmediginden birlestirilemedi.")
    return setler


def c_maskesi(y, anahtar=None):
    """anahtar None ise tum C sinifi; degilse tek kusur."""
    if anahtar is None:
        m = np.zeros(len(y), bool)
        for a in C_ANAHTAR.values():
            m |= np.array([a in k for k in y])
        return m
    return np.array([anahtar in k for k in y])


def deney(setler, ad, anahtar=None):
    print(f"\n\n{C}\n{ad}\n{C}")

    # once mevcudiyet kontrolu
    ilk = next(iter(setler.values()))
    y_tum = ilk[1]
    m = c_maskesi(y_tum, anahtar)
    if m.sum() == 0:
        print("  Bu kusur veri setinde bulunamadi.")
        print("  Mevcut kategoriler:")
        for k in sorted(set(y_tum)):
            print(f"    {k}")
        return

    bulunan = sorted(set(y_tum[m]))
    print(f"  Eslesen klasorler: {bulunan}")
    print(f"  Toplam ornek     : {int(m.sum())}")

    sonuclar = {}
    for isim, (X, y, gun, yol) in setler.items():
        hedef = c_maskesi(y, anahtar)
        te = np.isin(gun, list(TEST_GUNLERI))

        ytr = hedef[~te].astype(int)
        yte = hedef[te].astype(int)
        if ytr.sum() < 5 or yte.sum() < 2:
            print(f"\n  [{isim}] Yetersiz ornek "
                  f"(egitim {int(ytr.sum())}, test {int(yte.sum())}) - atlandi.")
            continue

        olcek = StandardScaler().fit(X[~te])
        model = LogisticRegression(max_iter=3000, C=1.0,
                                   class_weight="balanced")
        model.fit(olcek.transform(X[~te]), ytr)
        olas = model.predict_proba(olcek.transform(X[te]))[:, 1]

        try:
            auc = roc_auc_score(yte, olas)
        except ValueError:
            auc = float("nan")
        sonuclar[isim] = (yte, olas, auc, yol[te])

    if not sonuclar:
        print("  Hicbir oznitelik seti ile deney yapilamadi.")
        return

    # ---- AUC karsilastirmasi
    print(f"\n  {'oznitelik':<12} {'egitim+':>8} {'test+':>7} {'AUC':>8}")
    print("  " + "-" * 40)
    for isim, (yte, olas, auc, _) in sonuclar.items():
        X, y, gun, yol = setler[isim]
        hedef = c_maskesi(y, anahtar)
        te = np.isin(gun, list(TEST_GUNLERI))
        print(f"  {isim:<12} {int(hedef[~te].sum()):>8} "
              f"{int(yte.sum()):>7} {auc:>8.3f}")

    print("\n  AUC yorumu: 0.5 = rastgele, 0.7 = zayif, 0.8 = kullanilabilir,")
    print("              0.9 = iyi, 0.95+ = cok iyi")

    # ---- en iyi set ile esik analizi
    en_iyi = max(sonuclar, key=lambda k: sonuclar[k][2]
                 if not np.isnan(sonuclar[k][2]) else -1)
    yte, olas, auc, yol_te = sonuclar[en_iyi]
    print(f"\n  En iyi oznitelik seti: '{en_iyi}' (AUC {auc:.3f})")

    print(f"\n{C}\n  RECALL ODAKLI ESIK ANALIZI ({en_iyi})\n{C}")
    print("  Amac: kacirmamak. Dusuk esik = genis ag = yuksek recall.")
    print("  'operatore giden' = model supheli dedigi icin insana yollanan sayi\n")
    print(f"  {'esik':>6} {'isaret':>8} {'yakalanan':>10} {'RECALL':>8} "
          f"{'precision':>10} {'op.yuku':>9}")
    print("  " + "-" * 58)

    toplam_poz = int(yte.sum())
    N = len(yte)
    hedefe_ulasan = None
    for e in ESIKLER:
        isaret = olas >= e
        n = int(isaret.sum())
        if n == 0:
            print(f"  {e:>6.2f} {0:>8} {0:>10} {0.0:>8.2f} {'-':>10} {0.0:>8.1f}%")
            continue
        yak = int((yte[isaret] == 1).sum())
        rec = yak / max(toplam_poz, 1)
        prec = yak / n
        yuk = 100 * n / N
        isa = "  <<<" if rec >= HEDEF_RECALL and hedefe_ulasan is None else ""
        if rec >= HEDEF_RECALL and hedefe_ulasan is None:
            hedefe_ulasan = (e, n, yak, rec, prec, yuk)
        print(f"  {e:>6.2f} {n:>8} {yak:>10} {rec:>8.2f} {prec:>10.3f} "
              f"{yuk:>8.1f}%{isa}")

    print(f"\n  Test kumesinde toplam {toplam_poz} adet gercek ornek var "
          f"({N} gorsel icinde).")

    if hedefe_ulasan:
        e, n, yak, rec, prec, yuk = hedefe_ulasan
        print(f"\n  SONUC: %{100*HEDEF_RECALL:.0f} recall icin esik {e:.2f}")
        print(f"    {toplam_poz} ornekten {yak} tanesi yakalanir")
        print(f"    Operatore {n} gorsel gider (test kumesinin %{yuk:.1f}'i)")
        if yuk < 25:
            print(f"    -> KULLANILABILIR: on eleme olarak is gorur")
        elif yuk < 50:
            print(f"    -> SINIRDA: yuk azalmasi mutevazi")
        else:
            print(f"    -> YETERSIZ: neredeyse her seyi isaretliyor")
    else:
        print(f"\n  SONUC: %{100*HEDEF_RECALL:.0f} recall hicbir esikte "
              f"yakalanamadi.")
        print("    -> Bu kusur mevcut ozniteliklerle guvenilir tespit edilemiyor.")

    # hatalari kaydet (kacirilanlar - en kritik olanlar)
    RAPOR_DIZIN.mkdir(exist_ok=True)
    etiket = anahtar if anahtar else "C_sinifi"
    d = RAPOR_DIZIN / f"kacirilan_{etiket}.csv"
    with open(d, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["yol", "olasilik", "durum"])
        sira = np.argsort(olas)
        for i in sira:
            if yte[i] == 1:
                w.writerow([yol_te[i], f"{olas[i]:.4f}", "GERCEK_POZITIF"])
    print(f"\n  Gercek ornekler olasiliga gore sirali: {d}")
    print("  (en ustteki = modelin en cok kacirdigi)")


def main():
    print(f"{C}\nC SINIFI KUSUR TESPITI\n{C}")
    print("C sinifi kusurlar B'den farkli degerlendiriliyor.")
    print("Bu yuzden burada oncelik RECALL (kacirmamak), precision degil.\n")
    print("Oznitelik setleri:")
    setler = oznitelik_setleri()

    ilk = next(iter(setler.values()))
    gun = ilk[2]
    print(f"\nBolme: zamana gore")
    print(f"  egitim : {sorted(set(gun) - TEST_GUNLERI)}")
    print(f"  test   : {sorted(TEST_GUNLERI)}")

    # kategori mevcudiyeti
    y = ilk[1]
    print(f"\nC sinifi klasorleri araniyor:")
    for ad, anahtar in C_ANAHTAR.items():
        esl = sorted({k for k in set(y) if anahtar in k})
        n = int(sum(anahtar in k for k in y))
        print(f"  {ad:<16} '{anahtar}' -> {esl if esl else 'BULUNAMADI'}  ({n} gorsel)")

    # DENEY 1: hepsi birlikte
    deney(setler, "DENEY 1 - C SINIFI (hepsi birlikte) vs GERI KALAN")

    # DENEY 2: tek tek
    for ad, anahtar in C_ANAHTAR.items():
        deney(setler, f"DENEY 2 - {ad.upper()} vs GERI KALAN", anahtar=anahtar)

    print(f"\n\n{C}\nBITTI\n{C}")
    print("Yorumlama:")
    print("  - AUC en onemli sayi: kusur digerlerinden ayrisiyor mu?")
    print("  - Esik tablosunda 'op.yuku' dusukken recall yuksekse is gorur.")
    print("  - Ornek sayisi cok az (ozellikle baski plakasi); sonuclar")
    print("    yon gostericidir, kesin degildir.")


if __name__ == "__main__":
    main()
