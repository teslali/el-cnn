#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C KLASORU NEREDEN DOLUYOR? (9.27_AI, model calistirmadan, saniyeler)

1) Kaynaklar: C klasorundeki her hucre hangi yoldan geldi, ve her yol
   gunun gercek C'lerinden kacini getirdi.
2) Olasilik dilimleri: dedektor olasiligina gore C klasoru; her dilimde
   gercek C orani. 'Yuksek' / 'dusuk' alt klasor sinirini buna gore seceriz.

Gercek C'ler 9.27 altin kumesinden (gunun butun C'leri orada oldugu icin
tahmini isabet = altindaki gercek C / tam gundeki hucre sayisi).

  python c_kaynak.py
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict

import altin_olcum as ao

TAM_GUN = Path.home() / "Desktop" / "9.27_AI" / "_kayit.csv"
ALTIN_ONAY = Path("cikti_altin_927")
ALTIN_HATA = Path.home() / "Desktop" / "altin6_hatalar"
ESIK_ADAY, ESIK_DIGER = 0.05, 0.15
DILIMLER = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 1.01]


def c_klasoru_mu(k):
    return "supheli" in ao.normal(k) or any(x in k for x in ao.C_KEYS)


def yol(r):
    aday = r["A_adayi_mi"] == "evet"
    cd = float(r.get("C_dedektor") or 0)
    if aday:
        if cd >= ESIK_DIGER:
            return "1 A adayi, dedektor >= 0.15"
        if cd >= ESIK_ADAY:
            return "2 A adayi, dedektor 0.05-0.15 (dusuk esik)"
        return "3 A adayi, catlak vetosu / kategori modeli"
    if cd >= ESIK_DIGER:
        return "4 A adayi degil, dedektor >= 0.15"
    return "5 A adayi degil, kategori modeli 'catlak' dedi"


def dilim(cd):
    for a, b in zip(DILIMLER[:-1], DILIMLER[1:]):
        if a <= cd < b:
            return f"{a:.2f}-{min(b, 1.0):.2f}"
    return "?"


def main():
    if not TAM_GUN.exists():
        print(f"HATA: {TAM_GUN} yok"); return
    tam = [r for r in csv.DictReader(open(TAM_GUN, encoding="utf-8-sig")) if c_klasoru_mu(r["klasor"])]

    etiket = {}
    for kok in (ALTIN_ONAY, ALTIN_HATA):
        if kok.is_dir():
            for d in kok.iterdir():
                if d.is_dir() and not d.name.startswith("_") and "emin" not in ao.normal(d.name):
                    k = ao.anahtar(d.name)
                    if k:
                        for f in d.iterdir():
                            etiket[f.name] = k
    altin = {r["dosya"]: r for r in csv.DictReader(open(ALTIN_ONAY / "_kayit.csv", encoding="utf-8-sig"))}
    gercekC = [altin[d] for d, k in etiket.items() if d in altin and (k in ao.C_KEYS or k == "C")]
    gC_klasorde = [r for r in gercekC if c_klasoru_mu(r["klasor"])]
    print(f"9.27 tam gun C klasoru: {len(tam)} hucre  |  gunun gercek C'leri: {len(gercekC)} "
          f"(C klasorunde: {len(gC_klasorde)})\n")

    # 1) kaynaklar
    t_say = Counter(yol(r) for r in tam)
    g_say = Counter(yol(r) for r in gC_klasorde)
    print(f"{'KAYNAK':<48} {'hucre':>6} {'gercek C':>9} {'isabet':>7}")
    print("-" * 74)
    for k in sorted(t_say):
        print(f"  {k[2:]:<46} {t_say[k]:>6} {g_say[k]:>9} "
              f"{'~%'+str(round(100*g_say[k]/max(t_say[k],1))):>7}")

    # 2) olasilik dilimleri
    t_d = Counter(dilim(float(r.get("C_dedektor") or 0)) for r in tam)
    g_d = Counter(dilim(float(r.get("C_dedektor") or 0)) for r in gC_klasorde)
    print(f"\n{'DEDEKTOR OLASILIGI':<20} {'hucre':>6} {'gercek C':>9} {'isabet':>7} "
          f"{'ustundeki hucre':>16} {'ustundeki C':>12}")
    print("-" * 76)
    siralar = sorted(t_d, key=lambda x: float(x.split("-")[0]))
    for i, k in enumerate(siralar):
        ust_h = sum(t_d[x] for x in siralar[i:])
        ust_c = sum(g_d[x] for x in siralar[i:])
        print(f"  {k:<18} {t_d[k]:>6} {g_d[k]:>9} {'~%'+str(round(100*g_d[k]/max(t_d[k],1))):>7} "
              f"{ust_h:>16} {ust_c:>12}")
    print("\n  'ustundeki': bu dilim ve daha yuksek olasilikli hucreler. Ornegin 0.50")
    print("  satiri, sinir 0.50 secilirse 'yuksek' klasorune kac hucre ve kac gercek C")
    print("  gidecegini gosterir.")


if __name__ == "__main__":
    main()
