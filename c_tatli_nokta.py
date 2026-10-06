#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C ESIGI - TATLI NOKTA (model calistirmadan, saniyeler)

Iki tur yanlis yerlesim:
  - yanlis alarm : C olmayan hucre C klasorune gidiyor
  - kacan C      : gercek C baska klasore gidiyor
Toplam ceza = yanlis alarm + AGIRLIK x kacan C
AGIRLIK: "bir C'yi kacirmak kac yanlis alarma esit?" Birkac deger icin
en iyi esik cifti bulunur; sen fabrikanin onceligine uyani secersin.

Veri (C dedektoru v2 ile calismis gunler):
  kacan C     : 9.26 + 9.27 altin kumelerindeki gercek C'ler (gunlerin tum C'leri)
  yanlis alarm: 9.27 tam gun (9.27_AI) - C klasoru buyuklugu kesin
Sonuclar 9.27 gunune olceklenmis olarak verilir.

  python c_tatli_nokta.py
"""

import csv
from pathlib import Path

import altin_olcum as ao

D = Path.home() / "Desktop"
TAM_GUN = D / "9.27_AI" / "_kayit.csv"
ALTINLAR = [  # (onay, duzeltmeler, v2 ile calismis kayit)
    (Path("cikti_altin_926"), D / "altin5_hatalar", Path("cikti_altin_926b/_kayit.csv")),
    (Path("cikti_altin_927"), D / "altin6_hatalar", Path("cikti_altin_927/_kayit.csv")),
]
SIMDI = (0.05, 0.15)
TA = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
TG = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
AGIRLIKLAR = [1, 3, 5, 10, 20]


def c_klasoru_mu(k):
    return "supheli" in ao.normal(k) or any(x in k for x in ao.C_KEYS)


def oku(yol):
    R = []
    for r in csv.DictReader(open(yol, encoding="utf-8-sig")):
        aday = r["A_adayi_mi"] == "evet"
        cd = float(r.get("C_dedektor") or 0)
        sabit = c_klasoru_mu(r["klasor"]) and cd < (SIMDI[0] if aday else SIMDI[1])
        R.append({"dosya": r["dosya"], "aday": aday, "cd": cd, "sabit": sabit})
    return R


def c_de(r, ta, tg):
    return r["sabit"] or r["cd"] >= (ta if r["aday"] else tg)


def etiketler(onay, hatalar):
    et = {}
    for kok in (onay, hatalar):
        if kok.is_dir():
            for d in kok.iterdir():
                if d.is_dir() and not d.name.startswith("_") and "emin" not in ao.normal(d.name):
                    k = ao.anahtar(d.name)
                    if k:
                        for f in d.iterdir():
                            et[f.name] = k
    return et


def main():
    tam = oku(TAM_GUN)
    gercekC = []
    for onay, hatalar, kayit in ALTINLAR:
        if not kayit.exists():
            print(f"atlandi: {kayit} yok"); continue
        et = etiketler(onay, hatalar)
        gercekC += [r for r in oku(kayit)
                    if r["dosya"] in et and (et[r["dosya"]] in ao.C_KEYS or et[r["dosya"]] == "C")]
    nC = len(gercekC)
    gunluk_C = 95                     # 9.27'nin gercek C sayisi (olcek icin)
    print(f"Gercek C (9.26 + 9.27): {nC}  |  9.27 tam gun: {len(tam)} gorsel\n")

    sonuc = []
    for ta in TA:
        for tg in TG:
            if ta > tg:
                continue
            boyut = sum(c_de(r, ta, tg) for r in tam)
            kacan = sum(not c_de(r, ta, tg) for r in gercekC)
            kacan_aday = sum((not c_de(r, ta, tg)) and r["aday"] for r in gercekC)
            kacan_gun = kacan / nC * gunluk_C            # 9.27 gunune olcekli
            yakalanan_gun = gunluk_C - kacan_gun
            yanlis = max(boyut - yakalanan_gun, 0)
            sonuc.append((ta, tg, boyut, kacan, kacan_aday, kacan_gun, yanlis))

    print(f"{'AGIRLIK':>8} | {'en iyi esik':>11} | {'C klasoru':>10} {'yanlis alarm':>13} "
          f"{'kacan C/gun':>12} {'kacan A adayi':>14}")
    print("-" * 80)
    for w in AGIRLIKLAR:
        en = min(sonuc, key=lambda s: s[6] + w * s[5])
        print(f"{w:>8} | {en[0]:.2f} / {en[1]:.2f} | {en[2]:>10} {en[6]:>13.0f} "
              f"{en[5]:>12.1f} {en[4]:>14}")
    s0 = next(s for s in sonuc if (s[0], s[1]) == SIMDI)
    print(f"{'su an':>8} | {SIMDI[0]:.2f} / {SIMDI[1]:.2f} | {s0[2]:>10} {s0[6]:>13.0f} "
          f"{s0[5]:>12.1f} {s0[4]:>14}")

    print(f"\nTum secenekler (C klasoru kucukten buyuge; kacan C 9.27 gunune olcekli):")
    print(f"  {'A adayi':>7} {'diger':>6} {'C klasoru':>10} {'yanlis':>7} {'kacan C/gun':>12} "
          f"{'kacan (2 gun)':>14} {'A adayi olan':>13}")
    pareto = []
    for s in sorted(sonuc, key=lambda s: s[2]):
        if not any(o[2] <= s[2] and o[3] < s[3] for o in sonuc) and \
                not any(p[2] == s[2] and p[3] == s[3] for p in pareto):
            pareto.append(s)
    for s in pareto:
        isaret = "  <- su an" if (s[0], s[1]) == SIMDI else ""
        print(f"  {s[0]:>7.2f} {s[1]:>6.2f} {s[2]:>10} {s[6]:>7.0f} {s[5]:>12.1f} "
              f"{s[3]:>9}/{nC:<4} {s[4]:>13}{isaret}")
    print("\n  Listede sadece 'verimli' secenekler var: ayni ya da daha kucuk C klasoruyle")
    print("  daha az C kaciran baska bir secenegi olanlar elendi.")


if __name__ == "__main__":
    main()
