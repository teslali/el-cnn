#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C ESIGI DENEMESI (model calistirmadan, saniyeler)

Sistem her gorsel icin C dedektoru olasiligini _kayit.csv'ye yaziyor.
Farkli esiklerde C klasorunun ne olacagi buradan hesaplanir:

  9.27 tam gun (9.27_AI)  -> C klasoru kac hucre olur (kesin)
  9.27 altin kume (inceledigin) -> gunun gercek C'lerinden kaci yakalanir,
                                   kacanlar A adayi mi (A'ya sizma riski)

Iki esik var:
  A adaylarinda (su an 0.05) - dusuk, sizmaya karsi
  digerlerinde  (su an 0.15)

Not: kategori modelinin catlak/plaka dedigi ve catlak vetosuna takilan
hucreler dedektor esiginden bagimsiz olarak C klasorunde kalir; hesaba katildi.

  python c_esik_deneme.py
"""

import csv
from pathlib import Path

import altin_olcum as ao

TAM_GUN = Path.home() / "Desktop" / "9.27_AI" / "_kayit.csv"
ALTIN_ONAY = Path("cikti_altin_927")
ALTIN_HATA = Path.home() / "Desktop" / "altin6_hatalar"
ALTIN_KAYIT = ALTIN_ONAY / "_kayit.csv"
SIMDI = (0.05, 0.15)
DENEMELER = [(0.05, 0.15), (0.05, 0.25), (0.05, 0.40),
             (0.10, 0.15), (0.10, 0.25), (0.10, 0.40),
             (0.15, 0.25), (0.15, 0.40), (0.20, 0.40), (0.30, 0.50)]


def c_klasoru_mu(k):
    return "supheli" in ao.normal(k) or any(x in k for x in ao.C_KEYS)


def oku(yol):
    R = []
    for r in csv.DictReader(open(yol, encoding="utf-8-sig")):
        aday = r["A_adayi_mi"] == "evet"
        cd = float(r.get("C_dedektor") or 0)
        c_kl = c_klasoru_mu(r["klasor"])
        # dedektor disinda bir yoldan C klasorune gelmis mi (kategori / veto)
        dedektor_esigi = SIMDI[0] if aday else SIMDI[1]
        sabit = c_kl and cd < dedektor_esigi
        R.append({"dosya": r["dosya"], "aday": aday, "cd": cd, "sabit": sabit,
                  "a_olas": float(r["A_olasiligi"])})
    return R


def c_de_mi(r, ta, tg):
    return r["sabit"] or r["cd"] >= (ta if r["aday"] else tg)


def main():
    if not TAM_GUN.exists():
        print(f"HATA: {TAM_GUN} yok"); return
    tam = oku(TAM_GUN)

    etiket = {}
    for kok in (ALTIN_ONAY, ALTIN_HATA):
        if kok.is_dir():
            for d in kok.iterdir():
                if d.is_dir() and not d.name.startswith("_") and "emin" not in ao.normal(d.name):
                    k = ao.anahtar(d.name)
                    if k:
                        for f in d.iterdir():
                            etiket[f.name] = k
    altin = [r for r in oku(ALTIN_KAYIT) if r["dosya"] in etiket]
    gC = [r for r in altin if etiket[r["dosya"]] in ao.C_KEYS or etiket[r["dosya"]] == "C"]
    print(f"9.27 tam gun: {len(tam)} gorsel  |  altin kumede gercek C: {len(gC)}")
    print(f"(su anki ayar: A adaylarinda {SIMDI[0]}, digerlerinde {SIMDI[1]})\n")

    print(f"  {'A adayi':>8} {'diger':>6} | {'C klasoru':>10} {'yakalanan C':>12} "
          f"{'tahmini isabet':>15} | {'kacan C':>8} {'A adayi olan':>13}")
    print("  " + "-" * 84)
    for ta, tg in DENEMELER:
        boyut = sum(c_de_mi(r, ta, tg) for r in tam)
        yak = sum(c_de_mi(r, ta, tg) for r in gC)
        kacan = [r for r in gC if not c_de_mi(r, ta, tg)]
        risk = sum(r["aday"] for r in kacan)
        isabet = 100 * yak / max(boyut, 1)
        isaret = "  <- su an" if (ta, tg) == SIMDI else ""
        uyari = "  !! A'ya sizma riski" if risk else ""
        print(f"  {ta:>8.2f} {tg:>6.2f} | {boyut:>10} {yak:>7}/{len(gC):<4} "
              f"{'~%'+str(round(isabet)):>15} | {len(kacan):>8} {risk:>13}{isaret}{uyari}")
    print("\n  tahmini isabet: yakalanan gercek C / C klasoru buyuklugu (gunun C'lerinin")
    print("  hepsinin altin kumede oldugu varsayimiyla)")
    print("  A adayi olan kacan C: A klasorune gidebilir (ikinci gorus yakalamazsa)")


if __name__ == "__main__":
    main()
