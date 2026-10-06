#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C KLASORUNU UCE BOL (model calistirmadan, saniyeler)

Daha once tek C klasoruyle uretilmis bir ciktiyi, gunluk.py'nin simdiki
duzenine getirir. Kayit dosyasindaki olasiliklari kullanir:
  C dedektoru >= 0.90          -> C_supheli_1_yuksek
  degilse, A adayiysa          -> C_supheli_3 (yeniden degerlendir)
  degilse                      -> C_supheli_2_dusuk
Kurallar gunluk.py ile birebir ayni.

  python c_bol.py ~/Desktop/9.27_AI
"""

import sys
import csv
import shutil
from pathlib import Path

YUKSEK_ESIK = 0.90
C_YUKSEK = "C_supheli_1 隐裂压板疑似-高 (buyuk ihtimalle C)"
C_DUSUK = "C_supheli_2 隐裂压板疑似-低 (dusuk supheli)"
C_ADAY = "C_supheli_3 隐裂压板疑似-需复判 (yeniden degerlendir)"
YENI = {C_YUKSEK, C_DUSUK, C_ADAY}
ESKI_BOLUNMUS = {  # onceki surumun adlari -> yeni adlar
    "C_supheli_1_yuksek (buyuk ihtimalle C)": C_YUKSEK,
    "C_supheli_2_dusuk (catlak var mi bak)": C_DUSUK,
    "C_supheli_3_A_adayi (catlak yoksa A sinifina)": C_ADAY,
    "C_supheli_3 隐裂压板疑似-A级候选 (A adayiydi)": C_ADAY,
}


def c_klasoru_mu(ad):
    a = ad.lower()
    return (("supheli" in a or "隐裂" in ad or "压板" in ad)
            and ad not in YENI and ad not in ESKI_BOLUNMUS)


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python c_bol.py <cikti_klasoru>"); sys.exit(1)
    kok = Path(sys.argv[1]).expanduser()
    kayit_yolu = kok / "_kayit.csv"
    if not kayit_yolu.exists():
        print(f"HATA: {kayit_yolu} yok"); sys.exit(1)
    # onceden eski adlarla bolunmus klasorleri yeni adlara tasi
    for eski_ad, yeni_ad in ESKI_BOLUNMUS.items():
        e = kok / eski_ad
        if e.is_dir():
            (kok / yeni_ad).mkdir(exist_ok=True)
            for f in list(e.iterdir()):
                shutil.move(str(f), kok / yeni_ad / f.name)
            e.rmdir()
            print(f"yeniden adlandirildi: {eski_ad} -> {yeni_ad}")
            if (kok / "_kayit.csv").exists():
                sat = list(csv.DictReader(open(kok / "_kayit.csv", encoding="utf-8-sig")))
                for r in sat:
                    if r["klasor"] == eski_ad:
                        r["klasor"] = yeni_ad
                with open(kok / "_kayit.csv", "w", newline="", encoding="utf-8-sig") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(sat[0].keys()))
                    w.writeheader(); w.writerows(sat)
    eski = [d for d in kok.iterdir() if d.is_dir() and c_klasoru_mu(d.name)]
    if not eski:
        print("Bolunecek C klasoru bulunamadi (zaten bolunmus olabilir)."); return
    print("Bolunecek klasor(ler):", ", ".join(d.name for d in eski))

    satirlar = list(csv.DictReader(open(kayit_yolu, encoding="utf-8-sig")))
    bilgi = {r["dosya"]: r for r in satirlar}
    for ad in YENI:
        (kok / ad).mkdir(exist_ok=True)

    say = {ad: 0 for ad in YENI}
    bilinmeyen = 0
    for d in eski:
        for f in list(d.iterdir()):
            if not f.is_file():
                continue
            r = bilgi.get(f.name)
            if r is None:
                bilinmeyen += 1
                hedef = C_DUSUK
            else:
                cd = float(r.get("C_dedektor") or 0)
                aday = r.get("A_adayi_mi") == "evet"
                hedef = C_YUKSEK if cd >= YUKSEK_ESIK else (C_ADAY if aday else C_DUSUK)
                r["klasor"] = hedef
            shutil.move(str(f), kok / hedef / f.name)
            say[hedef] += 1
        if not any(d.iterdir()):
            d.rmdir()

    with open(kayit_yolu, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(satirlar[0].keys()))
        w.writeheader(); w.writerows(satirlar)

    for ad in (C_YUKSEK, C_DUSUK, C_ADAY):
        print(f"  {say[ad]:>5}  {ad}")
    if bilinmeyen:
        print(f"  (kayitta bulunamayan {bilinmeyen} gorsel 'dusuk' klasorune kondu)")


if __name__ == "__main__":
    main()
