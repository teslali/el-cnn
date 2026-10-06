#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YENI VERI ENVANTERI (9.15 - 9.22)

Klasor yapisi:  <gun>/<vardiya>/<bin>/<hat>/<kategori>/gorsel.jpg
                 9.15 / D(08-16) / 64  / 2A  / f黑斑Siyah leke... /

DOSYALARA DOKUNMAZ. Agaci tarar ve her gorsel icin tek satirlik kayit
cikarir: gun, vardiya, bin, hat, operator etiketi (kategori), dosya
adindaki makine etiketi ve zaman damgasi.

Ayrica:
  - gun / vardiya / hat bazinda dagilim
  - VARDIYA BAZINDA A ORANI: operatorlerin A/B davranisini karsilastirmak
    icin ilk isaret (bir vardiya belirgin az A diyorsa sinirda B'ye kayiyor
    olabilir)
  - gun bazinda C sayilari (gizli catlak, baski plakasi): C dedektorunun
    yeni yakiti
  - 2B ile cakisma, ayni dosyanin iki klasorde bulunmasi

ALTIN KUME (--altin-gun 9.22)
  O gunden vardiya ve hat dengeli rastgele N gorsel secilir ve
  altin_kume/ klasorune KOPYALANIR. Sonra gunluk.py ile ayiklanir, sen
  kontrol edersin; bu kume hicbir egitime girmez, tum modeller onunla
  olculur. O GUNUN TAMAMI egitimden haric tutulur (ayni gunun gorselleri
  birbirine benzer, sizinti yaratir).

Kullanim:
  python yeni_envanter.py ~/Desktop/fabrika-fotolar/yeni
  python yeni_envanter.py ~/Desktop/fabrika-fotolar/yeni \\
         --kaynak ~/Desktop/fabrika-fotolar/2B --altin-gun 9.22 --altin-adet 2000
"""

import re
import sys
import csv
import shutil
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
GUN_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
VARDIYA_RE = re.compile(r"^([DMN])\s*\(", re.I)
HAT_RE = re.compile(r"^\d[A-Z]$", re.I)
SAAT_RE = re.compile(r"^(\d{1,2})\s*-\s*(\d{1,2})$")      # 2 saatlik dilim: 8-10
BIN_RE = re.compile(r"^\d{2,3}$")
DOSYA_RE = re.compile(
    r"^(?P<hat>\d[A-Z])_(?P<id>\d+)_EL_plain_Bin(?P<bin>\d+)_Class\d+_"
    r"(?P<etiket>[A-Za-z0-9]+)_(?P<zaman>\d{14})")

A_ANAHTAR = "其他"
C_ANAHTARLAR = {"隐裂": "gizli catlak", "压板": "baski plakasi"}
YIL = "2026"
C = "=" * 72


def kisa(ad, n=30):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kok", help="9.15, 9.16 ... klasorlerinin bulundugu yer")
    ap.add_argument("--kaynak", default=None, help="2B klasoru (cakisma icin)")
    ap.add_argument("--altin-gun", default=None, help="ornek: 9.22")
    ap.add_argument("--altin-adet", type=int, default=2000)
    ap.add_argument("--cikti", default="yeni_envanter.csv")
    ap.add_argument("--altin-klasor", default="altin_kume")
    ap.add_argument("--altin-csv", default="altin_kume.csv")
    a = ap.parse_args()

    kok = Path(a.kok).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}"); sys.exit(1)

    print(f"{C}\nYENI VERI ENVANTERI\n{C}\nTaraniyor: {kok}")
    kayitlar, cozulemeyen, bos = [], [], 0
    for f in kok.rglob("*"):
        if not (f.is_file() and f.suffix.lower() in UZANTILAR):
            continue
        if f.name.startswith("._"):
            continue                       # macOS yardimci dosyasi
        parcalar = f.relative_to(kok).parts[:-1]
        gun = vardiya = bin_ = hat = saat = None
        for p in parcalar:
            if gun is None and GUN_RE.match(p):
                m = GUN_RE.match(p)
                gun = f"{YIL}{int(m.group(1)):02d}{int(m.group(2)):02d}"
            elif vardiya is None and VARDIYA_RE.match(p):
                vardiya = VARDIYA_RE.match(p).group(1).upper()
            elif hat is None and HAT_RE.match(p):
                hat = p.upper()
            elif saat is None and SAAT_RE.match(p):
                m = SAAT_RE.match(p)
                saat = f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"
            elif bin_ is None and BIN_RE.match(p):
                bin_ = p
        kategori = parcalar[-1] if parcalar else ""
        if not (gun and vardiya and hat and kategori):
            cozulemeyen.append(str(f.relative_to(kok))); continue
        try:
            if f.stat().st_size == 0:
                bos += 1; continue
        except OSError:
            bos += 1; continue
        m = DOSYA_RE.match(f.stem)
        etiketsiz = HAT_RE.match(kategori) is not None
        kayitlar.append({
            "yol": str(f), "gun": gun, "vardiya": vardiya, "bin": bin_ or "",
            "hat": hat, "saat": saat or "", "kategori": kategori,
            "makine_etiketi": m.group("etiket") if m else "",
            "hucre_id": m.group("id") if m else "",
            "dosya_zamani": m.group("zaman") if m else "",
            "kume": "etiketsiz" if etiketsiz else "egitim_adayi",
        })

    if not kayitlar:
        print("HATA: hic gorsel cozulemedi. Klasor yapisini kontrol edin.")
        for c in cozulemeyen[:5]:
            print(f"   {c}")
        sys.exit(1)

    N = len(kayitlar)
    print(f"  gorsel: {N}   cozulemeyen yol: {len(cozulemeyen)}   bos dosya: {bos}")

    # ---------------- ayni dosya birden fazla klasorde
    ad_say = Counter(Path(k["yol"]).name for k in kayitlar)
    cift = {ad for ad, n in ad_say.items() if n > 1}
    cift_farkli = 0
    if cift:
        etiketler = defaultdict(set)
        for k in kayitlar:
            if Path(k["yol"]).name in cift:
                etiketler[Path(k["yol"]).name].add(k["kategori"])
        cift_farkli = sum(1 for v in etiketler.values() if len(v) > 1)

    # ---------------- 2B cakismasi
    kaynak_cakisma = 0
    if a.kaynak:
        kay = Path(a.kaynak).expanduser().resolve()
        if kay.is_dir():
            eski = {f.name for f in kay.rglob("*") if f.suffix.lower() in UZANTILAR}
            kaynak_cakisma = sum(1 for k in kayitlar if Path(k["yol"]).name in eski)

    # ---------------- ozetler
    print(f"\n{C}\nGUN / VARDIYA\n{C}")
    gv = Counter((k["gun"], k["vardiya"]) for k in kayitlar)
    gunler = sorted({k["gun"] for k in kayitlar})
    print(f"  {'gun':<10} {'D':>7} {'M':>7} {'N':>7} {'toplam':>8}")
    for g in gunler:
        d, m_, n_ = gv[(g, "D")], gv[(g, "M")], gv[(g, "N")]
        print(f"  {g:<10} {d:>7} {m_:>7} {n_:>7} {d+m_+n_:>8}")

    print(f"\n{C}\nVARDIYA BAZINDA A ORANI (operator davranisi icin ilk isaret)\n{C}")
    print("  Not: vardiyalarin urun karisimi farkli olabilir; bu bir sucluluk")
    print("  olcusu degil, sadece nereye bakmamiz gerektigini gosterir.")
    print(f"  {'gun':<10} {'D':>8} {'M':>8} {'N':>8}")
    for g in gunler:
        satir = []
        for v in "DMN":
            grup = [k for k in kayitlar if k["gun"] == g and k["vardiya"] == v]
            if grup:
                oran = sum(A_ANAHTAR in k["kategori"] for k in grup) / len(grup)
                satir.append(f"{100*oran:>7.1f}%")
            else:
                satir.append(f"{'-':>8}")
        print(f"  {g:<10} " + " ".join(satir))

    print(f"\n{C}\nC SINIFI (gun bazinda) - C dedektorunun yeni verisi\n{C}")
    print(f"  {'gun':<10} " + " ".join(f"{ad:>15}" for ad in C_ANAHTARLAR.values()))
    for g in gunler:
        say = [sum(1 for k in kayitlar if k["gun"] == g and anahtar in k["kategori"])
               for anahtar in C_ANAHTARLAR]
        print(f"  {g:<10} " + " ".join(f"{n:>15}" for n in say))
    top_c = [sum(1 for k in kayitlar if an in k["kategori"]) for an in C_ANAHTARLAR]
    print(f"  {'TOPLAM':<10} " + " ".join(f"{n:>15}" for n in top_c))

    print(f"\n{C}\nKATEGORI DAGILIMI (operator etiketi)\n{C}")
    for kat, n in Counter(k["kategori"] for k in kayitlar).most_common():
        print(f"  {n:>7}  {100*n/N:>5.1f}%  {kisa(kat)}")

    print(f"\n{C}\nHAT / BIN\n{C}")
    for (h, b), n in sorted(Counter((k["hat"], k["bin"]) for k in kayitlar).items()):
        print(f"  {h}  bin {b or '?':<4} {n:>7}")

    print(f"\n{C}\nKONTROLLER\n{C}")
    print(f"  birden fazla klasorde bulunan dosya : {len(cift)} "
          f"(farkli etiketle: {cift_farkli})")
    if a.kaynak:
        print(f"  2B ile ortak dosya                 : {kaynak_cakisma}")
    if cozulemeyen:
        print(f"  yolu cozulemeyen ({len(cozulemeyen)}), ilk ornekler:")
        for c in cozulemeyen[:5]:
            print(f"     {c}")

    # ---------------- altin kume
    if a.altin_gun:
        m = GUN_RE.match(a.altin_gun)
        if not m:
            print(f"\nHATA: --altin-gun bicimi 9.22 gibi olmali"); sys.exit(1)
        ag = f"{YIL}{int(m.group(1)):02d}{int(m.group(2)):02d}"
        havuz = [i for i, k in enumerate(kayitlar)
                 if k["gun"] == ag and k["kume"] != "etiketsiz"]
        if not havuz:
            print(f"\nHATA: {a.altin_gun} gununde gorsel yok."); sys.exit(1)
        for i in havuz:
            kayitlar[i]["kume"] = "altin_gun_haric"
        rng = np.random.RandomState(0)
        katman = defaultdict(list)
        for i in havuz:
            katman[(kayitlar[i]["vardiya"], kayitlar[i]["hat"])].append(i)
        secilen = []
        hedef = min(a.altin_adet, len(havuz))
        for key, grup in katman.items():
            pay = max(1, round(hedef * len(grup) / len(havuz)))
            secilen += list(rng.choice(grup, size=min(pay, len(grup)), replace=False))
        secilen = secilen[:hedef]
        # C ornekleri nadir: o gunun TUM C'leri eklenir (sizinti olcumu icin)
        c_hepsi = [i for i in havuz if any(an in kayitlar[i]["kategori"]
                                            for an in C_ANAHTARLAR)]
        eklenen_c = [i for i in c_hepsi if i not in set(secilen)]
        secilen = list(secilen) + eklenen_c
        hedef_kl = Path(a.altin_klasor)
        if hedef_kl.exists():
            print(f"\nUYARI: {hedef_kl.resolve()} zaten var, uzerine yazilmadi.")
            print("  Eski kumeyi bilerek silmedikce altin kume degismemeli.")
        else:
            hedef_kl.mkdir()
            for i in secilen:
                k = kayitlar[i]; k["kume"] = "altin"
                shutil.copy2(k["yol"], hedef_kl / Path(k["yol"]).name)
            with open(a.altin_csv, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(kayitlar[0].keys()))
                w.writeheader(); w.writerows(kayitlar[i] for i in secilen)
            print(f"\n{C}\nALTIN KUME\n{C}")
            print(f"  gun {a.altin_gun}: {len(havuz)} gorsel -> {len(secilen)} secildi")
            print(f"    {len(secilen)-len(eklenen_c)} vardiya/hat dengeli orneklem "
                  f"+ {len(eklenen_c)} ek C ornegi (gunun tum C'leri)")
            print(f"  klasor: {hedef_kl.resolve()}")
            print(f"  liste : {a.altin_csv} (operator etiketleri karsilastirma icin)")
            print(f"  {a.altin_gun} gununun TAMAMI egitimden haric isaretlendi.")

    if Path(a.cikti).exists() and a.cikti == "yeni_envanter.csv" and a.altin_klasor != "altin_kume":
        print(f"\nUYARI: {a.cikti} egitimlerde kullaniliyor, uzerine YAZILMADI.")
        print("  --cikti ile farkli bir ad verin.")
    else:
        with open(a.cikti, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(kayitlar[0].keys()))
            w.writeheader(); w.writerows(kayitlar)
    print(f"\nEnvanter: {Path(a.cikti).resolve()}")
    print(C)


if __name__ == "__main__":
    main()
