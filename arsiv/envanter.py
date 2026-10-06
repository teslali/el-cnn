#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - Adim 1: Veri Envanteri

Kategori klasorlerini tarar, veri sagligini raporlar.
Model kurmadan once elimizde ne oldugunu gormek icin.

Kullanim:
    python envanter.py /gorsellerin/oldugu/klasor

Ornek:
    python envanter.py ~/Desktop/fabrika_fotolar/2B
"""

import sys
import csv
import hashlib
import re
from pathlib import Path
from collections import Counter, defaultdict

try:
    from PIL import Image
except ImportError:
    print("HATA: Pillow kurulu degil.  pip install pillow")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CIKTI_CSV = "envanter.csv"
# Dosya adindaki zaman damgasi: ..._20260904073539.jpg
ZAMAN_KALIBI = re.compile(r"(\d{14})")
# Seri numarasi: 2B_0563483306_...
SERI_KALIBI = re.compile(r"^[^_]+_(\d{6,})_")


def dosya_hash(yol, blok=1 << 20):
    """Dosyanin SHA1 ozeti - birebir ayni dosyalari bulmak icin."""
    h = hashlib.sha1()
    with open(yol, "rb") as f:
        while True:
            b = f.read(blok)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    kok = Path(sys.argv[1]).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor bulunamadi: {kok}")
        sys.exit(1)

    print(f"Taraniyor: {kok}\n")

    kategoriler = sorted([d for d in kok.iterdir() if d.is_dir()],
                         key=lambda p: p.name)
    if not kategoriler:
        print("HATA: Alt klasor yok. Kategori klasorlerini iceren dizini ver.")
        sys.exit(1)

    satirlar = []
    bozuk = []
    hash_map = defaultdict(list)

    boyut_sayaci = Counter()
    format_sayaci = Counter()
    mod_sayaci = Counter()
    kategori_sayaci = Counter()
    zaman_var = Counter()

    toplam = 0
    for kat in kategoriler:
        dosyalar = [f for f in kat.iterdir()
                    if f.is_file() and f.suffix.lower() in UZANTILAR]
        for f in dosyalar:
            toplam += 1
            if toplam % 500 == 0:
                print(f"  ... {toplam} dosya islendi", flush=True)

            kayit = {
                "yol": str(f.relative_to(kok)),
                "kategori": kat.name,
                "dosya_adi": f.name,
                "boyut_bayt": f.stat().st_size,
                "genislik": "",
                "yukseklik": "",
                "format": "",
                "mod": "",
                "zaman_damgasi": "",
                "seri_no": "",
                "hash": "",
                "durum": "ok",
            }

            # zaman damgasi / seri no
            zm = ZAMAN_KALIBI.search(f.stem)
            if zm:
                kayit["zaman_damgasi"] = zm.group(1)
                zaman_var[kat.name] += 1
            sm = SERI_KALIBI.match(f.name)
            if sm:
                kayit["seri_no"] = sm.group(1)

            # goruntu ac
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    kayit["genislik"], kayit["yukseklik"] = im.size
                    kayit["format"] = im.format or ""
                    kayit["mod"] = im.mode
                boyut_sayaci[(kayit["genislik"], kayit["yukseklik"])] += 1
                format_sayaci[kayit["format"]] += 1
                mod_sayaci[kayit["mod"]] += 1
            except Exception as e:
                kayit["durum"] = f"BOZUK: {type(e).__name__}"
                bozuk.append((str(f.relative_to(kok)), str(e)[:80]))

            if kayit["durum"] == "ok":
                try:
                    h = dosya_hash(f)
                    kayit["hash"] = h
                    hash_map[h].append(kayit["yol"])
                except Exception:
                    pass

            kategori_sayaci[kat.name] += 1
            satirlar.append(kayit)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    alanlar = ["yol", "kategori", "dosya_adi", "boyut_bayt", "genislik",
               "yukseklik", "format", "mod", "zaman_damgasi", "seri_no",
               "hash", "durum"]
    with open(CIKTI_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=alanlar)
        w.writeheader()
        w.writerows(satirlar)

    # ------------------------------------------------------------------
    # RAPOR
    # ------------------------------------------------------------------
    C = "=" * 62
    print(f"\n{C}\nVERI ENVANTERI\n{C}")
    print(f"Toplam gorsel : {toplam}")
    print(f"Kategori      : {len(kategoriler)}")
    print(f"Bozuk dosya   : {len(bozuk)}")

    print(f"\n{C}\nKATEGORI DAGILIMI\n{C}")
    print(f"{'adet':>7}  {'%':>6}  kategori")
    for kat, n in kategori_sayaci.most_common():
        print(f"{n:>7}  {100*n/max(toplam,1):>5.1f}%  {kat}")

    print(f"\n{C}\nGORUNTU TUTARLILIGI\n{C}")
    print("Boyutlar:")
    for (w_, h_), n in boyut_sayaci.most_common(8):
        print(f"   {w_}x{h_}  ->  {n} dosya")
    print("Formatlar:", dict(format_sayaci))
    print("Renk modlari:", dict(mod_sayaci))

    print(f"\n{C}\nZAMAN DAMGASI (bolme stratejisi icin kritik)\n{C}")
    zamanli = sum(zaman_var.values())
    print(f"Zaman damgasi olan : {zamanli} / {toplam}"
          f"  (%{100*zamanli/max(toplam,1):.1f})")
    if zamanli:
        tums = sorted(r["zaman_damgasi"] for r in satirlar if r["zaman_damgasi"])
        print(f"En eski : {tums[0]}")
        print(f"En yeni : {tums[-1]}")
        gunler = Counter(t[:8] for t in tums)
        print(f"Farkli gun sayisi : {len(gunler)}")
        print("Gun bazinda dagilim (ilk 12):")
        for g, n in sorted(gunler.items())[:12]:
            print(f"   {g[:4]}-{g[4:6]}-{g[6:8]}  ->  {n}")
        eksik = [k for k in kategori_sayaci if zaman_var.get(k, 0) < kategori_sayaci[k]]
        if eksik:
            print("\nUYARI - bu kategorilerde zaman damgasi eksik dosyalar var:")
            for k in eksik:
                print(f"   {k}: {zaman_var.get(k,0)}/{kategori_sayaci[k]}")

    print(f"\n{C}\nBIREBIR AYNI DOSYALAR\n{C}")
    tekrar = {h: yl for h, yl in hash_map.items() if len(yl) > 1}
    if not tekrar:
        print("Yok - temiz.")
    else:
        cok_kat = 0
        for h, yollar in tekrar.items():
            kats = {Path(y).parts[0] for y in yollar}
            if len(kats) > 1:
                cok_kat += 1
        print(f"Tekrarlanan dosya grubu : {len(tekrar)}")
        print(f"Bunlardan FARKLI kategorilerde olan : {cok_kat}  <-- onemli")
        for h, yollar in list(tekrar.items())[:10]:
            print(f"   {len(yollar)}x  {yollar[0]}")
            for y in yollar[1:3]:
                print(f"        = {y}")

    if bozuk:
        print(f"\n{C}\nBOZUK DOSYALAR\n{C}")
        for y, e in bozuk[:20]:
            print(f"   {y}  ->  {e}")
        if len(bozuk) > 20:
            print(f"   ... ve {len(bozuk)-20} tane daha")

    print(f"\n{C}")
    print(f"Ayrinti kaydedildi: {Path(CIKTI_CSV).resolve()}")
    print(C)


if __name__ == "__main__":
    main()
