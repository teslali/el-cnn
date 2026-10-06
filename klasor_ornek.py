#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HERHANGI BIR KLASORDEN RASTGELE ORNEKLEM

Ornek: masaustundeki bir vardiya klasorunden (ör. M(16-00)) rastgele 2000 gorsel.
Hat ve operator etiketi klasor yapisindan okunur (olcum icin), baska bir
orneklemde bulunan gorseller (--haric) disarida birakilir.

  python klasor_ornek.py ~/Desktop/"M(16-00)" --hedef altin_kume9 --haric altin_kume8
"""

import re
import csv
import shutil
import argparse
from pathlib import Path

import numpy as np

UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
HAT_RE = re.compile(r"^\d[A-Z]$", re.I)
VARDIYA_RE = re.compile(r"^([DMN])\s*\(", re.I)
ZAMAN = re.compile(r"(\d{14})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak")
    ap.add_argument("--adet", type=int, default=2000)
    ap.add_argument("--hedef", default="altin_kume9")
    ap.add_argument("--haric", default=None, help="bu klasordeki gorseller secilmez")
    a = ap.parse_args()

    kok = Path(a.kaynak).expanduser()
    hedef = Path(a.hedef)
    if hedef.exists():
        print(f"{hedef} zaten var, uzerine yazilmadi."); return
    haric = set()
    if a.haric and Path(a.haric).is_dir():
        haric = {f.name for f in Path(a.haric).iterdir()}

    satirlar = []
    for f in kok.rglob("*"):
        if not (f.is_file() and f.suffix.lower() in UZANTILAR) or f.name in haric:
            continue
        if "OverTime" in f.name:
            continue
        parcalar = f.relative_to(kok).parts[:-1]
        hat = next((p.upper() for p in parcalar if HAT_RE.match(p)), "")
        vardiya = next((VARDIYA_RE.match(p).group(1).upper() for p in (kok.name,) + parcalar
                        if VARDIYA_RE.match(p)), "")
        z = ZAMAN.search(f.name)
        satirlar.append({"yol": str(f), "gun": z.group(1)[:8] if z else "",
                         "vardiya": vardiya, "bin": "", "hat": hat,
                         "kategori": f.parent.name})
    print(f"Kaynakta uygun gorsel: {len(satirlar)}  (haric tutulan: {len(haric)})")
    if not satirlar:
        return
    rng = np.random.RandomState(930)
    secilen = [satirlar[i] for i in rng.permutation(len(satirlar))[:a.adet]]

    hedef.mkdir()
    for r in secilen:
        shutil.copy2(r["yol"], hedef / Path(r["yol"]).name)
    with open(f"{hedef}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(secilen[0].keys()))
        w.writeheader(); w.writerows(secilen)
    gunler = sorted({r["gun"] for r in secilen})
    print(f"{len(secilen)} gorsel -> {hedef.resolve()}  |  liste: {hedef}.csv")
    print(f"gun(ler): {', '.join(gunler)}  |  hatlar: "
          + ", ".join(sorted({r['hat'] for r in secilen if r['hat']})))


if __name__ == "__main__":
    main()
