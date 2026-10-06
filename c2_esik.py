#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C DEDEKTORU v2 - ESIK SECIMI

v2 tur 2 ve tur 3'u uc altin gunde, dusuk esikler dahil dener:
  - her esikte yakalanan C ve yanlis alarm (tum kumeye olceklenmis tahmin)
  - kacan her C icin: hangi gun, gercek etiketi, mevcut sistemde nereye gitti,
    A adayi mi (A olasiligi >= 0.4) -> A'ya sizma riski

c_dedektor2.py ile ayni klasorde calistirin (onun fonksiyonlarini kullanir).
Sure: ~30 dakika.
  python c2_esik.py
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import c_dedektor2 as cd

TURLAR = [2, 3]
ESIKLER = [0.05, 0.10, 0.15, 0.20, 0.30]
A_ESIK = 0.40


def main():
    kumeler = cd.altin_hazirla()
    toplam_C = sum(int(g.sum()) for _, _, g, _, _, _ in kumeler)
    kayitlar = {}
    for ad, onay, hatalar, kl, kayit in cd.ALTINLAR:
        if kayit.exists():
            kayitlar[ad] = {r["dosya"]: r for r in
                            csv.DictReader(open(kayit, encoding="utf-8-sig"))}

    sonuc = {}
    for tur in TURLAR:
        yol = cd.CIKTI / f"tur{tur}.pt"
        if not yol.exists():
            print(f"atlandi: {yol} yok"); continue
        net = models.resnet18(weights=None); net.fc = nn.Linear(512, 2)
        net.load_state_dict(torch.load(yol, map_location="cpu")["model"])
        print(f"\nv2 tur{tur} calisiyor...")
        sonuc[tur] = [cd.c_olasiligi(net, y) for _, y, _, _, _, _ in kumeler]

    print(f"\n{'='*72}\nUC GUN TOPLAM  ({toplam_C} gercek C)\n{'='*72}")
    print(f"  {'model':<10} {'esik':>5} {'yakalanan C':>12} {'yanlis alarm':>13}")
    for e in [0.3]:
        yak = sum(int(((p1 >= e) & g).sum()) for _, _, g, _, p1, _ in kumeler)
        yan = sum(((p1 >= e) & ~g).sum() * o for _, _, g, _, p1, o in kumeler)
        print(f"  {'v1':<10} {e:>5.2f} {yak:>8}/{toplam_C:<3} {'~'+str(int(yan)):>13}")
    for tur, P in sonuc.items():
        for e in ESIKLER:
            yak = sum(int(((p >= e) & g).sum()) for p, (_, _, g, _, _, _) in zip(P, kumeler))
            yan = sum(((p >= e) & ~g).sum() * o for p, (_, _, g, _, _, o) in zip(P, kumeler))
            print(f"  {'v2 tur'+str(tur):<10} {e:>5.2f} {yak:>8}/{toplam_C:<3} "
                  f"{'~'+str(int(yan)):>13}")
        print()

    print(f"{'='*72}\nKACAN C'LER (esik 0.20) - mevcut sistemde nereye gitti\n{'='*72}")
    for tur, P in sonuc.items():
        print(f"  v2 tur{tur}:")
        for p, (ad, y, g, k, p1, _) in zip(P, kumeler):
            for i in np.where(g & (p < 0.20))[0]:
                r = kayitlar.get(ad, {}).get(y[i].name, {})
                a_olas = float(r.get("A_olasiligi") or 0)
                risk = "  <-- A ADAYI, SIZMA RISKI" if a_olas >= A_ESIK else ""
                print(f"    {ad}  v2 {p[i]:.3f}  v1 {p1[i]:.3f}  gercek {k[i]:<4} "
                      f"A olas. {a_olas:.2f}  simdiki klasor: {r.get('klasor', '?')[:22]}{risk}")
        print()


if __name__ == "__main__":
    main()
