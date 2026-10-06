#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C v2 - A ADAYLARINDA DUSUK ESIGIN BEDELI

A adaylarinda (A olasiligi >= 0.40) C dedektoru v2 (tur 3) dusuk bir esikle
calisacak. Bu script her esikte:
  - A adaylari arasindaki gercek C'lerin kaci yakalanir (hedef: hepsi)
  - kac GERCEK A hucresi gereksiz yere C supheliye gider (bedel)

c_dedektor2.py ile ayni klasorde calistirin.  Sure: ~5 dakika.
  python c2_aday.py
"""

import csv

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import c_dedektor2 as cd

TUR = 3
A_ESIK = 0.40
ESIKLER = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]


def main():
    net = models.resnet18(weights=None); net.fc = nn.Linear(512, 2)
    net.load_state_dict(torch.load(cd.CIKTI / f"tur{TUR}.pt", map_location="cpu")["model"])
    top = {e: [0, 0, 0] for e in ESIKLER}        # yakalanan C, kaybedilen A
    top_C = top_A = 0
    for ad, onay, hatalar, kl, kayit in cd.ALTINLAR:
        if not kayit.exists():
            continue
        etiket = {}
        for kok in (onay, hatalar):
            if kok.is_dir():
                for d in kok.iterdir():
                    if d.is_dir() and "emin" not in cd.normal(d.name):
                        k = cd.anahtar(d.name)
                        if k:
                            for f in d.iterdir():
                                etiket[f.name] = k
        R = [r for r in csv.DictReader(open(kayit, encoding="utf-8-sig"))
             if float(r["A_olasiligi"]) >= A_ESIK and r["dosya"] in etiket
             and (kl / r["dosya"]).exists()]
        g_C = np.array([cd.c_mi(etiket[r["dosya"]]) for r in R])
        g_A = np.array([etiket[r["dosya"]] == cd.A for r in R])
        print(f"\n{ad}: A adayi {len(R)}  (gercek A {int(g_A.sum())}, gercek C {int(g_C.sum())})")
        p = cd.c_olasiligi(net, [kl / r["dosya"] for r in R])
        top_C += int(g_C.sum()); top_A += int(g_A.sum())
        for e in ESIKLER:
            top[e][0] += int(((p >= e) & g_C).sum())
            top[e][1] += int(((p >= e) & g_A).sum())

    print(f"\n{'='*60}\nUC GUN TOPLAM: A adaylari arasinda {top_C} gercek C, {top_A} gercek A\n{'='*60}")
    print(f"  {'esik':>5} {'yakalanan C':>12} {'C supheliye giden gercek A':>28}")
    for e in ESIKLER:
        print(f"  {e:>5.2f} {top[e][0]:>8}/{top_C:<3} {top[e][1]:>16}  "
              f"(A'larin %{100*top[e][1]/max(top_A,1):.1f}'i)")


if __name__ == "__main__":
    main()
