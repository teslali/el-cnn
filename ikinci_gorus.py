#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IKINCI GORUS: 384 px kategori birlesimi A adaylarini onaylasin mi?

SORUN
  A kararini veren A/B modeli 224 px'te calisiyor; ince cizikleri gormuyor.
  9.24'te model A dedigi hucrelerin cogu senin gozunde ince cizikli B.
  Kategori birlesimi (model3 + kat5, 384 px) ciziklerde cok daha iyi ve
  icinde bir A sinifi var; simdiye kadar A kararinda kullanilmiyordu.

DENEME
  A adaylari (A olasiligi >= 0.40, C dedektorunden gecmis) icin birlesimin
  kendi A olasiligina bakilir. Birlesim A demiyorsa A iptal.
  Iki altin kumede birden olculur; yeni egitim yok.

  9.22: dogru etiketler cikti_altin + altin_hatalar, kararlar cikti_altin2
  9.24: dogru etiketler cikti_altin_924 + altin2_hatalar, kararlar cikti_altin_924b

Sure: birkac dakika.
  python ikinci_gorus.py
"""

import sys
import csv
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})"); sys.exit(1)

A_ESIK = 0.40
C_ESIK = 0.30
MODEL3 = Path("model3/en_iyi_model.pt")
KAT5 = Path("model_kat5/en_iyi_model.pt")
D = Path.home() / "Desktop"
KUMELER = [
    ("9.22", Path("cikti_altin"), D / "altin_hatalar", Path("altin_kume"),
     Path("cikti_altin2/_kayit.csv")),
    ("9.24", Path("cikti_altin_924"), D / "altin2_hatalar", Path("altin_kume2"),
     Path("cikti_altin_924b/_kayit.csv")),
]
ESIKLER = [None, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
A = "其他"
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
torch.set_num_threads(4)


def normal(s):
    return " ".join(s.translate(str.maketrans("ıİşŞçÇğĞöÖüÜ", "iIsScCgGoOuU"))
                    .lower().replace("_", " ").split())


def a_mi(klasor_adi):
    return A in klasor_adi or "a sinif" in normal(klasor_adi)


def altin_A(onay, hatalar):
    etiket = {}
    for kok in (onay, hatalar):
        if kok.is_dir():
            for d in kok.iterdir():
                if d.is_dir() and "emin" not in normal(d.name):
                    for f in d.iterdir():
                        if f.suffix.lower() in UZANTILAR:
                            etiket[f.name] = a_mi(d.name)
    return etiket


class Kume(Dataset):
    def __init__(self, yollar, boyut):
        self.yollar, self.boyut = yollar, boyut
        self.don = transforms.Compose([
            transforms.Resize((boyut, boyut)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.don(im.convert("RGB")), i
        except Exception:
            return torch.zeros(3, self.boyut, self.boyut), i


def calistir(net, yollar, boyut, n):
    P = np.zeros((len(yollar), n), dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for x, idx in DataLoader(Kume(yollar, boyut), batch_size=12, num_workers=0):
            P[idx.numpy()] = torch.softmax(net(x), 1).numpy()
    return P


def yukle(yol):
    ck = torch.load(yol, map_location="cpu")
    net = models.resnet18(weights=None); net.fc = nn.Linear(512, len(ck["sinif_ad"]))
    net.load_state_dict(ck["model"])
    return net, ck


def main():
    for p in (MODEL3, KAT5):
        if not p.exists():
            print(f"HATA: {p} yok"); sys.exit(1)
    net3, ck3 = yukle(MODEL3)
    net5, ck5 = yukle(KAT5)
    k3 = [next((k for k in [A] if k in s), s) for s in ck3["sinif_ad"]]
    a3 = next(i for i, s in enumerate(ck3["sinif_ad"]) if A in s)
    a5 = next(i for i, s in enumerate(ck5["sinif_ad"]) if A in s)
    sayi, us = ck5.get("egitim_sayi", {}), ck5.get("denge_us", 0.5)
    anahtarlar5 = [next((k for k in sayi if k in s), None) for s in ck5["sinif_ad"]]
    carpan = np.array([sayi.get(k, 1) ** us if k else 1.0 for k in anahtarlar5])

    for ad, onay, hatalar, kl, kayit in KUMELER:
        print(f"\n{'='*70}\nALTIN KUME {ad}\n{'='*70}")
        if not kayit.exists():
            print(f"  atlandi: {kayit} yok"); continue
        etiket = altin_A(onay, hatalar)
        R = list(csv.DictReader(open(kayit, encoding="utf-8-sig")))
        gercek_A = sum(1 for r in R if etiket.get(r["dosya"]))
        aday = [r for r in R if float(r["A_olasiligi"]) >= A_ESIK
                and float(r.get("C_dedektor") or 0) < C_ESIK
                and r["dosya"] in etiket and (kl / r["dosya"]).exists()]
        g = np.array([etiket[r["dosya"]] for r in aday])
        print(f"  gercek A (senin etiketin): {gercek_A}")
        print(f"  A adayi (A/B >= {A_ESIK}, C dedektorunden gecmis): {len(aday)}, "
              f"dogru {int(g.sum())} -> isabet %{100*g.mean():.0f}")
        yollar = [kl / r["dosya"] for r in aday]
        print("  birlesim calisiyor...")
        P3 = calistir(net3, yollar, 224, len(ck3["sinif_ad"]))
        P5 = calistir(net5, yollar, ck5.get("girdi_boyut", 384), len(ck5["sinif_ad"]))
        P5 = P5 * carpan; P5 = P5 / P5.sum(1, keepdims=True)
        # birlesimin A olasiligi ve en olasi B sinifi
        pA = (P3[:, a3] + P5[:, a5]) / 2
        P5b = P5.copy(); P5b[:, a5] = -1
        en_B = [ck5["sinif_ad"][i] for i in P5b.argmax(1)]

        print(f"\n  {'kural':<26} {'kalan':>6} {'dogru A':>8} {'isabet':>7} "
              f"{'cikan B':>8} {'kaybedilen A':>13}")
        for e in ESIKLER:
            if e is None:
                tut = np.ones(len(aday), dtype=bool); kural = "ikinci gorus yok"
            else:
                tut = pA >= e; kural = f"birlesim A olas. >= {e:.2f}"
            dA = int((tut & g).sum())
            print(f"  {kural:<26} {int(tut.sum()):>6} {dA:>8} "
                  f"{'%'+str(round(100*dA/max(tut.sum(),1))):>7} "
                  f"{int((~tut & ~g).sum()):>8} {int(g.sum())-dA:>13}")
        cikan = [(en_B[i], g[i]) for i in range(len(aday)) if pA[i] < 0.20]
        if cikan:
            from collections import Counter
            print("\n  esik 0.20'de iptal edilenler, birlesime gore neydi:")
            for k, n in Counter(c[0][:24] for c in cikan).most_common(6):
                print(f"    {n:>4}  {k}")


if __name__ == "__main__":
    main()
