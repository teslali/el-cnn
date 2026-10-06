#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YENI TANIMLI KIRLENMEYE BENZEYEN ESKI GORSELLER

Sorun: fabrikanin yeni tanimiyla kirlenme sayilan bir tip, eski egitim verisinde
cogunlukla 'hava akisi' (bazen 'cizik') etiketiyle duruyor. Model celiskili
etiketler gordugu icin bu tipi ogrenemiyor.

Bu script:
  1) Ornekler: altin gunlerde (9.22-9.27) senin KIRLENMEYE duzelttigin hucreler
     (altin*_hatalar icindeki kirlenme klasorleri). 9.25'in dokunulmamis kismi
     (altin7) SINAV oldugu icin dahil degil.
  2) Havuz: eski egitim verisindeki hava akisi, duzensiz cizik ve kirlenme
     etiketli gorseller (2B + yeni veri).
  3) kat5'in "gozunden" orneklere en cok benzeyen havuz gorsellerini bulur,
     en benzer 400'unu masaustunde 'kirlenme_benzer' klasorune kopyalar.

INCELEME: klasordeki gorsellerden GERCEKTEN bu yeni kirlenme tipi olanlari
  ~/Desktop/yeni_duzeltmeler/kirlenme yeni tanim
klasorune TASI. Digerlerine dokunma (eski etiketleriyle kalirlar).
Dosya adinin basinda eski etiket ve benzerlik puani yazar.

Sure: ~30 dakika (havuzdaki gorsellerin ozellikleri cikariliyor).
  python kirlenme_benzer.py ~/Desktop/fabrika-fotolar/2B
"""

import sys
import csv
import shutil
import argparse
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

KAT5 = Path("model_kat5/en_iyi_model.pt")
GRUPLAR = Path("ayrim_gruplar.csv")
D = Path.home() / "Desktop"
ORNEK_KLASORLERI = [D / "altin_hatalar", D / "altin2_hatalar", D / "altin3_hatalar",
                    D / "altin4_hatalar", D / "altin5_hatalar", D / "altin6_hatalar"]
HEDEF = D / "kirlenme_benzer"
HAVUZ_KEYS = {"气流": "havaakisi", "不规则": "cizik", "脏污": "kirlenme"}
HAVUZ_SINIR = 4000          # yeni veriden etiket basina en fazla
EN_BENZER = 400
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
HARIC_GUN = {"20260922", "20260923", "20260924", "20260925", "20260926", "20260927"}
torch.set_num_threads(4)


def normal(s):
    return " ".join(s.translate(str.maketrans("ıİşŞçÇğĞöÖüÜ", "iIsScCgGoOuU"))
                    .lower().replace("_", " ").split())


class Kume(Dataset):
    def __init__(self, yollar):
        self.yollar = yollar
        self.don = transforms.Compose([
            transforms.Resize((384, 384)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.don(im.convert("RGB")), i
        except Exception:
            return torch.zeros(3, 384, 384), i


def ozellik(net, yollar, baslik):
    F = np.zeros((len(yollar), 512), dtype=np.float32)
    n = 0
    with torch.no_grad():
        for x, idx in DataLoader(Kume(yollar), batch_size=16, num_workers=2):
            F[idx.numpy()] = net(x).numpy()
            n += len(idx)
            print(f"\r  {baslik}: {n}/{len(yollar)}", end="", flush=True)
    print()
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    return F


def gun(ad):
    import re
    z = re.search(r"(\d{14})", ad)
    return z.group(1)[:8] if z else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak", help="2B klasoru")
    a = ap.parse_args()
    if HEDEF.exists() and any(HEDEF.iterdir()):
        print(f"{HEDEF} dolu; once bosaltin ya da silin."); return

    # ---- ornekler
    ornek = []
    for kok in ORNEK_KLASORLERI:
        if kok.is_dir():
            for d in kok.iterdir():
                n = normal(d.name)
                if d.is_dir() and "kirlen" in n and "temas" not in n:
                    ornek += [f for f in d.iterdir() if f.suffix.lower() in UZANTILAR]
    print(f"Ornek (kirlenmeye duzelttiklerin, 9.22-9.27): {len(ornek)}")
    if len(ornek) < 10:
        print("HATA: cok az ornek bulundu; altin*_hatalar klasorlerindeki kirlenme")
        print("klasorlerinin adlarini kontrol edin."); return

    # ---- havuz
    havuz = []
    kay = Path(a.kaynak).expanduser()
    for d in kay.iterdir():
        k = next((k for k in HAVUZ_KEYS if k in d.name), None)
        if d.is_dir() and k:
            havuz += [(f, HAVUZ_KEYS[k]) for f in d.iterdir() if f.suffix.lower() in UZANTILAR]
    rng = np.random.RandomState(0)
    if GRUPLAR.exists():
        R = [r for r in csv.DictReader(open(GRUPLAR, encoding="utf-8-sig"))
             if r["operator"] in HAVUZ_KEYS and gun(r["yol"]) not in HARIC_GUN]
        for k in HAVUZ_KEYS:
            g = [r for r in R if r["operator"] == k]
            for i in rng.permutation(len(g))[:HAVUZ_SINIR]:
                havuz.append((Path(g[i]["yol"]), HAVUZ_KEYS[k]))
    havuz = [(f, e) for f, e in havuz if f.exists()]
    print(f"Havuz (eski hava akisi / cizik / kirlenme): {len(havuz)}")

    ck = torch.load(KAT5, map_location="cpu")
    net = models.resnet18(weights=None); net.fc = nn.Linear(512, len(ck["sinif_ad"]))
    net.load_state_dict(ck["model"]); net.fc = nn.Identity(); net.eval()

    Fo = ozellik(net, ornek, "ornekler")
    Fh = ozellik(net, [f for f, _ in havuz], "havuz")
    # her havuz gorseli icin en benzer 3 ornege ortalama benzerlik
    puan = np.zeros(len(havuz))
    for i in range(0, len(havuz), 2000):
        S = Fh[i:i+2000] @ Fo.T
        puan[i:i+2000] = np.sort(S, axis=1)[:, -3:].mean(1)
    sira = np.argsort(-puan)[:EN_BENZER]

    HEDEF.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    say = Counter()
    for i in sira:
        f, e = havuz[i]
        say[e] += 1
        shutil.copy2(f, HEDEF / f"op-{e}_m-benzer_a{puan[i]:.2f}_{f.name}")
    print(f"\nEn benzer {len(sira)} gorsel -> {HEDEF}")
    print("  eski etiketleri: " + ", ".join(f"{k} {n}" for k, n in say.most_common()))
    print(f"  benzerlik araligi: {puan[sira[0]]:.2f} - {puan[sira[-1]]:.2f}")
    (D / "yeni_duzeltmeler" / "kirlenme yeni tanim").mkdir(parents=True, exist_ok=True)
    print("\nInceleme: gercekten YENI KIRLENME TIPI olanlari su klasore tasiyin:")
    print(f"  {D / 'yeni_duzeltmeler' / 'kirlenme yeni tanim'}")
    print("Digerlerine dokunmayin.")


if __name__ == "__main__":
    main()
