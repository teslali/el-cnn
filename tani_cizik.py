#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TANI: altin kumede "duzensiz cizik" klasoru neden kotu?

Uc supheli, her biri farkli iz birakir:

  1) KURAL DEGISIKLIGI (gunluk.py)
     A adayi olmayan hucre, kategori modeli ilk tercih olarak "Diger" dese
     bile Diger'e gidemiyor, IKINCI tercihine gidiyor. Temiz / A'ya yakin
     hucrelerin ikinci tercihi cizik olabilir.
     Iz: cizik klasorundekilerin cogunun ilk tercihi aslinda Diger.

  2) ALTIN KUMEDE COK CATLAK
     Gunun tum C ornekleri eklendi; kategori modeli catlagi cizik saniyor.
     Iz: operator etiketi "gizli catlak" olanlar cizik klasorunde yogun.

  3) YENI HATLAR (2A, 1B)
     Model bu hatlari hic gormemis olabilir.
     Iz: cizik tahmini bu hatlarda operatorun cizik oranindan cok yuksek.

Ayrica: simdiye kadar duzelttigin (altin_hatalar'a tasidigin) cizik
tahminlerinin aslinda ne oldugunu sayar.

Model sadece cizik klasorune dusen gorsellerde calisir; birkac dakika.

Kullanim:
  python tani_cizik.py
"""

import sys
import csv
import shutil
from pathlib import Path
from collections import Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})"); sys.exit(1)

CIKTI_KL = Path("cikti_altin")
ALTIN_KL = Path("altin_kume")
ALTIN_CSV = Path("altin_kume.csv")
HATALAR = Path.home() / "Desktop" / "altin_hatalar"
KAT_MODEL = Path("model3/en_iyi_model.pt")
TANI_KL = Path("tani_cizik")

CIZIK = "不规则"
A_ANAHTAR = "其他"
CATLAK = "隐裂"
C = "=" * 70
torch.set_num_threads(4)


def kisa(ad, n=30):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


class Kume(Dataset):
    def __init__(self, yollar):
        self.yollar = yollar
        self.don = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.don(im.convert("RGB")), i
        except Exception:
            return torch.zeros(3, 224, 224), -1


def main():
    for p in (CIKTI_KL / "_kayit.csv", ALTIN_CSV, ALTIN_KL, KAT_MODEL):
        if not p.exists():
            print(f"HATA: bulunamadi: {p}"); sys.exit(1)

    kayit = list(csv.DictReader(open(CIKTI_KL / "_kayit.csv", encoding="utf-8-sig")))
    operator = {}
    for r in csv.DictReader(open(ALTIN_CSV, encoding="utf-8-sig")):
        operator[Path(r["yol"]).name] = r

    toplam = len(kayit)
    cizik = [r for r in kayit if CIZIK in r["klasor"]]
    print(f"{C}\nTANI: DUZENSIZ CIZIK KLASORU\n{C}")
    print(f"Altin kume: {toplam} gorsel | modelin cizik dedigi: {len(cizik)} "
          f"(%{100*len(cizik)/max(toplam,1):.1f})")
    op_cizik = sum(1 for r in operator.values() if CIZIK in r["kategori"])
    print(f"Operatorlerin cizik dedigi: {op_cizik} (%{100*op_cizik/max(len(operator),1):.1f})")

    # ---- SUPHELI 1: kategori modelinin ham ilk tercihi
    ck = torch.load(KAT_MODEL, map_location="cpu")
    sinif_ad = ck["sinif_ad"]
    net = models.resnet18(weights=None); net.fc = nn.Linear(512, len(sinif_ad))
    net.load_state_dict(ck["model"]); net.eval()
    yollar = [ALTIN_KL / r["dosya"] for r in cizik]
    P = np.zeros((len(yollar), len(sinif_ad)), dtype=np.float32)
    yl = DataLoader(Kume(yollar), batch_size=24, num_workers=2)
    with torch.no_grad():
        for x, idx in yl:
            p = torch.softmax(net(x), 1).numpy()
            for j, i in enumerate(idx.tolist()):
                if i >= 0:
                    P[i] = p[j]
    ilk = [sinif_ad[int(P[i].argmax())] for i in range(len(yollar))]
    ilk_diger = [i for i, k in enumerate(ilk) if A_ANAHTAR in k]

    print(f"\n{C}\nSUPHELI 1 - kategori modelinin ILK tercihi\n{C}")
    for k, n in Counter(kisa(k) for k in ilk).most_common(6):
        print(f"  {n:>5}  {k}")
    print(f"  -> ilk tercihi Diger olup kural geregi cizige itilen: "
          f"{len(ilk_diger)} (%{100*len(ilk_diger)/max(len(cizik),1):.0f})")

    # ---- SUPHELI 2: operator etiketi
    print(f"\n{C}\nSUPHELI 2 - operator bunlara ne demis\n{C}")
    op = Counter(kisa(operator.get(r["dosya"], {}).get("kategori", "?")) for r in cizik)
    for k, n in op.most_common(8):
        print(f"  {n:>5}  {k}")
    catlak_n = sum(1 for r in cizik
                   if CATLAK in operator.get(r["dosya"], {}).get("kategori", ""))
    print(f"  -> operatorun gizli catlak dedigi: {catlak_n}")

    # ---- SUPHELI 3: hat bazinda
    print(f"\n{C}\nSUPHELI 3 - hat bazinda cizik orani\n{C}")
    print(f"  {'hat':<5} {'gorsel':>7} {'model cizik':>12} {'operator cizik':>15}")
    hat_of = {d: operator.get(d, {}).get("hat", "?") for d in (r["dosya"] for r in kayit)}
    for h in sorted(set(hat_of.values())):
        grp = [r for r in kayit if hat_of[r["dosya"]] == h]
        m = sum(1 for r in grp if CIZIK in r["klasor"])
        o = sum(1 for r in grp if CIZIK in operator.get(r["dosya"], {}).get("kategori", ""))
        print(f"  {h:<5} {len(grp):>7} {100*m/max(len(grp),1):>11.1f}% "
              f"{100*o/max(len(grp),1):>14.1f}%")

    # ---- simdiye kadarki duzeltmeler
    if HATALAR.is_dir():
        cizik_adlar = {r["dosya"] for r in cizik}
        duz = Counter()
        for d in HATALAR.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    if f.name in cizik_adlar:
                        duz[d.name] += 1
        if duz:
            print(f"\n{C}\nDUZELTTIKLERIN - cizik tahmini aslinda neymis\n{C}")
            for k, n in duz.most_common():
                print(f"  {n:>5}  {k}")

    # ---- goz kontrolu icin ornekler
    if TANI_KL.exists():
        shutil.rmtree(TANI_KL)
    (TANI_KL / "1_ilk_tercihi_Diger").mkdir(parents=True)
    (TANI_KL / "2_operator_gizli_catlak").mkdir(parents=True)
    for i in ilk_diger[:60]:
        shutil.copy2(yollar[i], TANI_KL / "1_ilk_tercihi_Diger" /
                     f"pDiger{P[i].max():.2f}_{yollar[i].name}")
    for r in cizik:
        if CATLAK in operator.get(r["dosya"], {}).get("kategori", ""):
            shutil.copy2(ALTIN_KL / r["dosya"], TANI_KL / "2_operator_gizli_catlak" / r["dosya"])

    print(f"\n{C}\nYORUM\n{C}")
    print("  Supheli 1 yuksekse: kural degisikligi sebep; A'ya yakin hucreler")
    print("    ciziye dokuluyor. 1_ilk_tercihi_Diger klasorune bakin.")
    print("  Supheli 2 yuksekse: catlaklar ciziye gidiyor; C dedektoru tum")
    print("    gorsellere uygulanarak cozulur.")
    print("  Bir hatta model cizik orani operatorunkinden cok yuksekse:")
    print("    o hat modele yabanci; yeni veriyle yeniden egitim gerekir.")
    print(f"\n  open {TANI_KL.resolve()}")
    print(C)


if __name__ == "__main__":
    main()
