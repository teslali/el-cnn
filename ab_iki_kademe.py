#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B SIDDET MODELI - IKI KADEMELI A SINIFI ONERISI

NEDEN IKI KADEME
  Gozle inceleme sunu gosterdi:
    - Ilk ~640 gorsel: neredeyse tamami A  (10-20 hata)
    - ~1400-1700 arasi: gecis bolgesi, karisik
    - Son %25: B'ler yogunlasiyor

  Tek liste verince operator hepsine ayni dikkatle bakmak zorunda kaliyor.
  Oysa ilk bolum neredeyse temiz, ikinci bolum dikkat istiyor.

  Bu yuzden iki ayri klasor:
    _A1_YUKSEK_GUVEN  -> hizli goz gezdirme yeter
    _A2_KONTROL       -> tek tek bakilmali

KULLANIM
  python ab_iki_kademe.py "/path/otomasyon_cikti"
  python ab_iki_kademe.py "/path" --kademe1 700 --kademe2 1400
  python ab_iki_kademe.py "/path" --oran1 0.13 --oran2 0.26

  --csv ile mevcut bir ab_duzeltme2.csv kullanilabilir (yeniden tahmin yapmaz)
"""

import sys
import csv
import time
import shutil
import argparse
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
    print(f"HATA: paket eksik ({e.name})")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_YOLU = Path("model_ab2") / "en_iyi_model.pt"
GIRDI_BOYUT = 224
BATCH = 24
ISCI = 2
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
B_SINIFI = {"黑斑", "黑边"}          # siyah nokta, siyah kenar

K1_AD = "_A1_YUKSEK_GUVEN"
K2_AD = "_A2_KONTROL"

C = "=" * 74
torch.set_num_threads(4)


class Kume(Dataset):
    def __init__(self, yollar, donusum):
        self.yollar, self.donusum = yollar, donusum

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.donusum(im.convert("RGB")), i
        except Exception:
            return torch.zeros(3, GIRDI_BOYUT, GIRDI_BOYUT), -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--kademe1", type=int, default=None,
                    help="yuksek guven bolumunun sinirı (adet)")
    ap.add_argument("--kademe2", type=int, default=None,
                    help="kontrol bolumunun sinirı (adet, kademe1 dahil)")
    ap.add_argument("--oran1", type=float, default=0.13)
    ap.add_argument("--oran2", type=float, default=0.26)
    ap.add_argument("--model", default=str(MODEL_YOLU))
    ap.add_argument("--csv", default=None,
                    help="mevcut ab_duzeltme2.csv (yeniden tahmin yapma)")
    a = ap.parse_args()

    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    print(f"{C}\nIKI KADEMELI A SINIFI ONERISI\n{C}")

    # ------------------------------------------------------------------
    # Olasiliklari al: ya csv'den ya yeniden hesapla
    # ------------------------------------------------------------------
    kayitlar = []      # (dosya_yolu, olasilik, eski_klasor)

    csv_yolu = Path(a.csv) if a.csv else kok / "ab_duzeltme2.csv"
    if csv_yolu.exists():
        print(f"Mevcut tahminler kullaniliyor: {csv_yolu.name}")
        harita = {}
        for f in kok.rglob("*"):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                harita.setdefault(f.name, f)
        with open(csv_yolu, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    p = float(r["A_olasiligi"])
                except (ValueError, KeyError):
                    continue
                yol = harita.get(r["dosya"])
                if yol is not None:
                    kayitlar.append((yol, p, r.get("eski_klasor", "")))
        print(f"  okunan: {len(kayitlar)}")
    else:
        mp = Path(a.model)
        if not mp.exists():
            print(f"HATA: model yok: {mp}")
            sys.exit(1)
        ck = torch.load(mp, map_location="cpu")
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(512, 2)
        net.load_state_dict(ck["model"])
        net.eval()
        print(f"Model: {mp}  (dogrulama F1 {ck.get('dog_f1',0):.3f})")

        hedefler = [d for d in sorted(kok.iterdir())
                    if d.is_dir() and any(b in d.name for b in B_SINIFI)]
        if not hedefler:
            print("HATA: siyah nokta / siyah kenar klasoru bulunamadi.")
            for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
                print(f"   {d.name}")
            sys.exit(1)

        dosyalar = []
        for d in hedefler:
            dosyalar += [f for f in sorted(d.iterdir())
                         if f.is_file() and f.suffix.lower() in UZANTILAR]
        print(f"Gorsel: {len(dosyalar)}\n")

        donusum = transforms.Compose([
            transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        yl = DataLoader(Kume(dosyalar, donusum), batch_size=BATCH,
                        shuffle=False, num_workers=ISCI)
        P = np.zeros(len(dosyalar))
        n, t0 = 0, time.time()
        with torch.no_grad():
            for x, idx in yl:
                pr = torch.softmax(net(x), 1)[:, 0].numpy()
                for j, i in enumerate(idx.tolist()):
                    if i >= 0:
                        P[i] = pr[j]
                n += len(idx)
                print(f"\r  {n}/{len(dosyalar)}", end="", flush=True)
        print(f"\n  bitti: {(time.time()-t0)/60:.1f} dk")
        kayitlar = [(dosyalar[i], P[i], dosyalar[i].parent.name)
                    for i in range(len(dosyalar))]

    if not kayitlar:
        print("HATA: hicbir kayit olusturulamadi.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Sirala ve kademe sinirlarini belirle
    # ------------------------------------------------------------------
    kayitlar.sort(key=lambda r: -r[1])
    N = len(kayitlar)
    k1 = a.kademe1 if a.kademe1 else max(1, int(N * a.oran1))
    k2 = a.kademe2 if a.kademe2 else max(k1 + 1, int(N * a.oran2))
    k1, k2 = min(k1, N), min(k2, N)

    olaslar = np.array([r[1] for r in kayitlar])
    print(f"\n{C}\nDAGILIM\n{C}")
    print(f"Toplam    : {N}")
    print(f"En yuksek : {olaslar[0]:.4f}")
    print(f"Medyan    : {np.median(olaslar):.4f}")
    print(f"En dusuk  : {olaslar[-1]:.4f}")

    print(f"\n{C}\nKADEMELER\n{C}")
    print(f"  KADEME 1 (yuksek guven) : 1 - {k1}")
    print(f"     olasilik araligi     : {olaslar[0]:.4f} - {olaslar[k1-1]:.4f}")
    print(f"  KADEME 2 (kontrol)      : {k1+1} - {k2}")
    if k2 > k1:
        print(f"     olasilik araligi     : {olaslar[k1]:.4f} - {olaslar[k2-1]:.4f}")
    print(f"  Onerilmeyen             : {k2+1} - {N}  ({N-k2} gorsel)")

    # ------------------------------------------------------------------
    # Kopyala
    # ------------------------------------------------------------------
    for ad in (K1_AD, K2_AD):
        p = kok / ad
        if p.exists():
            shutil.rmtree(p)
        (kok / ad).mkdir()

    sayac = Counter()
    for rank, (yol, p, eski) in enumerate(kayitlar[:k2], 1):
        hedef_ad = K1_AD if rank <= k1 else K2_AD
        yeni = kok / hedef_ad / f"{rank:04d}_a{p:.4f}_{yol.name}"
        try:
            shutil.copy2(yol, yeni)
            sayac[hedef_ad] += 1
        except Exception as e:
            print(f"  kopyalanamadi: {yol.name} ({e})")

    # rapor
    rap = kok / "ab_iki_kademe.csv"
    with open(rap, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["sira", "dosya", "eski_klasor", "A_olasiligi", "kademe"])
        for rank, (yol, p, eski) in enumerate(kayitlar, 1):
            kad = "1-yuksek" if rank <= k1 else ("2-kontrol" if rank <= k2
                                                 else "onerilmedi")
            w.writerow([rank, yol.name, eski, f"{p:.6f}", kad])

    # talimat
    with open(kok / K1_AD / "OKU.txt", "w", encoding="utf-8") as fh:
        fh.write("KADEME 1 - YUKSEK GUVEN\n" + "=" * 50 + "\n\n")
        fh.write("Bu bolumde onceki olcumde ~%97 dogruluk goruldu.\n")
        fh.write("Hizli goz gezdirmek yeterli olmali.\n\n")
        fh.write("Finder'da:\n")
        fh.write("  - Simge boyutunu buyutun (Cmd + 1, sonra kaydiriciyi saga)\n")
        fh.write("  - Ya da Gallery gorunumu (Cmd + 4)\n\n")
        fh.write("Goze carpan B varsa ayirin.\n")
    with open(kok / K2_AD / "OKU.txt", "w", encoding="utf-8") as fh:
        fh.write("KADEME 2 - KONTROL GEREKLI\n" + "=" * 50 + "\n\n")
        fh.write("Bu bolum GECIS BOLGESI: modelin kararsiz oldugu aralik.\n")
        fh.write("Gozle incelemede burada B'ler artmaya basliyor.\n\n")
        fh.write("Tek tek bakin. Emin olmadiginizi B birakin.\n\n")
        fh.write("Dosya adindaki 'a0.31' modelin A olasiligidir;\n")
        fh.write("kucuk oldukca B olma ihtimali artar.\n")

    print(f"\n{C}\nSONUC\n{C}")
    print(f"  {K1_AD:<22} {sayac[K1_AD]:>6} gorsel")
    print(f"  {K2_AD:<22} {sayac[K2_AD]:>6} gorsel")
    print(f"  toplam oneri           {sayac[K1_AD]+sayac[K2_AD]:>6}  "
          f"(%{100*(sayac[K1_AD]+sayac[K2_AD])/N:.1f})")
    print(f"\nRapor: {rap}")
    print(f"\n  open \"{kok/K1_AD}\"")
    print(f"  open \"{kok/K2_AD}\"")
    print(C)
    print("\nKademe 1: hizli goz gezdirme.")
    print("Kademe 2: tek tek bakin, gecis bolgesi.")
    print("\nSinirlari degistirmek icin:")
    print("  --kademe1 700 --kademe2 1400")


if __name__ == "__main__":
    main()
