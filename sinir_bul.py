#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YENI A/B MODELI (model_ab3) - GOZLE SINIR DOGRULAMA

model_ab3 "A sinifi vs geri kalan hersey" olarak egitildi ve TUM gorselleri
tarayabiliyor (eski model sadece siyah nokta+kenar'a bakiyordu).

Test sonuclari:
  dogrulama AUC 0.9653, test AUC 0.9600
  C sinifi sizintisi: %30 kesime kadar SIFIR (veto gerekmiyor)

AMA test kumesindeki precision sayilarina guvenmiyoruz. Iki kez olctuk:
test %78 derken gozle bakinca %97-98 cikti. Sebep, test kumesindeki "A"
taniminin operatorlerin o gunku kararlari olmasi - ve bizim tum bulgumuz
operatorlerin sinirda B'ye kaydigi yonunde.

Bu yuzden GERCEK SINIRI GOZLE BULACAGIZ.

NE YAPAR
  Verilen klasordeki tum gorselleri model_ab3'ten gecirir, A olasiligina
  gore siralar ve en yuksek N tanesini tek bir klasore kopyalar.
  Dosya adlari sirali: 0001 = modelin A olma ihtimalini en yuksek gordugu.

NASIL BAKACAKSINIZ
  Klasoru acin, Gallery gorunumune gecin (Cmd+4), ok tuslariyla ilerleyin.
  Tek soru: "bu gorsel A sinifi mi?"
  A'larin nerede bitip B'lerin basladigi sirayi tespit edin.
  O sira, gunluk kullanimda kesim noktasi olacak.

Kullanim:
    python sinir_bul.py "/path/gorseller"
    python sinir_bul.py "/path/gorseller" --adet 2500
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
AB_MODEL = Path("model_ab3") / "en_iyi_model.pt"
KATEGORI_MODEL = Path("model3") / "en_iyi_model.pt"

GIRDI_BOYUT = 224
BATCH = 24
ISCI = 2
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}

CIKTI = Path("sinir_bul")

# C sinifi kontrolu (guvenlik: sizinti olmamasi gerekiyor ama dogrulayalim)
C_ANAHTARLAR = {"隐裂", "压板"}

C = "=" * 74
torch.set_num_threads(4)


def kisa(ad, n=32):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


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


def donusum():
    return transforms.Compose([
        transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--adet", type=int, default=2500,
                    help="kac gorsel kopyalansin (sinir bunun icinde olmali)")
    ap.add_argument("--model", default=str(AB_MODEL))
    ap.add_argument("--kategori-de", action="store_true",
                    help="kategori modelini de calistir (C sinifi kontrolu)")
    a = ap.parse_args()

    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    mp = Path(a.model)
    if not mp.exists():
        print(f"HATA: model yok: {mp}")
        sys.exit(1)

    print(f"{C}\nSINIR BULMA - model_ab3\n{C}")

    ck = torch.load(mp, map_location="cpu")
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, 2)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"Model : {mp}")
    print(f"        dogrulama AUC {ck.get('dog_auc', 0):.4f}, "
          f"tur {ck.get('tur','?')}")

    dosyalar = sorted([f for f in kok.rglob("*")
                       if f.is_file() and f.suffix.lower() in UZANTILAR])
    if not dosyalar:
        print("HATA: gorsel bulunamadi.")
        sys.exit(1)
    print(f"Gorsel: {len(dosyalar)}\n")

    # ---------------- A/B tahmini
    yl = DataLoader(Kume(dosyalar, donusum()), batch_size=BATCH,
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
            hz = n / max(time.time() - t0, 1e-9)
            print(f"\r  A/B taramasi: {n}/{len(dosyalar)}  ({hz:.1f} g/sn)  "
                  f"kalan ~{(len(dosyalar)-n)/max(hz,1e-9)/60:.1f} dk",
                  end="", flush=True)
    print(f"   bitti ({(time.time()-t0)/60:.1f} dk)")

    # ---------------- istege bagli: kategori modeli (C sinifi kontrolu)
    kategori = None
    if a.kategori_de and Path(KATEGORI_MODEL).exists():
        ck2 = torch.load(KATEGORI_MODEL, map_location="cpu")
        sinif_ad = ck2["sinif_ad"]
        net2 = models.resnet18(weights=None)
        net2.fc = nn.Linear(512, len(sinif_ad))
        net2.load_state_dict(ck2["model"])
        net2.eval()
        K = np.zeros(len(dosyalar), dtype=int)
        n, t0 = 0, time.time()
        with torch.no_grad():
            for x, idx in yl:
                t = torch.softmax(net2(x), 1).argmax(1).numpy()
                for j, i in enumerate(idx.tolist()):
                    if i >= 0:
                        K[i] = t[j]
                n += len(idx)
                print(f"\r  kategori    : {n}/{len(dosyalar)}",
                      end="", flush=True)
        print(f"   bitti ({(time.time()-t0)/60:.1f} dk)")
        kategori = [sinif_ad[i] for i in K]

    # ---------------- sirala ve kopyala
    sira = np.argsort(-P)
    N = min(a.adet, len(dosyalar))

    if CIKTI.exists():
        shutil.rmtree(CIKTI)
    CIKTI.mkdir()

    print(f"\n{C}\nDAGILIM\n{C}")
    print(f"En yuksek : {P[sira[0]]:.4f}")
    print(f"Medyan    : {np.median(P):.4f}")
    print(f"En dusuk  : {P[sira[-1]]:.4f}")
    print(f"\nKopyalanacak: ilk {N} gorsel "
          f"(%{100*N/len(dosyalar):.1f})")
    print(f"Olasilik araligi: {P[sira[0]]:.4f} - {P[sira[N-1]]:.4f}")

    kopyalanan = 0
    for rank, i in enumerate(sira[:N], 1):
        f = dosyalar[i]
        yeni = CIKTI / f"{rank:04d}_a{P[i]:.4f}_{f.name}"
        try:
            shutil.copy2(f, yeni)
            kopyalanan += 1
        except Exception as e:
            print(f"  kopyalanamadi: {f.name} ({e})")

    # ---------------- rapor
    with open(CIKTI / "_kayit.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        basliklar = ["sira", "dosya", "A_olasiligi", "kaynak_klasor"]
        if kategori:
            basliklar.append("model_kategorisi")
        w.writerow(basliklar)
        for rank, i in enumerate(sira, 1):
            satir = [rank, dosyalar[i].name, f"{P[i]:.6f}",
                     dosyalar[i].parent.name]
            if kategori:
                satir.append(kategori[i])
            w.writerow(satir)

    # ---------------- C sinifi kontrolu
    if kategori:
        print(f"\n{C}\nC SINIFI KONTROLU\n{C}")
        for kesim in [500, 1000, 1500, 2000, N]:
            if kesim > N:
                continue
            sizan = [kategori[i] for i in sira[:kesim]
                     if any(k in kategori[i] for k in C_ANAHTARLAR)]
            if sizan:
                print(f"  ilk {kesim:>5}: {len(sizan)} C sinifi")
                for k, v in Counter(sizan).most_common():
                    print(f"       {v:>4}x  {kisa(k)}")
            else:
                print(f"  ilk {kesim:>5}: sizinti yok")

    # ---------------- talimat
    with open(CIKTI / "OKU.txt", "w", encoding="utf-8") as fh:
        fh.write("SINIR BULMA\n" + "=" * 56 + "\n\n")
        fh.write("Dosyalar A olasiligina gore SIRALI.\n")
        fh.write("0001 = modelin A olma ihtimalini EN YUKSEK gordugu.\n\n")
        fh.write("NASIL BAKACAKSINIZ\n\n")
        fh.write("  1) Finder'da Gallery gorunumu (Cmd + 4)\n")
        fh.write("  2) Ok tuslariyla bastan sona ilerleyin\n")
        fh.write("  3) Tek soru: 'bu gorsel A sinifi mi?'\n\n")
        fh.write("  Hepsine tek tek bakmaniz gerekmez. Ornekleme yapin:\n")
        fh.write("  500, 1000, 1500, 2000 civarlarindaki onar gorsele bakin.\n")
        fh.write("  B'ler nerede yogunlasmaya basliyor, onu tespit edin.\n\n")
        fh.write("  O sira numarasi gunluk kullanimda kesim noktasi olacak.\n\n")
        fh.write("NOT\n")
        fh.write("  Dosya adindaki 'a0.87' modelin A olasiligidir.\n")
        fh.write("  Test olcumu %18 kesimde precision 0.41 diyor ama bu sayiya\n")
        fh.write("  guvenmiyoruz - onceki iki olcumde gozle bakinca %97-98\n")
        fh.write("  cikmisti. Gercek siniri siz belirleyeceksiniz.\n")

    print(f"\n{C}\nHAZIR\n{C}")
    print(f"Kopyalanan : {kopyalanan}")
    print(f"Klasor     : {CIKTI.resolve()}")
    print(f"\n  open \"{CIKTI.resolve()}\"")
    print(C)
    print("\nOKU.txt dosyasini okuyun.")
    print("Sirayla bakip A'larin nerede bittigini tespit edin.")
    print("O sayiyi bana soyleyin, gunluk scripti ona gore ayarlayalim.")


if __name__ == "__main__":
    main()
