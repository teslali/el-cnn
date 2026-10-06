#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B SIDDET MODELI - TTA + MODEL BIRLESTIRME KARSILASTIRMASI

IKI IYILESTIRME DENENIYOR, HER IKISI DE YENIDEN EGITIM GEREKTIRMIYOR:

  1) TTA (Test Time Augmentation)
     Her gorsel 4 kez islenir: orijinal, yatay cevrilmis, dikey cevrilmis,
     ikisi birden. Olasiliklarin ortalamasi alinir.
     EL goruntuleri simetrik oldugu icin bu guvenli. Tipik kazanc %1-2 puan.

  2) MODEL BIRLESTIRME (ensemble)
     model_ab (6 tur) ve model_ab2 (12 tur) ayri ayri calistirilir,
     olasiliklar ortalanir. Farkli egitilmis modeller farkli hatalar yapar;
     ortalama alinca hatalar kismen birbirini goturur.

     ONEMLI: Model dosyalarina DOKUNULMAZ. Sadece ciktilar ortalanir.
     Sonuc kotu cikarsa hicbir sey bozulmaz, eski haline donulur.

BU SCRIPT KARAR VERMEZ, OLCER.
Hangi yontemin daha iyi SIRALADIGINI test verisiyle karsilastirir.
Cunku bizim icin onemli olan sıralama kalitesi (AUC), mutlak olasilik degil.

Kullanim:
    python tta_ensemble.py "/path/2B"
"""

import sys
import time
import re
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
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
GIRDI_BOYUT = 224
BATCH = 24
ISCI = 2
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")

A_SINIFI = "其他"
B_SINIFI = {"黑斑", "黑边"}
TEST_GUNLERI = {"20260910", "20260911"}

MODELLER = [
    ("model_ab",  Path("model_ab")  / "en_iyi_model.pt"),
    ("model_ab2", Path("model_ab2") / "en_iyi_model.pt"),
]

# sira bazli kesim noktalari - gozle inceleme ~1300'u isaret etti
KESIM_ORANLARI = [0.05, 0.10, 0.13, 0.20, 0.26, 0.35]

C = "=" * 76
torch.set_num_threads(4)


class Kume(Dataset):
    def __init__(self, yollar, etiketler, donusum):
        self.yollar, self.etiketler, self.donusum = yollar, etiketler, donusum

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.donusum(im.convert("RGB")), self.etiketler[i]
        except Exception:
            return torch.zeros(3, GIRDI_BOYUT, GIRDI_BOYUT), self.etiketler[i]


def temel_donusum():
    return transforms.Compose([
        transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def model_yukle(yol):
    ck = torch.load(yol, map_location="cpu")
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, 2)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


def tahmin_et(net, yukleyici, tta):
    """A sinifi olasiliklarini dondur. tta=True ise 4 varyantin ortalamasi."""
    P, G = [], []
    with torch.no_grad():
        for x, y in yukleyici:
            if tta:
                varyantlar = [
                    x,
                    torch.flip(x, dims=[3]),          # yatay
                    torch.flip(x, dims=[2]),          # dikey
                    torch.flip(x, dims=[2, 3]),       # ikisi
                ]
                toplam = None
                for v in varyantlar:
                    p = torch.softmax(net(v), 1)[:, 0]
                    toplam = p if toplam is None else toplam + p
                p = (toplam / len(varyantlar)).numpy()
            else:
                p = torch.softmax(net(x), 1)[:, 0].numpy()
            P.append(p)
            G.append(y.numpy())
    return np.concatenate(G), np.concatenate(P)


def sira_tablosu(G, P, baslik):
    """Sira bazli kesim: ilk %X'i alsak precision ne olur."""
    sira = np.argsort(-P)
    toplam_a = int((G == 0).sum())
    N = len(G)
    print(f"\n  {baslik}")
    print(f"    {'ilk %':>7} {'N':>6} {'A_bulunan':>10} "
          f"{'precision':>11} {'A_kapsama':>11}")
    print("    " + "-" * 50)
    satirlar = []
    for o in KESIM_ORANLARI:
        n = max(1, int(N * o))
        sec = sira[:n]
        bul = int((G[sec] == 0).sum())
        satirlar.append((o, n, bul, bul / n, bul / max(toplam_a, 1)))
        print(f"    {100*o:>6.0f}% {n:>6} {bul:>10} "
              f"{bul/n:>11.3f} {bul/max(toplam_a,1):>11.3f}")
    return satirlar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    a = ap.parse_args()
    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    print(f"{C}\nTTA + MODEL BIRLESTIRME KARSILASTIRMASI\n{C}")
    print("Model dosyalarina DOKUNULMUYOR. Sadece ciktilar karsilastiriliyor.\n")

    # ---------------- test verisi
    a_kl, b_kl = None, []
    for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
        if A_SINIFI in d.name:
            a_kl = d
        elif any(b in d.name for b in B_SINIFI):
            b_kl.append(d)
    if a_kl is None or not b_kl:
        print("HATA: gerekli klasorler bulunamadi.")
        sys.exit(1)

    yollar, etk = [], []
    for d, e in [(a_kl, 0)] + [(x, 1) for x in b_kl]:
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                z = ZAMAN.search(f.stem)
                if z and z.group(1)[:8] in TEST_GUNLERI:
                    yollar.append(f); etk.append(e)

    if not yollar:
        print("HATA: test gunlerinde gorsel bulunamadi.")
        print(f"Aranan gunler: {sorted(TEST_GUNLERI)}")
        sys.exit(1)

    etk = np.array(etk)
    print(f"Test kumesi : {len(yollar)} gorsel "
          f"(A {int((etk==0).sum())}, B {int((etk==1).sum())})")
    print(f"Test gunleri: {sorted(TEST_GUNLERI)}")

    yukleyici = DataLoader(Kume(yollar, etk, temel_donusum()),
                           batch_size=BATCH, shuffle=False, num_workers=ISCI)

    # ---------------- modelleri yukle
    yuklu = []
    for ad, yol in MODELLER:
        if not yol.exists():
            print(f"\nUYARI: {yol} yok, atlaniyor.")
            continue
        net, ck = model_yukle(yol)
        yuklu.append((ad, net, ck))
        print(f"  yuklendi: {ad}  (dogrulama F1 {ck.get('dog_f1',0):.3f}, "
              f"tur {ck.get('tur','?')})")

    if not yuklu:
        print("HATA: hicbir model yuklenemedi.")
        sys.exit(1)

    # ---------------- 4 senaryo
    sonuclar = {}

    for ad, net, _ in yuklu:
        for tta in (False, True):
            etiket = f"{ad}{' + TTA' if tta else ''}"
            t0 = time.time()
            G, P = tahmin_et(net, yukleyici, tta)
            sure = time.time() - t0
            try:
                auc = roc_auc_score(1 - G, P)
            except ValueError:
                auc = float("nan")
            sonuclar[etiket] = (G, P, auc, sure)
            print(f"\r  {etiket:<24} AUC {auc:.4f}  ({sure/60:.1f} dk)",
                  flush=True)

    # ---------------- birlestirme
    if len(yuklu) >= 2:
        for tta in (False, True):
            anahtarlar = [f"{ad}{' + TTA' if tta else ''}" for ad, _, _ in yuklu]
            if all(k in sonuclar for k in anahtarlar):
                G = sonuclar[anahtarlar[0]][0]
                P = np.mean([sonuclar[k][1] for k in anahtarlar], axis=0)
                try:
                    auc = roc_auc_score(1 - G, P)
                except ValueError:
                    auc = float("nan")
                etiket = f"BIRLESIK{' + TTA' if tta else ''}"
                sonuclar[etiket] = (G, P, auc, 0.0)
                print(f"  {etiket:<24} AUC {auc:.4f}")

    # ---------------- ozet
    print(f"\n{C}\nAUC KARSILASTIRMASI (yuksek olan daha iyi siralıyor)\n{C}")
    print(f"  {'yontem':<26} {'AUC':>8} {'fark':>9}")
    print("  " + "-" * 46)
    taban = sonuclar.get("model_ab2", list(sonuclar.values())[0])[2]
    sirali = sorted(sonuclar.items(), key=lambda kv: -kv[1][2])
    for ad, (G, P, auc, sure) in sirali:
        fark = auc - taban
        isaret = "  <-- en iyi" if ad == sirali[0][0] else ""
        print(f"  {ad:<26} {auc:>8.4f} {fark:>+9.4f}{isaret}")

    print(f"\n  Taban: model_ab2 (tek model, TTA yok) = {taban:.4f}")

    # ---------------- en iyi icin sira tablosu
    en_iyi_ad = sirali[0][0]
    G, P, auc, _ = sonuclar[en_iyi_ad]
    print(f"\n{C}\nSIRA BAZLI KESIM - {en_iyi_ad}\n{C}")
    yeni = sira_tablosu(G, P, f"{en_iyi_ad} (AUC {auc:.4f})")

    if en_iyi_ad != "model_ab2" and "model_ab2" in sonuclar:
        G2, P2, auc2, _ = sonuclar["model_ab2"]
        eski = sira_tablosu(G2, P2, f"model_ab2 - mevcut (AUC {auc2:.4f})")
        print(f"\n  KARSILASTIRMA (precision farki):")
        print(f"    {'ilk %':>7} {'mevcut':>9} {'yeni':>9} {'fark':>9}")
        print("    " + "-" * 38)
        for (o, n, b1, p1, k1), (_, _, b2, p2, k2) in zip(eski, yeni):
            print(f"    {100*o:>6.0f}% {p1:>9.3f} {p2:>9.3f} {p2-p1:>+9.3f}")

    # ---------------- karar
    print(f"\n{C}\nKARAR\n{C}")
    kazanc = sirali[0][1][2] - taban
    if kazanc > 0.005:
        print(f"  '{en_iyi_ad}' mevcut modelden {kazanc:+.4f} AUC daha iyi.")
        print("  KULLANMAYA DEGER.")
        if "TTA" in en_iyi_ad:
            print("\n  TTA icin: ayikla/ab scriptlerinde her gorseli 4 varyantla")
            print("  isleyip olasiliklari ortalayin. Islem suresi 4 katina cikar.")
        if "BIRLESIK" in en_iyi_ad:
            print("\n  Birlestirme icin: iki modeli de yukleyip olasiliklarin")
            print("  ortalamasini alin. Model dosyalari degismez.")
    else:
        print(f"  En iyi yontem tabandan sadece {kazanc:+.4f} AUC farkli.")
        print("  KAYDA DEGER KAZANC YOK - mevcut kurulumda kalin.")
        print("  Ek karmasiklik getirmeye degmez.")
    print(C)


if __name__ == "__main__":
    main()
