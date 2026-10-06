#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Ayiklama - TAM OTOMASYON

Savunulabilirlik olcumu (200 gorsel, temsili ornekleme) %97.5 cikti.
Hatalarin ortalama guveni 0.644, dogrularin 0.825 idi; hicbir hata 0.90
uzerinde degildi. Bu yuzden ESIK KULLANILMIYOR - her gorsel bir kategoriye
atanir.

Iki istisna: olcumde zayif cikan NOKTA BASKISI ve GIZLI CATLAK tahminleri
ayri bir _kontrol klasorune gider. Bu iki kategori toplam hacmin ~%3.5'i.

VARSAYILAN: kopyalar, tasimaz. Kaynak klasore dokunmaz.

Kullanim:
    python otomasyon.py /klasor
    python otomasyon.py /klasor --cikti sonuc
    python otomasyon.py /klasor --tasi              (DIKKAT: tasir)
    python otomasyon.py /klasor --kontrol-yok       (istisna klasoru olmasin)
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

MODEL_YOLU = Path("model3") / "en_iyi_model.pt"
BATCH = 24
ISCI = 2
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}

# Olcumde zayif cikan kategoriler -> ayri klasore, operator baksin
KONTROL_KATEGORILER = {"卡点", "隐裂"}     # nokta baskisi, gizli catlak
KONTROL_KLASOR = "_kontrol"

# Bu esigin altindakiler raporda isaretlenir (ayrilmaz, sadece bilgi)
DUSUK_GUVEN = 0.60

C = "=" * 72
torch.set_num_threads(4)


def kisa(ad, n=34):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def guvenli(ad):
    for ch in '/\\:*?"<>|':
        ad = ad.replace(ch, "-")
    return ad.strip()[:70] or "bilinmeyen"


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
            return torch.zeros(3, 224, 224), -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--cikti", default="otomasyon_cikti")
    ap.add_argument("--tasi", action="store_true")
    ap.add_argument("--kontrol-yok", action="store_true")
    ap.add_argument("--model", default=str(MODEL_YOLU))
    a = ap.parse_args()

    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    mp = Path(a.model)
    if not mp.exists():
        print(f"HATA: model yok: {mp}")
        sys.exit(1)

    ck = torch.load(mp, map_location="cpu")
    sinif_ad = ck["sinif_ad"]
    boyut = ck.get("girdi_boyut", 224)
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, len(sinif_ad))
    net.load_state_dict(ck["model"])
    net.eval()

    print(f"{C}\nTAM OTOMASYON\n{C}")
    print(f"Model     : {mp}  ({len(sinif_ad)} kategori)")
    print(f"Esik      : YOK - her gorsel bir kategoriye atanir")
    if not a.kontrol_yok:
        kn = [kisa(c) for c in sinif_ad
              if any(k in c for k in KONTROL_KATEGORILER)]
        print(f"Istisna   : {', '.join(kn)} -> {KONTROL_KLASOR}/")

    dosyalar = sorted([f for f in kok.rglob("*")
                       if f.is_file() and f.suffix.lower() in UZANTILAR])
    if not dosyalar:
        print("HATA: gorsel bulunamadi.")
        sys.exit(1)
    print(f"Gorsel    : {len(dosyalar)}")

    donusum = transforms.Compose([
        transforms.Resize((boyut, boyut)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    yukleyici = DataLoader(Kume(dosyalar, donusum), batch_size=BATCH,
                           shuffle=False, num_workers=ISCI)

    print("\nIsleniyor...")
    t0 = time.time()
    tah = np.zeros(len(dosyalar), dtype=int)
    guv = np.zeros(len(dosyalar), dtype=float)
    n = 0
    with torch.no_grad():
        for x, idx in yukleyici:
            p = torch.softmax(net(x), 1)
            g, t = p.max(1)
            for j, i in enumerate(idx.tolist()):
                if i >= 0:
                    tah[i] = int(t[j]); guv[i] = float(g[j])
            n += len(idx)
            hiz = n / max(time.time() - t0, 1e-9)
            print(f"\r  {n}/{len(dosyalar)}  ({hiz:.1f} g/sn)  "
                  f"kalan ~{(len(dosyalar)-n)/max(hiz,1e-9)/60:.1f} dk",
                  end="", flush=True)
    sure = time.time() - t0
    print(f"\n  bitti: {sure/60:.1f} dk  "
          f"({len(dosyalar)/max(sure,1e-9):.1f} gorsel/sn)")

    # karar
    kararlar = []
    for i, f in enumerate(dosyalar):
        kat = sinif_ad[tah[i]]
        kontrol = (not a.kontrol_yok and
                   any(k in kat for k in KONTROL_KATEGORILER))
        kararlar.append({
            "dosya": f,
            "tahmin": kat,
            "hedef": KONTROL_KLASOR if kontrol else kat,
            "guven": guv[i],
            "kontrol": kontrol,
        })

    # kopyala / tasi
    cikti = Path(a.cikti).resolve()
    cikti.mkdir(parents=True, exist_ok=True)
    islem = shutil.move if a.tasi else shutil.copy2
    print(f"\n{'TASINIYOR' if a.tasi else 'Kopyalaniyor'}: {cikti}")
    if a.tasi:
        print("  DIKKAT: dosyalar kaynaktan TASINIYOR.")

    sayac = Counter()
    hata = 0
    for k in kararlar:
        ad = guvenli(kisa(k["hedef"], 40)) if not k["kontrol"] else KONTROL_KLASOR
        hedef = cikti / ad
        hedef.mkdir(parents=True, exist_ok=True)
        # kontrol klasorunde hangi kategori oldugu dosya adinda gorunsun
        if k["kontrol"]:
            yeni_ad = f"{guvenli(kisa(k['tahmin'],20))}_g{k['guven']:.2f}_{k['dosya'].name}"
        else:
            yeni_ad = k["dosya"].name
        hedef_dosya = hedef / yeni_ad
        if hedef_dosya.exists():
            hedef_dosya = hedef / f"{Path(yeni_ad).stem}_{sayac[ad]}{k['dosya'].suffix}"
        try:
            islem(str(k["dosya"]), str(hedef_dosya))
            sayac[ad] += 1
        except Exception as e:
            hata += 1
            if hata <= 5:
                print(f"  hata: {k['dosya'].name} ({e})")

    # rapor
    rapor = cikti / "rapor.csv"
    with open(rapor, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dosya", "atanan_klasor", "model_tahmini",
                    "guven", "dusuk_guven_mi"])
        for k in kararlar:
            w.writerow([k["dosya"].name,
                        KONTROL_KLASOR if k["kontrol"] else k["tahmin"],
                        k["tahmin"], f"{k['guven']:.4f}",
                        "evet" if k["guven"] < DUSUK_GUVEN else "hayir"])

    # ozet
    kontrol_n = sum(1 for k in kararlar if k["kontrol"])
    oto_n = len(kararlar) - kontrol_n
    dusuk = sum(1 for k in kararlar if k["guven"] < DUSUK_GUVEN)

    print(f"\n{C}\nSONUC\n{C}")
    print(f"Toplam gorsel        : {len(dosyalar)}")
    print(f"Otomatik siniflanan  : {oto_n}  (%{100*oto_n/len(dosyalar):.1f})")
    if not a.kontrol_yok:
        print(f"Kontrole giden       : {kontrol_n}  "
              f"(%{100*kontrol_n/len(dosyalar):.1f})")
    print(f"Islem suresi         : {sure/60:.1f} dk")
    print(f"Dusuk guvenli (<{DUSUK_GUVEN}) : {dusuk}  "
          f"(%{100*dusuk/len(dosyalar):.1f})  - raporda isaretli")

    print(f"\n{C}\nKATEGORI DAGILIMI\n{C}")
    say = Counter(k["tahmin"] for k in kararlar)
    print(f"  {'kategori':<36} {'adet':>7} {'%':>7} {'ort.guven':>10}")
    print("  " + "-" * 64)
    for kat, adet in say.most_common():
        g = [k["guven"] for k in kararlar if k["tahmin"] == kat]
        isaret = "  <- kontrol" if any(x in kat for x in KONTROL_KATEGORILER) \
            and not a.kontrol_yok else ""
        print(f"  {kisa(kat):<36} {adet:>7} {100*adet/len(dosyalar):>6.1f}% "
              f"{np.mean(g):>10.3f}{isaret}")

    print(f"\n{C}")
    print(f"Cikti : {cikti}")
    print(f"Rapor : {rapor}")
    print(C)
    print("\nOlculen savunulabilirlik: %97.5 (200 gorsel, temsili ornek)")
    print("Yani ~100 gorselden 2-3'unde etiket tartismali olabilir.")
    print("\nKontrol icin: rapor.csv'de 'dusuk_guven_mi = evet' olanlara bakin;")
    print("olcumde hatalarin ortalama guveni 0.644, dogrularinki 0.825 idi.")


if __name__ == "__main__":
    main()
