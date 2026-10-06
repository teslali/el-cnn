#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - Adim 2: Oznitelik Cikarma

Tum gorselleri onceden egitilmis ResNet18'den BIR KEZ gecirir, 512 boyutlu
oznitelik vektorlerini diske kaydeder. Sonraki adimlarda bu vektorler
kullanilir; gorseller bir daha okunmaz.

GPU yok, CPU uzerinde calisir. Yarida kesilirse kaldigi yerden devam eder.

Kullanim:
    python oznitelik.py /gorsellerin/oldugu/klasor

Ornek:
    python oznitelik.py "/Users/umutcan/Desktop/fabrika fotolar/2B"
"""

import sys
import time
import json
import re
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
except ImportError as e:
    print(f"HATA: gerekli paket eksik ({e.name})")
    print("  pip install torch==2.2.2 torchvision==0.17.2 pillow 'numpy<2'")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
GIRDI_BOYUT = 224          # 224 hizli. Sonuc zayifsa 384 denenecek.
BATCH = 16                 # 8 GB RAM icin guvenli
ISCI = 2                   # DataLoader worker sayisi (eski CPU, dusuk tut)
CIKTI_DIZIN = Path("ozellikler")
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ZAMAN_KALIBI = re.compile(r"(\d{14})")
KAYDET_HER = 1000          # kac gorselde bir ara kayit

torch.set_num_threads(4)   # 2013 quad-core i7


class GorselKumesi(torch.utils.data.Dataset):
    def __init__(self, yollar, kok, donusum):
        self.yollar = yollar
        self.kok = kok
        self.donusum = donusum

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        p = self.kok / self.yollar[i]
        with Image.open(p) as im:
            im = im.convert("RGB")   # gri de olsa 3 kanala kopyala
            x = self.donusum(im)
        return x, i


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    kok = Path(sys.argv[1]).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor bulunamadi: {kok}")
        sys.exit(1)

    CIKTI_DIZIN.mkdir(exist_ok=True)
    ozn_yol = CIKTI_DIZIN / f"ozellikler_{GIRDI_BOYUT}.npy"
    meta_yol = CIKTI_DIZIN / f"meta_{GIRDI_BOYUT}.json"

    # ------------------------------------------------------------------
    # Dosya listesi (envanter ile ayni siralama mantigi)
    # ------------------------------------------------------------------
    print(f"Klasor taraniyor: {kok}")
    kayitlar = []
    for kat in sorted([d for d in kok.iterdir() if d.is_dir()], key=lambda p: p.name):
        for f in sorted(kat.iterdir()):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                zm = ZAMAN_KALIBI.search(f.stem)
                kayitlar.append({
                    "yol": str(f.relative_to(kok)),
                    "kategori": kat.name,
                    "zaman": zm.group(1) if zm else "",
                })
    N = len(kayitlar)
    if N == 0:
        print("HATA: gorsel bulunamadi.")
        sys.exit(1)
    print(f"Bulunan gorsel: {N}\n")

    # ------------------------------------------------------------------
    # Devam kontrolu
    # ------------------------------------------------------------------
    baslangic = 0
    ozellikler = None
    if ozn_yol.exists() and meta_yol.exists():
        try:
            eski = json.loads(meta_yol.read_text(encoding="utf-8"))
            if eski.get("toplam") == N and eski.get("girdi_boyut") == GIRDI_BOYUT:
                ozellikler = np.load(ozn_yol)
                baslangic = int(eski.get("tamamlanan", 0))
                if baslangic >= N:
                    print("Oznitelikler zaten cikarilmis. Yeniden yapmak icin")
                    print(f"  {CIKTI_DIZIN}/ klasorunu silin.")
                    return
                print(f"Onceki calisma bulundu, {baslangic}. gorselden devam ediliyor.\n")
        except Exception:
            ozellikler, baslangic = None, 0

    # ------------------------------------------------------------------
    # Model: ResNet18, son siniflandirma katmani cikarilmis
    # ------------------------------------------------------------------
    print("Model yukleniyor (ResNet18, ImageNet)...")
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()          # 512 boyutlu oznitelik
    net.eval()

    donusum = transforms.Compose([
        transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if ozellikler is None:
        ozellikler = np.zeros((N, 512), dtype=np.float32)

    kalan = [k["yol"] for k in kayitlar[baslangic:]]
    kume = GorselKumesi(kalan, kok, donusum)
    yukleyici = torch.utils.data.DataLoader(
        kume, batch_size=BATCH, shuffle=False,
        num_workers=ISCI, pin_memory=False)

    print(f"Islem basliyor. Girdi {GIRDI_BOYUT}x{GIRDI_BOYUT}, batch {BATCH}.")
    print("Bu adim bir kez yapilir; sonraki denemeler saniyeler surer.\n")

    t0 = time.time()
    islenen = 0
    with torch.no_grad():
        for x, idx in yukleyici:
            v = net(x).numpy()
            for j, i_local in enumerate(idx.tolist()):
                ozellikler[baslangic + i_local] = v[j]
            islenen += len(idx)

            if islenen % KAYDET_HER < BATCH:
                np.save(ozn_yol, ozellikler)
                meta_yol.write_text(json.dumps({
                    "toplam": N,
                    "tamamlanan": baslangic + islenen,
                    "girdi_boyut": GIRDI_BOYUT,
                    "kok": str(kok),
                    "kayitlar": kayitlar,
                }, ensure_ascii=False), encoding="utf-8")

            gecen = time.time() - t0
            hiz = islenen / max(gecen, 1e-9)
            kalan_sn = (len(kalan) - islenen) / max(hiz, 1e-9)
            print(f"\r  {baslangic+islenen}/{N}  "
                  f"({hiz:.1f} gorsel/sn)  "
                  f"kalan ~{kalan_sn/60:.1f} dk", end="", flush=True)

    # son kayit
    np.save(ozn_yol, ozellikler)
    meta_yol.write_text(json.dumps({
        "toplam": N,
        "tamamlanan": N,
        "girdi_boyut": GIRDI_BOYUT,
        "kok": str(kok),
        "kayitlar": kayitlar,
    }, ensure_ascii=False), encoding="utf-8")

    sure = time.time() - t0
    print(f"\n\n{'='*62}")
    print("TAMAMLANDI")
    print("=" * 62)
    print(f"Gorsel        : {N}")
    print(f"Sure          : {sure/60:.1f} dakika  ({N/max(sure,1e-9):.1f} gorsel/sn)")
    print(f"Oznitelik     : {ozellikler.shape}")
    print(f"Kaydedildi    : {ozn_yol.resolve()}")
    print(f"                {meta_yol.resolve()}")
    print(f"Dosya boyutu  : {ozn_yol.stat().st_size/1e6:.1f} MB")
    print("=" * 62)


if __name__ == "__main__":
    main()
