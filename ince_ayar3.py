#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - INCE AYAR v3

Veri 16.989 -> 43.050 cikti. Bu surumde iki onemli degisiklik var:

  1) DOGRULAMA ARTIK GUNE GORE AYRILIYOR
     v2'de dogrulama kumesi egitim gunlerinden RASTGELE ayriliyordu. Ayni
     partiden saniyeler arayla cekilmis hucreler birbirine cok benzedigi icin
     dogrulama bagimsiz degildi: dogrulama F1 0.861 gosterirken test 0.645
     cikiyordu. Artik dogrulama icin AYRI BIR GUN kullaniliyor; en iyi tur
     secimi gercekten bagimsiz bir olcume dayaniyor.

  2) MIN_ORNEK 300 -> 150
     Gizli catlak (164) dahil oluyor. Bu kusur C sinifi oldugu icin en kritik
     olani ama en az veriye sahip. Az veriyle kotu sonuc verebilir; amac
     sinyal olup olmadigini gormek. Kotu cikarsa esik analizinde disarida
     birakilir - diger kategorileri etkilemez.

UYARI: Egitim calisirken goruntu klasorune yeni dosya EKLEMEYIN.

Kullanim:
    caffeinate -i python ince_ayar3.py "/Users/umutcan/Desktop/fabrika-fotolar/2B"
"""

import sys
import json
import time
import csv
import re
import math
from pathlib import Path
from collections import Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
    from sklearn.metrics import precision_recall_fscore_support
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
GIRDI_BOYUT = 224
BATCH = 24
ISCI = 2
TUR_SAYISI = 10            # 43k gorsel: tur basina ~75 dk
LR_GOVDE = 1.5e-4
LR_BAS = 1.5e-3
ISINMA_TUR = 1
LABEL_SMOOTHING = 0.1
MIN_ORNEK = 150            # gizli catlak (164) dahil olsun

# Gun bazli bolme - sizinti yok
TEST_GUNLERI = {"20260910", "20260911"}     # son 2 gun: sadece final olcum
DOGRULAMA_GUNLERI = {"20260909"}            # en iyi tur secimi icin

CIKTI = Path("model3")
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")

torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)

C = "=" * 74


def kisa(ad, n=30):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


class ELKumesi(Dataset):
    def __init__(self, yollar, etiketler, kok, donusum):
        self.yollar, self.etiketler = yollar, etiketler
        self.kok, self.donusum = kok, donusum

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        with Image.open(self.kok / self.yollar[i]) as im:
            x = self.donusum(im.convert("RGB"))
        return x, self.etiketler[i]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    kok = Path(sys.argv[1]).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    CIKTI.mkdir(exist_ok=True)
    durum_yolu = CIKTI / "durum.json"
    en_iyi_yolu = CIKTI / "en_iyi_model.pt"
    son_yolu = CIKTI / "son_model.pt"
    gunluk = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gunluk, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nINCE AYAR v3 - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    # ---------------- veri
    kayitlar = []
    for kat in sorted([d for d in kok.iterdir() if d.is_dir()],
                      key=lambda p: p.name):
        for f in sorted(kat.iterdir()):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                z = ZAMAN.search(f.stem)
                kayitlar.append((str(f.relative_to(kok)), kat.name,
                                 z.group(1)[:8] if z else ""))

    sayim = Counter(k[1] for k in kayitlar)
    secili = sorted({k for k, n in sayim.items() if n >= MIN_ORNEK})
    atlanan = sorted({k: n for k, n in sayim.items() if n < MIN_ORNEK}.items(),
                     key=lambda kv: -kv[1])
    kayitlar = [k for k in kayitlar if k[1] in secili]
    sinif_ad = secili
    sinif_idx = {c: i for i, c in enumerate(sinif_ad)}

    yollar = np.array([k[0] for k in kayitlar])
    etk = np.array([sinif_idx[k[1]] for k in kayitlar])
    gun = np.array([k[2] for k in kayitlar])

    te_idx = np.where(np.isin(gun, list(TEST_GUNLERI)))[0]
    dog_idx = np.where(np.isin(gun, list(DOGRULAMA_GUNLERI)))[0]
    tr_idx = np.where(~np.isin(gun, list(TEST_GUNLERI | DOGRULAMA_GUNLERI)))[0]

    egitim_gun = sorted(set(gun) - TEST_GUNLERI - DOGRULAMA_GUNLERI)
    yaz(f"Toplam gorsel : {len(kayitlar)}")
    yaz(f"Kategori      : {len(sinif_ad)}  (>= {MIN_ORNEK} ornek)")
    yaz(f"\nGUN BAZLI BOLME (sizinti yok):")
    yaz(f"  egitim    : {egitim_gun}  -> {len(tr_idx)} gorsel")
    yaz(f"  dogrulama : {sorted(DOGRULAMA_GUNLERI)}  -> {len(dog_idx)} gorsel")
    yaz(f"  test      : {sorted(TEST_GUNLERI)}  -> {len(te_idx)} gorsel")
    if atlanan:
        yaz(f"\nAtlanan kategoriler (< {MIN_ORNEK} ornek):")
        for k, n in atlanan:
            yaz(f"  {n:>5}  {kisa(k)}")

    yaz(f"\nKategoriler:")
    for i, c in enumerate(sinif_ad):
        yaz(f"  {kisa(c):<32} egitim {int((etk[tr_idx]==i).sum()):>6}  "
            f"dog {int((etk[dog_idx]==i).sum()):>5}  "
            f"test {int((etk[te_idx]==i).sum()):>5}")

    if len(dog_idx) == 0 or len(te_idx) == 0 or len(tr_idx) == 0:
        yaz("\nHATA: bolmelerden biri bos. TEST/DOGRULAMA gunlerini kontrol edin.")
        sys.exit(1)

    # ---------------- donusumler
    egitim_don = transforms.Compose([
        transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    sabit_don = transforms.Compose([
        transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def yukleyici(idx, don, karistir):
        return DataLoader(
            ELKumesi(list(yollar[idx]), etk[idx], kok, don),
            batch_size=BATCH, shuffle=karistir, num_workers=ISCI,
            drop_last=karistir)

    tr_y = yukleyici(tr_idx, egitim_don, True)
    dog_y = yukleyici(dog_idx, sabit_don, False)
    te_y = yukleyici(te_idx, sabit_don, False)

    # ---------------- model
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, len(sinif_ad))
    bas_id = {id(p) for p in net.fc.parameters()}
    govde = [p for p in net.parameters() if id(p) not in bas_id]
    opt = torch.optim.AdamW([
        {"params": govde, "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS},
    ], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    adim_basina = max(len(tr_y), 1)
    toplam_adim = TUR_SAYISI * adim_basina
    isinma_adim = ISINMA_TUR * adim_basina

    def lr_carpani(adim):
        if adim < isinma_adim:
            return (adim + 1) / max(isinma_adim, 1)
        ilerleme = (adim - isinma_adim) / max(toplam_adim - isinma_adim, 1)
        return 0.5 * (1 + math.cos(math.pi * min(ilerleme, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_carpani)

    baslangic_tur, en_iyi_dog = 0, -1.0
    if durum_yolu.exists() and son_yolu.exists():
        try:
            d = json.loads(durum_yolu.read_text(encoding="utf-8"))
            if d.get("sinif_ad") == sinif_ad and d.get("toplam") == len(kayitlar):
                ck = torch.load(son_yolu, map_location="cpu")
                net.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                baslangic_tur = d["tur"]
                en_iyi_dog = d.get("en_iyi_dog", -1.0)
                yaz(f"\nDevam: {baslangic_tur}. turdan. "
                    f"en iyi dogrulama F1 {en_iyi_dog:.4f}")
            else:
                yaz("\nVeri degismis, sifirdan basliyor.")
        except Exception as e:
            yaz(f"\nKayit okunamadi ({e}), sifirdan.")

    def olc(yukle):
        net.eval()
        T, G, V = [], [], []
        with torch.no_grad():
            for x, y in yukle:
                p = torch.softmax(net(x), 1)
                g, t = p.max(1)
                T.append(t.numpy()); V.append(g.numpy()); G.append(y.numpy())
        return (np.concatenate(G), np.concatenate(T), np.concatenate(V))

    yaz(f"\n{C}\nEGITIM BASLIYOR\n{C}")
    yaz("En iyi tur AYRI BIR GUNUN verisiyle secilir (test'e bakilmaz).")
    yaz("Kesilirse ayni komutla devam eder.\n")

    for tur in range(baslangic_tur, TUR_SAYISI):
        net.train()
        t0, kayip_top, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(tr_y):
            opt.zero_grad()
            kayip = kayip_fn(net(x), y)
            kayip.backward()
            opt.step()
            sched.step()
            kayip_top += float(kayip) * len(y); n += len(y)
            if i % 20 == 0:
                hiz = n / max(time.time() - t0, 1e-9)
                kalan = (len(tr_idx) - n) / max(hiz, 1e-9)
                print(f"\r  tur {tur+1}/{TUR_SAYISI}  {n}/{len(tr_idx)}  "
                      f"kayip {kayip_top/max(n,1):.4f}  lr "
                      f"{opt.param_groups[0]['lr']:.2e}  "
                      f"kalan ~{kalan/60:.0f} dk", end="", flush=True)

        Gd, Td, _ = olc(dog_y)
        p, r, f1, s = precision_recall_fscore_support(
            Gd, Td, labels=range(len(sinif_ad)), zero_division=0)
        dog_f1 = float(np.mean(f1))
        sure = (time.time() - t0) / 60

        yaz(f"\n\n  --- TUR {tur+1}/{TUR_SAYISI}  ({sure:.0f} dk) ---")
        yaz(f"  egitim kaybi : {kayip_top/max(n,1):.4f}   "
            f"lr {opt.param_groups[0]['lr']:.2e}")
        yaz(f"  DOGRULAMA: dogruluk %{100*(Gd==Td).mean():.1f}  "
            f"makro-P {np.mean(p):.3f}  makro-F1 {dog_f1:.3f}")

        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son_yolu)

        if dog_f1 > en_iyi_dog:
            en_iyi_dog = dog_f1
            torch.save({"model": net.state_dict(), "sinif_ad": sinif_ad,
                        "girdi_boyut": GIRDI_BOYUT, "tur": tur + 1,
                        "dog_f1": dog_f1}, en_iyi_yolu)
            yaz(f"  >>> EN IYI (dogrulama F1 {dog_f1:.3f})")

            Gt, Tt, Vt = olc(te_y)
            pt, rt, f1t, st = precision_recall_fscore_support(
                Gt, Tt, labels=range(len(sinif_ad)), zero_division=0)
            yaz(f"    TEST: dogruluk %{100*(Gt==Tt).mean():.1f}  "
                f"makro-P {np.mean(pt):.3f}  makro-F1 {np.mean(f1t):.3f}")
            sirali = sorted(range(len(sinif_ad)), key=lambda i: -pt[i])
            for i in sirali:
                if st[i]:
                    isaret = "  <--" if pt[i] >= 0.90 else ""
                    yaz(f"     {kisa(sinif_ad[i]):<30} P {pt[i]:.2f}  "
                        f"R {rt[i]:.2f}  n {int(st[i])}{isaret}")
            with open(CIKTI / "tahminler.csv", "w", newline="",
                      encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["yol", "gercek", "tahmin", "guven"])
                for j in range(len(Gt)):
                    w.writerow([yollar[te_idx][j], sinif_ad[Gt[j]],
                                sinif_ad[Tt[j]], f"{Vt[j]:.4f}"])
        else:
            yaz(f"  (en iyi dogrulama hala {en_iyi_dog:.3f})")

        durum_yolu.write_text(json.dumps({
            "tur": tur + 1, "sinif_ad": sinif_ad,
            "toplam": len(kayitlar), "en_iyi_dog": en_iyi_dog,
        }, ensure_ascii=False), encoding="utf-8")

    yaz(f"\n{C}\nBITTI\n{C}")
    yaz(f"En iyi dogrulama F1 : {en_iyi_dog:.3f}")
    yaz(f"Model               : {en_iyi_yolu}")
    yaz(f"Tahminler           : {CIKTI/'tahminler.csv'}")
    yaz("\nOnceki sonuclar (16.989 gorsel, 9 kategori):")
    yaz("  dondurulmus oznitelik : test makro-F1 0.417")
    yaz("  ince ayar v1          : test makro-F1 0.643   (esik: %74.0 otomatik)")
    yaz("  ince ayar v2          : test makro-F1 0.645   (esik: %76.5 otomatik)")
    yaz("\nEsik analizi: esik.py'de MODEL_DIZIN = Path('model3') yapin.")


if __name__ == "__main__":
    main()
