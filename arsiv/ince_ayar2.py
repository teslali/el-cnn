#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - Adim 8: INCE AYAR v2

Ilk ince ayar makro-F1 0.643 verdi (dondurulmus oznitelikte 0.417 idi).
5. turda doydu, sonrasi asiri ogrenmeydi. Bu surum uc degisiklik yapar:

  1) OGRENME HIZI ZAMANLAYICISI (cosine)
     Sabit ogrenme hizi, modelin ince ayar yapmasi gereken noktada hala
     buyuk adimlar atmasina sebep oluyordu. Kademeli dusus asiri ogrenmeyi
     geciktirir ve daha uzun egitmeyi anlamli kilar.

  2) DAHA UZUN EGITIM (14 tur)
     Zamanlayici ile birlikte anlamli hale geliyor.

  3) LABEL SMOOTHING
     Modelin asiri ozguvenli olmasini engeller. Etiket gurultusu olan
     veride (coklu kusur nedeniyle var) hem dogrulugu hem de GUVEN
     KALIBRASYONUNU iyilestirir. Tum sistemimiz esik/guven skoruna
     dayandigi icin bu onemli.

  AYRICA: egitim setinin bir kismi DOGRULAMA icin ayrilir. Onceki surumde
  en iyi tur test kumesine bakilarak seciliyordu - bu teknik olarak test
  kumesini iki kez kullanmak demek. Ayri dogrulama kumesi bunu temizler;
  test skoru artik gercekten bagimsiz.

UYARI: Egitim calisirken goruntu klasorune YENI DOSYA EKLEMEYIN.
Yeni gorselleri gecici bir klasorde biriktirin, egitim bitince tasiyin.

Kullanim:
    python ince_ayar2.py "/Users/umutcan/Desktop/fabrika-fotolar/2B"
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
TUR_SAYISI = 14
LR_GOVDE = 1.5e-4
LR_BAS = 1.5e-3
ISINMA_TUR = 1             # ilk tur dusuk lr ile baslar
LABEL_SMOOTHING = 0.1
MIN_ORNEK = 300
DOGRULAMA_ORANI = 0.15     # egitim setinden ayrilan pay

TEST_GUNLERI = {"20260908", "20260909"}

CIKTI = Path("model2")
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")

torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)

C = "=" * 72


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

    yaz(f"\n{C}\nINCE AYAR v2 - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

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
    kayitlar = [k for k in kayitlar if k[1] in secili]
    sinif_ad = secili
    sinif_idx = {c: i for i, c in enumerate(sinif_ad)}

    yollar = np.array([k[0] for k in kayitlar])
    etk = np.array([sinif_idx[k[1]] for k in kayitlar])
    gun = np.array([k[2] for k in kayitlar])

    te_m = np.isin(gun, list(TEST_GUNLERI))
    egitim_idx = np.where(~te_m)[0]

    # egitimden dogrulama ayir (tabakali)
    rng = np.random.RandomState(0)
    dog_idx = []
    for c in range(len(sinif_ad)):
        ait = egitim_idx[etk[egitim_idx] == c]
        n_dog = max(1, int(len(ait) * DOGRULAMA_ORANI))
        dog_idx.extend(rng.choice(ait, size=min(n_dog, len(ait)),
                                  replace=False))
    dog_idx = np.array(sorted(dog_idx))
    tr_idx = np.array(sorted(set(egitim_idx) - set(dog_idx)))
    te_idx = np.where(te_m)[0]

    yaz(f"Toplam gorsel : {len(kayitlar)}")
    yaz(f"Kategori      : {len(sinif_ad)} (>= {MIN_ORNEK} ornek)")
    yaz(f"Egitim        : {len(tr_idx)}")
    yaz(f"Dogrulama     : {len(dog_idx)}   <- en iyi tur buna gore secilir")
    yaz(f"Test          : {len(te_idx)}   <- sadece final olcum")
    yaz(f"Tur / batch   : {TUR_SAYISI} / {BATCH}")
    yaz(f"LR            : govde {LR_GOVDE}, bas {LR_BAS}, cosine zamanlayici")
    yaz(f"Label smooth  : {LABEL_SMOOTHING}")
    yaz("\nKategoriler:")
    for i, c in enumerate(sinif_ad):
        yaz(f"  {kisa(c):<32} egitim {int((etk[tr_idx]==i).sum()):>5}  "
            f"dog {int((etk[dog_idx]==i).sum()):>4}  "
            f"test {int((etk[te_idx]==i).sum()):>5}")

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

    def rapor(G, T, baslik):
        p, r, f1, s = precision_recall_fscore_support(
            G, T, labels=range(len(sinif_ad)), zero_division=0)
        yaz(f"  {baslik}: dogruluk %{100*(G==T).mean():.1f}  "
            f"makro-P {np.mean(p):.3f}  makro-F1 {np.mean(f1):.3f}")
        return float(np.mean(f1)), (p, r, f1, s)

    yaz(f"\n{C}\nEGITIM BASLIYOR\n{C}")
    yaz("En iyi tur DOGRULAMA kumesine gore secilir (test'e bakilmaz).")
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
        sure = (time.time() - t0) / 60
        yaz(f"\n\n  --- TUR {tur+1}/{TUR_SAYISI}  ({sure:.0f} dk) ---")
        yaz(f"  egitim kaybi : {kayip_top/max(n,1):.4f}   "
            f"lr {opt.param_groups[0]['lr']:.2e}")
        dog_f1, _ = rapor(Gd, Td, "DOGRULAMA")

        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son_yolu)

        if dog_f1 > en_iyi_dog:
            en_iyi_dog = dog_f1
            torch.save({"model": net.state_dict(), "sinif_ad": sinif_ad,
                        "girdi_boyut": GIRDI_BOYUT, "tur": tur + 1,
                        "dog_f1": dog_f1}, en_iyi_yolu)
            yaz(f"  >>> EN IYI (dogrulama F1 {dog_f1:.3f})")

            # test olcumu + tahmin kaydi
            Gt, Tt, Vt = olc(te_y)
            te_f1, (p, r, f1, s) = rapor(Gt, Tt, "  test")
            for i, c in enumerate(sinif_ad):
                if s[i]:
                    yaz(f"     {kisa(c):<30} P {p[i]:.2f}  R {r[i]:.2f}  "
                        f"n {int(s[i])}")
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
    yaz("\nKarsilastirma:")
    yaz("  dondurulmus oznitelik : makro-F1 0.417")
    yaz("  ince ayar v1          : makro-F1 0.643")
    yaz("\nEsik analizi icin esik.py'de MODEL_DIZIN'i 'model2' yapin.")


if __name__ == "__main__":
    main()
