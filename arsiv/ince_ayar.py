#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - Adim 6: INCE AYAR (fine-tuning)

Dondurulmus oznitelik yaklasiminin sinirina geldik. Bu script ResNet18'i
kendi verimizle bastan egitir (ImageNet agirliklariyla baslayarak).

GPU YOK - bu Mac'te CPU uzerinde calisir, saatler surer.
Gece calistirilmak uzere tasarlandi:
  - Her turdan (epoch) sonra test performansi olculur
  - EN IYI tur kaydedilir; sonra kotulesse bile o nokta elde kalir
  - Yarida kesilirse kaldigi yerden devam eder
  - Her tur sonunda ozet dosyaya yazilir (sabah bakmak icin)

Kullanim:
    python ince_ayar.py /gorsellerin/oldugu/klasor

Ornek:
    python ince_ayar.py "/Users/umutcan/Desktop/fabrika-fotolar/2B"

Devam etmek icin ayni komutu tekrar calistirin.
"""

import sys
import json
import time
import csv
import re
from pathlib import Path
from collections import Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
GIRDI_BOYUT = 224          # 384'un daha iyi olmadigini olctuk
BATCH = 24
ISCI = 2
TUR_SAYISI = 8             # epoch
LR_GOVDE = 1e-4            # onceden egitilmis katmanlar: kucuk adim
LR_BAS = 1e-3              # yeni siniflandirma katmani: buyuk adim
MIN_ORNEK = 300            # bu sayinin altindaki kategoriler haric

TEST_GUNLERI = {"20260908", "20260909"}
IYI_KATEGORI = "zz其他Diger"

CIKTI = Path("model")
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


def veri_tara(kok):
    kayitlar = []
    for kat in sorted([d for d in kok.iterdir() if d.is_dir()],
                      key=lambda p: p.name):
        for f in sorted(kat.iterdir()):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                z = ZAMAN.search(f.stem)
                kayitlar.append((str(f.relative_to(kok)), kat.name,
                                 z.group(1)[:8] if z else ""))
    return kayitlar


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    kok = Path(sys.argv[1]).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    CIKTI.mkdir(exist_ok=True)
    kayit_yolu = CIKTI / "durum.json"
    en_iyi_yolu = CIKTI / "en_iyi_model.pt"
    son_yolu = CIKTI / "son_model.pt"
    gunluk = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gunluk, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nINCE AYAR - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    # ---------------- veri
    kayitlar = veri_tara(kok)
    sayim = Counter(k[1] for k in kayitlar)
    secili = sorted({k for k, n in sayim.items() if n >= MIN_ORNEK})
    if not secili:
        yaz("HATA: yeterli ornegi olan kategori yok.")
        sys.exit(1)

    kayitlar = [k for k in kayitlar if k[1] in secili]
    sinif_ad = secili
    sinif_idx = {c: i for i, c in enumerate(sinif_ad)}

    yollar = [k[0] for k in kayitlar]
    etk = np.array([sinif_idx[k[1]] for k in kayitlar])
    gun = np.array([k[2] for k in kayitlar])
    te = np.isin(gun, list(TEST_GUNLERI))

    yaz(f"Toplam gorsel : {len(kayitlar)}")
    yaz(f"Kategori      : {len(sinif_ad)}  (>= {MIN_ORNEK} ornek)")
    yaz(f"Egitim        : {int((~te).sum())}")
    yaz(f"Test          : {int(te.sum())}")
    yaz(f"Girdi         : {GIRDI_BOYUT}x{GIRDI_BOYUT}, batch {BATCH}")
    yaz("\nKategoriler:")
    for c in sinif_ad:
        n_tr = int(((etk == sinif_idx[c]) & ~te).sum())
        n_te = int(((etk == sinif_idx[c]) & te).sum())
        yaz(f"  {kisa(c):<32} egitim {n_tr:>5}  test {n_te:>5}")

    # ---------------- donusumler
    # egitimde hafif veri artirma: EL goruntuleri simetrik, cevirme guvenli
    egitim_donusum = transforms.Compose([
        transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_donusum = transforms.Compose([
        transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tr_yol = [yollar[i] for i in np.where(~te)[0]]
    tr_etk = etk[~te]
    te_yol = [yollar[i] for i in np.where(te)[0]]
    te_etk = etk[te]

    tr_yukle = DataLoader(ELKumesi(tr_yol, tr_etk, kok, egitim_donusum),
                          batch_size=BATCH, shuffle=True,
                          num_workers=ISCI, drop_last=True)
    te_yukle = DataLoader(ELKumesi(te_yol, te_etk, kok, test_donusum),
                          batch_size=BATCH, shuffle=False, num_workers=ISCI)

    # ---------------- model
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, len(sinif_ad))

    # govde ve bas icin farkli ogrenme hizi
    bas = list(net.fc.parameters())
    bas_id = {id(p) for p in bas}
    govde = [p for p in net.parameters() if id(p) not in bas_id]
    opt = torch.optim.AdamW([
        {"params": govde, "lr": LR_GOVDE},
        {"params": bas, "lr": LR_BAS},
    ], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss()

    baslangic_tur = 0
    en_iyi_skor = -1.0

    # devam kontrolu
    if kayit_yolu.exists() and son_yolu.exists():
        try:
            d = json.loads(kayit_yolu.read_text(encoding="utf-8"))
            if d.get("sinif_ad") == sinif_ad and d.get("toplam") == len(kayitlar):
                ckpt = torch.load(son_yolu, map_location="cpu")
                net.load_state_dict(ckpt["model"])
                opt.load_state_dict(ckpt["opt"])
                baslangic_tur = d["tur"]
                en_iyi_skor = d.get("en_iyi_skor", -1.0)
                yaz(f"\nOnceki calisma bulundu: {baslangic_tur}. turdan devam.")
                yaz(f"  en iyi makro-F1 su ana kadar: {en_iyi_skor:.4f}")
            else:
                yaz("\nOnceki kayit uyusmuyor (veri degismis), sifirdan.")
        except Exception as e:
            yaz(f"\nKayit okunamadi ({e}), sifirdan basliyor.")

    def degerlendir():
        net.eval()
        tahminler, gercekler, guvenler = [], [], []
        with torch.no_grad():
            for x, y in te_yukle:
                o = net(x)
                p = torch.softmax(o, 1)
                g, t = p.max(1)
                tahminler.append(t.numpy())
                guvenler.append(g.numpy())
                gercekler.append(y.numpy())
        return (np.concatenate(gercekler), np.concatenate(tahminler),
                np.concatenate(guvenler))

    # ---------------- egitim dongusu
    yaz(f"\n{C}\nEGITIM BASLIYOR\n{C}")
    yaz("Her tur sonunda test olculur ve en iyi model kaydedilir.")
    yaz("Kesilirse ayni komutla kaldigi yerden devam eder.\n")

    for tur in range(baslangic_tur, TUR_SAYISI):
        net.train()
        t0 = time.time()
        toplam_kayip, n = 0.0, 0
        for i, (x, y) in enumerate(tr_yukle):
            opt.zero_grad()
            kayip = kayip_fn(net(x), y)
            kayip.backward()
            opt.step()
            toplam_kayip += float(kayip) * len(y)
            n += len(y)
            if i % 20 == 0:
                gecen = time.time() - t0
                hiz = n / max(gecen, 1e-9)
                kalan = (len(tr_etk) - n) / max(hiz, 1e-9)
                print(f"\r  tur {tur+1}/{TUR_SAYISI}  {n}/{len(tr_etk)}  "
                      f"kayip {toplam_kayip/max(n,1):.4f}  "
                      f"{hiz:.1f} g/sn  kalan ~{kalan/60:.0f} dk",
                      end="", flush=True)

        gercek, tahmin, guven = degerlendir()
        p, r, f1, s = precision_recall_fscore_support(
            gercek, tahmin, labels=range(len(sinif_ad)), zero_division=0)
        makro_f1 = float(np.mean(f1))
        makro_p = float(np.mean(p))
        dog = float((gercek == tahmin).mean())
        sure = (time.time() - t0) / 60

        yaz(f"\n\n  --- TUR {tur+1}/{TUR_SAYISI}  ({sure:.0f} dk) ---")
        yaz(f"  egitim kaybi : {toplam_kayip/max(n,1):.4f}")
        yaz(f"  test dogruluk: {100*dog:.1f}%")
        yaz(f"  makro-P      : {makro_p:.3f}")
        yaz(f"  makro-F1     : {makro_f1:.3f}")
        for i, c in enumerate(sinif_ad):
            if s[i]:
                yaz(f"    {kisa(c):<30} P {p[i]:.2f}  R {r[i]:.2f}  n {int(s[i])}")

        # kayit
        torch.save({"model": net.state_dict(), "opt": opt.state_dict()},
                   son_yolu)
        if makro_f1 > en_iyi_skor:
            en_iyi_skor = makro_f1
            torch.save({"model": net.state_dict(),
                        "sinif_ad": sinif_ad,
                        "girdi_boyut": GIRDI_BOYUT,
                        "tur": tur + 1,
                        "makro_f1": makro_f1}, en_iyi_yolu)
            yaz(f"  >>> EN IYI (makro-F1 {makro_f1:.3f}) kaydedildi")

            # esik analizi sadece en iyi turda
            with open(CIKTI / "tahminler.csv", "w", newline="",
                      encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["yol", "gercek", "tahmin", "guven"])
                for i in range(len(gercek)):
                    w.writerow([te_yol[i], sinif_ad[gercek[i]],
                                sinif_ad[tahmin[i]], f"{guven[i]:.4f}"])
        else:
            yaz(f"  (en iyi hala {en_iyi_skor:.3f} - asiri ogrenme olabilir)")

        kayit_yolu.write_text(json.dumps({
            "tur": tur + 1, "sinif_ad": sinif_ad,
            "toplam": len(kayitlar), "en_iyi_skor": en_iyi_skor,
        }, ensure_ascii=False), encoding="utf-8")

    # ---------------- final
    yaz(f"\n{C}\nBITTI\n{C}")
    yaz(f"En iyi makro-F1 : {en_iyi_skor:.3f}")
    yaz(f"En iyi model    : {en_iyi_yolu}")
    yaz(f"Tahminler       : {CIKTI/'tahminler.csv'}")
    yaz("\nKarsilastirma icin: dondurulmus oznitelikle makro-F1 0.417 idi.")
    yaz("Esik analizi icin tahminler.csv kullanilabilir.")


if __name__ == "__main__":
    main()
