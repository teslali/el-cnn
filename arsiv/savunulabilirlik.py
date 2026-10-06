#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Ayiklama - SAVUNULABILIRLIK OLCUMU

Cevabini aradigimiz soru:
  Modelin tahminleri, operatorun sectigi etiketten FARKLI oldugunda,
  gercekten YANLIS mi - yoksa hucrede birden fazla kusur oldugu icin
  model baska bir gecerli kusuru mu gormus?

Bu, tam otomasyona gecilip gecilemeyecegini belirleyen tek olcum.
Cunku:
  - Model etiketleri cogunlukla SAVUNULABILIR ise, olctugumuz %78.8 dogruluk
    etiket gurultusunden kaynaklaniyordur ve gercek performans cok daha yuksektir.
  - Gercekten YANLIS ise, tam otomasyon erken demektir; esik gereklidir.

NASIL CALISIR
  Rastgele gorseller secilir ve modelin tahmin ettigi kategoriye gore
  klasorlenir. Siz her klasore bakip su soruyu sorarsiniz:

      "Bu klasordeki gorsellerde, klasorun adindaki kusur VAR MI?"

  Operatorun ne dedigi ONEMLI DEGIL. Sadece etiketin savunulabilir olup
  olmadigina bakin. Sonuclari sonda verilen tabloya yazin.

Kullanim:
    python savunulabilirlik.py /gorsellerin/oldugu/klasor
    python savunulabilirlik.py /klasor --adet 150
    python savunulabilirlik.py /klasor --sadece-farkli    (yalnizca model != etiket)
"""

import sys
import csv
import time
import shutil
import argparse
from pathlib import Path
from collections import Counter, defaultdict

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
CIKTI = Path("savunulabilirlik")
C = "=" * 74

torch.set_num_threads(4)


def kisa(ad, n=32):
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
    ap.add_argument("--adet", type=int, default=150,
                    help="incelenecek toplam gorsel sayisi")
    ap.add_argument("--sadece-farkli", action="store_true",
                    help="yalnizca model tahmini != klasor etiketi olanlar")
    ap.add_argument("--model", default=str(MODEL_YOLU))
    ap.add_argument("--tohum", type=int, default=0)
    a = ap.parse_args()

    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    ck = torch.load(Path(a.model), map_location="cpu")
    sinif_ad = ck["sinif_ad"]
    boyut = ck.get("girdi_boyut", 224)
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, len(sinif_ad))
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"Model: {a.model}  ({len(sinif_ad)} kategori)")

    # dosyalar - alt klasorlu ise klasor adi etiket sayilir
    dosyalar, etiketler = [], []
    altlar = [d for d in kok.iterdir() if d.is_dir()]
    if altlar:
        for d in sorted(altlar):
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in UZANTILAR:
                    dosyalar.append(f); etiketler.append(d.name)
    else:
        for f in sorted(kok.iterdir()):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                dosyalar.append(f); etiketler.append(kok.name)

    if not dosyalar:
        print("HATA: gorsel yok.")
        sys.exit(1)
    print(f"Bulunan gorsel: {len(dosyalar)}")

    # ornekleme: once rastgele buyuk bir alt kume uzerinde tahmin yap
    rng = np.random.RandomState(a.tohum)
    on_adet = min(len(dosyalar), max(a.adet * 12, 2000))
    on_idx = rng.choice(len(dosyalar), size=on_adet, replace=False)
    alt_dosya = [dosyalar[i] for i in on_idx]
    alt_etiket = [etiketler[i] for i in on_idx]

    donusum = transforms.Compose([
        transforms.Resize((boyut, boyut)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    yukleyici = DataLoader(Kume(alt_dosya, donusum), batch_size=BATCH,
                           shuffle=False, num_workers=ISCI)

    print(f"\n{on_adet} gorsel uzerinde tahmin yapiliyor...")
    tah = np.zeros(len(alt_dosya), dtype=int)
    guv = np.zeros(len(alt_dosya), dtype=float)
    t0, n = time.time(), 0
    with torch.no_grad():
        for x, idx in yukleyici:
            p = torch.softmax(net(x), 1)
            g, t = p.max(1)
            for j, i in enumerate(idx.tolist()):
                if i >= 0:
                    tah[i] = int(t[j]); guv[i] = float(g[j])
            n += len(idx)
            hiz = n / max(time.time() - t0, 1e-9)
            print(f"\r  {n}/{on_adet}  ({hiz:.1f} g/sn)", end="", flush=True)
    print()

    # secim havuzu
    havuz = list(range(len(alt_dosya)))
    if a.sadece_farkli:
        havuz = [i for i in havuz if sinif_ad[tah[i]] != alt_etiket[i]]
        print(f"\nModel tahmini klasor etiketinden FARKLI olan: {len(havuz)}")
        if not havuz:
            print("Farkli tahmin yok.")
            sys.exit(0)

    # kategorilere dengeli dagit
    kat_gruplari = defaultdict(list)
    for i in havuz:
        kat_gruplari[sinif_ad[tah[i]]].append(i)

    secilen = []
    kats = sorted(kat_gruplari, key=lambda k: -len(kat_gruplari[k]))
    kalan = a.adet
    for sira, kat in enumerate(kats):
        pay = max(1, int(round(kalan / max(len(kats) - sira, 1))))
        grup = kat_gruplari[kat]
        rng.shuffle(grup)
        al = grup[:min(pay, len(grup))]
        secilen.extend(al)
        kalan -= len(al)
        if kalan <= 0:
            break

    # klasorleme
    if CIKTI.exists():
        shutil.rmtree(CIKTI)
    CIKTI.mkdir()

    kayitlar = []
    for i in secilen:
        kat = sinif_ad[tah[i]]
        hedef = CIKTI / guvenli(kisa(kat, 40))
        hedef.mkdir(parents=True, exist_ok=True)
        yeni = hedef / f"guven{guv[i]:.2f}_{alt_dosya[i].name}"
        try:
            shutil.copy2(alt_dosya[i], yeni)
        except Exception as e:
            print(f"  kopyalanamadi: {e}")
            continue
        kayitlar.append({
            "dosya": alt_dosya[i].name,
            "klasor_etiketi": alt_etiket[i],
            "model_tahmini": kat,
            "guven": guv[i],
            "farkli_mi": "evet" if kat != alt_etiket[i] else "hayir",
        })

    with open(CIKTI / "kayit.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dosya", "klasor_etiketi", "model_tahmini",
                    "guven", "farkli_mi"])
        for k in kayitlar:
            w.writerow([k["dosya"], k["klasor_etiketi"], k["model_tahmini"],
                        f"{k['guven']:.4f}", k["farkli_mi"]])

    # talimat dosyasi
    say = Counter(k["model_tahmini"] for k in kayitlar)
    farkli = sum(1 for k in kayitlar if k["farkli_mi"] == "evet")
    with open(CIKTI / "OKU.txt", "w", encoding="utf-8") as fh:
        fh.write("SAVUNULABILIRLIK OLCUMU\n" + "=" * 66 + "\n\n")
        fh.write("HER KLASORE BAKIN VE SU SORUYU SORUN:\n\n")
        fh.write("   'Bu gorselde, klasorun adindaki kusur VAR MI?'\n\n")
        fh.write("Operatorun daha once ne dedigi ONEMLI DEGIL.\n")
        fh.write("Hucrede birden fazla kusur olabilir; model baska birini\n")
        fh.write("gormus olabilir ve bu YANLIS SAYILMAZ.\n\n")
        fh.write("Sadece sunu sorun: bu etiket savunulabilir mi?\n\n")
        fh.write("=" * 66 + "\nSAYIM TABLOSU (doldurun)\n" + "=" * 66 + "\n\n")
        fh.write(f"{'kategori':<40} {'toplam':>7} {'savunulabilir':>14}\n")
        fh.write("-" * 64 + "\n")
        for kat, n_ in say.most_common():
            fh.write(f"{kisa(kat,38):<40} {n_:>7} {'':>14}\n")
        fh.write("-" * 64 + "\n")
        fh.write(f"{'TOPLAM':<40} {len(kayitlar):>7}\n\n")
        fh.write("=" * 66 + "\nSONUC YORUMU\n" + "=" * 66 + "\n\n")
        fh.write("  %90 ve uzeri savunulabilir\n")
        fh.write("     -> Olctugumuz hata orani buyuk olcude etiket gurultusu.\n")
        fh.write("        TAM OTOMASYON makul. Esik gevsetilebilir.\n\n")
        fh.write("  %80-90 arasi\n")
        fh.write("     -> Kismi. Bazi kategoriler otomatik, bazilari esikli.\n\n")
        fh.write("  %80 altinda\n")
        fh.write("     -> Model gercekten yaniliyor. Esik gerekli,\n")
        fh.write("        tam otomasyon erken.\n")

    print(f"\n{C}\nHAZIR\n{C}")
    print(f"Incelenecek gorsel : {len(kayitlar)}")
    print(f"Klasor etiketinden farkli olan : {farkli} "
          f"(%{100*farkli/max(len(kayitlar),1):.0f})")
    print(f"\nKategori dagilimi:")
    for kat, n_ in say.most_common():
        print(f"  {kisa(kat):<40} {n_:>5}")
    print(f"\nKlasor: {CIKTI.resolve()}")
    print(f"\n  open {CIKTI.resolve()}")
    print(f"\nOnce OKU.txt dosyasini okuyun. Her klasore bakip")
    print("'bu kusur var mi?' sorusunu cevaplayin, tabloyu doldurun.")
    print(C)


if __name__ == "__main__":
    main()
