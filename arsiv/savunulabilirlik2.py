#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Ayiklama - SAVUNULABILIRLIK OLCUMU v2 (temsili ornekleme)

v1'de kategorilere ESIT dagitim yapiliyordu; bu, gercek uretim dagilimini
temsil etmiyordu. Bu surum GERCEK DAGILIMA gore ornek alir.

Cevabini aradigimiz soru:
  "Esik olmadan, HER SEYI otomatik siniflandirsak, etiketlerin yuzde kaci
   savunulabilir olur?"

v1 sonucu (dengeli ornekleme, 150 gorsel): ~%95 savunulabilir
  - operator etiketine gore olculen dogruluk    : %78.8
  - insan gozuyle savunulabilirlik              : ~%95
  Aradaki fark COKLU KUSUR etiket gurultusudur.

IS AKISI
  1) python savunulabilirlik2.py <klasor>        -> gorselleri hazirlar
  2) Klasorlere bakip HATALI olanlari _HATALI/ klasorune TASIYIN
  3) python savunulabilirlik2.py --sayim         -> sonucu hesaplar
                                                     (guven skorlariyla)

Kullanim:
    python savunulabilirlik2.py /gorsellerin/oldugu/klasor
    python savunulabilirlik2.py /klasor --adet 200
    python savunulabilirlik2.py --sayim
"""

import sys
import csv
import json
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
CIKTI = Path("savunulabilirlik2")
HATALI = "_HATALI"
C = "=" * 74

torch.set_num_threads(4)


def kisa(ad, n=34):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def guvenli(ad):
    for ch in '/\\:*?"<>|':
        ad = ad.replace(ch, "-")
    return ad.strip()[:70] or "bilinmeyen"


def wilson(k, n, z=1.96):
    """Wilson guven araligi - kucuk orneklemde dogru olan yontem."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    pay = p + z * z / (2 * n)
    kok = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    bol = 1 + z * z / n
    return (max(0.0, (pay - kok) / bol), min(1.0, (pay + kok) / bol))


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


# ----------------------------------------------------------------------
# SAYIM MODU
# ----------------------------------------------------------------------
def sayim():
    kayit_yolu = CIKTI / "kayit.csv"
    if not kayit_yolu.exists():
        print(f"HATA: {kayit_yolu} yok. Once hazirlik calistirin.")
        sys.exit(1)

    kayitlar = []
    with open(kayit_yolu, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            r["guven"] = float(r["guven"])
            kayitlar.append(r)

    hatali_dizin = CIKTI / HATALI
    hatali_adlar = set()
    if hatali_dizin.is_dir():
        for f in hatali_dizin.rglob("*"):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                hatali_adlar.add(f.name)

    toplam = len(kayitlar)
    hatali_n = 0
    kat_top, kat_hata = Counter(), Counter()
    hata_guvenleri, dogru_guvenleri = [], []

    for k in kayitlar:
        kat = k["model_tahmini"]
        kat_top[kat] += 1
        if k["kopya_adi"] in hatali_adlar:
            hatali_n += 1
            kat_hata[kat] += 1
            hata_guvenleri.append(k["guven"])
        else:
            dogru_guvenleri.append(k["guven"])

    dogru = toplam - hatali_n
    oran = dogru / max(toplam, 1)
    alt, ust = wilson(dogru, toplam)

    print(f"\n{C}\nSAVUNULABILIRLIK SONUCU (temsili ornekleme)\n{C}")
    print(f"Incelenen gorsel : {toplam}")
    print(f"Savunulabilir    : {dogru}")
    print(f"Hatali           : {hatali_n}")
    print(f"\n  ORAN : %{100*oran:.1f}")
    print(f"  %95 guven araligi : %{100*alt:.1f} - %{100*ust:.1f}")

    print(f"\n{C}\nKATEGORI BAZINDA\n{C}")
    print(f"  {'kategori':<36} {'toplam':>7} {'hata':>6} {'oran':>8}")
    print("  " + "-" * 60)
    for kat, n in kat_top.most_common():
        h = kat_hata[kat]
        print(f"  {kisa(kat):<36} {n:>7} {h:>6} "
              f"{100*(n-h)/max(n,1):>7.0f}%")

    if hata_guvenleri:
        print(f"\n{C}\nGUVEN SKORU ANALIZI\n{C}")
        print(f"  Hatali olanlarin ort. guveni  : "
              f"{np.mean(hata_guvenleri):.3f}")
        print(f"  Dogru olanlarin ort. guveni   : "
              f"{np.mean(dogru_guvenleri):.3f}")
        print(f"\n  Hatalilarin guven dagilimi:")
        for e in [0.5, 0.7, 0.9, 0.95]:
            n = sum(1 for g in hata_guvenleri if g >= e)
            print(f"    guven >= {e:.2f} : {n}/{len(hata_guvenleri)}")
        if np.mean(hata_guvenleri) < np.mean(dogru_guvenleri) - 0.05:
            print("\n  -> Hatalar DUSUK guvenli. Esik koymak bunlari eleyebilir;")
            print("     dusuk esikli bir filtre kaliteyi artirir.")
        else:
            print("\n  -> Hatalar YUKSEK guvenli. Esik koymak bunlari elemez;")
            print("     esik sadece kapsamayi dusurur, kaliteyi artirmaz.")
            print("     Bu durumda TAM OTOMASYON mantikli.")

    print(f"\n{C}\nYORUM\n{C}")
    if oran >= 0.93:
        print("  Etiketler buyuk olcude savunulabilir.")
        print("  TAM OTOMASYON makul: esik kaldirilip tum gorseller")
        print("  otomatik siniflandirilabilir.")
    elif oran >= 0.85:
        print("  Kismi. Bazi kategoriler otomatik olabilir, bazilari esikli")
        print("  kalmali. Kategori bazindaki tabloya bakin.")
    else:
        print("  Model gercekten yaniliyor. Esik gerekli,")
        print("  tam otomasyon erken.")
    print(C)


# ----------------------------------------------------------------------
# HAZIRLIK MODU
# ----------------------------------------------------------------------
def hazirla(a):
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

    print(f"Bulunan gorsel : {len(dosyalar)}")
    print("\nGercek dagilim (ilk 10):")
    for k, n in Counter(etiketler).most_common(10):
        print(f"  {kisa(k):<36} {n:>7}  (%{100*n/len(dosyalar):.1f})")

    # TEMSILI ornekleme: tum havuzdan duz rastgele
    rng = np.random.RandomState(a.tohum)
    adet = min(a.adet, len(dosyalar))
    sec = rng.choice(len(dosyalar), size=adet, replace=False)
    sec_dosya = [dosyalar[i] for i in sec]
    sec_etiket = [etiketler[i] for i in sec]

    print(f"\nTemsili ornek secildi: {adet} gorsel")
    print("(gercek dagilima gore, dengeli DEGIL)")

    donusum = transforms.Compose([
        transforms.Resize((boyut, boyut)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    yukleyici = DataLoader(Kume(sec_dosya, donusum), batch_size=BATCH,
                           shuffle=False, num_workers=ISCI)

    print("\nTahmin yapiliyor...")
    tah = np.zeros(adet, dtype=int)
    guv = np.zeros(adet, dtype=float)
    t0, n = time.time(), 0
    with torch.no_grad():
        for x, idx in yukleyici:
            p = torch.softmax(net(x), 1)
            g, t = p.max(1)
            for j, i in enumerate(idx.tolist()):
                if i >= 0:
                    tah[i] = int(t[j]); guv[i] = float(g[j])
            n += len(idx)
            print(f"\r  {n}/{adet}", end="", flush=True)
    print()

    if CIKTI.exists():
        shutil.rmtree(CIKTI)
    CIKTI.mkdir()
    (CIKTI / HATALI).mkdir()

    kayitlar = []
    for i in range(adet):
        kat = sinif_ad[tah[i]]
        hedef = CIKTI / guvenli(kisa(kat, 40))
        hedef.mkdir(parents=True, exist_ok=True)
        kopya_adi = f"g{guv[i]:.2f}_{sec_dosya[i].name}"
        try:
            shutil.copy2(sec_dosya[i], hedef / kopya_adi)
        except Exception as e:
            print(f"  kopyalanamadi: {e}")
            continue
        kayitlar.append({
            "kopya_adi": kopya_adi,
            "orijinal": sec_dosya[i].name,
            "klasor_etiketi": sec_etiket[i],
            "model_tahmini": kat,
            "guven": f"{guv[i]:.4f}",
            "farkli_mi": "evet" if kat != sec_etiket[i] else "hayir",
        })

    with open(CIKTI / "kayit.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kayitlar[0].keys()))
        w.writeheader(); w.writerows(kayitlar)

    say = Counter(k["model_tahmini"] for k in kayitlar)
    farkli = sum(1 for k in kayitlar if k["farkli_mi"] == "evet")

    with open(CIKTI / "OKU.txt", "w", encoding="utf-8") as fh:
        fh.write("SAVUNULABILIRLIK OLCUMU - TEMSILI ORNEKLEME\n")
        fh.write("=" * 66 + "\n\n")
        fh.write("NE YAPACAKSINIZ\n\n")
        fh.write("  1) Her klasore bakin. Klasor adi = modelin tahmini.\n\n")
        fh.write("  2) Her gorsel icin TEK SORU:\n")
        fh.write("       'Bu gorselde, klasorun adindaki kusur VAR MI?'\n\n")
        fh.write("  3) Kusur YOKSA, o dosyayi _HATALI/ klasorune TASIYIN.\n")
        fh.write("     (kopyalamayin, tasiyin)\n\n")
        fh.write("  4) Bitince su komutu calistirin:\n")
        fh.write("       python savunulabilirlik2.py --sayim\n\n")
        fh.write("ONEMLI\n\n")
        fh.write("  Operatorun daha once ne dedigi ONEMLI DEGIL.\n")
        fh.write("  Hucrede birden fazla kusur olabilir; model baska bir\n")
        fh.write("  gecerli kusuru gormus olabilir - bu YANLIS SAYILMAZ.\n\n")
        fh.write("  Dosya adindaki 'g0.87' modelin guven skorudur.\n")
        fh.write("  Bakarken dikkate almayin; sayim sirasinda otomatik\n")
        fh.write("  analiz edilecek.\n\n")
        fh.write("=" * 66 + "\n")
        fh.write(f"Toplam gorsel : {len(kayitlar)}\n")
        fh.write(f"Klasor etiketinden farkli : {farkli} "
                 f"(%{100*farkli/max(len(kayitlar),1):.0f})\n\n")
        for kat, n_ in say.most_common():
            fh.write(f"  {kisa(kat,40):<42} {n_:>5}\n")

    print(f"\n{C}\nHAZIR\n{C}")
    print(f"Incelenecek : {len(kayitlar)} gorsel (temsili ornek)")
    print(f"Klasor etiketinden farkli : {farkli} "
          f"(%{100*farkli/max(len(kayitlar),1):.0f})")
    print(f"\nKategori dagilimi:")
    for kat, n_ in say.most_common():
        print(f"  {kisa(kat):<40} {n_:>5}")
    print(f"\n  open {CIKTI.resolve()}")
    print("\nOKU.txt dosyasini okuyun.")
    print("Hatali buldugunuz dosyalari _HATALI/ klasorune TASIYIN,")
    print("sonra:  python savunulabilirlik2.py --sayim")
    print(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor", nargs="?")
    ap.add_argument("--adet", type=int, default=200)
    ap.add_argument("--sayim", action="store_true")
    ap.add_argument("--model", default=str(MODEL_YOLU))
    ap.add_argument("--tohum", type=int, default=1)
    a = ap.parse_args()

    if a.sayim:
        sayim()
    elif a.klasor:
        hazirla(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
