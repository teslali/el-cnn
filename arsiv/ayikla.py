#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Ayiklama Araci

Egitilmis modeli alir, bir klasordeki gorselleri isler ve GUVEN ESIGINE gore
ayirir:
  - Esigi gecenler  -> kategori klasorune  (otomatik karar)
  - Gecemeyenler    -> belirsiz/ klasorune (operatore gider)

Her kategorinin KENDI esigi vardir; bu esikler esik.py analiziyle olculmustur.
Esikte olculen precision ~%95'tir.

VARSAYILAN MOD: KOPYALAR, tasimaz. Orijinal klasore dokunmaz.
Tasima icin --tasi bayragi gerekir (dikkatli kullanin).

Kullanim:
    python ayikla.py /islenecek/klasor
    python ayikla.py /islenecek/klasor --cikti sonuc
    python ayikla.py /islenecek/klasor --tasi          (DIKKAT: dosyalari tasir)
    python ayikla.py /islenecek/klasor --rapor-only    (dosya kopyalamaz)
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

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_YOLU = Path("model3") / "en_iyi_model.pt"
BATCH = 24
ISCI = 2
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}

# esik.py 0.93 analizinden olculen kategori bazli esikler.
# Kategori adi -> esik.  Listede olmayan kategori ASLA otomatik karara
# baglanmaz; hepsi belirsiz/ klasorune gider.
ESIKLER = {
    "c偏移Baski kaymasi":            0.50,
    "r分片Ayirma kaynakli cizik":    0.70,
    "l舟框印Tekne cercevesi izi":     0.80,
    "k黑边Siyah kenar + siyah kose":  0.80,
    "o云雾Bulutlu hücre":            0.90,
    "w隐裂Gizli catlak":             0.60,
    "f黑斑Siyah leke + siyah nokta":  0.90,
    "n脏污Kirlenme":                 0.80,
    "m气流Hava akisi izi":           0.70,
    "zz其他Diger":                   0.80,
}

BELIRSIZ = "_belirsiz"
C = "=" * 72

torch.set_num_threads(4)


def kisa(ad, n=32):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def guvenli(ad):
    for ch in '/\\:*?"<>|':
        ad = ad.replace(ch, "-")
    return ad.strip()[:80] or "bilinmeyen"


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
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("klasor", help="islenecek gorsellerin bulundugu klasor")
    ap.add_argument("--cikti", default="ayiklama", help="cikti klasoru")
    ap.add_argument("--tasi", action="store_true",
                    help="DIKKAT: kopyalamak yerine TASIR")
    ap.add_argument("--rapor-only", action="store_true",
                    help="dosya kopyalamaz, sadece rapor uretir")
    ap.add_argument("--model", default=str(MODEL_YOLU))
    a = ap.parse_args()

    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    model_yolu = Path(a.model)
    if not model_yolu.exists():
        print(f"HATA: model yok: {model_yolu}")
        sys.exit(1)

    # ---------------- model
    print(f"Model yukleniyor: {model_yolu}")
    ck = torch.load(model_yolu, map_location="cpu")
    sinif_ad = ck["sinif_ad"]
    boyut = ck.get("girdi_boyut", 224)
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, len(sinif_ad))
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"  kategori sayisi : {len(sinif_ad)}")
    print(f"  girdi boyutu    : {boyut}x{boyut}")
    print(f"  egitim turu     : {ck.get('tur','?')}")

    eksik = [c for c in sinif_ad if c not in ESIKLER]
    if eksik:
        print(f"\n  Esigi tanimli OLMAYAN kategoriler "
              f"(hepsi belirsiz'e gider):")
        for c in eksik:
            print(f"    {kisa(c)}")

    # ---------------- dosyalar
    dosyalar = sorted([f for f in kok.rglob("*")
                       if f.is_file() and f.suffix.lower() in UZANTILAR])
    if not dosyalar:
        print("HATA: gorsel bulunamadi.")
        sys.exit(1)
    print(f"\nIslenecek gorsel: {len(dosyalar)}")

    donusum = transforms.Compose([
        transforms.Resize((boyut, boyut)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    yukleyici = DataLoader(Kume(dosyalar, donusum), batch_size=BATCH,
                           shuffle=False, num_workers=ISCI)

    # ---------------- tahmin
    print("\nIsleniyor...")
    t0 = time.time()
    tahminler = np.zeros(len(dosyalar), dtype=int)
    guvenler = np.zeros(len(dosyalar), dtype=float)
    hatali = []
    n = 0
    with torch.no_grad():
        for x, idx in yukleyici:
            p = torch.softmax(net(x), 1)
            g, t = p.max(1)
            for j, i in enumerate(idx.tolist()):
                if i < 0:
                    continue
                tahminler[i] = int(t[j])
                guvenler[i] = float(g[j])
            n += len(idx)
            hiz = n / max(time.time() - t0, 1e-9)
            print(f"\r  {n}/{len(dosyalar)}  ({hiz:.1f} g/sn)  "
                  f"kalan ~{(len(dosyalar)-n)/max(hiz,1e-9)/60:.1f} dk",
                  end="", flush=True)
    sure = time.time() - t0
    print(f"\n  bitti: {sure/60:.1f} dk  ({len(dosyalar)/max(sure,1e-9):.1f} g/sn)")

    # ---------------- karar
    kararlar = []
    for i, f in enumerate(dosyalar):
        kat = sinif_ad[tahminler[i]]
        esik = ESIKLER.get(kat)
        otomatik = esik is not None and guvenler[i] >= esik
        kararlar.append({
            "dosya": f,
            "kategori": kat if otomatik else BELIRSIZ,
            "model_tahmini": kat,
            "guven": guvenler[i],
            "otomatik": otomatik,
        })

    oto = [k for k in kararlar if k["otomatik"]]
    bel = [k for k in kararlar if not k["otomatik"]]

    # ---------------- kopyala / tasi
    cikti = Path(a.cikti).resolve()
    if not a.rapor_only:
        cikti.mkdir(parents=True, exist_ok=True)
        islem = shutil.move if a.tasi else shutil.copy2
        etiket = "TASINIYOR" if a.tasi else "kopyalaniyor"
        print(f"\n{etiket}: {cikti}")
        if a.tasi:
            print("  DIKKAT: dosyalar kaynak klasorden TASINIYOR.")
        sayac = Counter()
        for k in kararlar:
            hedef = cikti / guvenli(k["kategori"])
            hedef.mkdir(parents=True, exist_ok=True)
            yeni = hedef / k["dosya"].name
            if yeni.exists():
                yeni = hedef / f"{k['dosya'].stem}_{sayac[k['kategori']]}{k['dosya'].suffix}"
            try:
                islem(str(k["dosya"]), str(yeni))
                sayac[k["kategori"]] += 1
            except Exception as e:
                print(f"  hata: {k['dosya'].name} ({e})")

    # ---------------- rapor
    rapor = (cikti if not a.rapor_only else Path(".")) / "ayiklama_raporu.csv"
    rapor.parent.mkdir(parents=True, exist_ok=True)
    with open(rapor, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dosya", "atanan_klasor", "model_tahmini",
                    "guven", "otomatik_mi"])
        for k in kararlar:
            w.writerow([k["dosya"].name, k["kategori"], k["model_tahmini"],
                        f"{k['guven']:.4f}", "evet" if k["otomatik"] else "hayir"])

    # ---------------- ozet
    print(f"\n{C}\nSONUC\n{C}")
    print(f"Toplam gorsel      : {len(dosyalar)}")
    print(f"Otomatik siniflanan: {len(oto)}  (%{100*len(oto)/len(dosyalar):.1f})")
    print(f"Operatore giden    : {len(bel)}  (%{100*len(bel)/len(dosyalar):.1f})")
    print(f"Islem suresi       : {sure/60:.1f} dk")

    print(f"\n{C}\nKATEGORI DAGILIMI (otomatik kararlar)\n{C}")
    say = Counter(k["kategori"] for k in oto)
    if say:
        print(f"  {'kategori':<34} {'adet':>6} {'esik':>6} {'ort.guven':>10}")
        print("  " + "-" * 60)
        for kat, adet in say.most_common():
            ilgili = [k["guven"] for k in oto if k["kategori"] == kat]
            print(f"  {kisa(kat):<34} {adet:>6} {ESIKLER.get(kat,0):>6.2f} "
                  f"{np.mean(ilgili):>10.3f}")
    else:
        print("  (hicbiri esigi gecemedi)")

    print(f"\n{C}\nBELIRSIZ KLASORU (operatore gidecek)\n{C}")
    bel_say = Counter(k["model_tahmini"] for k in bel)
    print("  Model ne tahmin etti ama esigi gecemedi:")
    print(f"  {'model tahmini':<34} {'adet':>6} {'ort.guven':>10}")
    print("  " + "-" * 54)
    for kat, adet in bel_say.most_common(12):
        ilgili = [k["guven"] for k in bel if k["model_tahmini"] == kat]
        print(f"  {kisa(kat):<34} {adet:>6} {np.mean(ilgili):>10.3f}")

    print(f"\n{C}")
    if not a.rapor_only:
        print(f"Cikti klasoru : {cikti}")
    print(f"Rapor         : {rapor}")
    print(C)
    print("\nNOT: Otomatik kararlarin olculen dogrulugu ~%95'tir.")
    print("Yani her 100 otomatik karardan ~5'i yanlis olacaktir.")
    print("Ilk kullanimda otomatik klasorleri de gozle kontrol edin.")


if __name__ == "__main__":
    main()
