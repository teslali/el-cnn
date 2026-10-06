#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C SINIFI DEDEKTORU - TAM COZUNURLUK (512 px)

NEDEN
  Baski plakasi izi orta boy nokta buyuklugunde bir X. Modellerimiz
  goruntuyu 224'e kucultuyor; X'in kollari kayboluyor, model onu
  "tolere edilebilir tek nokta" sanip A diyor. Gozle dogrulandi:
  224'te cogunlukla nokta, 384'te bir kismi hala nokta.
  Bu yuzden bu model goruntuyu KUCULTMEDEN, 512'de gorur.

GOREV
  Tek soru: "Bu hucrede gizli catlak ya da baski plakasi var mi?"
  Yakalama isabetten onemli: bos alarm operatorun birkac saniyesi,
  kacirma musteriye giden kusurlu hucre.

VERI
  Pozitif: egitim gunlerindeki tum C ornekleri (2B + duzeltmeler)
  Negatif: ~6000 secilmis ornek, en cok karistirilanlar agirlikli
           (siyah nokta, A sinifi, duzensiz cizik, siyah kenar)
  Baslangic: model_ab4 govdesi (EL goruntulerini zaten taniyor)

SINAV (test gunleri, egitime hic girmez)
  model_ab4'un test kumesindeki A adaylari icinde 28 C var
  (26 baski plakasi, 2 catlak). Her esikte kacinin yakalandigi ve
  karsiliginda kac gercek A hucresinin bosa isaretlendigi yazilir.

Sure: bu Mac'te ~10-14 saat. Baska egitim/gunluk.py calisirken baslatmayin.

Kullanim:
  caffeinate -i python c_dedektor.py "/path/2B"
"""

import sys
import csv
import json
import re
import math
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import models, transforms
    from PIL import Image
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BOYUT = 512
AB_BOYUT = 224
BATCH = 8                    # 512 px'te bellek icin kucuk
ISCI = 2
TUR_SAYISI = 6
LR_GOVDE = 1e-4
LR_BAS = 1e-3
LABEL_SMOOTHING = 0.05
POZITIF_PAYI = 0.30          # her batch'in ~%30'u C ornegi olsun

A_ANAHTAR = "其他"
C_ANAHTARLAR = {"隐裂", "压板"}
PLAKA = "压板"

TEST_GUNLERI = {"20260910", "20260911"}
DOGRULAMA_GUNLERI = {"20260909"}
A_ORANI = 0.187

# negatif kotalari (egitim): karistirilan tipler agirlikli
NEG_KOTA = {"黑斑": 1800, "其他": 1500, "不规则": 1000, "黑边": 800}
NEG_DIGER = 900              # geri kalan kategorilere orantili
DOG_NEG = 1500               # dogrulama icin negatif ornek

AB_ADAY = [Path("model_ab4/en_iyi_model.pt"), Path("model_ab3/en_iyi_model.pt")]
MANIFEST = Path("duzeltme_listesi.csv")
CIKTI = Path("model_c")
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")
ESIKLER = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]

C = "=" * 76
torch.set_num_threads(4)
torch.manual_seed(0)
rng = np.random.RandomState(0)


def gun_bul(p):
    z = ZAMAN.search(Path(p).stem)
    return z.group(1)[:8] if z else ""


def kisa(ad, n=28):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


class Kume(Dataset):
    def __init__(self, yollar, etiketler, donusum, boyut):
        self.yollar, self.etiketler = yollar, etiketler
        self.donusum, self.boyut = donusum, boyut

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                x = self.donusum(im.convert("RGB"))
        except Exception:
            x = torch.zeros(3, self.boyut, self.boyut)
        return x, self.etiketler[i]


def don(boyut, egitim=False):
    t = [transforms.Resize((boyut, boyut))]
    if egitim:
        t += [transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.ColorJitter(brightness=0.10, contrast=0.10)]
    t += [transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


def olasilik(net, yollar, boyut, etiket):
    yl = DataLoader(Kume(yollar, [0] * len(yollar), don(boyut), boyut),
                    batch_size=24 if boyut <= 224 else BATCH,
                    shuffle=False, num_workers=ISCI)
    P, n, t0 = [], 0, time.time()
    net.eval()
    with torch.no_grad():
        for x, _ in yl:
            P.append(torch.softmax(net(x), 1)[:, 1].numpy())
            n += len(x)
            hz = n / max(time.time() - t0, 1e-9)
            print(f"\r  {etiket}: {n}/{len(yollar)}  "
                  f"kalan ~{(len(yollar)-n)/max(hz,1e-9)/60:.0f} dk",
                  end="", flush=True)
    print(f"   ({(time.time()-t0)/60:.0f} dk)")
    return np.concatenate(P) if P else np.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor", help="2B klasoru")
    ap.add_argument("--manifest", default=str(MANIFEST))
    a = ap.parse_args()
    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}"); sys.exit(1)
    ab_yol = next((p for p in AB_ADAY if p.exists()), None)
    if ab_yol is None:
        print("HATA: model_ab4 / model_ab3 bulunamadi."); sys.exit(1)

    CIKTI.mkdir(exist_ok=True)
    gunluk_y = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gunluk_y, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nC DEDEKTORU (512 px) - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    # ---------------- veri havuzu: (yol, kategori, gun)
    kayit = []
    adlar = set()
    man = Path(a.manifest)
    if man.exists():
        with open(man, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                p = Path(r["yol"])
                g = gun_bul(p)
                if p.exists() and g not in TEST_GUNLERI | DOGRULAMA_GUNLERI:
                    kayit.append((p, r["etiket"], g)); adlar.add(p.name)
    n_man = len(kayit)
    for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in UZANTILAR and f.name not in adlar:
                kayit.append((f, d.name, gun_bul(f)))

    def c_mi(k):
        return any(x in k for x in C_ANAHTARLAR)

    tr = [r for r in kayit if r[2] not in TEST_GUNLERI | DOGRULAMA_GUNLERI]
    dg = [r for r in kayit if r[2] in DOGRULAMA_GUNLERI]
    te = [r for r in kayit if r[2] in TEST_GUNLERI]

    # ---------------- egitim: tum pozitifler + kotali negatifler
    tr_pos = [r for r in tr if c_mi(r[1])]
    tr_neg_hepsi = [r for r in tr if not c_mi(r[1])]
    secilen, kullanilan = [], set()
    for anahtar, kota in NEG_KOTA.items():
        grup = [i for i, r in enumerate(tr_neg_hepsi) if anahtar in r[1]]
        sec = rng.choice(grup, size=min(kota, len(grup)), replace=False) if grup else []
        secilen += [tr_neg_hepsi[i] for i in sec]; kullanilan.update(int(i) for i in sec)
    kalan = [i for i in range(len(tr_neg_hepsi)) if i not in kullanilan
             and not any(k in tr_neg_hepsi[i][1] for k in NEG_KOTA)]
    if kalan:
        sec = rng.choice(kalan, size=min(NEG_DIGER, len(kalan)), replace=False)
        secilen += [tr_neg_hepsi[i] for i in sec]
    tr_set = tr_pos + secilen
    tr_y = [1] * len(tr_pos) + [0] * len(secilen)

    # ---------------- dogrulama: tum pozitifler + orneklem negatif
    dg_pos = [r for r in dg if c_mi(r[1])]
    dg_neg = [r for r in dg if not c_mi(r[1])]
    if len(dg_neg) > DOG_NEG:
        dg_neg = [dg_neg[i] for i in rng.choice(len(dg_neg), DOG_NEG, replace=False)]
    dg_set = dg_pos + dg_neg
    dg_y = np.array([1] * len(dg_pos) + [0] * len(dg_neg))

    yaz(f"Duzeltme listesinden : {n_man}")
    yaz(f"Egitim   : {len(tr_pos)} C  +  {len(secilen)} negatif")
    for k, n in Counter(kisa(r[1]) for r in tr_pos).most_common():
        yaz(f"             {n:>5}  {k}")
    yaz(f"Dogrulama: {len(dg_pos)} C  +  {len(dg_neg)} negatif")
    if len(tr_pos) == 0 or len(dg_pos) == 0:
        yaz("HATA: pozitif ornek yok."); sys.exit(1)

    # ---------------- SINAV: ab modelinin test A adaylari + tum test C
    yaz(f"\nSinav kumesi hazirlaniyor ({ab_yol.parent.name}, 224 px)...")
    ck_ab = torch.load(ab_yol, map_location="cpu")
    net_ab = models.resnet18(weights=None); net_ab.fc = nn.Linear(512, 2)
    net_ab.load_state_dict(ck_ab["model"])
    onbellek = CIKTI / "sinav_ab.npy"
    te_yol = [r[0] for r in te]
    if onbellek.exists() and len(np.load(onbellek)) == len(te):
        p_ab = np.load(onbellek)
    else:
        # A olasiligi = sinif 0; olasilik() sinif 1 dondurdugu icin 1 - p
        p_ab = 1.0 - olasilik(net_ab, te_yol, AB_BOYUT, "A/B test")
        np.save(onbellek, p_ab)
    N = max(1, int(len(te) * A_ORANI))
    aday = set(np.argsort(-p_ab)[:N].tolist())
    te_c = [i for i, r in enumerate(te) if c_mi(r[1])]
    sinav_idx = sorted(aday | set(te_c))
    sinav = [te[i] for i in sinav_idx]
    s_aday = np.array([i in aday for i in sinav_idx])
    s_c = np.array([c_mi(r[1]) for r in sinav])
    s_plaka = np.array([PLAKA in r[1] for r in sinav])
    s_a = np.array([A_ANAHTAR in r[1] for r in sinav])
    yaz(f"  A adaylari: {N}  |  test C: {len(te_c)}  |  "
        f"adaylara sizan C: {int((s_aday & s_c).sum())} "
        f"(baski plakasi {int((s_aday & s_plaka).sum())})")

    # ---------------- model: ab govdesi, 2 cikisli bas
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, 2)
    govde_agirlik = {k: v for k, v in ck_ab["model"].items()
                     if not k.startswith("fc.")}
    net.load_state_dict(govde_agirlik, strict=False)
    yaz(f"Baslangic: {ab_yol.parent.name} govdesi, yeni C basi")

    bas_id = {id(p) for p in net.fc.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in bas_id],
         "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS},
    ], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    w = np.where(np.array(tr_y) == 1, POZITIF_PAYI / len(tr_pos),
                 (1 - POZITIF_PAYI) / len(secilen))
    ornekleyici = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                        num_samples=len(tr_set), replacement=True)
    tr_yl = DataLoader(Kume([r[0] for r in tr_set], tr_y, don(BOYUT, True), BOYUT),
                       batch_size=BATCH, sampler=ornekleyici, num_workers=ISCI)

    adim = max(len(tr_yl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / (TUR_SAYISI * adim), 1.0))))

    durum_y, son_y, iyi_y = CIKTI / "durum.json", CIKTI / "son_model.pt", CIKTI / "en_iyi_model.pt"
    bas_tur, en_iyi = 0, -1.0
    if durum_y.exists() and son_y.exists():
        try:
            d = json.loads(durum_y.read_text(encoding="utf-8"))
            if d.get("egitim") == len(tr_set):
                ck = torch.load(son_y, map_location="cpu")
                net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                bas_tur, en_iyi = d["tur"], d.get("en_iyi", -1.0)
                yaz(f"Kaldigi yerden devam: {bas_tur}. tur")
        except Exception as e:
            yaz(f"Kayit okunamadi ({e}), bastan.")

    yaz(f"\n{C}\nEGITIM ({TUR_SAYISI} tur, 512 px, batch {BATCH})\n{C}\n")

    for tur in range(bas_tur, TUR_SAYISI):
        net.train(); t0, kt, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(tr_yl):
            opt.zero_grad()
            k = kayip_fn(net(x), y); k.backward(); opt.step(); sched.step()
            kt += float(k) * len(y); n += len(y)
            if i % 20 == 0:
                hz = n / max(time.time() - t0, 1e-9)
                print(f"\r  tur {tur+1}/{TUR_SAYISI}  {n}/{len(tr_set)}  "
                      f"kayip {kt/max(n,1):.4f}  "
                      f"kalan ~{(len(tr_set)-n)/max(hz,1e-9)/60:.0f} dk",
                      end="", flush=True)

        pd = olasilik(net, [r[0] for r in dg_set], BOYUT, "\n  dogrulama")
        try:
            auc = roc_auc_score(dg_y, pd)
        except ValueError:
            auc = float("nan")
        yaz(f"\n  --- TUR {tur+1}/{TUR_SAYISI}  ({(time.time()-t0)/60:.0f} dk) ---")
        yaz(f"  kayip {kt/max(n,1):.4f}   DOGRULAMA AUC {auc:.4f}  "
            f"(C {len(dg_pos)} / negatif {len(dg_neg)})")

        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son_y)

        if auc > en_iyi:
            en_iyi = auc
            torch.save({"model": net.state_dict(), "girdi_boyut": BOYUT,
                        "tur": tur + 1, "dog_auc": auc,
                        "gorev": "C (gizli catlak + baski plakasi) vs diger"}, iyi_y)
            yaz("  >>> EN IYI - sinav olcuyor")
            ps = olasilik(net, [r[0] for r in sinav], BOYUT, "  sinav")
            yaz(f"\n    Test C yakalama ve A adaylarindaki sizinti:")
            yaz(f"    {'esik':>6} {'C yakalandi':>13} {'sizan kalan':>12} "
                f"{'(plaka)':>8} {'bosa A':>8} {'isaretli B':>11}")
            yaz("    " + "-" * 62)
            for e in ESIKLER:
                isaret = ps >= e
                yak = int((s_c & isaret).sum())
                kalan_s = int((s_aday & s_c & ~isaret).sum())
                kalan_p = int((s_aday & s_plaka & ~isaret).sum())
                bosa = int((s_aday & s_a & isaret).sum())
                b_is = int((s_aday & ~s_a & ~s_c & isaret).sum())
                yaz(f"    {e:>6.2f} {yak:>6}/{int(s_c.sum()):<6} {kalan_s:>12} "
                    f"{kalan_p:>8} {bosa:>8} {b_is:>11}")
            yaz("    sizan kalan : vetodan sonra A'ya gecen C (simdi "
                f"{int((s_aday & s_c).sum())}, hedef 0)")
            yaz("    bosa A      : yanlislikla veto edilen gercek A (bedel)")
        else:
            yaz(f"  (en iyi hala {en_iyi:.4f})")

        durum_y.write_text(json.dumps({"tur": tur + 1, "egitim": len(tr_set),
                                       "en_iyi": en_iyi}), encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama AUC {en_iyi:.4f}")
    yaz(f"Model: {iyi_y}")
    yaz("Sonraki adim: sinav tablosuna gore esik secip gunluk.py'ye eklemek.")
    yaz(C)


if __name__ == "__main__":
    main()
