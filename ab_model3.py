#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B SIDDET MODELI v3 - "A SINIFI vs GERI KALAN HERSEY"

NEDEN YENI MODEL
  Mevcut model_ab2 sadece "Diger vs siyah nokta+kenar" ile egitildi.
  Bu yuzden A/B taramasi sadece o iki kategoriye uygulanabiliyor (5360 gorsel).
  Diger kategorilerdeki hafif kusurlu hucreler hic taranmiyor.

  Bu surum TUM kategorilerle egitilir: "Diger (A) vs geri kalan hepsi (B)".
  Boylece A/B taramasi tum gorsellere (7400) uygulanabilir ve once
  calistirilabilir.

YENI AKIS
  Tum gorseller -> A/B modeli -> en yuksek %N: Diger klasoru, biter
                               -> kalanlar: model3 ile kategorilere

KRITIK RISK: C SINIFI SIZINTISI
  Gizli catlak (164 ornek) ve baski plakasi (107) C sinifidir. Bunlarin
  A'ya kacmasi en pahali hatadir. Iki katmanli koruma:

  1) EGITIMDE: C sinifi orneklerine ekstra agirlik verilir, model onlari
     A demekten kacinmayi ogrenir.
  2) CALISMA ANINDA: A adayi secilen her gorsel model3'ten de gecer;
     model3 "gizli catlak" veya "baski plakasi" derse A adayligi iptal.
     (bu kisim gunluk scriptte uygulanir)

  Bu script egitim sonunda SIZINTIYI OLCER: A adaylarinin icine kac C
  sinifi ornek karismis.

Kullanim:
    caffeinate -i python ab_model3.py "/path/2B"
"""

import sys
import json
import time
import csv
import re
import math
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
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
GIRDI_BOYUT = 224
BATCH = 24
ISCI = 2
TUR_SAYISI = 12
LR_GOVDE = 1.5e-4
LR_BAS = 1.5e-3
ISINMA_TUR = 1
LABEL_SMOOTHING = 0.05

A_ANAHTAR = "其他"                      # Diger = A sinifi
C_ANAHTARLAR = {"隐裂", "压板"}         # gizli catlak, baski plakasi
C_AGIRLIK_CARPANI = 3.0                 # C sinifi orneklerine ekstra agirlik

# Karsilastirma icin: mevcut sistemin kapsadigi kategoriler
ESKI_KAPSAM = {"黑斑", "黑边"}

TEST_GUNLERI = {"20260910", "20260911"}
DOGRULAMA_GUNLERI = {"20260909"}

CIKTI = Path("model_ab3")
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")

KESIM_ORANLARI = [0.05, 0.10, 0.15, 0.18, 0.22, 0.26, 0.30]

C = "=" * 76
torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)


def kisa(ad, n=32):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


class Kume(Dataset):
    def __init__(self, yollar, etiketler, agirliklar, donusum):
        self.yollar = yollar
        self.etiketler = etiketler
        self.agirliklar = agirliklar
        self.donusum = donusum

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                x = self.donusum(im.convert("RGB"))
        except Exception:
            x = torch.zeros(3, GIRDI_BOYUT, GIRDI_BOYUT)
        return x, self.etiketler[i], self.agirliklar[i]


def don(egitim):
    t = [transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT))]
    if egitim:
        t += [transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.ColorJitter(brightness=0.10, contrast=0.10)]
    t += [transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    a = ap.parse_args()
    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        sys.exit(1)

    CIKTI.mkdir(exist_ok=True)
    gunluk_y = CIKTI / "gunluk.txt"
    durum_y = CIKTI / "durum.json"
    son_y = CIKTI / "son_model.pt"
    iyi_y = CIKTI / "en_iyi_model.pt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gunluk_y, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nA/B MODELI v3 - A vs HEPSI - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    # ---------------- veri
    yollar, etk, gun, kat, c_mi, eski_mi = [], [], [], [], [], []
    for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
        a_sinifi = A_ANAHTAR in d.name
        c_sinifi = any(k in d.name for k in C_ANAHTARLAR)
        eski_kapsam = any(k in d.name for k in ESKI_KAPSAM)
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                z = ZAMAN.search(f.stem)
                yollar.append(f)
                etk.append(0 if a_sinifi else 1)
                gun.append(z.group(1)[:8] if z else "")
                kat.append(d.name)
                c_mi.append(c_sinifi)
                eski_mi.append(eski_kapsam)

    if not yollar:
        yaz("HATA: gorsel bulunamadi.")
        sys.exit(1)

    yollar = np.array(yollar)
    etk = np.array(etk)
    gun = np.array(gun)
    kat = np.array(kat)
    c_mi = np.array(c_mi)
    eski_mi = np.array(eski_mi)

    te = np.where(np.isin(gun, list(TEST_GUNLERI)))[0]
    dg = np.where(np.isin(gun, list(DOGRULAMA_GUNLERI)))[0]
    tr = np.where(~np.isin(gun, list(TEST_GUNLERI | DOGRULAMA_GUNLERI)))[0]

    yaz(f"Toplam gorsel : {len(yollar)}")
    yaz(f"A sinifi      : {int((etk==0).sum())}")
    yaz(f"B/C sinifi    : {int((etk==1).sum())}")
    yaz(f"C sinifi      : {int(c_mi.sum())}  (ekstra agirlik x{C_AGIRLIK_CARPANI})")
    yaz(f"\nGun bazli bolme:")
    for ad, ix in [("egitim", tr), ("dogrulama", dg), ("test", te)]:
        yaz(f"  {ad:<10}: {len(ix):>6}  "
            f"(A {int((etk[ix]==0).sum())}, B/C {int((etk[ix]==1).sum())}, "
            f"C {int(c_mi[ix].sum())})")

    if min(len(tr), len(dg), len(te)) == 0:
        yaz("\nHATA: bolmelerden biri bos. Gunleri kontrol edin.")
        sys.exit(1)

    # ---------------- ornek bazli agirlik
    n_a = int((etk[tr] == 0).sum())
    n_b = int((etk[tr] == 1).sum())
    w_a = (n_a + n_b) / (2 * max(n_a, 1))
    w_b = (n_a + n_b) / (2 * max(n_b, 1))
    yaz(f"\nSinif agirliklari : A {w_a:.2f}, B/C {w_b:.2f}")

    agirlik = np.where(etk == 0, w_a, w_b).astype(np.float32)
    agirlik[c_mi] *= C_AGIRLIK_CARPANI
    yaz(f"C sinifi ornekleri ayrica x{C_AGIRLIK_CARPANI} agirlikli")

    def yukle(ix, egitim):
        return DataLoader(
            Kume(list(yollar[ix]), etk[ix], agirlik[ix], don(egitim)),
            batch_size=BATCH, shuffle=egitim, num_workers=ISCI,
            drop_last=egitim)

    tr_y, dg_y, te_y = yukle(tr, True), yukle(dg, False), yukle(te, False)

    # ---------------- model
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, 2)
    bas_id = {id(p) for p in net.fc.parameters()}
    govde = [p for p in net.parameters() if id(p) not in bas_id]
    opt = torch.optim.AdamW([
        {"params": govde, "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS},
    ], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING,
                                   reduction="none")

    adim = max(len(tr_y), 1)
    toplam_adim = TUR_SAYISI * adim
    isinma = ISINMA_TUR * adim

    def lr_c(s):
        if s < isinma:
            return (s + 1) / max(isinma, 1)
        p = (s - isinma) / max(toplam_adim - isinma, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_c)

    bas_tur, en_iyi = 0, -1.0
    if durum_y.exists() and son_y.exists():
        try:
            d = json.loads(durum_y.read_text(encoding="utf-8"))
            if d.get("toplam") == len(yollar):
                ck = torch.load(son_y, map_location="cpu")
                net.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                bas_tur = d["tur"]
                en_iyi = d.get("en_iyi", -1.0)
                yaz(f"\nDevam: {bas_tur}. turdan (en iyi AUC {en_iyi:.4f})")
        except Exception as e:
            yaz(f"\nKayit okunamadi ({e}), sifirdan.")

    def olc(yl):
        net.eval()
        G, P = [], []
        with torch.no_grad():
            for x, y, _ in yl:
                pr = torch.softmax(net(x), 1)[:, 0]
                P.append(pr.numpy()); G.append(y.numpy())
        return np.concatenate(G), np.concatenate(P)

    yaz(f"\n{C}\nEGITIM ({TUR_SAYISI} tur)\n{C}\n")

    for tur in range(bas_tur, TUR_SAYISI):
        net.train()
        t0, kt, n = time.time(), 0.0, 0
        for i, (x, y, w) in enumerate(tr_y):
            opt.zero_grad()
            kayip = (kayip_fn(net(x), y) * w).mean()
            kayip.backward(); opt.step(); sched.step()
            kt += float(kayip) * len(y); n += len(y)
            if i % 20 == 0:
                hz = n / max(time.time() - t0, 1e-9)
                print(f"\r  tur {tur+1}/{TUR_SAYISI}  {n}/{len(tr)}  "
                      f"kayip {kt/max(n,1):.4f}  lr "
                      f"{opt.param_groups[0]['lr']:.2e}  "
                      f"kalan ~{(len(tr)-n)/max(hz,1e-9)/60:.0f} dk",
                      end="", flush=True)

        Gd, Pd = olc(dg_y)
        try:
            auc_d = roc_auc_score(1 - Gd, Pd)
        except ValueError:
            auc_d = float("nan")
        Td = (Pd < 0.5).astype(int)
        p, r, f1, s = precision_recall_fscore_support(
            Gd, Td, labels=[0, 1], zero_division=0)

        yaz(f"\n\n  --- TUR {tur+1}/{TUR_SAYISI}  "
            f"({(time.time()-t0)/60:.0f} dk) ---")
        yaz(f"  kayip {kt/max(n,1):.4f}   lr {opt.param_groups[0]['lr']:.2e}")
        yaz(f"  DOGRULAMA: AUC {auc_d:.4f}  "
            f"A: P {p[0]:.3f} R {r[0]:.3f}  n {int(s[0])}")

        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son_y)

        if auc_d > en_iyi:
            en_iyi = auc_d
            torch.save({"model": net.state_dict(),
                        "girdi_boyut": GIRDI_BOYUT, "tur": tur + 1,
                        "dog_auc": auc_d}, iyi_y)
            yaz(f"  >>> EN IYI")

            # ---- TEST
            Gt, Pt = olc(te_y)
            try:
                auc_t = roc_auc_score(1 - Gt, Pt)
            except ValueError:
                auc_t = float("nan")
            yaz(f"    TEST AUC: {auc_t:.4f}")

            sira = np.argsort(-Pt)
            toplam_a = int((Gt == 0).sum())
            c_te = c_mi[te]
            kat_te = kat[te]
            eski_te = eski_mi[te]

            yaz(f"\n    SIRA BAZLI KESIM ({len(Gt)} gorsel, "
                f"{toplam_a} gercek A, {int(c_te.sum())} C sinifi)")
            yaz(f"    {'ilk %':>7} {'N':>6} {'A_bulundu':>10} "
                f"{'precision':>10} {'A_kapsama':>10} {'C_SIZAN':>9}")
            yaz("    " + "-" * 60)
            for o in KESIM_ORANLARI:
                nn_ = max(1, int(len(Gt) * o))
                sec = sira[:nn_]
                bul = int((Gt[sec] == 0).sum())
                c_sizan = int(c_te[sec].sum())
                yaz(f"    {100*o:>6.0f}% {nn_:>6} {bul:>10} "
                    f"{bul/nn_:>10.3f} {bul/max(toplam_a,1):>10.3f} "
                    f"{c_sizan:>9}")

            # ---- C sinifi sizintisi detay
            yaz(f"\n    C SINIFI SIZINTISI (en kritik risk)")
            for o in [0.18, 0.26]:
                nn_ = max(1, int(len(Gt) * o))
                sec = sira[:nn_]
                sizan = [kat_te[i] for i in sec if c_te[i]]
                if sizan:
                    yaz(f"      ilk %{100*o:.0f}: {len(sizan)} C sinifi sizdi")
                    for k, v in Counter(sizan).most_common():
                        yaz(f"         {v:>4}x  {kisa(k)}")
                else:
                    yaz(f"      ilk %{100*o:.0f}: SIZINTI YOK")

            # ---- eski kapsamla karsilastirma
            if eski_te.sum() > 0:
                alt = np.where(eski_te)[0]
                yaz(f"\n    ESKI KAPSAM KARSILASTIRMASI")
                yaz(f"      siyah nokta+kenar: {len(alt)} gorsel "
                    f"({int((Gt[alt]==0).sum())} A)")
                yaz(f"      Yeni model bu alt kumede de calisiyor; ayrica")
                yaz(f"      diger {len(Gt)-len(alt)} gorseli de tariyor.")

            with open(CIKTI / "test_tahmin.csv", "w", newline="",
                      encoding="utf-8-sig") as fh:
                w_ = csv.writer(fh)
                w_.writerow(["sira", "dosya", "kategori", "gercek",
                             "A_olasiligi", "C_sinifi_mi"])
                for rank, i in enumerate(sira, 1):
                    w_.writerow([rank, Path(yollar[te][i]).name,
                                 kat_te[i], "A" if Gt[i] == 0 else "B",
                                 f"{Pt[i]:.6f}",
                                 "evet" if c_te[i] else "hayir"])
        else:
            yaz(f"  (en iyi hala {en_iyi:.4f})")

        durum_y.write_text(json.dumps(
            {"tur": tur + 1, "toplam": len(yollar), "en_iyi": en_iyi},
            ensure_ascii=False), encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama AUC {en_iyi:.4f}\n{C}")
    yaz(f"Model     : {iyi_y}")
    yaz(f"Tahminler : {CIKTI/'test_tahmin.csv'}")
    yaz("\nKarsilastirma:")
    yaz("  model_ab2 (sadece siyah nokta+kenar) test AUC: 0.9241")
    yaz("\nDIKKAT: Bu iki AUC dogrudan kiyaslanamaz - farkli test kumeleri.")
    yaz("Onemli olan C_SIZAN sutunu ve sira bazli precision.")
    yaz("\nSizinti varsa gunluk scriptte model3 vetosu sart.")


if __name__ == "__main__":
    main()
