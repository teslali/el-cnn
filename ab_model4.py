#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B MODELI v4 - DUZELTMELERLE YENIDEN EGITIM

model_ab3'un uzerine, insan tarafindan dogrulanmis yeni veriyi ekler:
  - duzeltme_listesi.csv (duzeltme_hazirla.py ciktisi)
      duzeltme    : modelin yanildigi gorseller, dogru etiketle, x3 agirlik
      dogrulanmis : gunluk.py ciktisinin kontrol edilmis kismi, x1 agirlik

VARSAYILAN: model_ab3'ten DEVAM (warm start), 5 tur, dusuk ogrenme hizi.
  Model zaten iyi; sifirdan 12 tur egitmek yerine yeni veriyle ince ayar
  yeterli ve yaklasik yarim surede biter.
  --sifirdan ile ImageNet'ten 10 tur tam egitim yapilabilir.

TEST KUMESI DEGISMEDI (10-11 Eylul), dolayisiyla ab3 ile birebir
karsilastirilabilir:
  model_ab3: test AUC 0.9600, C sizintisi %30'a kadar 0

Guvenlik:
  - Listede test/dogrulama gunlerine ait gorsel varsa atlanir.
  - C sinifi (gizli catlak, baski plakasi) ornekleri ayrica x3 agirlikli.

Kullanim:
  caffeinate -i python ab_model4.py "/path/2B"
  caffeinate -i python ab_model4.py "/path/2B" --sifirdan
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
LABEL_SMOOTHING = 0.05

A_ANAHTAR = "其他"
C_ANAHTARLAR = {"隐裂", "压板"}
C_AGIRLIK_CARPANI = 3.0

TEST_GUNLERI = {"20260910", "20260911"}
DOGRULAMA_GUNLERI = {"20260909"}

BASLANGIC_MODEL = Path("model_ab3") / "en_iyi_model.pt"
MANIFEST = Path("duzeltme_listesi.csv")
CIKTI = Path("model_ab4")

UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")
KESIM_ORANLARI = [0.05, 0.10, 0.15, 0.187, 0.22, 0.26, 0.30]

C = "=" * 76
torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)


def kisa(ad, n=32):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def gun_bul(p):
    z = ZAMAN.search(Path(p).stem)
    return z.group(1)[:8] if z else ""


class Kume(Dataset):
    def __init__(self, yollar, etiketler, agirliklar, donusum):
        self.yollar, self.etiketler = yollar, etiketler
        self.agirliklar, self.donusum = agirliklar, donusum

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
    ap.add_argument("klasor", help="2B klasoru")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--sifirdan", action="store_true",
                    help="model_ab3'ten devam etme, ImageNet'ten 10 tur egit")
    a = ap.parse_args()

    kok = Path(a.klasor).expanduser().resolve()
    man = Path(a.manifest)
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}"); sys.exit(1)
    if not man.exists():
        print(f"HATA: {man} yok. Once duzeltme_hazirla.py calistirin.")
        sys.exit(1)

    devam = (not a.sifirdan) and BASLANGIC_MODEL.exists()
    TUR_SAYISI = 5 if devam else 10
    LR_GOVDE = 5e-5 if devam else 1.5e-4
    LR_BAS = 5e-4 if devam else 1.5e-3
    ISINMA_TUR = 0 if devam else 1

    CIKTI.mkdir(exist_ok=True)
    gunluk_y = CIKTI / "gunluk.txt"
    durum_y = CIKTI / "durum.json"
    son_y = CIKTI / "son_model.pt"
    iyi_y = CIKTI / "en_iyi_model.pt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gunluk_y, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nA/B MODELI v4 - DUZELTMELERLE - "
        f"{time.strftime('%Y-%m-%d %H:%M')}\n{C}")
    yaz(f"Baslangic : {'model_ab3 (devam)' if devam else 'ImageNet (sifirdan)'}")
    yaz(f"Tur / LR  : {TUR_SAYISI} / govde {LR_GOVDE}, bas {LR_BAS}")

    # ---------------- manifest
    ek = []
    atlanan = Counter()
    with open(man, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            p = Path(r["yol"])
            if not p.exists():
                atlanan["dosya bulunamadi"] += 1; continue
            g = gun_bul(p)
            if g in TEST_GUNLERI | DOGRULAMA_GUNLERI:
                atlanan["test/dogrulama gunu"] += 1; continue
            ek.append((p, r["etiket"], g, float(r.get("agirlik") or 1.0),
                       r.get("kaynak", "")))
    ek_adlar = {p.name for p, *_ in ek}

    # ---------------- 2B
    yollar, etk, gun, kat, c_mi, carpan, kaynak = [], [], [], [], [], [], []

    def ekle(p, klasor_adi, g, crp, kyn):
        yollar.append(p)
        etk.append(0 if A_ANAHTAR in klasor_adi else 1)
        gun.append(g)
        kat.append(klasor_adi)
        c_mi.append(any(k in klasor_adi for k in C_ANAHTARLAR))
        carpan.append(crp)
        kaynak.append(kyn)

    cift = 0
    for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                if f.name in ek_adlar:
                    cift += 1; continue      # duzeltilmis etiket oncelikli
                ekle(f, d.name, gun_bul(f), 1.0, "2B")
    for p, etiket, g, crp, kyn in ek:
        ekle(p, etiket, g, crp, kyn)

    yollar = np.array(yollar); etk = np.array(etk); gun = np.array(gun)
    kat = np.array(kat); c_mi = np.array(c_mi)
    carpan = np.array(carpan, dtype=np.float32); kaynak = np.array(kaynak)

    te = np.where(np.isin(gun, list(TEST_GUNLERI)))[0]
    dg = np.where(np.isin(gun, list(DOGRULAMA_GUNLERI)))[0]
    tr = np.where(~np.isin(gun, list(TEST_GUNLERI | DOGRULAMA_GUNLERI)))[0]

    yaz(f"\nVeri:")
    yaz(f"  2B              : {int((kaynak=='2B').sum())}")
    yaz(f"  duzeltme  (x3)  : {int((kaynak=='duzeltme').sum())}  "
        f"(A: {int(((kaynak=='duzeltme') & (etk==0)).sum())})")
    yaz(f"  dogrulanmis(x1) : {int((kaynak=='dogrulanmis').sum())}  "
        f"(A: {int(((kaynak=='dogrulanmis') & (etk==0)).sum())})")
    if cift:
        yaz(f"  2B'deki eski kopyasi yerine duzeltilmis etiket: {cift}")
    for k, n in atlanan.items():
        yaz(f"  atlandi ({k}): {n}")
    yaz(f"\nBolme (test ab3 ile ayni):")
    for ad, ix in [("egitim", tr), ("dogrulama", dg), ("test", te)]:
        yaz(f"  {ad:<10}: {len(ix):>6}  (A {int((etk[ix]==0).sum())}, "
            f"B/C {int((etk[ix]==1).sum())}, C {int(c_mi[ix].sum())})")
    if min(len(tr), len(dg), len(te)) == 0:
        yaz("HATA: bolmelerden biri bos."); sys.exit(1)

    # ---------------- agirliklar
    n_a = int((etk[tr] == 0).sum()); n_b = int((etk[tr] == 1).sum())
    w_a = (n_a + n_b) / (2 * max(n_a, 1)); w_b = (n_a + n_b) / (2 * max(n_b, 1))
    agirlik = np.where(etk == 0, w_a, w_b).astype(np.float32) * carpan
    agirlik[c_mi] *= C_AGIRLIK_CARPANI
    yaz(f"\nSinif agirliklari: A {w_a:.2f}, B/C {w_b:.2f}  "
        f"(+duzeltme x3, +C x{C_AGIRLIK_CARPANI:.0f})")

    def yukle(ix, egitim):
        return DataLoader(Kume(list(yollar[ix]), etk[ix], agirlik[ix], don(egitim)),
                          batch_size=BATCH, shuffle=egitim,
                          num_workers=ISCI, drop_last=egitim)

    tr_y, dg_y, te_y = yukle(tr, True), yukle(dg, False), yukle(te, False)

    # ---------------- model
    net = models.resnet18(weights=None if devam
                          else models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, 2)
    if devam:
        ck0 = torch.load(BASLANGIC_MODEL, map_location="cpu")
        net.load_state_dict(ck0["model"])
        yaz(f"model_ab3 agirliklari yuklendi (tur {ck0.get('tur','?')}, "
            f"dogrulama AUC {ck0.get('dog_auc',0):.4f})")

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
            if d.get("toplam") == len(yollar) and d.get("devam") == devam:
                ck = torch.load(son_y, map_location="cpu")
                net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                bas_tur = d["tur"]; en_iyi = d.get("en_iyi", -1.0)
                yaz(f"\nKaldigi yerden devam: {bas_tur}. tur")
        except Exception as e:
            yaz(f"\nKayit okunamadi ({e}), bastan.")

    def olc(yl):
        net.eval(); G, P = [], []
        with torch.no_grad():
            for x, y, _ in yl:
                P.append(torch.softmax(net(x), 1)[:, 0].numpy())
                G.append(y.numpy())
        return np.concatenate(G), np.concatenate(P)

    # ---- baslangic olcumu (devam ediyorsa ab3'un kendi skoru)
    if devam and bas_tur == 0:
        Gt0, Pt0 = olc(te_y)
        try:
            yaz(f"\nBaslangic (model_ab3) test AUC: {roc_auc_score(1-Gt0, Pt0):.4f}")
        except ValueError:
            pass

    yaz(f"\n{C}\nEGITIM ({TUR_SAYISI} tur)\n{C}\n")

    for tur in range(bas_tur, TUR_SAYISI):
        net.train(); t0, kt, n = time.time(), 0.0, 0
        for i, (x, y, w) in enumerate(tr_y):
            opt.zero_grad()
            kayip = (kayip_fn(net(x), y) * w).mean()
            kayip.backward(); opt.step(); sched.step()
            kt += float(kayip) * len(y); n += len(y)
            if i % 20 == 0:
                hz = n / max(time.time() - t0, 1e-9)
                print(f"\r  tur {tur+1}/{TUR_SAYISI}  {n}/{len(tr)}  "
                      f"kayip {kt/max(n,1):.4f}  "
                      f"kalan ~{(len(tr)-n)/max(hz,1e-9)/60:.0f} dk",
                      end="", flush=True)

        Gd, Pd = olc(dg_y)
        try:
            auc_d = roc_auc_score(1 - Gd, Pd)
        except ValueError:
            auc_d = float("nan")
        yaz(f"\n\n  --- TUR {tur+1}/{TUR_SAYISI}  "
            f"({(time.time()-t0)/60:.0f} dk) ---")
        yaz(f"  kayip {kt/max(n,1):.4f}   DOGRULAMA AUC {auc_d:.4f}")

        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son_y)

        if auc_d > en_iyi:
            en_iyi = auc_d
            torch.save({"model": net.state_dict(), "girdi_boyut": GIRDI_BOYUT,
                        "tur": tur + 1, "dog_auc": auc_d}, iyi_y)
            yaz("  >>> EN IYI")

            Gt, Pt = olc(te_y)
            try:
                auc_t = roc_auc_score(1 - Gt, Pt)
            except ValueError:
                auc_t = float("nan")
            yaz(f"    TEST AUC: {auc_t:.4f}   (model_ab3: 0.9600)")

            sira = np.argsort(-Pt)
            toplam_a = int((Gt == 0).sum())
            c_te = c_mi[te]
            yaz(f"\n    {'ilk %':>7} {'N':>6} {'A_bulundu':>10} "
                f"{'precision':>10} {'A_kapsama':>10} {'C_SIZAN':>9}")
            yaz("    " + "-" * 60)
            for o in KESIM_ORANLARI:
                nn_ = max(1, int(len(Gt) * o))
                sec = sira[:nn_]
                bul = int((Gt[sec] == 0).sum())
                yaz(f"    {100*o:>6.1f}% {nn_:>6} {bul:>10} {bul/nn_:>10.3f} "
                    f"{bul/max(toplam_a,1):>10.3f} {int(c_te[sec].sum()):>9}")
        else:
            yaz(f"  (en iyi hala {en_iyi:.4f})")

        durum_y.write_text(json.dumps(
            {"tur": tur + 1, "toplam": len(yollar), "en_iyi": en_iyi,
             "devam": devam}, ensure_ascii=False), encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama AUC {en_iyi:.4f}\n{C}")
    yaz(f"Model: {iyi_y}")
    yaz("\nKarsilastirma icin model_ab3: dogrulama 0.9653, test 0.9600")
    yaz("Kullanmak icin gunluk.py icinde AB_MODEL satirini model_ab4 yapin.")
    yaz("ASIL OLCUM: yeni bir gunun verisinde gunluk.py + goz kontrolu.")


if __name__ == "__main__":
    main()
