#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Ayiklama - IKINCI KADEME: A/B SIDDET KARARI

SORUN
  Ana model (model3) "hangi kusur var?" sorusunu iyi cevapliyor ama
  "bu kusur A sinifini engelleyecek kadar SIDDETLI mi?" sorusunu hic
  ogrenmedi. Bu yuzden 1-2 soluk noktasi olan, aslinda A sinifi olmasi
  gereken hucreleri "siyah nokta" ya da "siyah kenar" olarak isaretliyor.

  Bu, tek gercek MALIYETLI hata turu: A olmasi gereken hucre B'de kaliyor.
  Ve bu iki kategori toplam hacmin ~%72'si.

COZUM
  Dar kapsamli, ikinci bir model. Tek soru soruyor:
      "Bu hucre A sinifi mi (Diger), yoksa gercekten B mi?"

  Sadece siyah nokta + siyah kenar + Diger klasorleriyle egitilir.
  Gorev dar oldugu icin ana modelden daha iyi ogrenir.

KULLANIM SIRASI
  1) python ab_model.py egit  "/path/2B"          -> modeli egitir
  2) python ab_model.py uygula "/path/cikti"      -> otomasyon ciktisini duzeltir

Egitim bu Mac'te ~2-4 saat surer (yaklasik 11.000 gorsel, 6 tur).
"""

import sys
import json
import time
import csv
import re
import math
import shutil
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
TUR_SAYISI = 6
LR_GOVDE = 1.5e-4
LR_BAS = 1.5e-3
LABEL_SMOOTHING = 0.05

# Bu kademe SADECE su klasorlerle calisir
A_SINIFI = "其他"                      # Diger = A sinifi (tolere edilebilir)
B_SINIFI = {"黑斑", "黑边"}            # siyah nokta, siyah kenar

TEST_GUNLERI = {"20260910", "20260911"}
DOGRULAMA_GUNLERI = {"20260909"}

CIKTI = Path("model_ab")
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")

# uygula modunda: A sinifi olasiligi bu esigi gecerse A'ya tasinir.
# Yuksek tutuldu cunku yanlis A kararı (kusurlu hucre A'ya gider) pahali.
A_ESIGI = 0.80

C = "=" * 72
torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)


class Kume(Dataset):
    def __init__(self, yollar, etiketler, donusum):
        self.yollar, self.etiketler, self.donusum = yollar, etiketler, donusum

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.donusum(im.convert("RGB")), self.etiketler[i]
        except Exception:
            return torch.zeros(3, GIRDI_BOYUT, GIRDI_BOYUT), self.etiketler[i]


def don(egitim):
    t = [transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT))]
    if egitim:
        t += [transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.ColorJitter(brightness=0.10, contrast=0.10)]
    t += [transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


# ======================================================================
# EGITIM
# ======================================================================
def egit(kok):
    CIKTI.mkdir(exist_ok=True)
    gunluk = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gunluk, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nA/B SIDDET MODELI - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    # ilgili klasorleri bul
    a_klasor, b_klasorler = None, []
    for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
        if A_SINIFI in d.name:
            a_klasor = d
        elif any(b in d.name for b in B_SINIFI):
            b_klasorler.append(d)

    if a_klasor is None or not b_klasorler:
        yaz("HATA: gerekli klasorler bulunamadi.")
        yaz(f"  A sinifi ('{A_SINIFI}') : {a_klasor}")
        yaz(f"  B sinifi {B_SINIFI} : {[d.name for d in b_klasorler]}")
        yaz("\nMevcut klasorler:")
        for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
            yaz(f"    {d.name}")
        sys.exit(1)

    yaz(f"A sinifi klasoru : {a_klasor.name}")
    for d in b_klasorler:
        yaz(f"B sinifi klasoru : {d.name}")

    yollar, etk, gun = [], [], []
    for d, e in [(a_klasor, 0)] + [(x, 1) for x in b_klasorler]:
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in UZANTILAR:
                z = ZAMAN.search(f.stem)
                yollar.append(f); etk.append(e)
                gun.append(z.group(1)[:8] if z else "")

    yollar = np.array(yollar); etk = np.array(etk); gun = np.array(gun)

    te = np.where(np.isin(gun, list(TEST_GUNLERI)))[0]
    dg = np.where(np.isin(gun, list(DOGRULAMA_GUNLERI)))[0]
    tr = np.where(~np.isin(gun, list(TEST_GUNLERI | DOGRULAMA_GUNLERI)))[0]

    yaz(f"\nGun bazli bolme:")
    yaz(f"  egitim    : {len(tr):>6}  "
        f"(A {int((etk[tr]==0).sum())}, B {int((etk[tr]==1).sum())})")
    yaz(f"  dogrulama : {len(dg):>6}  "
        f"(A {int((etk[dg]==0).sum())}, B {int((etk[dg]==1).sum())})")
    yaz(f"  test      : {len(te):>6}  "
        f"(A {int((etk[te]==0).sum())}, B {int((etk[te]==1).sum())})")

    if len(tr) == 0 or len(dg) == 0 or len(te) == 0:
        yaz("\nHATA: bolmelerden biri bos. TEST/DOGRULAMA gunlerini kontrol edin.")
        sys.exit(1)

    def yukle(idx, egitim):
        return DataLoader(Kume(list(yollar[idx]), etk[idx], don(egitim)),
                          batch_size=BATCH, shuffle=egitim,
                          num_workers=ISCI, drop_last=egitim)

    tr_y, dg_y, te_y = yukle(tr, True), yukle(dg, False), yukle(te, False)

    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, 2)
    bas_id = {id(p) for p in net.fc.parameters()}
    govde = [p for p in net.parameters() if id(p) not in bas_id]
    opt = torch.optim.AdamW([
        {"params": govde, "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS},
    ], weight_decay=1e-4)

    # dengesizlik: B cok daha fazla. A'yi kacirmamak icin agirlik ver.
    n_a, n_b = int((etk[tr] == 0).sum()), int((etk[tr] == 1).sum())
    w = torch.tensor([(n_a + n_b) / (2 * max(n_a, 1)),
                      (n_a + n_b) / (2 * max(n_b, 1))], dtype=torch.float)
    yaz(f"\nSinif agirliklari: A {w[0]:.2f}, B {w[1]:.2f}")
    kayip_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)

    adim = max(len(tr_y), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / (TUR_SAYISI * adim), 1.0))))

    def olc(yk):
        net.eval(); G, T, P = [], [], []
        with torch.no_grad():
            for x, y in yk:
                p = torch.softmax(net(x), 1)
                T.append(p.argmax(1).numpy()); P.append(p[:, 0].numpy())
                G.append(y.numpy())
        return np.concatenate(G), np.concatenate(T), np.concatenate(P)

    yaz(f"\n{C}\nEGITIM BASLIYOR ({TUR_SAYISI} tur)\n{C}\n")
    en_iyi = -1.0
    for tur in range(TUR_SAYISI):
        net.train(); t0, kt, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(tr_y):
            opt.zero_grad()
            k = kayip_fn(net(x), y); k.backward(); opt.step(); sched.step()
            kt += float(k) * len(y); n += len(y)
            if i % 20 == 0:
                hiz = n / max(time.time() - t0, 1e-9)
                print(f"\r  tur {tur+1}/{TUR_SAYISI}  {n}/{len(tr)}  "
                      f"kayip {kt/max(n,1):.4f}  "
                      f"kalan ~{(len(tr)-n)/max(hiz,1e-9)/60:.0f} dk",
                      end="", flush=True)

        G, T, _ = olc(dg_y)
        p, r, f1, s = precision_recall_fscore_support(
            G, T, labels=[0, 1], zero_division=0)
        f1m = float(np.mean(f1))
        yaz(f"\n\n  --- TUR {tur+1}  ({(time.time()-t0)/60:.0f} dk) ---")
        yaz(f"  DOGRULAMA: dogruluk %{100*(G==T).mean():.1f}  makro-F1 {f1m:.3f}")
        yaz(f"    A sinifi : P {p[0]:.3f}  R {r[0]:.3f}  n {int(s[0])}")
        yaz(f"    B sinifi : P {p[1]:.3f}  R {r[1]:.3f}  n {int(s[1])}")

        if f1m > en_iyi:
            en_iyi = f1m
            torch.save({"model": net.state_dict(),
                        "girdi_boyut": GIRDI_BOYUT, "tur": tur + 1,
                        "dog_f1": f1m}, CIKTI / "en_iyi_model.pt")
            yaz(f"  >>> EN IYI")
            Gt, Tt, Pt = olc(te_y)
            pt, rt, f1t, st = precision_recall_fscore_support(
                Gt, Tt, labels=[0, 1], zero_division=0)
            yaz(f"    TEST: dogruluk %{100*(Gt==Tt).mean():.1f}  "
                f"makro-F1 {np.mean(f1t):.3f}")
            yaz(f"      A : P {pt[0]:.3f}  R {rt[0]:.3f}  n {int(st[0])}")
            yaz(f"      B : P {pt[1]:.3f}  R {rt[1]:.3f}  n {int(st[1])}")
            # esik tablosu: A olasiligina gore
            yaz(f"\n      A esigi analizi (A dedigimizde ne kadar hakliyiz):")
            yaz(f"      {'esik':>6} {'A_karar':>9} {'dogru':>7} {'precision':>11}")
            for e in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                m = Pt >= e
                nn_ = int(m.sum())
                if nn_ == 0:
                    yaz(f"      {e:>6.2f} {0:>9} {0:>7} {'-':>11}")
                    continue
                d = int((Gt[m] == 0).sum())
                yaz(f"      {e:>6.2f} {nn_:>9} {d:>7} {d/nn_:>11.3f}")
            with open(CIKTI / "test_tahmin.csv", "w", newline="",
                      encoding="utf-8-sig") as fh:
                wr = csv.writer(fh)
                wr.writerow(["dosya", "gercek", "tahmin", "A_olasiligi"])
                for j in range(len(Gt)):
                    wr.writerow([Path(yollar[te][j]).name,
                                 "A" if Gt[j] == 0 else "B",
                                 "A" if Tt[j] == 0 else "B", f"{Pt[j]:.4f}"])
        else:
            yaz(f"  (en iyi hala {en_iyi:.3f})")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama F1 {en_iyi:.3f}")
    yaz(f"Model: {CIKTI/'en_iyi_model.pt'}")
    yaz(f"\nSimdi:  python ab_model.py uygula <otomasyon_cikti_klasoru>")
    yaz(C)


# ======================================================================
# UYGULAMA
# ======================================================================
def uygula(cikti_kok, esik):
    mp = CIKTI / "en_iyi_model.pt"
    if not mp.exists():
        print(f"HATA: {mp} yok. Once 'egit' calistirin.")
        sys.exit(1)

    ck = torch.load(mp, map_location="cpu")
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, 2)
    net.load_state_dict(ck["model"]); net.eval()

    hedef_klasorler = []
    for d in sorted([x for x in cikti_kok.iterdir() if x.is_dir()]):
        if any(b in d.name for b in B_SINIFI):
            hedef_klasorler.append(d)
    if not hedef_klasorler:
        print("HATA: siyah nokta / siyah kenar klasoru bulunamadi.")
        print("Mevcut klasorler:")
        for d in sorted([x for x in cikti_kok.iterdir() if x.is_dir()]):
            print(f"   {d.name}")
        sys.exit(1)

    print(f"{C}\nA/B SIDDET DUZELTMESI\n{C}")
    print(f"Model : {mp}  (dogrulama F1 {ck.get('dog_f1',0):.3f})")
    print(f"Esik  : A olasiligi >= {esik}")
    for d in hedef_klasorler:
        print(f"Taranacak: {d.name}")

    dosyalar = []
    for d in hedef_klasorler:
        dosyalar += [f for f in sorted(d.iterdir())
                     if f.is_file() and f.suffix.lower() in UZANTILAR]
    print(f"Toplam: {len(dosyalar)} gorsel\n")

    yk = DataLoader(Kume(dosyalar, np.zeros(len(dosyalar), int), don(False)),
                    batch_size=BATCH, shuffle=False, num_workers=ISCI)
    olas = np.zeros(len(dosyalar))
    t0, n = time.time(), 0
    with torch.no_grad():
        for x, _ in yk:
            p = torch.softmax(net(x), 1)[:, 0].numpy()
            olas[n:n+len(p)] = p; n += len(p)
            print(f"\r  {n}/{len(dosyalar)}", end="", flush=True)
    print(f"\n  bitti: {(time.time()-t0)/60:.1f} dk\n")

    a_hedef = cikti_kok / "_A_SINIFI_ONERISI"
    a_hedef.mkdir(exist_ok=True)
    tasinan = 0
    with open(cikti_kok / "ab_duzeltme.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["dosya", "eski_klasor", "A_olasiligi", "tasindi_mi"])
        for i, f in enumerate(dosyalar):
            tasi = olas[i] >= esik
            if tasi:
                try:
                    shutil.copy2(f, a_hedef / f"a{olas[i]:.2f}_{f.name}")
                    tasinan += 1
                except Exception:
                    tasi = False
            wr.writerow([f.name, f.parent.name, f"{olas[i]:.4f}",
                         "evet" if tasi else "hayir"])

    print(f"{C}\nSONUC\n{C}")
    print(f"Incelenen        : {len(dosyalar)}")
    print(f"A sinifi onerisi : {tasinan}  (%{100*tasinan/max(len(dosyalar),1):.1f})")
    print(f"\nKlasor : {a_hedef}")
    print(f"Rapor  : {cikti_kok/'ab_duzeltme.csv'}")
    print(C)
    print("\nBu klasordeki gorseller KOPYALANDI (orijinaller yerinde).")
    print("Gozle kontrol edin: bunlar gercekten A sinifi mi?")
    print("Dosya adindaki 'a0.93' modelin A olasiligidir.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mod", choices=["egit", "uygula"])
    ap.add_argument("klasor")
    ap.add_argument("--esik", type=float, default=A_ESIGI)
    a = ap.parse_args()
    p = Path(a.klasor).expanduser().resolve()
    if not p.is_dir():
        print(f"HATA: klasor yok: {p}")
        sys.exit(1)
    if a.mod == "egit":
        egit(p)
    else:
        uygula(p, a.esik)


if __name__ == "__main__":
    main()
