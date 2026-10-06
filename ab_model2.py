#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B SIDDET MODELI v2 - daha uzun egitim + SIRA BAZLI kesim analizi

NEDEN v2
  Ilk A/B modeli 6 tur egitildi ve dogrulama F1'i hala yukseliyordu
  (0.779 -> 0.829, duzenli artis, doymamis). Yarida birakilmisti.

  Bu model artik projenin en degerli parcasi: gunde ~1500 hucreyi B'den
  A'ya tasiyabiliyor. Iyilestirmesi, kategori siniflandirmasini
  iyilestirmekten daha karli.

DEGISIKLIKLER
  1) 14 tur (6 yerine) + cosine ogrenme hizi zamanlayicisi
  2) Label smoothing
  3) SIRA BAZLI kesim analizi

SIRA BAZLI KESIM NEDEN GEREKLI
  Mutlak olasilik esikleri bu veride ise yaramiyor. Ana model zaten en
  "B gibi" gorunenleri secti; A/B modeli de hepsini B'nin uc bolgesine
  itiyor. Ortalama A olasiligi %0.2 ciktı - ama SIRALAMA dogru.

  Gozle inceleme, rank ~1520 / a~0.1365 civarina kadar gorsellerin
  buyuk olcude A sinifi oldugunu gosterdi.

  Bu yuzden "olasilik >= X" yerine "en yuksek ilk N" mantigi kullaniyoruz.

KULLANIM
  1) python ab_model2.py egit "/path/2B"
  2) python ab_model2.py uygula "/path/otomasyon_cikti" --ilk-n 1500

Egitim bu Mac'te ~5-7 saat surer. Gece icin uygun.
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
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
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
ISINMA_TUR = 1
LABEL_SMOOTHING = 0.05

A_SINIFI = "其他"                  # Diger = A sinifi (tolere edilebilir)
B_SINIFI = {"黑斑", "黑边"}        # siyah nokta, siyah kenar

TEST_GUNLERI = {"20260910", "20260911"}
DOGRULAMA_GUNLERI = {"20260909"}

CIKTI = Path("model_ab2")
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")

# sira bazli kesim: test kumesinde bu oranlarda ne oluyor
KESIM_ORANLARI = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

C = "=" * 74
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
def egit(kok):
    CIKTI.mkdir(exist_ok=True)
    gunluk = CIKTI / "gunluk.txt"
    durum_y = CIKTI / "durum.json"
    son_y = CIKTI / "son_model.pt"
    iyi_y = CIKTI / "en_iyi_model.pt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gunluk, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nA/B SIDDET MODELI v2 - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    a_kl, b_kl = None, []
    for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
        if A_SINIFI in d.name:
            a_kl = d
        elif any(b in d.name for b in B_SINIFI):
            b_kl.append(d)
    if a_kl is None or not b_kl:
        yaz("HATA: klasorler bulunamadi.")
        for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
            yaz(f"   {d.name}")
        sys.exit(1)

    yaz(f"A sinifi : {a_kl.name}")
    for d in b_kl:
        yaz(f"B sinifi : {d.name}")

    yollar, etk, gun = [], [], []
    for d, e in [(a_kl, 0)] + [(x, 1) for x in b_kl]:
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
    for ad, ix in [("egitim", tr), ("dogrulama", dg), ("test", te)]:
        yaz(f"  {ad:<10}: {len(ix):>6}  "
            f"(A {int((etk[ix]==0).sum())}, B {int((etk[ix]==1).sum())})")
    if min(len(tr), len(dg), len(te)) == 0:
        yaz("\nHATA: bolmelerden biri bos.")
        sys.exit(1)

    def yk(ix, eg):
        return DataLoader(Kume(list(yollar[ix]), etk[ix], don(eg)),
                          batch_size=BATCH, shuffle=eg,
                          num_workers=ISCI, drop_last=eg)

    tr_y, dg_y, te_y = yk(tr, True), yk(dg, False), yk(te, False)

    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, 2)
    bas_id = {id(p) for p in net.fc.parameters()}
    govde = [p for p in net.parameters() if id(p) not in bas_id]
    opt = torch.optim.AdamW([
        {"params": govde, "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS},
    ], weight_decay=1e-4)

    n_a, n_b = int((etk[tr] == 0).sum()), int((etk[tr] == 1).sum())
    w = torch.tensor([(n_a+n_b)/(2*max(n_a,1)), (n_a+n_b)/(2*max(n_b,1))],
                     dtype=torch.float)
    yaz(f"\nSinif agirliklari: A {w[0]:.2f}, B {w[1]:.2f}")
    kayip_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)

    adim = max(len(tr_y), 1)
    toplam = TUR_SAYISI * adim
    isinma = ISINMA_TUR * adim

    def lr_c(s):
        if s < isinma:
            return (s + 1) / max(isinma, 1)
        p = (s - isinma) / max(toplam - isinma, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_c)

    bas_tur, en_iyi = 0, -1.0
    if durum_y.exists() and son_y.exists():
        try:
            d = json.loads(durum_y.read_text(encoding="utf-8"))
            if d.get("toplam") == len(yollar):
                ck = torch.load(son_y, map_location="cpu")
                net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                bas_tur = d["tur"]; en_iyi = d.get("en_iyi", -1.0)
                yaz(f"\nDevam: {bas_tur}. turdan (en iyi {en_iyi:.4f})")
        except Exception as e:
            yaz(f"\nKayit okunamadi ({e}), sifirdan.")

    def olc(yl):
        net.eval(); G, P = [], []
        with torch.no_grad():
            for x, y in yl:
                pr = torch.softmax(net(x), 1)
                P.append(pr[:, 0].numpy()); G.append(y.numpy())
        return np.concatenate(G), np.concatenate(P)

    yaz(f"\n{C}\nEGITIM ({TUR_SAYISI} tur)\n{C}\n")

    for tur in range(bas_tur, TUR_SAYISI):
        net.train(); t0, kt, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(tr_y):
            opt.zero_grad()
            k = kayip_fn(net(x), y); k.backward(); opt.step(); sched.step()
            kt += float(k)*len(y); n += len(y)
            if i % 20 == 0:
                hz = n / max(time.time()-t0, 1e-9)
                print(f"\r  tur {tur+1}/{TUR_SAYISI}  {n}/{len(tr)}  "
                      f"kayip {kt/max(n,1):.4f}  lr "
                      f"{opt.param_groups[0]['lr']:.2e}  "
                      f"kalan ~{(len(tr)-n)/max(hz,1e-9)/60:.0f} dk",
                      end="", flush=True)

        G, P = olc(dg_y)
        T = (P < 0.5).astype(int)
        p, r, f1, s = precision_recall_fscore_support(
            G, T, labels=[0, 1], zero_division=0)
        f1m = float(np.mean(f1))
        try:
            auc = roc_auc_score(1 - G, P)
        except ValueError:
            auc = float("nan")

        yaz(f"\n\n  --- TUR {tur+1}/{TUR_SAYISI}  "
            f"({(time.time()-t0)/60:.0f} dk) ---")
        yaz(f"  kayip {kt/max(n,1):.4f}   lr {opt.param_groups[0]['lr']:.2e}")
        yaz(f"  DOGRULAMA: dogruluk %{100*(G==T).mean():.1f}  "
            f"makro-F1 {f1m:.3f}  AUC {auc:.3f}")
        yaz(f"    A : P {p[0]:.3f}  R {r[0]:.3f}  n {int(s[0])}")

        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son_y)

        if f1m > en_iyi:
            en_iyi = f1m
            torch.save({"model": net.state_dict(), "girdi_boyut": GIRDI_BOYUT,
                        "tur": tur+1, "dog_f1": f1m}, iyi_y)
            yaz(f"  >>> EN IYI")

            Gt, Pt = olc(te_y)
            Tt = (Pt < 0.5).astype(int)
            pt, rt, f1t, st = precision_recall_fscore_support(
                Gt, Tt, labels=[0, 1], zero_division=0)
            try:
                auct = roc_auc_score(1 - Gt, Pt)
            except ValueError:
                auct = float("nan")
            yaz(f"    TEST: dogruluk %{100*(Gt==Tt).mean():.1f}  "
                f"makro-F1 {np.mean(f1t):.3f}  AUC {auct:.3f}")
            yaz(f"      A : P {pt[0]:.3f}  R {rt[0]:.3f}  n {int(st[0])}")

            # ---- SIRA BAZLI KESIM (asil onemli tablo)
            sira = np.argsort(-Pt)          # en yuksek A olasiligi basta
            toplam_a = int((Gt == 0).sum())
            yaz(f"\n      SIRA BAZLI KESIM (test: {len(Gt)} gorsel, "
                f"{toplam_a} gercek A)")
            yaz(f"      {'ilk %':>7} {'N':>7} {'A_bulunan':>10} "
                f"{'precision':>11} {'A_kapsama':>11} {'son_olasilik':>13}")
            yaz("      " + "-" * 62)
            for o in KESIM_ORANLARI:
                N = max(1, int(len(Gt) * o))
                sec = sira[:N]
                bul = int((Gt[sec] == 0).sum())
                yaz(f"      {100*o:>6.0f}% {N:>7} {bul:>10} "
                    f"{bul/N:>11.3f} {bul/max(toplam_a,1):>11.3f} "
                    f"{Pt[sec[-1]]:>13.4f}")

            with open(CIKTI / "test_tahmin.csv", "w", newline="",
                      encoding="utf-8-sig") as fh:
                wr = csv.writer(fh)
                wr.writerow(["dosya", "gercek", "A_olasiligi"])
                for j in sira:
                    wr.writerow([Path(yollar[te][j]).name,
                                 "A" if Gt[j] == 0 else "B", f"{Pt[j]:.6f}"])
        else:
            yaz(f"  (en iyi hala {en_iyi:.3f})")

        durum_y.write_text(json.dumps(
            {"tur": tur+1, "toplam": len(yollar), "en_iyi": en_iyi},
            ensure_ascii=False), encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama F1 {en_iyi:.3f}")
    yaz(f"Onceki model (6 tur): 0.829")
    yaz(f"\nSimdi:")
    yaz(f"  python ab_model2.py uygula <otomasyon_cikti> --ilk-n 1500")
    yaz(C)


# ======================================================================
def uygula(kok, ilk_n, oran):
    mp = CIKTI / "en_iyi_model.pt"
    if not mp.exists():
        print(f"HATA: {mp} yok. Once 'egit' calistirin.")
        sys.exit(1)
    ck = torch.load(mp, map_location="cpu")
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, 2)
    net.load_state_dict(ck["model"]); net.eval()

    hedefler = [d for d in sorted(kok.iterdir())
                if d.is_dir() and any(b in d.name for b in B_SINIFI)]
    if not hedefler:
        print("HATA: siyah nokta / siyah kenar klasoru yok.")
        for d in sorted([x for x in kok.iterdir() if x.is_dir()]):
            print(f"   {d.name}")
        sys.exit(1)

    dosyalar = []
    for d in hedefler:
        dosyalar += [f for f in sorted(d.iterdir())
                     if f.is_file() and f.suffix.lower() in UZANTILAR]

    print(f"{C}\nA/B SIDDET DUZELTMESI v2\n{C}")
    print(f"Model  : dogrulama F1 {ck.get('dog_f1',0):.3f} (tur {ck.get('tur','?')})")
    print(f"Gorsel : {len(dosyalar)}")

    N = ilk_n if ilk_n else max(1, int(len(dosyalar) * oran))
    N = min(N, len(dosyalar))
    print(f"Kesim  : en yuksek ilk {N} gorsel "
          f"(%{100*N/max(len(dosyalar),1):.1f})\n")

    yl = DataLoader(Kume(dosyalar, np.zeros(len(dosyalar), int), don(False)),
                    batch_size=BATCH, shuffle=False, num_workers=ISCI)
    P = np.zeros(len(dosyalar)); n = 0
    t0 = time.time()
    with torch.no_grad():
        for x, _ in yl:
            pr = torch.softmax(net(x), 1)[:, 0].numpy()
            P[n:n+len(pr)] = pr; n += len(pr)
            print(f"\r  {n}/{len(dosyalar)}", end="", flush=True)
    print(f"\n  bitti: {(time.time()-t0)/60:.1f} dk\n")

    sira = np.argsort(-P)
    sec = sira[:N]

    hedef = kok / "_A_SINIFI_ONERISI"
    if hedef.exists():
        shutil.rmtree(hedef)
    hedef.mkdir()

    for rank, i in enumerate(sec, 1):
        f = dosyalar[i]
        try:
            shutil.copy2(f, hedef / f"{rank:04d}_a{P[i]:.4f}_{f.name}")
        except Exception as e:
            print(f"  kopyalanamadi: {f.name} ({e})")

    with open(kok / "ab_duzeltme2.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["sira", "dosya", "eski_klasor", "A_olasiligi", "secildi_mi"])
        for rank, i in enumerate(sira, 1):
            wr.writerow([rank, dosyalar[i].name, dosyalar[i].parent.name,
                         f"{P[i]:.6f}", "evet" if rank <= N else "hayir"])

    print(f"{C}\nSONUC\n{C}")
    print(f"A sinifi onerisi : {N} gorsel")
    print(f"Olasilik araligi : {P[sec[0]]:.4f} - {P[sec[-1]]:.4f}")
    print(f"\nKlasor : {hedef}")
    print(f"Rapor  : {kok/'ab_duzeltme2.csv'}")
    print(C)
    print("\nDosyalar SIRALI: 0001 = en yuksek A olasiligi.")
    print("Sirayla bakip A'larin nerede bittigini tespit edin.")
    print("Sinir degisirse --ilk-n ile yeniden calistirin.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mod", choices=["egit", "uygula"])
    ap.add_argument("klasor")
    ap.add_argument("--ilk-n", type=int, default=None,
                    help="en yuksek kac gorsel A onerilsin")
    ap.add_argument("--oran", type=float, default=0.28,
                    help="--ilk-n verilmezse bu oran kullanilir")
    a = ap.parse_args()
    p = Path(a.klasor).expanduser().resolve()
    if not p.is_dir():
        print(f"HATA: klasor yok: {p}")
        sys.exit(1)
    if a.mod == "egit":
        egit(p)
    else:
        uygula(p, a.ilk_n, a.oran)


if __name__ == "__main__":
    main()
