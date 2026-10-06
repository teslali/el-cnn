#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A DOGRULAYICISI (512 px)

Sadece A adaylarina bakan ikinci model. Tek soru:
  "Bu hucre gercekten A mi, yoksa A sinirini asan bir kusuru mu var?"

NEDEN
  Altin kumede A klasoru isabeti %81. Yanlislarin cogu kucuk / ince / soluk
  kusurlar (siyah nokta, duzensiz cizik, siyah kenar). A/B modeli 224 px'te
  calisiyor ve bu detaylar orada kayboluyor. C dedektoru ayni sorunu 512 px ile
  cozdu; bu model de 512 px'te calisir.

  A sinifi hucrelerin cogunda kucuk kusur var; yani model kusurun VARLIGINA degil
  MIKTARINA bakmayi ogrenmeli. Bu yuzden en degerli B ornekleri, modelin A sanip
  yanildigi hucreler (inceleme 2a klasoru): x3 agirlik.

VERI
  A : ayrim_gruplar.csv anlasan_A + yeni_duzeltmeler/a sinifi
      + duzeltme_listesi.csv'deki A'lar (automation-test, dogrulanmis)
  B : yeni_duzeltmeler'deki A olmayanlar (x3, zor ornekler)
      + anlasan_B orneklemi (A klasorunu kirleten tipler agirlikli)
      + duzeltme_listesi.csv'deki B'ler
  Haric: 9.22 (altin gun), OverTime gorselleri
  Dogrulama: 9.21 gunu (en iyi tur secimi)

SINAV (altin kume, hic gorulmemis gun)
  cikti_altin2 A klasoru: 375 hucre, 304 gercek A (%81).
  Her esikte: kalan hucre, isabet, kaybedilen A.

Sure: ~8-10 saat. Kesilirse ayni komutla devam eder.
  caffeinate -i python a_dogrulayici.py
"""

import sys
import csv
import json
import math
import re
import time
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
    print(f"HATA: paket eksik ({e.name})"); sys.exit(1)

# ----------------------------------------------------------------------
BOYUT = 512
BATCH = 8
ISCI = 2
TUR_SAYISI = 3
LR_GOVDE, LR_BAS = 5e-5, 5e-4
ZOR_AGIRLIK = 3.0

AB_MODEL = next((p for p in [Path("model_ab4/en_iyi_model.pt"),
                             Path("model_ab3/en_iyi_model.pt")] if p.exists()), None)
GRUPLAR = Path("ayrim_gruplar.csv")
DUZELTMELER = Path.home() / "Desktop" / "yeni_duzeltmeler"
MANIFEST = Path("duzeltme_listesi.csv")
ALTIN_KL = Path("altin_kume")
ALTIN_ONAY = Path("cikti_altin")
ALTIN_HATA = Path.home() / "Desktop" / "altin_hatalar"
SINAV_KAYIT = Path("cikti_altin2/_kayit.csv")
CIKTI = Path("model_a")

HARIC_GUN = {"20260922"}
DOG_GUN = {"20260921"}
A = "其他"
A_POZ_ORNEK = 3500
NEG_KOTA = {"黑斑": 2000, "不规则": 1200, "黑边": 1200, "断栅": 600, "偏移": 600}
NEG_DIGER = 800
DOG_NEG = 1500
MANIFEST_B = 1500          # automation-test B'lerinden orneklem
TUR_ORNEK = 7000           # her turda gorulen ornek (512 px'te sure icin)
ESIKLER = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")
ONEK = re.compile(r"^op-.*?_m-.*?_a\d\.\d+_")      # inceleme dosya adi oneki
C = "=" * 74
torch.set_num_threads(4)
rng = np.random.RandomState(0)
torch.manual_seed(0)


def normal(s):
    return " ".join(s.translate(str.maketrans("ıİşŞçÇğĞöÖüÜ", "iIsScCgGoOuU"))
                    .lower().replace("_", " ").split())


def a_mi_turkce(ad):
    return "a sinif" in normal(ad)


def gun(p):
    z = ZAMAN.search(Path(p).name)
    return z.group(1)[:8] if z else ""


def bozuk(p):
    n = Path(p).name
    return "OverTime" in n or "_Class400_" in n


def orijinal_ad(n):
    return ONEK.sub("", n)


class Kume(Dataset):
    def __init__(self, yollar, etiket, don):
        self.yollar, self.etiket, self.don = yollar, etiket, don

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                x = self.don(im.convert("RGB"))
        except Exception:
            x = torch.zeros(3, BOYUT, BOYUT)
        return x, self.etiket[i], i


def don(egitim):
    t = [transforms.Resize((BOYUT, BOYUT))]
    if egitim:
        t += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
              transforms.ColorJitter(brightness=0.15, contrast=0.15)]
    t += [transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


def olasilik(net, yollar):
    yl = DataLoader(Kume(yollar, [0] * len(yollar), don(False)),
                    batch_size=BATCH, shuffle=False, num_workers=ISCI)
    P = np.zeros(len(yollar)); n, t0 = 0, time.time()
    net.eval()
    with torch.no_grad():
        for x, _, idx in yl:
            P[idx.numpy()] = torch.softmax(net(x), 1)[:, 1].numpy()
            n += len(idx)
            hz = n / max(time.time() - t0, 1e-9)
            print(f"\r    {n}/{len(yollar)}  kalan ~{(len(yollar)-n)/max(hz,1e-9)/60:.0f} dk",
                  end="", flush=True)
    print()
    return P


def main():
    if AB_MODEL is None:
        print("HATA: model_ab4/ab3 yok."); sys.exit(1)
    CIKTI.mkdir(exist_ok=True)
    gunluk = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gunluk, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nA DOGRULAYICISI (512 px) - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    # (yol, etiket 1=A 0=B, agirlik, kaynak)
    ornek = {}

    def ekle(p, etiket, agirlik, kaynak):
        p = Path(p)
        if not p.exists() or bozuk(p) or gun(p) in HARIC_GUN:
            return
        ad = orijinal_ad(p.name)
        if ad in ornek and ornek[ad][3] == "inceleme":
            return                            # senin kararin her seyin onunde
        ornek[ad] = (p, etiket, agirlik, kaynak)

    # ---- 1) senin incelemen (en guvenilir)
    if DUZELTMELER.is_dir():
        for d in DUZELTMELER.iterdir():
            if not d.is_dir() or "emin" in normal(d.name):
                continue
            a_ = a_mi_turkce(d.name)
            for f in d.iterdir():
                if f.suffix.lower() in UZANTILAR:
                    ekle(f, 1 if a_ else 0, 1.0 if a_ else ZOR_AGIRLIK, "inceleme")
    n_inc = sum(1 for v in ornek.values() if v[3] == "inceleme")

    # ---- 2) automation-test dogrulanmis gun
    if MANIFEST.exists():
        man = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
        man_B = [r for r in man if A not in r["etiket"]]
        secilen_B = {man_B[i]["yol"] for i in rng.permutation(len(man_B))[:MANIFEST_B]}
        for r in man:
            if A in r["etiket"] or r["yol"] in secilen_B:
                ekle(r["yol"], 1 if A in r["etiket"] else 0, 1.0, "dogrulanmis")

    # ---- 3) ayrim gruplari: anlasan A ve anlasan B
    if not GRUPLAR.exists():
        yaz(f"HATA: {GRUPLAR} yok (ayrim.py ozet)."); sys.exit(1)
    R = list(csv.DictReader(open(GRUPLAR, encoding="utf-8-sig")))
    anl_A = [r for r in R if r["grup"] == "anlasan_A"]
    anl_B = [r for r in R if r["grup"] == "anlasan_B"]
    for i in rng.permutation(len(anl_A))[:A_POZ_ORNEK]:
        ekle(anl_A[i]["yol"], 1, 1.0, "anlasan")
    for k, kota in NEG_KOTA.items():
        g = [r for r in anl_B if r["operator"] == k]
        for i in rng.permutation(len(g))[:kota]:
            ekle(g[i]["yol"], 0, 1.0, "anlasan")
    diger = [r for r in anl_B if r["operator"] not in NEG_KOTA]
    for i in rng.permutation(len(diger))[:NEG_DIGER]:
        ekle(diger[i]["yol"], 0, 1.0, "anlasan")
    # dogrulama gunu icin ek negatif (orneklem disinda kalanlardan)
    dog_B = [r for r in anl_B if r["gun"] in DOG_GUN]
    for i in rng.permutation(len(dog_B))[:DOG_NEG]:
        ekle(dog_B[i]["yol"], 0, 1.0, "anlasan")

    kayit = list(ornek.values())
    tr = [k for k in kayit if gun(k[0]) not in DOG_GUN]
    dg = [k for k in kayit if gun(k[0]) in DOG_GUN]
    yaz(f"Senin incelemen : {n_inc}")
    yaz(f"Egitim  : A {sum(k[1] for k in tr):>5}  B {sum(1-k[1] for k in tr):>5}  "
        f"(zor B x{ZOR_AGIRLIK:.0f}: {sum(1 for k in tr if k[1]==0 and k[3]=='inceleme')})")
    yaz(f"Dogrulama (9.21): A {sum(k[1] for k in dg)}  B {sum(1-k[1] for k in dg)}")
    for kyn, n in Counter(k[3] for k in tr).most_common():
        yaz(f"    {kyn:<12} {n}")
    if not tr or sum(k[1] for k in dg) == 0:
        yaz("HATA: egitim ya da dogrulama bos."); sys.exit(1)

    # ---- sinav: altin kumedeki A klasoru
    sinav = []
    if SINAV_KAYIT.exists():
        altin_A = {}
        if ALTIN_ONAY.is_dir():
            for d in ALTIN_ONAY.iterdir():
                if d.is_dir():
                    for f in d.iterdir():
                        altin_A[f.name] = A in d.name
        if ALTIN_HATA.is_dir():
            for d in ALTIN_HATA.iterdir():
                if d.is_dir() and "emin" not in normal(d.name):
                    for f in d.iterdir():
                        altin_A[f.name] = a_mi_turkce(d.name)
        for r in csv.DictReader(open(SINAV_KAYIT, encoding="utf-8-sig")):
            if A in r["klasor"] and r["dosya"] in altin_A:
                sinav.append((ALTIN_KL / r["dosya"], altin_A[r["dosya"]]))
        s_dogru = sum(v for _, v in sinav)
        yaz(f"Sinav   : altin A klasoru {len(sinav)} hucre, {s_dogru} gercek A "
            f"(%{100*s_dogru/max(len(sinav),1):.0f})")
    else:
        yaz(f"UYARI: {SINAV_KAYIT} yok, sinav atlanacak.")

    # ---- model
    ck = torch.load(AB_MODEL, map_location="cpu")
    net = models.resnet18(weights=None); net.fc = nn.Linear(512, 2)
    net.load_state_dict({k: v for k, v in ck["model"].items()
                         if not k.startswith("fc.")}, strict=False)
    yaz(f"Baslangic: {AB_MODEL.parent.name} govdesi")

    bas_id = {id(p) for p in net.fc.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in bas_id], "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS}], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    tr_y = [k[1] for k in tr]
    agirlik = torch.tensor([k[2] for k in tr], dtype=torch.float)
    nA, nB = sum(tr_y), len(tr_y) - sum(tr_y)
    ornekleme = [0.5 / nA if y else 0.5 / nB for y in tr_y]
    yl = DataLoader(Kume([k[0] for k in tr], tr_y, don(True)), batch_size=BATCH,
                    sampler=WeightedRandomSampler(ornekleme, min(len(tr), TUR_ORNEK),
                                                  replacement=True),
                    num_workers=ISCI)
    adim = max(len(yl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / (TUR_SAYISI * adim), 1.0))))

    durum, son, iyi = CIKTI / "durum.json", CIKTI / "son_model.pt", CIKTI / "en_iyi_model.pt"
    bas_tur, en_iyi = 0, -1.0
    if durum.exists() and son.exists():
        d = json.loads(durum.read_text(encoding="utf-8"))
        if d.get("egitim") == len(tr):
            c2 = torch.load(son, map_location="cpu")
            net.load_state_dict(c2["model"]); opt.load_state_dict(c2["opt"])
            sched.load_state_dict(c2["sched"])
            bas_tur, en_iyi = d["tur"], d["en_iyi"]
            yaz(f"Kaldigi yerden devam: {bas_tur}. tur")

    yaz(f"\n{C}\nEGITIM ({TUR_SAYISI} tur, 512 px)\n{C}")
    for tur in range(bas_tur, TUR_SAYISI):
        net.train(); t0, kt, n = time.time(), 0.0, 0
        for b, (x, y, idx) in enumerate(yl):
            opt.zero_grad()
            k = (kayip_fn(net(x), y) * agirlik[idx]).mean()
            k.backward(); opt.step(); sched.step()
            kt += float(k) * len(y); n += len(y)
            if b % 20 == 0:
                hz = n / max(time.time() - t0, 1e-9)
                top = min(len(tr), TUR_ORNEK)
                print(f"\r  tur {tur+1}/{TUR_SAYISI}  {n}/{top}  kayip {kt/max(n,1):.4f}  "
                      f"kalan ~{(top-n)/max(hz,1e-9)/60:.0f} dk", end="", flush=True)
        print()
        yaz(f"\n  --- TUR {tur+1}  ({(time.time()-t0)/60:.0f} dk) ---")
        yaz("  dogrulama:")
        pd = olasilik(net, [k[0] for k in dg])
        auc = roc_auc_score([k[1] for k in dg], pd)
        yaz(f"  DOGRULAMA AUC {auc:.4f}")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son)
        if auc > en_iyi:
            en_iyi = auc
            torch.save({"model": net.state_dict(), "girdi_boyut": BOYUT, "tur": tur + 1,
                        "dog_auc": auc, "gorev": "A dogrulayici (1 = gercek A)"}, iyi)
            yaz("  >>> EN IYI")
            if sinav:
                yaz("  sinav (altin A klasoru):")
                ps = olasilik(net, [s[0] for s in sinav])
                g = np.array([s[1] for s in sinav])
                top_A = int(g.sum())
                yaz(f"    su an: {len(g)} hucre, {top_A} A, isabet %{100*top_A/len(g):.0f}")
                yaz(f"    {'esik':>6} {'kalan':>6} {'dogru A':>8} {'isabet':>7} "
                    f"{'cikan B':>8} {'kaybedilen A':>13}")
                for e in ESIKLER:
                    tut = ps >= e
                    dA = int((tut & g).sum())
                    yaz(f"    {e:>6.2f} {int(tut.sum()):>6} {dA:>8} "
                        f"{'%'+str(round(100*dA/max(tut.sum(),1))):>7} "
                        f"{int((~tut & ~g).sum()):>8} {top_A-dA:>13}")
        durum.write_text(json.dumps({"tur": tur + 1, "egitim": len(tr), "en_iyi": en_iyi}),
                         encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama AUC {en_iyi:.4f}\nModel: {iyi}\n{C}")


if __name__ == "__main__":
    main()
