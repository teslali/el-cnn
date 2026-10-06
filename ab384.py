#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B MODELI 384 px (ab384) - gece egitimi

NEDEN
  A kararini veren ab4 224 px'te calisiyor ve ince cizikleri goremiyor.
  9.24'te A klasorunun yarisi ince cizikli B. Kategori modeli (kat5, 384 px)
  cizikleri ayirt etmeyi ogrendi; bu model onun govdesinden baslar.

GOREV: ab4 ile ayni (her hucre icin A mi B mi). Sinif 0 = A, 1 = B, ayni duzen;
  gunluk.py'de dogrudan ab4'un yerine gecebilir.

VERI
  A: anlasan_A, dogrulanmis gun (automation-test), incelemelerdeki A'lar (x2)
  B: incelemelerdeki B'ler (x3, modelin A sanip yanildigi hucreler)
     + anlasan_B (ince kusurlu kategoriler agirlikli) + dogrulanmis gun B'leri
  Haric: 9.22 ve 9.24 (altin gunler), OverTime. Dogrulama: 9.21.

SINAV: her turdan sonra iki altin kumede, ayni gun icin ab4 ile yan yana.
  Her tur model_ab384/tur<N>.pt olarak ayrica kaydedilir.

Sure: ~8 saat. Kesilirse ayni komutla devam eder.
  caffeinate -i python ab384.py
"""

import sys
import csv
import json
import math
import re
import time
from pathlib import Path
from collections import Counter, defaultdict

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

BOYUT = 384
BATCH = 12
ISCI = 2
TUR_SAYISI = 4
TUR_ORNEK = 15000
LR_GOVDE, LR_BAS = 5e-5, 1e-3

GOVDE = Path("model_kat5/en_iyi_model.pt")
GRUPLAR = Path("ayrim_gruplar.csv")
MANIFEST = Path("duzeltme_listesi.csv")
D = Path.home() / "Desktop"
INCELEMELER = [D / "yeni_duzeltmeler", D / "cift_inceleme_hatalar"]
ALTINLAR = [
    ("9.22", Path("cikti_altin"), D / "altin_hatalar", Path("altin_kume"),
     Path("cikti_altin2/_kayit.csv")),
    ("9.24", Path("cikti_altin_924"), D / "altin2_hatalar", Path("altin_kume2"),
     Path("cikti_altin_924b/_kayit.csv")),
]
CIKTI = Path("model_ab384")

HARIC_GUN = {"20260922", "20260924"}
DOG_GUN = {"20260921"}
A = "其他"
# anlasan_B'den alinacak B'ler: ince kusurlu kategoriler agirlikli
B_KOTA = {"不规则": 1500, "黑斑": 2000, "黑边": 1500, "断栅": 800,
          "偏移": 800, "气流": 800}
B_DIGER = 1500
MANIFEST_B = 2000
AGIRLIK = {"inceleme_A": 2.0, "inceleme_B": 3.0}
ESIKLER = [0.4, 0.5, 0.7]
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")
ONEK = re.compile(r"^op-.*?_m-.*?_a\d\.\d+_")
C = "=" * 74
torch.set_num_threads(4)
torch.manual_seed(0)
rng = np.random.RandomState(0)


def normal(s):
    return " ".join(s.translate(str.maketrans("ıİşŞçÇğĞöÖüÜ", "iIsScCgGoOuU"))
                    .lower().replace("_", " ").split())


def a_mi(ad):
    return A in ad or "a sinif" in normal(ad)


def gun(p):
    z = ZAMAN.search(Path(p).name)
    return z.group(1)[:8] if z else ""


def bozuk(p):
    return "OverTime" in Path(p).name or "_Class400_" in Path(p).name


class Kume(Dataset):
    def __init__(self, yollar, etiket, egitim=False):
        self.yollar, self.etiket = yollar, etiket
        t = [transforms.Resize((BOYUT, BOYUT))]
        if egitim:
            t += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.ColorJitter(brightness=0.25, contrast=0.25)]
        t += [transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.don = transforms.Compose(t)

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                x = self.don(im.convert("RGB"))
        except Exception:
            x = torch.zeros(3, BOYUT, BOYUT)
        return x, self.etiket[i], i


def a_olasiligi(net, yollar):
    yl = DataLoader(Kume(yollar, [0] * len(yollar)), batch_size=BATCH,
                    shuffle=False, num_workers=ISCI)
    P = np.zeros(len(yollar)); n, t0 = 0, time.time()
    net.eval()
    with torch.no_grad():
        for x, _, idx in yl:
            P[idx.numpy()] = torch.softmax(net(x), 1)[:, 0].numpy()
            n += len(idx)
            hz = n / max(time.time() - t0, 1e-9)
            print(f"\r    {n}/{len(yollar)}  kalan ~{(len(yollar)-n)/max(hz,1e-9)/60:.0f} dk",
                  end="", flush=True)
    print()
    return P


def altin_hazirla():
    """her altin gun icin: yollar, gercek A (bool), ab4'un A olasiligi"""
    kumeler = []
    for ad, onay, hatalar, kl, kayit in ALTINLAR:
        if not (kayit.exists() and kl.is_dir()):
            continue
        etiket = {}
        for kok in (onay, hatalar):
            if kok.is_dir():
                for d in kok.iterdir():
                    if d.is_dir() and "emin" not in normal(d.name):
                        for f in d.iterdir():
                            if f.suffix.lower() in UZANTILAR:
                                etiket[f.name] = a_mi(d.name)
        R = [r for r in csv.DictReader(open(kayit, encoding="utf-8-sig"))
             if r["dosya"] in etiket and (kl / r["dosya"]).exists()]
        kumeler.append((ad, [kl / r["dosya"] for r in R],
                        np.array([etiket[r["dosya"]] for r in R]),
                        np.array([float(r["A_olasiligi"]) for r in R])))
    return kumeler


def sinav(yaz, baslik, p, g):
    auc = roc_auc_score(g, p) if 0 < g.sum() < len(g) else float("nan")
    s = f"    {baslik:<12} AUC {auc:.3f} |"
    for e in ESIKLER:
        sec = p >= e
        d = int((sec & g).sum())
        s += f" {e:.1f}: {int(sec.sum()):>4} sec %{100*d/max(sec.sum(),1):>3.0f} isb %{100*d/max(g.sum(),1):>3.0f} yak |"
    yaz(s)


def main():
    for p in (GOVDE, GRUPLAR):
        if not p.exists():
            print(f"HATA: {p} yok."); sys.exit(1)
    CIKTI.mkdir(exist_ok=True)
    gl = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gl, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nA/B MODELI 384 px - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    ornek = {}   # orijinal ad -> (yol, etiket 0=A 1=B, agirlik, kaynak)
    oncelik = {"inceleme": 3, "dogrulanmis": 2, "anlasan": 1}

    def ekle(p, a_, kaynak, agirlik=1.0):
        p = Path(p)
        if bozuk(p) or gun(p) in HARIC_GUN or not p.exists():
            return
        ad = ONEK.sub("", p.name)
        if ad in ornek and oncelik[ornek[ad][3]] >= oncelik[kaynak]:
            return
        ornek[ad] = (p, 0 if a_ else 1, agirlik, kaynak)

    for kl in INCELEMELER:
        if kl.is_dir():
            for d in kl.iterdir():
                if d.is_dir() and "emin" not in normal(d.name):
                    a_ = a_mi(d.name)
                    for f in d.iterdir():
                        if f.suffix.lower() in UZANTILAR:
                            ekle(f, a_, "inceleme",
                                 AGIRLIK["inceleme_A" if a_ else "inceleme_B"])
    if MANIFEST.exists():
        man = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
        man_B = [r for r in man if A not in r["etiket"]]
        sec_B = {man_B[i]["yol"] for i in rng.permutation(len(man_B))[:MANIFEST_B]}
        for r in man:
            if A in r["etiket"] or r["yol"] in sec_B:
                ekle(r["yol"], A in r["etiket"], "dogrulanmis")
    R = list(csv.DictReader(open(GRUPLAR, encoding="utf-8-sig")))
    for r in R:
        if r["grup"] == "anlasan_A":
            ekle(r["yol"], True, "anlasan")
    anl_B = defaultdict(list)
    for r in R:
        if r["grup"] == "anlasan_B":
            anl_B[r["operator"]].append(r["yol"])
    for k, ys in anl_B.items():
        kota = B_KOTA.get(k, 0)
        for i in rng.permutation(len(ys))[:kota]:
            ekle(ys[i], False, "anlasan")
    diger = [y for k, ys in anl_B.items() if k not in B_KOTA for y in ys]
    for i in rng.permutation(len(diger))[:B_DIGER]:
        ekle(diger[i], False, "anlasan")

    kayit = list(ornek.values())
    tr = [x for x in kayit if gun(x[0]) not in DOG_GUN]
    dg = [x for x in kayit if gun(x[0]) in DOG_GUN]
    yaz(f"Egitim  : A {sum(x[1]==0 for x in tr):>5}  B {sum(x[1]==1 for x in tr):>5}")
    yaz(f"Dogrulama (9.21): A {sum(x[1]==0 for x in dg)}  B {sum(x[1]==1 for x in dg)}")
    for (kyn, et), n in sorted(Counter((x[3], "A" if x[1] == 0 else "B") for x in tr).items()):
        yaz(f"    {kyn:<12} {et}  {n}")
    if not tr or not any(x[1] == 0 for x in dg):
        yaz("HATA: egitim ya da dogrulama bos."); sys.exit(1)

    altin = altin_hazirla()
    yaz(f"\nSinav: " + ", ".join(f"{ad} ({len(y)} gorsel, {int(g.sum())} A)"
                                  for ad, y, g, _ in altin))
    yaz("  (her esikte: secilen A sayisi, isabet, gercek A'larin yakalanma orani)")
    for ad, y, g, p4 in altin:
        yaz(f"  {ad}")
        sinav(yaz, "ab4 (224)", p4, g)

    # ---- model: kat5 govdesi, yeni A/B basi
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, 2)
    net.load_state_dict({k: v for k, v in torch.load(GOVDE, map_location="cpu")["model"].items()
                         if not k.startswith("fc.")}, strict=False)
    yaz(f"\nBaslangic: {GOVDE.parent.name} govdesi (384 px, cizikleri taniyor), yeni A/B basi")

    bas_id = {id(p) for p in net.fc.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in bas_id], "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS}], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    tr_y = [x[1] for x in tr]
    nA, nB = tr_y.count(0), tr_y.count(1)
    ornekleme = [0.5 / nA if y == 0 else 0.5 / nB for y in tr_y]
    agirlik = torch.tensor([x[2] for x in tr], dtype=torch.float)
    yl = DataLoader(Kume([x[0] for x in tr], tr_y, egitim=True), batch_size=BATCH,
                    sampler=WeightedRandomSampler(ornekleme, min(len(tr), TUR_ORNEK),
                                                  replacement=True), num_workers=ISCI)
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

    yaz(f"\n{C}\nEGITIM ({TUR_SAYISI} tur, {BOYUT} px)\n{C}")
    for tur in range(bas_tur, TUR_SAYISI):
        net.train(); t0, kt, n = time.time(), 0.0, 0
        top = min(len(tr), TUR_ORNEK)
        for b, (x, y, idx) in enumerate(yl):
            opt.zero_grad()
            k = (kayip_fn(net(x), y) * agirlik[idx]).mean()
            k.backward(); opt.step(); sched.step()
            kt += float(k) * len(y); n += len(y)
            if b % 20 == 0:
                hz = n / max(time.time() - t0, 1e-9)
                print(f"\r  tur {tur+1}/{TUR_SAYISI}  {n}/{top}  kayip {kt/max(n,1):.4f}  "
                      f"kalan ~{(top-n)/max(hz,1e-9)/60:.0f} dk", end="", flush=True)
        print()
        yaz(f"\n  --- TUR {tur+1}  ({(time.time()-t0)/60:.0f} dk) ---")
        yaz("  dogrulama (9.21):")
        pd = a_olasiligi(net, [x[0] for x in dg])
        auc = roc_auc_score([x[1] == 0 for x in dg], pd)
        yaz(f"  DOGRULAMA AUC {auc:.4f}")
        kayit_ck = {"model": net.state_dict(), "girdi_boyut": BOYUT, "tur": tur + 1,
                    "dog_auc": float(auc), "sinif": ["A", "B"],
                    "gorev": "A/B (sinif 0 = A)"}
        torch.save(kayit_ck, CIKTI / f"tur{tur+1}.pt")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son)
        if auc > en_iyi:
            en_iyi = float(auc)
            torch.save(kayit_ck, iyi)
            yaz("  >>> EN IYI (dogrulamaya gore)")
        yaz("  sinav:")
        for ad, y, g, p4 in altin:
            yaz(f"  {ad}")
            sinav(yaz, "ab4 (224)", p4, g)
            sinav(yaz, f"ab384 tur{tur+1}", a_olasiligi(net, y), g)
        durum.write_text(json.dumps({"tur": tur + 1, "egitim": len(tr), "en_iyi": en_iyi}),
                         encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama AUC {en_iyi:.4f}\nModel: {iyi}\n"
        f"Her tur ayrica: {CIKTI}/tur1.pt ... tur{TUR_SAYISI}.pt\n{C}")


if __name__ == "__main__":
    main()
