#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C DEDEKTORU v2 (512 px) - gece egitimi

SORUN
  C supheli klasoru C'leri neredeyse eksiksiz yakaliyor (3 gunde 366'da 364),
  ama icinde cizik gibi yanlis alarmlar var (9.23'te isabet %78).
  v1 sadece ~250 C ornegiyle egitildi, cizikle catlagi tam ayirt edemiyor.

v2
  Pozitif (C): 2B + dogrulanmis gun + yeni verideki operator C'leri (~700)
               + incelemelerde C dediklerin (x2)
  Negatif    : incelemelerde C OLMADIGINI soyledigin hucreler (x3; cizikler,
               operatorun C deyip senin B dedigin hucreler) + cizik agirlikli
               anlasan orneklem + A orneklemi
  Baslangic  : v1 (model_c) - bildiklerini korur, yeni veriyle ince ayar
  Haric      : 9.22, 9.23, 9.24 (altin gunler), OverTime. Dogrulama: 9.21.

SINAV (her tur, uc altin gunde):
  Butun gercek C'ler + 500 rastgele C-olmayan hucre. Ayni hucrelerde v1 ile
  yan yana: yakalanan C, yanlis alarm (tum kumeye olceklenmis tahmin) ve
  yanlis alarmlarin kaci cizik.
  Her tur model_c2/tur<N>.pt olarak ayrica kaydedilir.

Sure: ~6 saat. Kesilirse ayni komutla devam eder.
  caffeinate -i python c_dedektor2.py ~/Desktop/fabrika-fotolar/2B
"""

import sys
import csv
import json
import math
import re
import time
import argparse
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

BOYUT = 512
BATCH = 8
ISCI = 2
TUR_SAYISI = 3
TUR_ORNEK = 8000
POZITIF_PAYI = 0.25
LR_GOVDE, LR_BAS = 5e-5, 5e-4
AGIRLIK = {"inceleme_C": 2.0, "inceleme_degil": 3.0}
NEG_KOTA = {"不规则": 2500, "黑斑": 1500, "断栅": 800, "黑边": 800, "偏移": 500}
NEG_DIGER = 1000
NEG_A = 1000
MANIFEST_NEG = 1500
SINAV_NEG = 500
ESIKLER = [0.2, 0.3, 0.4, 0.5]

V1 = Path("model_c/en_iyi_model.pt")
GRUPLAR = Path("ayrim_gruplar.csv")
MANIFEST = Path("duzeltme_listesi.csv")
D = Path.home() / "Desktop"
INCELEMELER = [D / "yeni_duzeltmeler", D / "cift_inceleme_hatalar"]
ALTINLAR = [
    ("9.22", Path("cikti_altin"), D / "altin_hatalar", Path("altin_kume"),
     Path("cikti_altin2/_kayit.csv")),
    ("9.24", Path("cikti_altin_924"), D / "altin2_hatalar", Path("altin_kume2"),
     Path("cikti_altin_924b/_kayit.csv")),
    ("9.23", Path("cikti_altin_923"), D / "altin3_hatalar", Path("altin_kume3"),
     Path("cikti_altin_923/_kayit.csv")),
]
CIKTI = Path("model_c2")

HARIC_GUN = {"20260922", "20260923", "20260924"}
DOG_GUN = {"20260921"}
C_KEYS = {"隐裂", "压板"}
A = "其他"
ANAHTARLAR = ["舟框印", "舟划", "舟印", "其他", "黑斑", "黑边", "偏移", "隐裂",
              "压板", "气流", "吸笔", "吸盘", "云雾", "断栅", "接触", "脏污",
              "分片", "不规则", "卡点", "皮带", "滚轮", "暗片"]
TURKCE = [("c supheli", "C"), ("a sinif", "其他"), ("nokta bask", "卡点"),
          ("siyah nokta", "黑斑"), ("siyah leke", "黑斑"), ("siyah kenar", "黑边"),
          ("siyah kose", "黑边"), ("baski kay", "偏移"), ("baski plaka", "压板"),
          ("gizli", "隐裂"), ("catlak", "隐裂"), ("grafit", "舟划"), ("grafik", "舟划"),
          ("kuvars", "舟印"), ("cercev", "舟框印"), ("hava", "气流"), ("kalem", "吸笔"),
          ("emici", "吸盘"), ("bulut", "云雾"), ("grid", "断栅"), ("temas", "接触"),
          ("kirlen", "脏污"), ("ayirma", "分片"), ("duzensiz", "不规则"),
          ("cizik", "不规则"), ("konveyor", "皮带"), ("rulo", "滚轮"), ("koyu", "暗片")]
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")
ONEK = re.compile(r"^op-.*?_m-.*?_a\d\.\d+_")
C = "=" * 76
torch.set_num_threads(4)
torch.manual_seed(0)
rng = np.random.RandomState(0)


def normal(s):
    return " ".join(s.translate(str.maketrans("ıİşŞçÇğĞöÖüÜ", "iIsScCgGoOuU"))
                    .lower().replace("_", " ").split())


def anahtar(ad):
    n = normal(ad)
    if "c supheli" in n:
        return "C"
    for k in ANAHTARLAR:
        if k in ad:
            return k
    for ifade, k in TURKCE:
        if ifade in n:
            return k
    return None


def c_mi(k):
    return k in C_KEYS or k == "C"


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
                  transforms.ColorJitter(brightness=0.2, contrast=0.2)]
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


def c_olasiligi(net, yollar):
    yl = DataLoader(Kume(yollar, [0] * len(yollar)), batch_size=BATCH,
                    shuffle=False, num_workers=ISCI)
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


def altin_hazirla():
    """her gun: yollar, gercek C (bool), gercek anahtar, v1 olasiligi, olcek"""
    kumeler = []
    r2 = np.random.RandomState(7)
    for ad, onay, hatalar, kl, kayit in ALTINLAR:
        if not (kayit.exists() and kl.is_dir()):
            continue
        etiket = {}
        for kok in (onay, hatalar):
            if kok.is_dir():
                for d in kok.iterdir():
                    if d.is_dir() and "emin" not in normal(d.name):
                        k = anahtar(d.name)
                        if k:
                            for f in d.iterdir():
                                if f.suffix.lower() in UZANTILAR:
                                    etiket[f.name] = k
        R = [r for r in csv.DictReader(open(kayit, encoding="utf-8-sig"))
             if r["dosya"] in etiket and (kl / r["dosya"]).exists() and r.get("C_dedektor")]
        cler = [r for r in R if c_mi(etiket[r["dosya"]])]
        degil = [r for r in R if not c_mi(etiket[r["dosya"]])]
        secilen = [degil[i] for i in r2.permutation(len(degil))[:SINAV_NEG]]
        olcek = len(degil) / max(len(secilen), 1)
        S = cler + secilen
        kumeler.append((ad, [kl / r["dosya"] for r in S],
                        np.array([c_mi(etiket[r["dosya"]]) for r in S]),
                        np.array([etiket[r["dosya"]] for r in S], dtype=object),
                        np.array([float(r["C_dedektor"]) for r in S]), olcek))
    return kumeler


def sinav(yaz, baslik, p, g, k, olcek):
    s = f"    {baslik:<11} |"
    for e in ESIKLER:
        sec = p >= e
        yak = int((sec & g).sum())
        yanlis = sec & ~g
        cz = int((yanlis & (k == "不规则")).sum())
        s += (f" {e:.1f}: C {yak:>3}/{int(g.sum()):<3} yanlis ~{int(yanlis.sum()*olcek):>4}"
              f" (cizik ~{int(cz*olcek):>3}) |")
    yaz(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak", help="2B klasoru")
    a = ap.parse_args()
    kay = Path(a.kaynak).expanduser().resolve()
    for p in (V1, GRUPLAR):
        if not p.exists():
            print(f"HATA: {p} yok."); sys.exit(1)
    CIKTI.mkdir(exist_ok=True)
    gl = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gl, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nC DEDEKTORU v2 (512 px) - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    ornek = {}   # ad -> (yol, etiket 1=C, agirlik, kaynak)
    oncelik = {"inceleme": 4, "dogrulanmis": 3, "operator": 2, "2B": 2, "anlasan": 1}

    def ekle(p, c_, kaynak, agirlik=1.0):
        p = Path(p)
        if bozuk(p) or gun(p) in HARIC_GUN or not p.exists():
            return
        ad = ONEK.sub("", p.name)
        if ad in ornek and oncelik[ornek[ad][3]] >= oncelik[kaynak]:
            return
        ornek[ad] = (p, 1 if c_ else 0, agirlik, kaynak)

    # 1) incelemeler (en guvenilir)
    for kl in INCELEMELER:
        if kl.is_dir():
            for d in kl.iterdir():
                if d.is_dir() and "emin" not in normal(d.name):
                    k = anahtar(d.name)
                    if not k:
                        continue
                    c_ = c_mi(k)
                    for f in d.iterdir():
                        if f.suffix.lower() in UZANTILAR:
                            ekle(f, c_, "inceleme",
                                 AGIRLIK["inceleme_C" if c_ else "inceleme_degil"])
    # 2) dogrulanmis gun
    if MANIFEST.exists():
        man = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
        neg = [r for r in man if not c_mi(anahtar(r["etiket"]))]
        sec = {neg[i]["yol"] for i in rng.permutation(len(neg))[:MANIFEST_NEG]}
        for r in man:
            c_ = c_mi(anahtar(r["etiket"]))
            if c_ or r["yol"] in sec:
                ekle(r["yol"], c_, "dogrulanmis")
    # 3) 2B klasorundeki C'ler
    for d in kay.iterdir():
        if d.is_dir() and c_mi(anahtar(d.name)):
            for f in d.iterdir():
                if f.suffix.lower() in UZANTILAR:
                    ekle(f, True, "2B")
    # 4) yeni veri: operator C'leri + negatif orneklem
    R = list(csv.DictReader(open(GRUPLAR, encoding="utf-8-sig")))
    for r in R:
        if r["operator"] in C_KEYS:
            ekle(r["yol"], True, "operator")
    anl = defaultdict(list)
    for r in R:
        if r["grup"] == "anlasan_B" and r["operator"] not in C_KEYS:
            anl[r["operator"]].append(r["yol"])
        if r["grup"] == "anlasan_A":
            anl[A].append(r["yol"])
    for k, ys in anl.items():
        kota = NEG_KOTA.get(k, NEG_A if k == A else 0)
        for i in rng.permutation(len(ys))[:kota]:
            ekle(ys[i], False, "anlasan")
    diger = [y for k, ys in anl.items() if k not in NEG_KOTA and k != A for y in ys]
    for i in rng.permutation(len(diger))[:NEG_DIGER]:
        ekle(diger[i], False, "anlasan")

    kayit = list(ornek.values())
    tr = [x for x in kayit if gun(x[0]) not in DOG_GUN]
    dg = [x for x in kayit if gun(x[0]) in DOG_GUN]
    yaz(f"Egitim  : C {sum(x[1] for x in tr):>5}  degil {sum(1-x[1] for x in tr):>5}")
    yaz(f"Dogrulama (9.21): C {sum(x[1] for x in dg)}  degil {sum(1-x[1] for x in dg)}")
    for (kyn, et), n in sorted(Counter((x[3], "C" if x[1] else "degil") for x in tr).items()):
        yaz(f"    {kyn:<12} {et:<6} {n}")
    if not tr or not any(x[1] for x in dg):
        yaz("HATA: egitim ya da dogrulamada C yok."); sys.exit(1)

    altin = altin_hazirla()
    yaz(f"\nSinav: her gunde tum gercek C'ler + {SINAV_NEG} rastgele C-olmayan hucre")
    yaz("  (yanlis alarm ve cizik sayilari tum kumeye olceklenmis tahmin)")
    for ad, y, g, k, p1, olc in altin:
        yaz(f"  {ad}  ({int(g.sum())} C)")
        sinav(yaz, "v1", p1, g, k, olc)

    net = models.resnet18(weights=None); net.fc = nn.Linear(512, 2)
    net.load_state_dict(torch.load(V1, map_location="cpu")["model"])
    yaz(f"\nBaslangic: v1 (model_c), yeni veriyle ince ayar")

    bas_id = {id(p) for p in net.fc.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in bas_id], "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS}], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    tr_y = [x[1] for x in tr]
    nC, nD = sum(tr_y), len(tr_y) - sum(tr_y)
    ornekleme = [POZITIF_PAYI / nC if y else (1 - POZITIF_PAYI) / nD for y in tr_y]
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
        pd = c_olasiligi(net, [x[0] for x in dg])
        auc = float(roc_auc_score([x[1] for x in dg], pd))
        yaz(f"  DOGRULAMA AUC {auc:.4f}")
        ck = {"model": net.state_dict(), "girdi_boyut": BOYUT, "tur": tur + 1,
              "dog_auc": auc, "gorev": "C dedektoru (sinif 1 = C)"}
        torch.save(ck, CIKTI / f"tur{tur+1}.pt")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son)
        if auc > en_iyi:
            en_iyi = auc
            torch.save(ck, iyi)
            yaz("  >>> EN IYI (dogrulamaya gore)")
        yaz("  sinav:")
        for ad, y, g, k, p1, olc in altin:
            yaz(f"  {ad}  ({int(g.sum())} C)")
            sinav(yaz, "v1", p1, g, k, olc)
            sinav(yaz, f"v2 tur{tur+1}", c_olasiligi(net, y), g, k, olc)
        durum.write_text(json.dumps({"tur": tur + 1, "egitim": len(tr), "en_iyi": en_iyi}),
                         encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama AUC {en_iyi:.4f}\nModel: {iyi}\n"
        f"Her tur ayrica: {CIKTI}/tur1.pt ... tur{TUR_SAYISI}.pt\n{C}")


if __name__ == "__main__":
    main()
