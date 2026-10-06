#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C DEDEKTORU v3 (512 px, uc kanalli kontrast gorunumu)

SORUN: 9.27'de C klasoru 360 hucre, sadece ~%26'si gercek C.
v3'te iki degisiklik:
  1) Kontrast kanallari (onisleme.py): ham + yerel kontrast + ince cizgi.
     Literaturde EL icin standart; catlak/cizik ayrimina yardim etmeli.
  2) Altin gunlerden (9.22-9.26) senin C OLMADIGINI soyledigin hucreler.
     Ozellikle C klasorune dusup senin cikardiklarin: x3 agirlik. Bunlar
     dedektorun tam yanildigi ornekler ve v2'de hic kullanilmadi.

Baslangic: v2 (tur 3). Dogrulama: 9.21.
SINAV: 9.27 (C icin hic dokunulmamis tek gun), v2 ile yan yana:
  - gunun 95 gercek C'sinden kaci yakalanir
  - kacan C'lerden kaci A adayi (A'ya sizma riski)
  - tam gunde C klasoru tahmini buyuklugu ve isabeti
Her tur model_c3/tur<N>.pt olarak kaydedilir.

Sure: ~6-7 saat. Kesilirse ayni komutla devam eder.
  caffeinate -i python c_dedektor3.py ~/Desktop/fabrika-fotolar/2B
"""

import sys
import csv
import json
import math
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import models
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    import onisleme
except ImportError as e:
    print(f"HATA: eksik ({e.name}). onisleme.py bu klasorde mi?"); sys.exit(1)

import c_dedektor2 as v2        # veri toplama yardimcilari ve sabitler

BOYUT = 512
BATCH = 8
ISCI = 2
TUR_SAYISI = 3
TUR_ORNEK = 8000
POZITIF_PAYI = 0.25
LR_GOVDE, LR_BAS = 5e-5, 5e-4
ALTIN_ZOR_AGIRLIK = 3.0          # C klasorune dusup senin cikardigin hucreler
ALTIN_NEG_SINIR = 3000           # altin gunlerdeki diger C-olmayanlardan orneklem

BASLANGIC = Path("model_c2/tur3.pt")
CIKTI = Path("model_c3")
D = Path.home() / "Desktop"
EGITIM_ALTIN = [  # (onay klasoru, duzeltmeler) - 9.27 HARIC
    (Path("cikti_altin"), D / "altin_hatalar"),
    (Path("cikti_altin_924"), D / "altin2_hatalar"),
    (Path("cikti_altin_923"), D / "altin3_hatalar"),
    (Path("cikti_altin_925"), D / "altin4_hatalar"),
    (Path("cikti_altin_926"), D / "altin5_hatalar"),
]
SINAV_ONAY = Path("cikti_altin_927")
SINAV_HATA = D / "altin6_hatalar"
SINAV_KL = Path("altin_kume6")
TAM_GUN_SAYISI = 10866           # 9.27'nin tamami (9.27_AI)
HARIC_GUN = {"20260927"}
DOG_GUN = {"20260921"}
V2_ESIK = (0.05, 0.15)
DENEME_ESIK = [(0.05, 0.15), (0.05, 0.30), (0.10, 0.30), (0.10, 0.50), (0.20, 0.50)]
C = "=" * 78
torch.set_num_threads(4)
torch.manual_seed(0)
rng = np.random.RandomState(0)


class Kume(Dataset):
    def __init__(self, yollar, etiket, egitim=False):
        self.yollar, self.etiket, self.egitim = yollar, etiket, egitim

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                x = onisleme.kanal3(im, BOYUT)
        except Exception:
            x = np.zeros((3, BOYUT, BOYUT), dtype=np.float32)
        if self.egitim:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
        return torch.from_numpy(np.ascontiguousarray(x)), self.etiket[i], i


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


def etiket_oku(onay, hatalar):
    et = {}
    for kok in (onay, hatalar):
        if kok.is_dir():
            for d in kok.iterdir():
                if d.is_dir() and not d.name.startswith("_") and "emin" not in v2.normal(d.name):
                    k = v2.anahtar(d.name)
                    if k:
                        for f in d.iterdir():
                            if f.suffix.lower() in v2.UZANTILAR:
                                et[f] = k
    return et


def c_klasoru_mu(ad):
    return "supheli" in v2.normal(ad) or any(x in ad for x in v2.C_KEYS)


def sinav_hazirla():
    et = {f.name: k for f, k in etiket_oku(SINAV_ONAY, SINAV_HATA).items()}
    R = [r for r in csv.DictReader(open(SINAV_ONAY / "_kayit.csv", encoding="utf-8-sig"))
         if r["dosya"] in et and (SINAV_KL / r["dosya"]).exists()]
    g = np.array([v2.c_mi(et[r["dosya"]]) for r in R])
    aday = np.array([r["A_adayi_mi"] == "evet" for r in R])
    p_v2 = np.array([float(r.get("C_dedektor") or 0) for r in R])
    olcek = (TAM_GUN_SAYISI - g.sum()) / max((~g).sum(), 1)
    return [SINAV_KL / r["dosya"] for r in R], g, aday, p_v2, olcek


def sinav_rapor(yaz, baslik, p, g, aday, olcek, esikler):
    for ta, tg in esikler:
        sec = np.where(aday, p >= ta, p >= tg)
        yak = int((sec & g).sum())
        kacan_aday = int((~sec & g & aday).sum())
        yanlis = (sec & ~g).sum() * olcek
        boyut = yak + yanlis
        uyari = "  !! A'ya sizma riski" if kacan_aday else ""
        yaz(f"    {baslik:<10} {ta:.2f}/{tg:.2f} | C {yak:>3}/{int(g.sum()):<3} "
            f"| tam gun C klasoru ~{int(boyut):>4}  isabet ~%{100*yak/max(boyut,1):>3.0f} "
            f"| kacan A adayi {kacan_aday}{uyari}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak", help="2B klasoru")
    a = ap.parse_args()
    kay = Path(a.kaynak).expanduser().resolve()
    for p in (BASLANGIC, v2.GRUPLAR, SINAV_ONAY / "_kayit.csv"):
        if not p.exists():
            print(f"HATA: {p} yok."); sys.exit(1)
    CIKTI.mkdir(exist_ok=True)
    gl = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gl, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nC DEDEKTORU v3 (512 px, {onisleme.SURUM}) - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    ornek = {}   # ad -> (yol, etiket 1=C, agirlik, kaynak)
    oncelik = {"altin_zor": 6, "altin": 5, "inceleme": 4, "dogrulanmis": 3,
               "operator": 2, "2B": 2, "anlasan": 1}

    def ekle(p, c_, kaynak, agirlik=1.0):
        p = Path(p)
        if v2.bozuk(p) or v2.gun(p) in HARIC_GUN or not p.exists():
            return
        ad = v2.ONEK.sub("", p.name)
        if ad in ornek and oncelik[ornek[ad][3]] >= oncelik[kaynak]:
            return
        ornek[ad] = (p, 1 if c_ else 0, agirlik, kaynak)

    # ---- altin gunler (9.27 haric)
    zor_say = Counter()
    for onay, hatalar in EGITIM_ALTIN:
        if not onay.is_dir():
            continue
        klasor = {}
        if (onay / "_kayit.csv").exists():
            klasor = {r["dosya"]: r["klasor"] for r in
                      csv.DictReader(open(onay / "_kayit.csv", encoding="utf-8-sig"))}
        et = etiket_oku(onay, hatalar)
        normal_neg = []
        for f, k in et.items():
            if v2.c_mi(k):
                ekle(f, True, "altin", 2.0)
            elif c_klasoru_mu(klasor.get(f.name, "")):
                ekle(f, False, "altin_zor", ALTIN_ZOR_AGIRLIK)   # C klasorunden cikardigin
                zor_say[onay.name] += 1
            else:
                normal_neg.append(f)
        for i in rng.permutation(len(normal_neg))[:ALTIN_NEG_SINIR // len(EGITIM_ALTIN)]:
            ekle(normal_neg[i], False, "altin")

    # ---- v2'nin veri kaynaklari (incelemeler, dogrulanmis gun, 2B, operator, anlasan)
    for kl in v2.INCELEMELER:
        if kl.is_dir():
            for d in kl.iterdir():
                if d.is_dir() and "emin" not in v2.normal(d.name):
                    k = v2.anahtar(d.name)
                    if not k:
                        continue
                    c_ = v2.c_mi(k)
                    for f in d.iterdir():
                        if f.suffix.lower() in v2.UZANTILAR:
                            ekle(f, c_, "inceleme",
                                 v2.AGIRLIK["inceleme_C" if c_ else "inceleme_degil"])
    if v2.MANIFEST.exists():
        man = list(csv.DictReader(open(v2.MANIFEST, encoding="utf-8-sig")))
        neg = [r for r in man if not v2.c_mi(v2.anahtar(r["etiket"]))]
        sec = {neg[i]["yol"] for i in rng.permutation(len(neg))[:v2.MANIFEST_NEG]}
        for r in man:
            c_ = v2.c_mi(v2.anahtar(r["etiket"]))
            if c_ or r["yol"] in sec:
                ekle(r["yol"], c_, "dogrulanmis")
    for d in kay.iterdir():
        if d.is_dir() and v2.c_mi(v2.anahtar(d.name)):
            for f in d.iterdir():
                if f.suffix.lower() in v2.UZANTILAR:
                    ekle(f, True, "2B")
    R = list(csv.DictReader(open(v2.GRUPLAR, encoding="utf-8-sig")))
    for r in R:
        if r["operator"] in v2.C_KEYS:
            ekle(r["yol"], True, "operator")
    anl = defaultdict(list)
    for r in R:
        if r["grup"] == "anlasan_B" and r["operator"] not in v2.C_KEYS:
            anl[r["operator"]].append(r["yol"])
        if r["grup"] == "anlasan_A":
            anl[v2.A].append(r["yol"])
    for k, ys in anl.items():
        kota = v2.NEG_KOTA.get(k, v2.NEG_A if k == v2.A else 0)
        for i in rng.permutation(len(ys))[:kota]:
            ekle(ys[i], False, "anlasan")

    kayit = list(ornek.values())
    tr = [x for x in kayit if v2.gun(x[0]) not in DOG_GUN]
    dg = [x for x in kayit if v2.gun(x[0]) in DOG_GUN]
    yaz(f"Egitim  : C {sum(x[1] for x in tr):>5}  degil {sum(1-x[1] for x in tr):>5}")
    yaz(f"Dogrulama (9.21): C {sum(x[1] for x in dg)}  degil {sum(1-x[1] for x in dg)}")
    for (kyn, et_), n in sorted(Counter((x[3], "C" if x[1] else "degil") for x in tr).items()):
        yaz(f"    {kyn:<12} {et_:<6} {n}")
    yaz(f"  C klasorunden cikardigin (zor ornek), gun bazinda: {dict(zor_say)}")
    if not tr or not any(x[1] for x in dg) or sum(zor_say.values()) == 0:
        yaz("HATA: egitim/dogrulama bos ya da altin gunlerden zor ornek okunamadi."); sys.exit(1)

    s_yol, s_g, s_aday, s_p2, s_olcek = sinav_hazirla()
    yaz(f"\nSinav 9.27: {len(s_yol)} gorsel, {int(s_g.sum())} gercek C "
        f"(C-olmayanlar tam gune x{s_olcek:.1f} olceklenir)")
    sinav_rapor(yaz, "v2 (simdi)", s_p2, s_g, s_aday, s_olcek, [V2_ESIK])

    net = models.resnet18(weights=None); net.fc = nn.Linear(512, 2)
    net.load_state_dict(torch.load(BASLANGIC, map_location="cpu")["model"])
    yaz(f"\nBaslangic: {BASLANGIC}")

    bas_id = {id(p) for p in net.fc.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in bas_id], "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS}], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.02, reduction="none")

    tr_y = [x[1] for x in tr]
    nC, nD = sum(tr_y), len(tr_y) - sum(tr_y)
    ornekleme = [POZITIF_PAYI / nC if y else (1 - POZITIF_PAYI) / nD for y in tr_y]
    agirlik = torch.tensor([x[2] for x in tr], dtype=torch.float)
    yl = DataLoader(Kume([x[0] for x in tr], tr_y, egitim=True), batch_size=BATCH,
                    sampler=WeightedRandomSampler(ornekleme, min(len(tr), TUR_ORNEK),
                                                  replacement=True), num_workers=ISCI)
    adim = max(len(yl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s_: 0.5 * (1 + math.cos(math.pi * min(s_ / (TUR_SAYISI * adim), 1.0))))

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
              "dog_auc": auc, "on_isleme": onisleme.SURUM,
              "gorev": "C dedektoru (sinif 1 = C)"}
        torch.save(ck, CIKTI / f"tur{tur+1}.pt")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son)
        if auc > en_iyi:
            en_iyi = auc
            torch.save(ck, iyi)
            yaz("  >>> EN IYI (dogrulamaya gore)")
        yaz("  sinav 9.27:")
        p3 = c_olasiligi(net, s_yol)
        sinav_rapor(yaz, "v2 (simdi)", s_p2, s_g, s_aday, s_olcek, [V2_ESIK])
        sinav_rapor(yaz, f"v3 tur{tur+1}", p3, s_g, s_aday, s_olcek, DENEME_ESIK)
        np.save(CIKTI / f"sinav_tur{tur+1}.npy", p3)
        durum.write_text(json.dumps({"tur": tur + 1, "egitim": len(tr), "en_iyi": en_iyi}),
                         encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama AUC {en_iyi:.4f}\n"
        f"Her tur: {CIKTI}/tur1.pt ... tur{TUR_SAYISI}.pt\n{C}")


if __name__ == "__main__":
    main()
