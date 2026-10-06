#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KATEGORI MODELI v6 (384 px)

v5'ten farklar:
  - DORT ALTIN GUN egitime giriyor (9.22, 9.23, 9.24, 9.25; ~8.000 gorsel,
    senin standardinla etiketli). Duzelttiklerin x3, onayladiklarin x1.5.
  - Nokta baskisi: eski tanimli etiketler (2B ve eski modelle anlasmaya dayali)
    DISARIDA; sadece yeni tanima gore etiketlenenler kaliyor.
  - Kirlenme: "kirlenme yeni tanim" klasorleri kirlenme olarak giriyor.
  - Baslangic: kat5 govdesi. 3 tur, her tur model_kat6/tur<N>.pt olarak kaydedilir.
  - Tarafsiz sinav yok (butun altin gunler egitimde); secim 9.21 dogrulamasiyla,
    asil olcum yeni bir gunle (9.26+) yapilacak.

Sure: ~5.5 saat. Kesilirse ayni komutla devam eder.
  caffeinate -i python kategori6.py ~/Desktop/fabrika-fotolar/2B
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
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})"); sys.exit(1)

# ----------------------------------------------------------------------
BOYUT = 384
BATCH = 12
ISCI = 2
TUR_SAYISI = 3
TUR_ORNEK = 14000
LR_GOVDE, LR_BAS = 5e-5, 1e-3
DENGE_US = 0.3             # ornek agirligi 1/n^0.3 -> sinif payi n^0.7
MIN_ORNEK = 60
ANLASAN_SINIR = 2000
ESKI_SINIR = 800
AGIRLIK = {"altin_duzeltme": 3.0, "altin_onay": 1.5, "inceleme": 3.0,
           "dogrulanmis": 1.0, "anlasan": 1.0, "op_ozel": 1.0, "2B": 1.0}
HAT_AGIRLIK = {"3A": 1.5}
ESKI_HARIC = {"断栅", "卡点"}      # 2B'de: kirik grid guvenilmez, nokta baskisi eski tanim
ANLASAN_HARIC = {"卡点"}           # eski modelle anlasma = eski tanim
OP_OZEL = {"压板", "滚轮"}   # eski modelde sinifi olmadigi icin anlasan'a giremeyenler

GRUPLAR = Path("ayrim_gruplar.csv")
ENVANTER = Path("yeni_envanter.csv")
MANIFEST = Path("duzeltme_listesi.csv")
INCELEMELER = [Path.home() / "Desktop" / "yeni_duzeltmeler",
               Path.home() / "Desktop" / "cift_inceleme_hatalar"]
ALTIN_KL = Path("altin_kume")
ALTIN_ONAY = Path("cikti_altin")
ALTIN_HATA = Path.home() / "Desktop" / "altin_hatalar"
MODEL3 = Path("model3/en_iyi_model.pt")
KAT5 = Path("model_kat5/en_iyi_model.pt")
CIKTI = Path("model_kat6")
D = Path.home() / "Desktop"
ALTINLAR = [(Path("cikti_altin"), D / "altin_hatalar"),
            (Path("cikti_altin_924"), D / "altin2_hatalar"),
            (Path("cikti_altin_923"), D / "altin3_hatalar"),
            (Path("cikti_altin_925"), D / "altin4_hatalar")]
KAT4_EGITIM = {"不规则": 3308, "云雾": 1080, "偏移": 3046, "其他": 4162, "分片": 1425,
               "卡点": 601, "压板": 199, "吸笔": 888, "接触": 461, "断栅": 1239,
               "气流": 2212, "脏污": 1525, "舟划": 1002, "舟框印": 995, "隐裂": 791,
               "黑斑": 6444, "黑边": 3778}

HARIC_GUN = set()     # altin gunler artik egitimde
DOG_GUN = {"20260921"}
A = "其他"
ANAHTARLAR = ["舟框印", "舟划", "舟印", "其他", "黑斑", "黑边", "偏移", "隐裂",
              "压板", "气流", "吸笔", "吸盘", "云雾", "断栅", "接触", "脏污",
              "分片", "不规则", "卡点", "皮带", "滚轮", "暗片"]
AD = {"其他": "A sinifi", "黑斑": "siyah nokta", "黑边": "siyah kenar",
      "偏移": "baski kaymasi", "隐裂": "gizli catlak", "压板": "baski plakasi",
      "舟划": "grafit tekne", "舟框印": "tekne cercevesi", "舟印": "kuvars tekne",
      "气流": "hava akisi", "吸笔": "vakum kalemi", "吸盘": "vakum emici",
      "云雾": "bulutlu", "断栅": "kirik grid", "接触": "temas kirl.",
      "脏污": "kirlenme", "分片": "ayirma cizik", "不规则": "duzensiz cizik",
      "卡点": "nokta baskisi", "皮带": "konveyor", "滚轮": "rulo izi", "暗片": "koyu hucre"}
TURKCE = [("a sinif", "其他"), ("diger", "其他"), ("nokta bask", "卡点"),
          ("siyah nokta", "黑斑"), ("siyah leke", "黑斑"), ("siyah kenar", "黑边"),
          ("siyah kose", "黑边"), ("baski kay", "偏移"), ("baski plaka", "压板"),
          ("gizli", "隐裂"), ("catlak", "隐裂"), ("grafit", "舟划"), ("grafik", "舟划"),
          ("kuvars", "舟印"), ("cercev", "舟框印"), ("hava", "气流"), ("kalem", "吸笔"),
          ("emici", "吸盘"), ("bulut", "云雾"), ("grid", "断栅"), ("temas", "接触"),
          ("kirlen", "脏污"), ("ayirma", "分片"), ("duzensiz", "不规则"),
          ("cizik", "不规则"), ("konveyor", "皮带"), ("rulo", "滚轮"), ("koyu", "暗片")]
RAPOR_KAT = ["黑斑", "黑边", "不规则", "偏移", "气流", "断栅", "脏污"]
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
    if "c supheli" in normal(ad):
        return None                  # catlak/plaka ayrimi yok, kategori egitimine girmez
    for k in ANAHTARLAR:
        if k in ad:
            return k
    n = normal(ad)
    for ifade, k in TURKCE:
        if ifade in n:
            return k
    return None


def gun(p):
    z = ZAMAN.search(Path(p).name)
    return z.group(1)[:8] if z else ""


def hat(p):
    m = re.match(r"^(?:op-.*?_a\d\.\d+_)?(\d[A-Z])_", Path(p).name)
    return m.group(1) if m else ""


def bozuk(p):
    return "OverTime" in Path(p).name or "_Class400_" in Path(p).name


class Kume(Dataset):
    def __init__(self, yollar, etiket, boyut, egitim=False):
        self.yollar, self.etiket, self.boyut = yollar, etiket, boyut
        t = [transforms.Resize((boyut, boyut))]
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
            x = torch.zeros(3, self.boyut, self.boyut)
        return x, self.etiket[i], i


def model_yukle(yol):
    ck = torch.load(yol, map_location="cpu")
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, len(ck["sinif_ad"]))
    net.load_state_dict(ck["model"])
    return net, ck


def olasilik(net, yollar, boyut, n_sinif):
    yl = DataLoader(Kume(yollar, [0] * len(yollar), boyut),
                    batch_size=24 if boyut <= 224 else BATCH, shuffle=False, num_workers=ISCI)
    P = np.zeros((len(yollar), n_sinif), dtype=np.float32)
    n, t0 = 0, time.time()
    net.eval()
    with torch.no_grad():
        for x, _, idx in yl:
            P[idx.numpy()] = torch.softmax(net(x), 1).numpy()
            n += len(idx)
            hz = n / max(time.time() - t0, 1e-9)
            print(f"\r    {n}/{len(yollar)}  kalan ~{(len(yollar)-n)/max(hz,1e-9)/60:.0f} dk",
                  end="", flush=True)
    print()
    return P


def hizala(P, keys_kaynak, keys_hedef):
    Q = np.zeros((P.shape[0], len(keys_hedef)), dtype=np.float32)
    for j, k in enumerate(keys_hedef):
        if k in keys_kaynak:
            Q[:, j] = P[:, keys_kaynak.index(k)]
    return Q


def ayarla(P, keys, sayi, us):
    c = np.array([sayi.get(k, 1) ** us for k in keys])
    Q = P * c
    return Q / Q.sum(1, keepdims=True)


def rapor(yaz, baslik, P, keys, g):
    P = P.copy()
    if A in keys:
        P[:, keys.index(A)] = -1
    t = np.array([keys[i] for i in P.argmax(1)], dtype=object)
    s = f"  {baslik:<20} genel %{100*(t==g).mean():>4.1f} |"
    for k in RAPOR_KAT:
        sec, ger = t == k, g == k
        isb = (g[sec] == k).mean() if sec.any() else float("nan")
        yak = (t[ger] == k).mean() if ger.any() else float("nan")
        s += f" {AD[k][:9]} {100*isb:>3.0f}/{100*yak:<3.0f}|"
    yaz(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak", help="2B klasoru")
    a = ap.parse_args()
    kay = Path(a.kaynak).expanduser().resolve()
    for p in (GRUPLAR, ENVANTER, MODEL3):
        if not p.exists():
            print(f"HATA: {p} yok."); sys.exit(1)
    CIKTI.mkdir(exist_ok=True)
    gl = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gl, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nKATEGORI MODELI v5 (384 px) - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    ornek = {}
    oncelik = {"altin_duzeltme": 7, "altin_onay": 6, "inceleme": 5, "dogrulanmis": 4,
               "op_ozel": 3, "anlasan": 2, "2B": 1}

    def ekle(p, k, kaynak):
        p = Path(p)
        if not k or bozuk(p) or gun(p) in HARIC_GUN or not p.exists():
            return
        ad = ONEK.sub("", p.name)
        if ad in ornek and oncelik[ornek[ad][2]] >= oncelik[kaynak]:
            return
        ornek[ad] = (p, k, kaynak)

    altin_say = Counter()
    for onay, hatalar in ALTINLAR:
        for kok, kaynak in ((onay, "altin_onay"), (hatalar, "altin_duzeltme")):
            if not kok.is_dir():
                continue
            for d in kok.iterdir():
                if not d.is_dir() or d.name.startswith("_") or "emin" in normal(d.name):
                    continue
                k = anahtar(d.name)
                if not k:
                    continue
                for f in d.iterdir():
                    if f.suffix.lower() in UZANTILAR:
                        ekle(f, k, kaynak)
                        altin_say[kaynak] += 1
    inc_say = {}
    for kl in INCELEMELER:
        n0 = len(ornek)
        if kl.is_dir():
            for d in kl.iterdir():
                if d.is_dir() and "emin" not in normal(d.name):
                    k = anahtar(d.name)
                    for f in d.iterdir():
                        if f.suffix.lower() in UZANTILAR:
                            ekle(f, k, "inceleme")
        inc_say[kl.name] = len(ornek) - n0
    if MANIFEST.exists():
        for r in csv.DictReader(open(MANIFEST, encoding="utf-8-sig")):
            ekle(r["yol"], anahtar(r["etiket"]), "dogrulanmis")
    R = list(csv.DictReader(open(GRUPLAR, encoding="utf-8-sig")))
    for r in R:
        if r["operator"] in OP_OZEL:
            ekle(r["yol"], r["operator"], "op_ozel")
    anl = defaultdict(list)
    for r in R:
        if r["grup"] in ("anlasan_A", "anlasan_B") and r["operator"]:
            anl[r["operator"]].append(r["yol"])
    for k, ys in anl.items():
        if k in ANLASAN_HARIC:
            continue
        for i in rng.permutation(len(ys))[:ANLASAN_SINIR]:
            ekle(ys[i], k, "anlasan")
    for d in sorted([x for x in kay.iterdir() if x.is_dir()]):
        k = anahtar(d.name)
        if not k or k in ESKI_HARIC:
            continue
        fs = [f for f in d.iterdir() if f.suffix.lower() in UZANTILAR]
        for i in rng.permutation(len(fs))[:ESKI_SINIR]:
            ekle(fs[i], k, "2B")

    say = Counter(v[1] for v in ornek.values())
    sinif_keys = sorted(k for k, n in say.items() if n >= MIN_ORNEK)
    ad_map = {}
    for d in kay.iterdir():
        k = anahtar(d.name)
        if d.is_dir() and k and k not in ad_map:
            ad_map[k] = d.name
    for r in csv.DictReader(open(ENVANTER, encoding="utf-8-sig")):
        k = anahtar(r["kategori"])
        if k and k not in ad_map:
            ad_map[k] = r["kategori"]
    sinif_ad = [ad_map.get(k, f"{k} {AD.get(k, '')}") for k in sinif_keys]
    idx_of = {k: i for i, k in enumerate(sinif_keys)}
    kayit = [(v[0], idx_of[v[1]], v[2]) for v in ornek.values() if v[1] in idx_of]
    tr = [x for x in kayit if gun(x[0]) not in DOG_GUN]
    dg = [x for x in kayit if gun(x[0]) in DOG_GUN]

    yaz(f"Altin gunler: onay {altin_say['altin_onay']}  duzeltme {altin_say['altin_duzeltme']}")
    for kl, n in inc_say.items():
        yaz(f"Inceleme {kl:<24}: {n}")
    if not any(inc_say.values()):
        yaz("UYARI: inceleme klasorlerinden hic gorsel okunamadi!")
    yaz(f"Siniflar ({len(sinif_keys)}):")
    for k in sinif_keys:
        yaz(f"    {say[k]:>6}  {AD.get(k, k)}")
    yaz(f"Egitim {len(tr)}   Dogrulama (9.21) {len(dg)}")
    for kyn, n in Counter(x[2] for x in tr).most_common():
        yaz(f"    {kyn:<12} {n}")
    if not tr or not dg:
        yaz("HATA: egitim ya da dogrulama bos."); sys.exit(1)
    egitim_sayi = {sinif_keys[i]: n for i, n in Counter(x[1] for x in tr).items()}

    # ---- dogrulama referansi: kat5 ayni 9.21 kumesinde (kategori bazinda)
    g_dg = np.array([sinif_keys[x[1]] for x in dg], dtype=object)
    iB = np.where(g_dg != A)[0]          # kategori raporu B hucrelerinde
    yaz("\nDogrulama (9.21), kategori bazinda isabet/yakalama:")
    if KAT5.exists():
        net5, ck5 = model_yukle(KAT5)
        k5 = [anahtar(s_) for s_ in ck5["sinif_ad"]]
        P5 = ayarla(olasilik(net5, [x[0] for x in dg], 384, len(k5)), k5,
                    ck5.get("egitim_sayi", {}), ck5.get("denge_us", 0.3))
        rapor(yaz, "kat5 (su an)", hizala(P5, k5, sinif_keys)[iB], sinif_keys, g_dg[iB])

    # ---- model: kat5 govdesi, yeni bas (sinif listesi degismis olabilir)
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, len(sinif_keys))
    kaynak_govde = KAT5 if KAT5.exists() else MODEL3
    net.load_state_dict({k: v for k, v in torch.load(kaynak_govde, map_location="cpu")
                         ["model"].items() if not k.startswith("fc.")}, strict=False)
    yaz(f"Baslangic: {kaynak_govde.parent.name} govdesi, yeni siniflandirma basi")

    bas_id = {id(p) for p in net.fc.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in bas_id], "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS}], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    tr_y = [x[1] for x in tr]
    sinif_say = Counter(tr_y)
    ornekleme = [1.0 / (sinif_say[y] ** DENGE_US) for y in tr_y]
    agirlik = torch.tensor([AGIRLIK[x[2]] * HAT_AGIRLIK.get(hat(x[0]), 1.0) for x in tr])
    yl = DataLoader(Kume([x[0] for x in tr], tr_y, BOYUT, egitim=True), batch_size=BATCH,
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
        yaz("  dogrulama:")
        Pd = olasilik(net, [x[0] for x in dg], BOYUT, len(sinif_keys))
        dog = float((Pd.argmax(1) == np.array([x[1] for x in dg])).mean())
        yaz(f"  DOGRULAMA dogruluk %{100*dog:.1f}")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son)
        ck_tur = {"model": net.state_dict(), "sinif_ad": sinif_ad,
                  "girdi_boyut": BOYUT, "tur": tur + 1, "dog": dog,
                  "egitim_sayi": egitim_sayi, "denge_us": DENGE_US}
        torch.save(ck_tur, CIKTI / f"tur{tur+1}.pt")
        if dog > en_iyi:
            en_iyi = dog
            torch.save(ck_tur, iyi)
            yaz("  >>> EN IYI (dogrulamaya gore)")
        rapor(yaz, f"kat6 tur {tur+1}", ayarla(Pd, sinif_keys, egitim_sayi, DENGE_US)[iB],
              sinif_keys, g_dg[iB])
        durum.write_text(json.dumps({"tur": tur + 1, "egitim": len(tr), "en_iyi": en_iyi}),
                         encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama %{100*en_iyi:.1f}\nModel: {iyi}\n{C}")


if __name__ == "__main__":
    main()
