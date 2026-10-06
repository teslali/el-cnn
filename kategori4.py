#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KATEGORI MODELI v4 (384 px) - yeniden egitim

HEDEF (altin kume olcumleri)
  duzensiz cizik klasoru isabeti %51; kirik grid yakalama %20;
  3A hattinda kategori uyumu %65; siyah kenar yakalama %64.
  Ortak neden: kucuk detay (kirik grid, soluk cizik, kucuk nokta) 224 px'te
  kayboluyor. Bu surum 384 px'te calisir.

VERI (guvenilirlik sirasiyla)
  inceleme     : ~/Desktop/yeni_duzeltmeler (senin kararin)          x3
  dogrulanmis  : duzeltme_listesi.csv (automation-test, kontrol edildi) x1
  anlasan      : ayrim_gruplar.csv, operator = model (sinif basi sinir) x1
  op_plaka     : yeni veride operatorun baski plakasi dedikleri        x1
  2B           : eski veri, sinif basi sinir; KIRIK GRID HARIC
                 (operator kirik grid etiketi altin kumede %24 isabetli)
  3A hatti x1.5 agirlik; guclu parlaklik/kontrast cesitlemesi.
  Haric: 9.22 (altin gun), OverTime.  Dogrulama: 9.21.

SINAV: altin kume, HER TURDAN SONRA. Basta mevcut model3 (224 px) ayni
  sinavdan gecer; karsilastirma birebir. Model secimi dogrulamaya gore yapilir
  (altin kume sinavdir, secim icin kullanilmaz).

Sure: ~9-10 saat. Kesilirse ayni komutla devam eder.
  caffeinate -i python kategori4.py ~/Desktop/fabrika-fotolar/2B
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
TUR_SAYISI = 4
TUR_ORNEK = 12000
LR_GOVDE, LR_BAS = 1e-4, 1e-3
MIN_ORNEK = 150
ANLASAN_SINIR = 2000
ESKI_SINIR = 800
AGIRLIK = {"inceleme": 3.0, "dogrulanmis": 1.0, "anlasan": 1.0,
           "op_plaka": 1.0, "2B": 1.0}
HAT_AGIRLIK = {"3A": 1.5}
ESKI_HARIC = {"断栅"}

GRUPLAR = Path("ayrim_gruplar.csv")
ENVANTER = Path("yeni_envanter.csv")
MANIFEST = Path("duzeltme_listesi.csv")
DUZELTMELER = Path.home() / "Desktop" / "yeni_duzeltmeler"
ALTIN_KL = Path("altin_kume")
ALTIN_ONAY = Path("cikti_altin")
ALTIN_HATA = Path.home() / "Desktop" / "altin_hatalar"
ESKI_MODEL = Path("model3/en_iyi_model.pt")
CIKTI = Path("model_kat4")

HARIC_GUN = {"20260922"}
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


def anahtar(ad):
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
    def __init__(self, yollar, etiket, don):
        self.yollar, self.etiket, self.don = yollar, etiket, don

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                x = self.don(im.convert("RGB"))
        except Exception:
            x = torch.zeros(3, self.don.transforms[0].size[0], self.don.transforms[0].size[0])
        return x, self.etiket[i], i


def don(boyut, egitim=False):
    t = [transforms.Resize((boyut, boyut))]
    if egitim:
        t += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
              transforms.ColorJitter(brightness=0.25, contrast=0.25)]
    t += [transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


def olasilik(net, yollar, boyut, n_sinif):
    yl = DataLoader(Kume(yollar, [0] * len(yollar), don(boyut)),
                    batch_size=24 if boyut <= 224 else BATCH,
                    shuffle=False, num_workers=ISCI)
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


def altin_etiketler():
    etiket = {}
    for kok in (ALTIN_ONAY, ALTIN_HATA):
        if not kok.is_dir():
            continue
        for d in kok.iterdir():
            if d.is_dir() and "emin" not in normal(d.name):
                k = anahtar(d.name)
                if k:
                    for f in d.iterdir():
                        if f.suffix.lower() in UZANTILAR:
                            etiket[f.name] = k
    return etiket


def sinav_raporu(yaz, baslik, P, sinif_ad, g_keys):
    """Altin kumede B hucreleri: A haric en yuksek sinif vs altin etiket."""
    keys = [anahtar(s) for s in sinif_ad]
    a_i = keys.index(A) if A in keys else None
    P = P.copy()
    if a_i is not None:
        P[:, a_i] = -1
    tahmin = np.array([keys[i] for i in P.argmax(1)], dtype=object)
    g = np.array(g_keys, dtype=object)
    dogru = (tahmin == g).mean()
    cz_sec = tahmin == "不规则"
    cz_isb = (g[cz_sec] == "不规则").mean() if cz_sec.any() else float("nan")
    cz_yak = (tahmin[g == "不规则"] == "不规则").mean() if (g == "不规则").any() else float("nan")
    kg_yak = (tahmin[g == "断栅"] == "断栅").mean() if (g == "断栅").any() else float("nan")
    sk_yak = (tahmin[g == "黑边"] == "黑边").mean() if (g == "黑边").any() else float("nan")
    yaz(f"  {baslik:<22} dogruluk %{100*dogru:.1f} | cizik isabet %{100*cz_isb:.0f} "
        f"yakalama %{100*cz_yak:.0f} | kirik grid yak. %{100*kg_yak:.0f} | "
        f"siyah kenar yak. %{100*sk_yak:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak", help="2B klasoru")
    a = ap.parse_args()
    kay = Path(a.kaynak).expanduser().resolve()
    for p in (GRUPLAR, ENVANTER):
        if not p.exists():
            print(f"HATA: {p} yok."); sys.exit(1)
    CIKTI.mkdir(exist_ok=True)
    gl = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gl, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nKATEGORI MODELI v4 (384 px) - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    ornek = {}   # orijinal ad -> (yol, anahtar, kaynak)
    oncelik = {"inceleme": 5, "dogrulanmis": 4, "op_plaka": 3, "anlasan": 2, "2B": 1}

    def ekle(p, k, kaynak):
        p = Path(p)
        if not k or bozuk(p) or gun(p) in HARIC_GUN or not p.exists():
            return
        ad = ONEK.sub("", p.name)
        if ad in ornek and oncelik[ornek[ad][2]] >= oncelik[kaynak]:
            return
        ornek[ad] = (p, k, kaynak)

    if DUZELTMELER.is_dir():
        for d in DUZELTMELER.iterdir():
            if d.is_dir() and "emin" not in normal(d.name):
                k = anahtar(d.name)
                for f in d.iterdir():
                    if f.suffix.lower() in UZANTILAR:
                        ekle(f, k, "inceleme")
    if MANIFEST.exists():
        for r in csv.DictReader(open(MANIFEST, encoding="utf-8-sig")):
            ekle(r["yol"], anahtar(r["etiket"]), "dogrulanmis")
    R = list(csv.DictReader(open(GRUPLAR, encoding="utf-8-sig")))
    for r in R:
        if r["operator"] == "压板":
            ekle(r["yol"], "压板", "op_plaka")
    anl = defaultdict(list)
    for r in R:
        if r["grup"] in ("anlasan_A", "anlasan_B") and r["operator"]:
            anl[r["operator"]].append(r["yol"])
    for k, yollar in anl.items():
        for i in rng.permutation(len(yollar))[:ANLASAN_SINIR]:
            ekle(yollar[i], k, "anlasan")
    for d in sorted([x for x in kay.iterdir() if x.is_dir()]):
        k = anahtar(d.name)
        if not k or k in ESKI_HARIC:
            continue
        fs = [f for f in d.iterdir() if f.suffix.lower() in UZANTILAR]
        for i in rng.permutation(len(fs))[:ESKI_SINIR]:
            ekle(fs[i], k, "2B")

    say = Counter(v[1] for v in ornek.values())
    sinif_keys = sorted(k for k, n in say.items() if n >= MIN_ORNEK)
    disari = {k: n for k, n in say.items() if n < MIN_ORNEK}
    # klasor adi: 2B'deki ad varsa o, yoksa envanterdeki, yoksa anahtar+ad
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
    yaz(f"Siniflar ({len(sinif_keys)}):")
    for k in sinif_keys:
        yaz(f"    {say[k]:>6}  {AD.get(k, k)}")
    if disari:
        yaz(f"  <{MIN_ORNEK} ornek, disarida: " +
            ", ".join(f"{AD.get(k, k)} {n}" for k, n in disari.items()))
    yaz(f"Egitim {len(tr)}   Dogrulama (9.21) {len(dg)}")
    for kyn, n in Counter(x[2] for x in tr).most_common():
        yaz(f"    {kyn:<12} {n}")
    if not tr or not dg:
        yaz("HATA: egitim ya da dogrulama bos."); sys.exit(1)

    # ---- sinav kumesi: altin B hucreleri, sinif listesinde olanlar
    ae = altin_etiketler()
    ortak_keys = set(idx_of)
    if ESKI_MODEL.exists():
        ortak_keys &= {anahtar(s) for s in torch.load(ESKI_MODEL, map_location="cpu")["sinif_ad"]}
    sinav = [(ALTIN_KL / d, k) for d, k in ae.items()
             if k != A and k in ortak_keys and (ALTIN_KL / d).exists()]
    yaz(f"Sinav: altin kumede {len(sinav)} B hucresi "
        f"(adil karsilastirma icin sadece iki modelde de olan siniflar)")
    if sinav and ESKI_MODEL.exists():
        ck0 = torch.load(ESKI_MODEL, map_location="cpu")
        eski = models.resnet18(weights=None)
        eski.fc = nn.Linear(512, len(ck0["sinif_ad"]))
        eski.load_state_dict(ck0["model"])
        yaz("  mevcut model3 (224 px) sinavda:")
        P0 = olasilik(eski, [s[0] for s in sinav], 224, len(ck0["sinif_ad"]))
        sinav_raporu(yaz, "model3 (mevcut)", P0, ck0["sinif_ad"], [s[1] for s in sinav])

    # ---- model: model3 govdesi, yeni bas
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, len(sinif_keys))
    if ESKI_MODEL.exists():
        net.load_state_dict({k: v for k, v in torch.load(ESKI_MODEL, map_location="cpu")
                             ["model"].items() if not k.startswith("fc.")}, strict=False)
        yaz("Baslangic: model3 govdesi, yeni siniflandirma basi")

    bas_id = {id(p) for p in net.fc.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in bas_id], "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS}], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    tr_y = [x[1] for x in tr]
    sinif_say = Counter(tr_y)
    ornekleme = [1.0 / math.sqrt(sinif_say[y]) for y in tr_y]      # dengeli ama asiri degil
    agirlik = torch.tensor([AGIRLIK[x[2]] * HAT_AGIRLIK.get(hat(x[0]), 1.0) for x in tr])
    yl = DataLoader(Kume([x[0] for x in tr], tr_y, don(BOYUT, True)), batch_size=BATCH,
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
        if dog > en_iyi:
            en_iyi = dog
            torch.save({"model": net.state_dict(), "sinif_ad": sinif_ad,
                        "girdi_boyut": BOYUT, "tur": tur + 1, "dog": dog}, iyi)
            yaz("  >>> EN IYI (dogrulamaya gore)")
        if sinav:
            yaz("  sinav (altin kume):")
            Ps = olasilik(net, [s[0] for s in sinav], BOYUT, len(sinif_keys))
            sinav_raporu(yaz, f"yeni model tur {tur+1}", Ps, sinif_ad, [s[1] for s in sinav])
        durum.write_text(json.dumps({"tur": tur + 1, "egitim": len(tr), "en_iyi": en_iyi}),
                         encoding="utf-8")

    yaz(f"\n{C}\nBITTI - en iyi dogrulama %{100*en_iyi:.1f}\nModel: {iyi}\n{C}")


if __name__ == "__main__":
    main()
