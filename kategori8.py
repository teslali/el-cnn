#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KATEGORI MODELI v8: YENI CIZIK TANIMLARI (kat7'nin uzerine ince ayar)

Ogretilecekler:
  - "yeni tanim": eskiden kirik grid denen, artik duzensiz cizik sayilan tip
    (model kirik gridle karistiriyor)
  - "yeni tur" : ilk kez gorulen cizik turu (model baski kaymasiyla karistiriyor)
kat7'nin ogrendigi kirlenme tanimi korunur.

Veri:
  - yeni tanim / yeni tur ornekleri (yeni_duzeltmeler + 9.30 altin) -> duzensiz cizik, x5
  - Sinirlar: altin gunlerde onayladigin cizik, kirik grid, baski kaymasi,
    kirlenme, hava akisi kararlarin x2
  - yeni_duzeltmeler/kirlenme yeni tanim x5 (kat7'den, unutmasin)
  - genel veri (kat5'in kaynaklari)
  Haric: 9.25 ve 10.01 (sinav). Yeni tiplerin rastgele 1/4'u da sinav icin ayrilir.

Karar kurali (sinavda karsilastirilir):
  SU AN : model3 + kat5; kat7 kirlenme derse kirlenme
  YENI  : model3 + kat5; kat8 kirlenme derse kirlenme; ayrica hem birlesim hem
          kat8 karisan gruptaysa (cizik, kirik grid, baski kaymasi, kirlenme,
          hava akisi) son sozu kat8 soyler

Sure: ~3 saat. Kesilirse ayni komutla devam eder.
  caffeinate -i python kategori8.py ~/Desktop/fabrika-fotolar/2B
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
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})"); sys.exit(1)

import kategori6 as k6

TUR_SAYISI = 2
TUR_ORNEK = 10000
LR_GOVDE, LR_BAS = 2e-5, 2e-4
DENGE_US = 0.3
GRUP = ["不规则", "断栅", "偏移", "脏污", "气流"]
CIZIK, KIR = "不规则", "脏污"
AGIRLIK = {"yeni_tip": 5.0, "yeni_kirlenme": 5.0, "altin_odak": 2.0, "inceleme": 3.0,
           "dogrulanmis": 1.0, "op_ozel": 1.0, "anlasan": 1.0, "2B": 1.0}
ONCELIK = {"yeni_tip": 9, "yeni_kirlenme": 8, "altin_odak": 7, "inceleme": 5,
           "dogrulanmis": 4, "op_ozel": 3, "anlasan": 2, "2B": 1}

KAT5 = Path("model_kat5/en_iyi_model.pt")
KAT7 = Path("model_kat7/tur2.pt")
CIKTI = Path("model_kat8")
D = Path.home() / "Desktop"
YD = D / "yeni_duzeltmeler"
YENI_TIP_KLASORLERI = [YD / "duzensiz cizik yeni tanim", YD / "duzensiz cizik yeni tur",
                       D / "altin8_hatalar" / "duzensiz cizik yeni tanim",
                       D / "altin8_hatalar" / "duzensiz cizik yeni tur"]
YENI_KIRLENME = YD / "kirlenme yeni tanim"
ALTIN_EGITIM = [
    (Path("cikti_altin"), D / "altin_hatalar"),
    (Path("cikti_altin_924"), D / "altin2_hatalar"),
    (Path("cikti_altin_923"), D / "altin3_hatalar"),
    (Path("cikti_altin_926"), D / "altin5_hatalar"),
    (Path("cikti_altin_927"), D / "altin6_hatalar"),
    (Path("cikti_altin_yeni"), D / "altin8_hatalar"),      # 9.30
]
SINAVLAR = [  # (ad, onay, hatalar, gorsel klasoru)
    ("9.25 geri kalani", Path("cikti_altin_925k"), D / "altin7_hatalar", Path("altin_kume7")),
    ("10.01 (M)", Path("cikti_altin_m"), D / "altin9_hatalar", Path("altin_kume9")),
]
HARIC_GUN = {"20260925", "20261001"}
C = "=" * 78
torch.manual_seed(0)
rng = np.random.RandomState(0)


def etiketler(onay, hatalar):
    et = {}
    for kok in (onay, hatalar):
        if kok.is_dir():
            for d in kok.iterdir():
                if not d.is_dir() or d.name.startswith("_") or "emin" in k6.normal(d.name):
                    continue
                k = k6.anahtar(d.name)
                if k:
                    for f in d.iterdir():
                        if f.suffix.lower() in k6.UZANTILAR:
                            et[f] = k
    return et


def karar(P3, P5, Pk, keys, grup_kurali):
    """birlesim + kat7/kat8 kurali -> her satir icin bir kategori indeksi"""
    taban = (P3 + P5) / 2
    a_i = keys.index(k6.A)
    t = taban.copy(); t[:, a_i] = -1
    tb = t.argmax(1)
    q = Pk.copy(); q[:, a_i] = -1
    kq = q.argmax(1)
    kir_i = keys.index(KIR)
    g = {keys.index(k) for k in GRUP if k in keys}
    out = tb.copy()
    out[kq == kir_i] = kir_i
    if grup_kurali:
        m = np.array([(tb[i] in g) and (kq[i] in g) for i in range(len(tb))])
        out[m] = kq[m]
    return out


def rapor_satir(yaz, baslik, tahmin, g, keys, tekrar):
    t = np.repeat(np.array([keys[i] for i in tahmin], dtype=object), tekrar)
    g = np.repeat(g, tekrar)
    s = f"  {baslik:<20} genel %{100*(t==g).mean():>4.1f} |"
    for k in GRUP:
        sec, ger = t == k, g == k
        isb = (g[sec] == k).mean() if sec.any() else float("nan")
        yak = (t[ger] == k).mean() if ger.any() else float("nan")
        s += f" {k6.AD[k][:9]} {100*isb:>3.0f}/{100*yak:<3.0f}|"
    yaz(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak", help="2B klasoru")
    a = ap.parse_args()
    kay = Path(a.kaynak).expanduser().resolve()
    for p in (KAT5, KAT7, k6.GRUPLAR):
        if not p.exists():
            print(f"HATA: {p} yok."); sys.exit(1)
    CIKTI.mkdir(exist_ok=True)
    gl = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gl, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nKATEGORI v8 - yeni cizik tanimlari - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")
    ck5 = torch.load(KAT5, map_location="cpu")
    ck7 = torch.load(KAT7, map_location="cpu")
    sinif_ad = ck7["sinif_ad"]
    keys = [k6.anahtar(s) for s in sinif_ad]
    idx_of = {k: i for i, k in enumerate(keys)}

    # ---- yeni tip ornekleri: 1/4'u sinav
    yeni = []
    for kl in YENI_TIP_KLASORLERI:
        if kl.is_dir():
            yeni += [f for f in kl.iterdir() if f.suffix.lower() in k6.UZANTILAR]
    adlar = sorted({k6.ONEK.sub("", f.name) for f in yeni})
    perm = rng.permutation(len(adlar))
    sinav_yeni_ad = {adlar[i] for i in perm[:len(adlar) // 4]}
    yeni_egitim = [f for f in yeni if k6.ONEK.sub("", f.name) not in sinav_yeni_ad]
    yeni_sinav = {}
    for f in yeni:
        ad = k6.ONEK.sub("", f.name)
        if ad in sinav_yeni_ad and ad not in yeni_sinav:
            yeni_sinav[ad] = f
    yaz(f"Yeni tip ornekleri: {len(adlar)} (egitim {len(adlar)-len(sinav_yeni_ad)}, "
        f"sinav {len(sinav_yeni_ad)})")
    if len(adlar) < 40:
        yaz("HATA: yeni tip ornegi cok az."); sys.exit(1)

    sinav_adlari = set(sinav_yeni_ad)
    for _, _, _, kl in SINAVLAR:
        if kl.is_dir():
            sinav_adlari |= {f.name for f in kl.iterdir()}

    ornek = {}

    def ekle(p, k, kaynak):
        p = Path(p)
        if not k or k not in idx_of or k6.bozuk(p) or not p.exists():
            return
        ad = k6.ONEK.sub("", p.name)
        if k6.gun(ad) in HARIC_GUN or ad in sinav_adlari:
            return
        if ad in ornek and ONCELIK[ornek[ad][2]] >= ONCELIK[kaynak]:
            return
        ornek[ad] = (p, k, kaynak)

    for f in yeni_egitim:
        ekle(f, CIZIK, "yeni_tip")
    if YENI_KIRLENME.is_dir():
        for f in YENI_KIRLENME.iterdir():
            if f.suffix.lower() in k6.UZANTILAR:
                ekle(f, KIR, "yeni_kirlenme")
    n_odak = Counter()
    for onay, hatalar in ALTIN_EGITIM:
        for f, k in etiketler(onay, hatalar).items():
            if k in GRUP:
                ekle(f, k, "altin_odak"); n_odak[k] += 1
    for kl in k6.INCELEMELER:
        if kl.is_dir():
            for d in kl.iterdir():
                if d.is_dir() and "emin" not in k6.normal(d.name) and \
                        d not in YENI_TIP_KLASORLERI and d != YENI_KIRLENME:
                    k = k6.anahtar(d.name)
                    for f in d.iterdir():
                        if f.suffix.lower() in k6.UZANTILAR:
                            ekle(f, k, "inceleme")
    if k6.MANIFEST.exists():
        for r in csv.DictReader(open(k6.MANIFEST, encoding="utf-8-sig")):
            ekle(r["yol"], k6.anahtar(r["etiket"]), "dogrulanmis")
    R = list(csv.DictReader(open(k6.GRUPLAR, encoding="utf-8-sig")))
    for r in R:
        if r["operator"] in k6.OP_OZEL:
            ekle(r["yol"], r["operator"], "op_ozel")
    anl = defaultdict(list)
    for r in R:
        if r["grup"] in ("anlasan_A", "anlasan_B") and r["operator"] and \
                r["operator"] not in k6.ANLASAN_HARIC:
            anl[r["operator"]].append(r["yol"])
    for k, ys in anl.items():
        for i in rng.permutation(len(ys))[:k6.ANLASAN_SINIR]:
            ekle(ys[i], k, "anlasan")
    for d in sorted(x for x in kay.iterdir() if x.is_dir()):
        k = k6.anahtar(d.name)
        if not k or k in k6.ESKI_HARIC:
            continue
        fs = [f for f in d.iterdir() if f.suffix.lower() in k6.UZANTILAR]
        for i in rng.permutation(len(fs))[:k6.ESKI_SINIR]:
            ekle(fs[i], k, "2B")

    tr = [(v[0], idx_of[v[1]], v[2]) for v in ornek.values()]
    yaz(f"Altin gun sinir kararlari: " + ", ".join(f"{k6.AD[k]} {n}" for k, n in n_odak.items()))
    yaz(f"Egitim: {len(tr)}")
    for kyn, n in Counter(x[2] for x in tr).most_common():
        yaz(f"    {kyn:<14} {n}")
    egitim_sayi = {keys[i]: n for i, n in Counter(x[1] for x in tr).items()}

    # ---- sinavlar: referans tahminleri bir kez
    net3, ck3 = k6.model_yukle(k6.MODEL3)
    k3 = [k6.anahtar(s) for s in ck3["sinif_ad"]]
    net5, _ = k6.model_yukle(KAT5)
    net7, _ = k6.model_yukle(KAT7)
    sinav = []
    for ad, onay, hatalar, kl in SINAVLAR:
        et = {f.name: k for f, k in etiketler(onay, hatalar).items()}
        kayit = {r["dosya"]: r for r in csv.DictReader(open(onay / "_kayit.csv", encoding="utf-8-sig"))}
        ds = [d for d in et if (kl / d).exists() and d in kayit and et[d] in idx_of
              and et[d] != k6.A and et[d] not in {"隐裂", "压板"}]
        w, ok_ = 1, None
        oj = onay / "_ornekleme.json"
        if oj.exists():
            o = json.loads(oj.read_text(encoding="utf-8"))
            w, ok_ = max(1, int(round(o["agirlik"]))), k6.anahtar(o["klasor"])
        tekrar = np.array([w if k6.anahtar(kayit[d]["klasor"]) == ok_ else 1 for d in ds])
        yollar = [kl / d for d in ds]
        yaz(f"\nSinav {ad}: {len(ds)} B hucresi - referans hesaplaniyor")
        P3 = k6.hizala(k6.olasilik(net3, yollar, 224, len(k3)), k3, keys)
        P5 = k6.ayarla(k6.olasilik(net5, yollar, 384, len(keys)), keys,
                       ck5.get("egitim_sayi", k6.KAT4_EGITIM), ck5.get("denge_us", 0.5))
        P7 = k6.ayarla(k6.olasilik(net7, yollar, 384, len(keys)), keys,
                       ck7["egitim_sayi"], ck7["denge_us"])
        g = np.array([et[d] for d in ds], dtype=object)
        sinav.append((ad, yollar, g, tekrar, P3, P5, P7))
    yy = list(yeni_sinav.values())
    yaz(f"\nSinav yeni tipler: {len(yy)} gorsel - referans hesaplaniyor")
    Y3 = k6.hizala(k6.olasilik(net3, yy, 224, len(k3)), k3, keys)
    Y5 = k6.ayarla(k6.olasilik(net5, yy, 384, len(keys)), keys,
                   ck5.get("egitim_sayi", k6.KAT4_EGITIM), ck5.get("denge_us", 0.5))
    Y7 = k6.ayarla(k6.olasilik(net7, yy, 384, len(keys)), keys, ck7["egitim_sayi"], ck7["denge_us"])
    cz_i = idx_of[CIZIK]

    def sinav_yaz(baslik, Pk_list, Yk):
        for (ad, _, g, tekrar, P3, P5, P7), Pk in zip(sinav, Pk_list):
            yaz(f"  [{ad}]")
            rapor_satir(yaz, "SU AN", karar(P3, P5, P7, keys, False), g, keys, tekrar)
            if Pk is not None:
                rapor_satir(yaz, f"{baslik} kural", karar(P3, P5, Pk, keys, True), g, keys, tekrar)
        su_an = (karar(Y3, Y5, Y7, keys, False) == cz_i).mean()
        s = f"  [yeni tipler] cizik olarak yakalanan: SU AN %{100*su_an:.0f}"
        if Yk is not None:
            s += f" | {baslik} kural %{100*(karar(Y3, Y5, Yk, keys, True) == cz_i).mean():.0f}"
        yaz(s)

    sinav_yaz("", [None] * len(sinav), None)

    # ---- egitim: kat7'den
    net = k6.models.resnet18(weights=None)
    net.fc = nn.Linear(512, len(keys))
    net.load_state_dict(ck7["model"])
    bas_id = {id(p) for p in net.fc.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in bas_id], "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS}], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
    tr_y = [x[1] for x in tr]
    ss = Counter(tr_y)
    gidx = {idx_of[k] for k in GRUP}
    ornekleme = [1.0 / (ss[y] ** DENGE_US) * (2.0 if y in gidx else 1.0) for y in tr_y]
    agirlik = torch.tensor([AGIRLIK[x[2]] for x in tr], dtype=torch.float)
    yl = DataLoader(k6.Kume([x[0] for x in tr], tr_y, 384, egitim=True), batch_size=12,
                    sampler=WeightedRandomSampler(ornekleme, min(len(tr), TUR_ORNEK), replacement=True),
                    num_workers=2)
    adim = max(len(yl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s_: 0.5 * (1 + math.cos(math.pi * min(s_ / (TUR_SAYISI * adim), 1.0))))

    durum, son = CIKTI / "durum.json", CIKTI / "son_model.pt"
    bas_tur = 0
    if durum.exists() and son.exists():
        dd = json.loads(durum.read_text(encoding="utf-8"))
        if dd.get("egitim") == len(tr):
            c2 = torch.load(son, map_location="cpu")
            net.load_state_dict(c2["model"]); opt.load_state_dict(c2["opt"])
            sched.load_state_dict(c2["sched"]); bas_tur = dd["tur"]
            yaz(f"Kaldigi yerden devam: {bas_tur}. tur")

    yaz(f"\n{C}\nINCE AYAR ({TUR_SAYISI} tur)\n{C}")
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
        torch.save({"model": net.state_dict(), "sinif_ad": sinif_ad, "girdi_boyut": 384,
                    "tur": tur + 1, "egitim_sayi": egitim_sayi, "denge_us": DENGE_US},
                   CIKTI / f"tur{tur+1}.pt")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son)
        P8 = [k6.ayarla(k6.olasilik(net, s[1], 384, len(keys)), keys, egitim_sayi, DENGE_US)
              for s in sinav]
        Y8 = k6.ayarla(k6.olasilik(net, yy, 384, len(keys)), keys, egitim_sayi, DENGE_US)
        sinav_yaz(f"kat8 t{tur+1}", P8, Y8)
        durum.write_text(json.dumps({"tur": tur + 1, "egitim": len(tr)}), encoding="utf-8")

    yaz(f"\n{C}\nBITTI - her tur: {CIKTI}/tur1.pt, tur2.pt\n{C}")


if __name__ == "__main__":
    main()
