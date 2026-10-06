#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KATEGORI MODELI v7: KIRLENME ODAKLI INCE AYAR (kat5'ten)

Sorun: yeni tanimla kirlenme sayilan bir tip, model tarafindan duzensiz cizik
saniliyor (9.25 geri kalaninda kirlenme yakalamasi %34). Eski veride bu tip
hava akisi / cizik etiketli; kat6 altin gun ornekleriyle bile ogrenemedi.

v7:
  - Baslangic kat5'in KENDISI (sinif listesi dahil), dusuk ogrenme hizi, 2 tur
  - yeni_duzeltmeler/kirlenme yeni tanim (benzer arama + onceki) x5 agirlik;
    bunlar eski etiketlerin ONUNE gecer
  - Altin gunlerdeki (9.22-9.27, 9.25 HARIC) kirlenme / cizik / hava akisi
    kararlarin x2 (siniri ogretmek icin)
  - kat5'in genel veri kaynaklari (unutmasin diye)
  - 9.25 tamamen egitim disi: SINAV 9.25'in dokunulmamis kismi (altin_kume7)
  Her turdan sonra sinav, su anki sistemle (m3+kat5) yan yana.

Sure: ~3 saat. Kesilirse ayni komutla devam eder.
  caffeinate -i python kategori7.py ~/Desktop/fabrika-fotolar/2B
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

import kategori6 as k6        # yardimci fonksiyonlar ve veri kaynaklari

TUR_SAYISI = 2
TUR_ORNEK = 10000
LR_GOVDE, LR_BAS = 2e-5, 2e-4
DENGE_US = 0.3
AGIRLIK_YENI_KIRLENME = 5.0
AGIRLIK_ALTIN_ODAK = 2.0
ODAK = {"脏污", "不规则", "气流"}

KAT5 = Path("model_kat5/en_iyi_model.pt")
CIKTI = Path("model_kat7")
D = Path.home() / "Desktop"
YENI_KIRLENME = D / "yeni_duzeltmeler" / "kirlenme yeni tanim"
ALTIN_EGITIM = [  # 9.25 (altin4) HARIC
    (Path("cikti_altin"), D / "altin_hatalar"),
    (Path("cikti_altin_924"), D / "altin2_hatalar"),
    (Path("cikti_altin_923"), D / "altin3_hatalar"),
    (Path("cikti_altin_926"), D / "altin5_hatalar"),
    (Path("cikti_altin_927"), D / "altin6_hatalar"),
]
SINAV_ONAY = Path("cikti_altin_925k")
SINAV_HATA = D / "altin7_hatalar"
SINAV_KL = Path("altin_kume7")
HARIC_GUN = {"20260925"}
DOG_GUN = {"20260921"}
A_ESIK, C_ESIK_ADAY, IKINCI_ESIK = 0.40, 0.20, 0.20
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak", help="2B klasoru")
    a = ap.parse_args()
    kay = Path(a.kaynak).expanduser().resolve()
    for p in (KAT5, k6.GRUPLAR, SINAV_ONAY / "_kayit.csv"):
        if not p.exists():
            print(f"HATA: {p} yok."); sys.exit(1)
    CIKTI.mkdir(exist_ok=True)
    gl = CIKTI / "gunluk.txt"

    def yaz(s=""):
        print(s, flush=True)
        with open(gl, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    yaz(f"\n{C}\nKATEGORI v7 - kirlenme odakli ince ayar - {time.strftime('%Y-%m-%d %H:%M')}\n{C}")

    ck5 = torch.load(KAT5, map_location="cpu")
    sinif_ad = ck5["sinif_ad"]
    sinif_keys = [k6.anahtar(s) for s in sinif_ad]
    idx_of = {k: i for i, k in enumerate(sinif_keys)}
    sinav_adlar = {f.name for f in SINAV_KL.iterdir()} if SINAV_KL.is_dir() else set()

    ornek = {}
    oncelik = {"yeni_kirlenme": 8, "altin_odak": 7, "inceleme": 5, "dogrulanmis": 4,
               "op_ozel": 3, "anlasan": 2, "2B": 1}
    agirlik_kaynak = {"yeni_kirlenme": AGIRLIK_YENI_KIRLENME, "altin_odak": AGIRLIK_ALTIN_ODAK,
                      "inceleme": 3.0, "dogrulanmis": 1.0, "op_ozel": 1.0, "anlasan": 1.0, "2B": 1.0}

    def ekle(p, k, kaynak):
        p = Path(p)
        if not k or k not in idx_of or k6.bozuk(p) or not p.exists():
            return
        ad = k6.ONEK.sub("", p.name)
        if k6.gun(ad) in HARIC_GUN or ad in sinav_adlar:
            return
        if ad in ornek and oncelik[ornek[ad][2]] >= oncelik[kaynak]:
            return
        ornek[ad] = (p, k, kaynak)

    # 1) yeni tanimli kirlenme (en yuksek oncelik)
    n_yeni = 0
    if YENI_KIRLENME.is_dir():
        for f in YENI_KIRLENME.iterdir():
            if f.suffix.lower() in k6.UZANTILAR:
                ekle(f, "脏污", "yeni_kirlenme"); n_yeni += 1
    # 2) altin gunlerde odak kategorilerdeki kararlarin
    n_altin = Counter()
    for onay, hatalar in ALTIN_EGITIM:
        for f, k in etiketler(onay, hatalar).items():
            if k in ODAK:
                ekle(f, k, "altin_odak"); n_altin[k] += 1
    # 3) incelemeler
    for kl in k6.INCELEMELER:
        if kl.is_dir():
            for d in kl.iterdir():
                if d.is_dir() and "emin" not in k6.normal(d.name) and d != YENI_KIRLENME:
                    k = k6.anahtar(d.name)
                    for f in d.iterdir():
                        if f.suffix.lower() in k6.UZANTILAR:
                            ekle(f, k, "inceleme")
    # 4) genel veri (kat5'in kaynaklari)
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

    kayit = [(v[0], idx_of[v[1]], v[2]) for v in ornek.values()]
    tr = [x for x in kayit if k6.gun(x[0].name) not in DOG_GUN]
    yaz(f"Yeni tanimli kirlenme (klasorun): {n_yeni}")
    yaz(f"Altin gun odak kararlari: " + ", ".join(f"{k6.AD.get(k, k)} {n}" for k, n in n_altin.items()))
    yaz(f"Egitim: {len(tr)}")
    for kyn, n in Counter(x[2] for x in tr).most_common():
        yaz(f"    {kyn:<14} {n}")
    if n_yeni < 10:
        yaz("HATA: 'kirlenme yeni tanim' klasorunden cok az ornek okundu."); sys.exit(1)
    egitim_sayi = {sinif_keys[i]: n for i, n in Counter(x[1] for x in tr).items()}

    # ---- sinav: 9.25 geri kalani
    et = {f.name: k for f, k in etiketler(SINAV_ONAY, SINAV_HATA).items()}
    kayit_s = {r["dosya"]: r for r in csv.DictReader(open(SINAV_ONAY / "_kayit.csv", encoding="utf-8-sig"))}
    dosyalar = [d for d in et if (SINAV_KL / d).exists() and d in kayit_s]
    yollar = [SINAV_KL / d for d in dosyalar]
    g_all = np.array([et[d] for d in dosyalar], dtype=object)
    w = 1; orn_k = None
    oj = SINAV_ONAY / "_ornekleme.json"
    if oj.exists():
        o = json.loads(oj.read_text(encoding="utf-8"))
        w, orn_k = max(1, int(round(o["agirlik"]))), k6.anahtar(o["klasor"])
    tekrar = np.array([w if k6.anahtar(kayit_s[d]["klasor"]) == orn_k else 1 for d in dosyalar])
    B = np.array([(gi != k6.A) and (gi in idx_of) and (gi not in {"隐裂", "压板"}) for gi in g_all])
    aday = np.array([float(kayit_s[d]["A_olasiligi"]) >= A_ESIK and
                     float(kayit_s[d].get("C_dedektor") or 0) < C_ESIK_ADAY for d in dosyalar])
    gA = g_all == k6.A
    a_i = idx_of[k6.A]

    def gen(x, m=None):
        return np.repeat(x, tekrar if m is None else tekrar[m], axis=0)

    net3, ck3 = k6.model_yukle(k6.MODEL3)
    k3 = [k6.anahtar(s) for s in ck3["sinif_ad"]]
    yaz(f"\nSinav (9.25 geri kalani): {len(dosyalar)} gorsel")
    yaz("  referans hesaplaniyor...")
    P3 = k6.hizala(k6.olasilik(net3, yollar, 224, len(k3)), k3, sinif_keys)
    net5, _ = k6.model_yukle(KAT5)
    P5 = k6.ayarla(k6.olasilik(net5, yollar, 384, len(sinif_keys)), sinif_keys,
                   ck5.get("egitim_sayi", k6.KAT4_EGITIM), ck5.get("denge_us", 0.5))

    def sinav_yaz(baslik, Pk):
        Q = (P3 + Pk) / 2
        k6.rapor(yaz, baslik, gen(Q[B], B), sinif_keys, gen(g_all[B], B))
        sec = aday & (Q[:, a_i] >= IKINCI_ESIK)
        d_ = int(gen(sec & gA).sum()); n_ = int(gen(sec).sum()); t_ = int(gen(gA).sum())
        yaz(f"  {'':<20} A klasoru {n_}, isabet %{100*d_/max(n_,1):.0f}, A yakalama %{100*d_/max(t_,1):.0f}")

    sinav_yaz("SU AN (m3+kat5)", P5)

    # ---- model: kat5 aynen
    net = k6.models.resnet18(weights=None)
    net.fc = nn.Linear(512, len(sinif_keys))
    net.load_state_dict(ck5["model"])
    bas_id = {id(p) for p in net.fc.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in bas_id], "lr": LR_GOVDE},
        {"params": list(net.fc.parameters()), "lr": LR_BAS}], weight_decay=1e-4)
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
    tr_y = [x[1] for x in tr]
    ss = Counter(tr_y)
    ornekleme = [1.0 / (ss[y] ** DENGE_US) * (2.0 if sinif_keys[y] in ODAK else 1.0) for y in tr_y]
    agirlik = torch.tensor([agirlik_kaynak[x[2]] for x in tr], dtype=torch.float)
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
        ck = {"model": net.state_dict(), "sinif_ad": sinif_ad, "girdi_boyut": 384,
              "tur": tur + 1, "egitim_sayi": egitim_sayi, "denge_us": DENGE_US}
        torch.save(ck, CIKTI / f"tur{tur+1}.pt")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, son)
        P7 = k6.ayarla(k6.olasilik(net, yollar, 384, len(sinif_keys)),
                       sinif_keys, egitim_sayi, DENGE_US)
        sinav_yaz("SU AN (m3+kat5)", P5)
        sinav_yaz(f"m3+kat7 tur{tur+1}", P7)
        durum.write_text(json.dumps({"tur": tur + 1, "egitim": len(tr)}), encoding="utf-8")

    yaz(f"\n{C}\nBITTI - her tur: {CIKTI}/tur1.pt, tur2.pt\n{C}")


if __name__ == "__main__":
    main()
