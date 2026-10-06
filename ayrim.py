#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YENI VERI AYRIMI (9.15 - 9.21, altin gun 9.22 HARIC)

Her gorselin iki etiketi var: operatorun ve modelin. Karsilastirarak
gorselleri gruplara ayirir; 105 bin gorsele tek tek bakman gerekmez.

ASAMA 1 - tahmin  (uzun, ~3-5 saat, gece)
  python ayrim.py tahmin
  Her gorsel A/B modeli ve kategori modelinden gecer (ayni goruntu bir kez
  okunur, iki model birden calisir). Sonuclar ayrim_tahmin.csv'ye yazilir.
  Kesilirse ayni komutla KALDIGI YERDEN devam eder.

ASAMA 2 - ozet  (saniyeler)
  python ayrim.py ozet
  Gruplar:
    anlasan_A     operator A + model guclu A      -> A dogrulayicisinin temiz A'lari
    anlasan_B     operator = model, A olasiligi dusuk -> kategori egitimi icin temiz B
    A_anlasmazlik operator A / model B ya da tersi -> inceleme (oncelik 2)
    C_anlasmazlik operator C / model C degil ya da tersi -> inceleme (oncelik 1)
    sinir         ikisi de B ama A olasiligi orta -> kacmis A olabilir (oncelik 3)
    kat_anlasmazlik ikisi de B, kategori farkli  -> dusuk oncelik
  Ayrica vardiya ve hat bazinda operator-model uyumu.
  Inceleme icin ayrim_inceleme/ altina oncelikli ornekler KOPYALANIR.

Esikler altin kumeden: A olasiligi 0.5 ustu -> isabet ~%83; 0.2 alti -> A nadir.
"""

import sys
import csv
import time
import shutil
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image
    _DON = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
except ImportError:          # ozet asamasi torch olmadan da calisir
    Dataset = object


class Kume(Dataset):
    """Modul seviyesinde: macOS'ta paralel yukleme icin sart."""
    def __init__(self, yollar):
        self.yollar = yollar

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return _DON(im.convert("RGB")), i, 1
        except Exception:
            return torch.zeros(3, 224, 224), i, 0

ENVANTER = Path("yeni_envanter.csv")
TAHMIN = Path("ayrim_tahmin.csv")
INCELEME = Path("ayrim_inceleme")
AB_MODEL = next((p for p in [Path("model_ab4/en_iyi_model.pt"),
                             Path("model_ab3/en_iyi_model.pt")] if p.exists()), None)
KAT_MODEL = Path("model3/en_iyi_model.pt")

A_GUCLU, A_ZAYIF = 0.5, 0.2
INCELEME_LIMIT = 400
A = "其他"
C_KEYS = {"隐裂", "压板"}
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
C = "=" * 72


def anahtar(ad):
    for k in ANAHTARLAR:
        if k in ad:
            return k
    return ""


def ad(k):
    return AD.get(k, k or "?")


# ----------------------------------------------------------------------
# ASAMA 1
# ----------------------------------------------------------------------
def tahmin():
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import models
    torch.set_num_threads(4)

    if AB_MODEL is None or not KAT_MODEL.exists():
        print("HATA: model_ab4/ab3 ya da model3 bulunamadi."); sys.exit(1)
    if not ENVANTER.exists():
        print(f"HATA: {ENVANTER} yok. Once yeni_envanter.py."); sys.exit(1)

    kayit = [r for r in csv.DictReader(open(ENVANTER, encoding="utf-8-sig"))
             if r["kume"] == "egitim_adayi"]
    bitti = set()
    if TAHMIN.exists():
        bitti = {r["yol"] for r in csv.DictReader(open(TAHMIN, encoding="utf-8-sig"))}
    kalan = [r for r in kayit if r["yol"] not in bitti]
    print(f"{C}\nAYRIM - ASAMA 1: TAHMIN\n{C}")
    print(f"egitim adayi: {len(kayit)}   onceden biten: {len(bitti)}   "
          f"kalan: {len(kalan)}")
    if not kalan:
        print("Hepsi bitmis. Simdi:  python ayrim.py ozet"); return

    ck1 = torch.load(AB_MODEL, map_location="cpu")
    ab = models.resnet18(weights=None); ab.fc = nn.Linear(512, 2)
    ab.load_state_dict(ck1["model"]); ab.eval()
    ck2 = torch.load(KAT_MODEL, map_location="cpu")
    sinif = ck2["sinif_ad"]
    kat = models.resnet18(weights=None); kat.fc = nn.Linear(512, len(sinif))
    kat.load_state_dict(ck2["model"]); kat.eval()
    a_idx = next((i for i, c in enumerate(sinif) if A in c), None)
    print(f"A/B: {AB_MODEL.parent.name}   kategori: {len(sinif)} sinif")

    yl = DataLoader(Kume([r["yol"] for r in kalan]), batch_size=32,
                    shuffle=False, num_workers=2)
    yeni = not TAHMIN.exists()
    fh = open(TAHMIN, "a", newline="", encoding="utf-8-sig")
    w = csv.writer(fh)
    if yeni:
        w.writerow(["yol", "gun", "vardiya", "hat", "bin", "operator",
                    "A_olas", "model_kat", "model_kat_olas", "model_kat2", "okundu"])
    n, t0 = 0, time.time()
    with torch.no_grad():
        for b, (x, idx, ok) in enumerate(yl):
            pa = torch.softmax(ab(x), 1)[:, 0].numpy()
            pk = torch.softmax(kat(x), 1).numpy()
            if a_idx is not None:
                pk[:, a_idx] = -1          # kategori B icinde: A karari ab'nin
            sira = np.argsort(-pk, axis=1)
            for j, i in enumerate(idx.tolist()):
                r = kalan[i]
                w.writerow([r["yol"], r["gun"], r["vardiya"], r["hat"], r["bin"],
                            anahtar(r["kategori"]), f"{pa[j]:.4f}",
                            anahtar(sinif[sira[j, 0]]), f"{pk[j, sira[j, 0]]:.4f}",
                            anahtar(sinif[sira[j, 1]]), int(ok[j])])
            n += len(idx)
            if b % 20 == 0:
                fh.flush()
                hz = n / max(time.time() - t0, 1e-9)
                print(f"\r  {n}/{len(kalan)}  ({hz:.1f} g/sn)  "
                      f"kalan ~{(len(kalan)-n)/max(hz,1e-9)/3600:.1f} saat",
                      end="", flush=True)
    fh.close()
    print(f"\n  bitti ({(time.time()-t0)/3600:.1f} saat)")
    print("Simdi:  python ayrim.py ozet")


# ----------------------------------------------------------------------
# ASAMA 2
# ----------------------------------------------------------------------
def grup(r):
    op, mk = r["operator"], r["model_kat"]
    pa = float(r["A_olas"])
    op_c, m_c = op in C_KEYS, mk in C_KEYS
    if op_c != m_c:
        return "C_anlasmazlik"
    if op == A and pa >= A_GUCLU:
        return "anlasan_A"
    if (op == A and pa < A_ZAYIF) or (op != A and pa >= A_GUCLU):
        return "A_anlasmazlik"
    if op == A:
        return "sinir"              # operator A, model kararsiz
    if pa >= A_ZAYIF:
        return "sinir"              # ikisi B, ama A olasiligi orta
    if op == mk:
        return "anlasan_B"
    return "kat_anlasmazlik"


def ozet():
    if not TAHMIN.exists():
        print(f"HATA: {TAHMIN} yok. Once:  python ayrim.py tahmin"); sys.exit(1)
    R = list(csv.DictReader(open(TAHMIN, encoding="utf-8-sig")))
    okunamayan = sum(1 for r in R if r["okundu"] == "0")
    R = [r for r in R if r["okundu"] == "1"]
    for r in R:
        r["grup"] = grup(r)
    N = len(R)
    print(f"{C}\nAYRIM - ASAMA 2: OZET  ({N} gorsel, okunamayan {okunamayan})\n{C}")

    sira = ["anlasan_A", "anlasan_B", "C_anlasmazlik", "A_anlasmazlik",
            "sinir", "kat_anlasmazlik"]
    aciklama = {"anlasan_A": "temiz A (A dogrulayicisi icin)",
                "anlasan_B": "temiz B (kategori egitimi icin)",
                "C_anlasmazlik": "INCELE - oncelik 1",
                "A_anlasmazlik": "INCELE - oncelik 2",
                "sinir": "INCELE - oncelik 3 (kacmis A olabilir)",
                "kat_anlasmazlik": "dusuk oncelik (B ici)"}
    say = Counter(r["grup"] for r in R)
    for g in sira:
        print(f"  {g:<16} {say[g]:>7}  %{100*say[g]/N:>4.1f}   {aciklama[g]}")

    print(f"\n{C}\nVARDIYA BAZINDA OPERATOR-MODEL A/B UYUMU\n{C}")
    print(f"  {'vardiya':<8} {'n':>7} {'op A':>7} {'model A (>=0.5)':>16} "
          f"{'op B / model A':>15} {'op A / model B':>15}")
    for v in "DMN":
        s = [r for r in R if r["vardiya"] == v]
        if not s:
            continue
        opA = sum(r["operator"] == A for r in s)
        mA = sum(float(r["A_olas"]) >= A_GUCLU for r in s)
        bA = sum(r["operator"] != A and float(r["A_olas"]) >= A_GUCLU for r in s)
        aB = sum(r["operator"] == A and float(r["A_olas"]) < A_ZAYIF for r in s)
        print(f"  {v:<8} {len(s):>7} {100*opA/len(s):>6.1f}% {100*mA/len(s):>15.1f}% "
              f"{bA:>15} {aB:>15}")
    print("  op B / model A: operatorun B dedigi, modelin guclu A dedigi (kacmis A?)")

    print(f"\n{C}\nHAT BAZINDA KATEGORI UYUMU (ikisinin de B dedigi hucrelerde)\n{C}")
    for h in sorted({r["hat"] for r in R}):
        s = [r for r in R if r["hat"] == h and r["operator"] != A
             and float(r["A_olas"]) < A_ZAYIF and r["operator"]]
        if s:
            u = sum(r["operator"] == r["model_kat"] for r in s)
            print(f"  {h:<4} n={len(s):>6}  uyum %{100*u/len(s):.1f}")
    print("  Bir hatta uyum belirgin dusukse model o hatta zorlaniyor olabilir.")

    print(f"\n{C}\nC SINIFI\n{C}")
    opC = [r for r in R if r["operator"] in C_KEYS]
    print(f"  operatorun C dedigi: {len(opC)} "
          f"(catlak {sum(r['operator']=='隐裂' for r in opC)}, "
          f"baski plakasi {sum(r['operator']=='压板' for r in opC)})"
          f" -> C dedektoru v2'nin pozitifleri")

    # ---- en sik kategori anlasmazliklari
    print(f"\n{C}\nEN SIK KATEGORI ANLASMAZLIKLARI (operator -> model)\n{C}")
    ks = Counter((r["operator"], r["model_kat"]) for r in R if r["grup"] == "kat_anlasmazlik")
    for (o, m), n in ks.most_common(10):
        print(f"  {n:>6}  {ad(o):<16} -> {ad(m)}")

    # ---- inceleme klasorleri
    if INCELEME.exists():
        shutil.rmtree(INCELEME)
    oncelik = [("1a_model_C_operator_degil", "C_anlasmazlik",
                lambda r: -float(r["model_kat_olas"])),
               ("1b_operator_C_model_degil", "C_anlasmazlik",
                lambda r: -float(r["A_olas"])),
               ("2a_operator_B_model_A", "A_anlasmazlik",
                lambda r: -float(r["A_olas"]) if r["operator"] != A else 9),
               ("2b_operator_A_model_B", "A_anlasmazlik",
                lambda r: float(r["A_olas"]) if r["operator"] == A else 9),
               ("3_sinir", "sinir", lambda r: -float(r["A_olas"]))]
    print(f"\n{C}\nINCELEME KLASORLERI (her biri en fazla {INCELEME_LIMIT})\n{C}")
    for klasor, g, anahtar_f in oncelik:
        s = [r for r in R if r["grup"] == g]
        if klasor.startswith("1a"):
            s = [r for r in s if r["model_kat"] in C_KEYS]
        if klasor.startswith("1b"):
            s = [r for r in s if r["operator"] in C_KEYS]
        if klasor.startswith("2a"):
            s = [r for r in s if r["operator"] != A]
        if klasor.startswith("2b"):
            s = [r for r in s if r["operator"] == A]
        s.sort(key=anahtar_f)
        hedef = INCELEME / klasor
        hedef.mkdir(parents=True)
        for r in s[:INCELEME_LIMIT]:
            p = Path(r["yol"])
            yeni_ad = (f"op-{ad(r['operator']).replace(' ', '')}_"
                       f"m-{ad(r['model_kat']).replace(' ', '')}_"
                       f"a{float(r['A_olas']):.2f}_{p.name}")
            try:
                shutil.copy2(p, hedef / yeni_ad)
            except Exception:
                pass
        print(f"  {klasor:<24} {min(len(s), INCELEME_LIMIT):>5} / {len(s)}")
    with open("ayrim_gruplar.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(R[0].keys()))
        w.writeheader(); w.writerows(R)
    print(f"\n  Dosya adlari: op-<operator>_m-<model>_a<A olasiligi>")
    print(f"  Tum gruplar: ayrim_gruplar.csv")
    print(f"  open {INCELEME.resolve()}")
    print(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asama", choices=["tahmin", "ozet"])
    a = ap.parse_args()
    tahmin() if a.asama == "tahmin" else ozet()


if __name__ == "__main__":
    main()
