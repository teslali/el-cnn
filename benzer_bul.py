#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BENZER GORSEL ARAMA (genel arac)

Senin dogru etiketledigin birkac "ornek" gorselden yola cikip, eski egitim
verisinde bunlara en cok benzeyen gorselleri bulur ve masaustunde bir
klasore kopyalar. Sen bakip gercekten o tipte olanlari hedef klasore tasirsin;
egitimde senin karar eski etiketin ONUNE gecer.

Kirlenmenin yeni taniminda bu yontemle yakalama %34 -> %88-90 oldu.

Kullanim:
  python benzer_bul.py --ornek KLASOR [KLASOR ...] --havuz ETIKET [ETIKET ...]
                       --cikti AD --hedef-klasor AD [--adet 400]
                       [--kaynak ~/Desktop/fabrika-fotolar/2B]

  --ornek        : ornek gorsellerin oldugu klasor(ler)
  --havuz        : eski veride taranacak etiketler (Cince anahtar ya da Turkce ad):
                   ör. 断栅 (kirik grid), 滚轮 (rulo izi), 皮带 (konveyor), 不规则 (cizik)
  --cikti        : masaustunde olusacak inceleme klasorunun adi
  --hedef-klasor : gercekten o tipte olanlarin tasinacagi klasorun adi
                   (~/Desktop/yeni_duzeltmeler altinda olusturulur)

Ornekler:
  Kirik gridin yeni tanimi (artik duzensiz cizik):
    python benzer_bul.py \\
      --ornek ~/Desktop/altin8_hatalar/"duzensiz cizik yeni tanim" \\
              ~/Desktop/altin9_hatalar/"duzensiz cizik yeni tanim" \\
      --havuz 断栅 --cikti kirikgrid_benzer \\
      --hedef-klasor "duzensiz cizik yeni tanim"

  Konveyor bant izi:
    python benzer_bul.py --ornek ~/Desktop/altin8_hatalar/"konveyor bant izi" \\
      --havuz 滚轮 皮带 --cikti konveyor_benzer --hedef-klasor "konveyor bant izi"
"""

import sys
import csv
import re
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
except ImportError as e:
    print(f"HATA: paket eksik ({e.name})"); sys.exit(1)

MODEL = next((p for p in [Path("model_kat7/tur2.pt"), Path("model_kat5/en_iyi_model.pt")]
              if p.exists()), None)
GRUPLAR = Path("ayrim_gruplar.csv")
D = Path.home() / "Desktop"
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
TURKCE = {"kirik grid": "断栅", "rulo": "滚轮", "konveyor": "皮带", "cizik": "不规则",
          "siyah nokta": "黑斑", "siyah kenar": "黑边", "hava": "气流", "kirlenme": "脏污",
          "baski kaymasi": "偏移", "ayirma": "分片", "vakum emici": "吸盘",
          "vakum kalemi": "吸笔", "bulut": "云雾", "temas": "接触"}
ALTIN_GUNLER = {"20260922", "20260923", "20260924", "20260925", "20260926",
                "20260927", "20260930", "20261001"}
torch.set_num_threads(4)


def gun(ad):
    z = re.search(r"(\d{14})", ad)
    return z.group(1)[:8] if z else ""


def anahtarla(e):
    if any("\u4e00" <= c <= "\u9fff" for c in e):
        return e
    return TURKCE.get(e.lower(), e)


class Kume(Dataset):
    def __init__(self, yollar):
        self.yollar = yollar
        self.don = transforms.Compose([
            transforms.Resize((384, 384)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.don(im.convert("RGB")), i
        except Exception:
            return torch.zeros(3, 384, 384), i


def ozellik(net, yollar, baslik):
    F = np.zeros((len(yollar), 512), dtype=np.float32)
    n = 0
    with torch.no_grad():
        for x, idx in DataLoader(Kume(yollar), batch_size=16, num_workers=2):
            F[idx.numpy()] = net(x).numpy()
            n += len(idx)
            print(f"\r  {baslik}: {n}/{len(yollar)}", end="", flush=True)
    print()
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ornek", nargs="+", required=True)
    ap.add_argument("--havuz", nargs="*", default=[],
                    help="eski veride taranacak etiketler")
    ap.add_argument("--havuz-klasor", nargs="*", default=[],
                    help="etiketine bakmadan taranacak klasor(ler), ör. bir gunun tamami")
    ap.add_argument("--haric", nargs="*", default=[],
                    help="bu klasorlerdeki gorseller (ör. incelenmis altin kume) taranmaz")
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--hedef-klasor", required=True)
    ap.add_argument("--adet", type=int, default=400)
    ap.add_argument("--kaynak", default=str(D / "fabrika-fotolar" / "2B"))
    ap.add_argument("--havuz-sinir", type=int, default=5000)
    a = ap.parse_args()

    hedef = D / a.cikti
    if hedef.exists() and any(hedef.iterdir()):
        print(f"{hedef} dolu; once bosaltin ya da silin."); return
    if MODEL is None:
        print("HATA: model_kat7 ya da model_kat5 bulunamadi."); return

    ornek = []
    for k in a.ornek:
        kl = Path(k).expanduser()
        if not kl.is_dir():
            print(f"UYARI: ornek klasoru yok: {kl}"); continue
        ornek += [f for f in kl.rglob("*") if f.suffix.lower() in UZANTILAR]
    print(f"Ornek gorsel: {len(ornek)}")
    if len(ornek) < 5:
        print("HATA: cok az ornek; klasor adlarini kontrol edin."); return

    keys = {anahtarla(e) for e in a.havuz}
    if not keys and not a.havuz_klasor:
        print("HATA: --havuz ya da --havuz-klasor verilmeli."); return
    haric = set()
    for k in a.haric:
        kl = Path(k).expanduser()
        if kl.is_dir():
            haric |= {f.name for f in kl.rglob("*") if f.is_file()}
    havuz = []
    for k in a.havuz_klasor:
        kl = Path(k).expanduser()
        if not kl.is_dir():
            print(f"UYARI: havuz klasoru yok: {kl}"); continue
        for f in kl.rglob("*"):
            if f.suffix.lower() in UZANTILAR and f.name not in haric and "OverTime" not in f.name:
                havuz.append((f, "klasor"))
    kay = Path(a.kaynak).expanduser()
    if keys and kay.is_dir():
        for d in kay.iterdir():
            k = next((k for k in keys if k in d.name), None)
            if d.is_dir() and k:
                havuz += [(f, k) for f in d.iterdir() if f.suffix.lower() in UZANTILAR]
    rng = np.random.RandomState(0)
    if keys and GRUPLAR.exists():
        R = [r for r in csv.DictReader(open(GRUPLAR, encoding="utf-8-sig"))
             if r["operator"] in keys and gun(r["yol"]) not in ALTIN_GUNLER]
        for k in keys:
            g = [r for r in R if r["operator"] == k]
            for i in rng.permutation(len(g))[:a.havuz_sinir]:
                havuz.append((Path(g[i]["yol"]), k))
    gorulen = set()
    temiz = []
    for f, k in havuz:
        if f.exists() and f.name not in gorulen:
            gorulen.add(f.name); temiz.append((f, k))
    havuz = temiz
    print(f"Havuz: {len(havuz)} gorsel  (haric tutulan: {len(haric)})")
    if not havuz:
        print("HATA: havuz bos; --havuz etiketlerini kontrol edin."); return

    ck = torch.load(MODEL, map_location="cpu")
    net = models.resnet18(weights=None); net.fc = nn.Linear(512, len(ck["sinif_ad"]))
    net.load_state_dict(ck["model"]); net.fc = nn.Identity(); net.eval()
    print(f"Ozellikler: {MODEL}")
    Fo = ozellik(net, ornek, "ornekler")
    Fh = ozellik(net, [f for f, _ in havuz], "havuz")
    puan = np.zeros(len(havuz))
    for i in range(0, len(havuz), 2000):
        S = Fh[i:i+2000] @ Fo.T
        puan[i:i+2000] = np.sort(S, axis=1)[:, -3:].mean(1)
    sira = np.argsort(-puan)[:a.adet]

    hedef.mkdir(parents=True, exist_ok=True)
    say = Counter()
    for i in sira:
        f, k = havuz[i]
        if k == "klasor":
            k = next((x for x in TURKCE.values() if x in f.parent.name), "diger")
        say[k] += 1
        shutil.copy2(f, hedef / f"op-{k}_m-benzer_a{puan[i]:.2f}_{f.name}")
    tasima = D / "yeni_duzeltmeler" / a.hedef_klasor
    tasima.mkdir(parents=True, exist_ok=True)
    print(f"\nEn benzer {len(sira)} gorsel -> {hedef}")
    print("  eski etiketleri: " + ", ".join(f"{k} {n}" for k, n in say.most_common()))
    print(f"  benzerlik araligi: {puan[sira[0]]:.2f} - {puan[sira[-1]]:.2f}")
    print(f"\nInceleme: gercekten aranan tipte olanlari su klasore TASIYIN:\n  {tasima}")
    print("Digerlerine dokunmayin.")


if __name__ == "__main__":
    main()
