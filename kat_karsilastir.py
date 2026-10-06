#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KATEGORI KARSILASTIRMASI (altin kume, B hucreleri)

model3 (224 px)  vs  model_kat4 (384 px)  vs  model_kat4 + siklik ayari

SIKLIK AYARI
  kategori4.py egitimde ornekleri 1/sqrt(n) ile dengeledi; model her kategoriyi
  egitim sayisinin karekoku oraninda gordu. Bu, kalabalik kategorileri (siyah
  nokta) olduklarindan az gosterir. Tahminde olasiliklar sqrt(n) ile carpilip
  normalize edilerek geri alinir. Miktar egitim formulunden gelir; altin kumeye
  bakilarak AYAR YAPILMAZ.

Sure: ~10 dakika.
  python kat_karsilastir.py
"""

import sys
import re
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

ESKI = Path("model3/en_iyi_model.pt")
YENI = Path("model_kat4/en_iyi_model.pt")
ALTIN_KL = Path("altin_kume")
ALTIN_ONAY = Path("cikti_altin")
ALTIN_HATA = Path.home() / "Desktop" / "altin_hatalar"
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
A = "其他"

# kategori4.py gunlugundeki egitim sayilari
EGITIM_SAYI = {"不规则": 3308, "云雾": 1080, "偏移": 3046, "其他": 4162, "分片": 1425,
               "卡点": 601, "压板": 199, "吸笔": 888, "接触": 461, "断栅": 1239,
               "气流": 2212, "脏污": 1525, "舟划": 1002, "舟框印": 995, "隐裂": 791,
               "黑斑": 6444, "黑边": 3778}

ANAHTARLAR = ["舟框印", "舟划", "舟印", "其他", "黑斑", "黑边", "偏移", "隐裂",
              "压板", "气流", "吸笔", "吸盘", "云雾", "断栅", "接触", "脏污",
              "分片", "不规则", "卡点", "皮带", "滚轮", "暗片"]
AD = {"其他": "A sinifi", "黑斑": "siyah nokta", "黑边": "siyah kenar",
      "偏移": "baski kaymasi", "隐裂": "gizli catlak", "压板": "baski plakasi",
      "舟划": "grafit tekne", "舟框印": "tekne cercevesi", "气流": "hava akisi",
      "吸笔": "vakum kalemi", "云雾": "bulutlu", "断栅": "kirik grid",
      "接触": "temas kirl.", "脏污": "kirlenme", "分片": "ayirma cizik",
      "不规则": "duzensiz cizik", "卡点": "nokta baskisi"}
TURKCE = [("a sinif", "其他"), ("nokta bask", "卡点"), ("siyah nokta", "黑斑"),
          ("siyah leke", "黑斑"), ("siyah kenar", "黑边"), ("siyah kose", "黑边"),
          ("baski kay", "偏移"), ("baski plaka", "压板"), ("gizli", "隐裂"),
          ("catlak", "隐裂"), ("grafit", "舟划"), ("grafik", "舟划"), ("cercev", "舟框印"),
          ("hava", "气流"), ("kalem", "吸笔"), ("emici", "吸盘"), ("bulut", "云雾"),
          ("grid", "断栅"), ("temas", "接触"), ("kirlen", "脏污"), ("ayirma", "分片"),
          ("duzensiz", "不规则"), ("cizik", "不规则"), ("rulo", "滚轮")]
torch.set_num_threads(4)


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


class Kume(Dataset):
    def __init__(self, yollar, boyut):
        self.yollar, self.boyut = yollar, boyut
        self.don = transforms.Compose([
            transforms.Resize((boyut, boyut)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.don(im.convert("RGB")), i
        except Exception:
            return torch.zeros(3, self.boyut, self.boyut), i


def calistir(yol, yollar, boyut):
    ck = torch.load(yol, map_location="cpu")
    net = models.resnet18(weights=None); net.fc = nn.Linear(512, len(ck["sinif_ad"]))
    net.load_state_dict(ck["model"]); net.eval()
    P = np.zeros((len(yollar), len(ck["sinif_ad"])), dtype=np.float32); n = 0
    with torch.no_grad():
        for x, idx in DataLoader(Kume(yollar, boyut), batch_size=16, num_workers=0):
            P[idx.numpy()] = torch.softmax(net(x), 1).numpy()
            n += len(idx); print(f"\r    {n}/{len(yollar)}", end="", flush=True)
    print()
    return P, [anahtar(s) for s in ck["sinif_ad"]]


def tahmin(P, keys):
    P = P.copy()
    if A in keys:
        P[:, keys.index(A)] = -1
    return np.array([keys[i] for i in P.argmax(1)], dtype=object)


def main():
    for p in (ESKI, YENI):
        if not p.exists():
            print(f"HATA: {p} yok"); sys.exit(1)
    altin = {}
    for kok in (ALTIN_ONAY, ALTIN_HATA):
        if kok.is_dir():
            for d in kok.iterdir():
                if d.is_dir() and "emin" not in normal(d.name):
                    k = anahtar(d.name)
                    if k:
                        for f in d.iterdir():
                            if f.suffix.lower() in UZANTILAR:
                                altin[f.name] = k
    eski_keys = {anahtar(s) for s in torch.load(ESKI, map_location="cpu")["sinif_ad"]}
    yeni_keys = {anahtar(s) for s in torch.load(YENI, map_location="cpu")["sinif_ad"]}
    ortak = eski_keys & yeni_keys
    sinav = [(ALTIN_KL / d, k) for d, k in altin.items()
             if k != A and k in ortak and (ALTIN_KL / d).exists()]
    yollar = [s[0] for s in sinav]
    g = np.array([s[1] for s in sinav], dtype=object)
    print(f"Altin kume B hucresi: {len(sinav)}")

    kayit = Path("kat_karsilastir_tahmin.npz")
    if kayit.exists() and len(np.load(kayit, allow_pickle=True)["g"]) == len(sinav):
        z = np.load(kayit, allow_pickle=True)
        P1, k1, P2, k2 = z["P1"], list(z["k1"]), z["P2"], list(z["k2"])
        print("  kaydedilmis tahminler kullanildi")
    else:
        print("  model3 (224 px):")
        P1, k1 = calistir(ESKI, yollar, 224)
        print("  model_kat4 (384 px):")
        P2, k2 = calistir(YENI, yollar, 384)
        np.savez(kayit, P1=P1, k1=np.array(k1, dtype=object), P2=P2,
                 k2=np.array(k2, dtype=object), g=g)
    carpan = np.array([np.sqrt(EGITIM_SAYI.get(k, 1)) for k in k2])
    P3 = P2 * carpan
    P3 = P3 / P3.sum(1, keepdims=True)

    # birlestirme: model3 olasiliklarini yeni modelin sinif sirasina hizala
    P1h = np.zeros_like(P2)
    for j, k in enumerate(k2):
        if k in k1:
            P1h[:, j] = P1[:, k1.index(k)]
    birlesik = (P1h + P3) / 2
    t = {"model3": tahmin(P1, k1), "yeni+ayar": tahmin(P3, k2),
         "birlesik": tahmin(birlesik, k2)}
    print(f"\n{'':<18}" + "".join(f"{ad:>14}" for ad in t))
    print(f"{'GENEL DOGRULUK':<18}" +
          "".join(f"{'%'+format(100*(v==g).mean(),'.1f'):>14}" for v in t.values()))
    print(f"\n{'kategori':<16}{'n':>5}  " +
          "".join(f"{ad+' isb/yak':>18}" for ad in t))
    for k, n in Counter(g).most_common():
        satir = f"{AD.get(k, k):<16}{n:>5}  "
        for v in t.values():
            sec = v == k
            isb = (g[sec] == k).mean() if sec.any() else float("nan")
            yak = (v[g == k] == k).mean()
            s = ("  -" if np.isnan(isb) else f"%{100*isb:.0f}") + f" / %{100*yak:.0f}"
            satir += f"{s:>18}"
        print(satir)
    print("\nisb = o kategoriye gidenlerin dogru orani, yak = gercek orneklerin yakalanma orani")


if __name__ == "__main__":
    main()
