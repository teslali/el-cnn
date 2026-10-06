#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KUSUR HARITALARI: kategori ve hat bazinda ortalama goruntu + yogunluk haritasi

Ayiklanmis bir ciktida (ör. 9.27_AI) her kategori icin:
  - ORTALAMA GORUNTU : tum hucrelerin piksel piksel ortalamasi. Hep ayni yerde
                       cikan kusurlar (rulo izi, vakum kalemi, tekne izleri...)
                       koyu bir bolge olarak belirir.
  - YOGUNLUK HARITASI: her hucrede cevresinden belirgin sekilde koyu pikseller
                       isaretlenir ve ust uste toplanir. "Kusurlar hucrenin
                       neresinde ne siklikla cikiyor" sorusunun cevabi.
Her kategori icin hat bazinda (dosya adinin basindaki 1B, 2A...) ve tum hatlar.
Her gorselde hucre sinirlari bulunup ayni kadraja oturtulur, parlaklik dengelenir.

Model kullanmaz; egitimle ayni anda calisabilir.

  python kusur_haritasi.py ~/Desktop/9.27_AI
  python kusur_haritasi.py ~/Desktop/9.27_AI --min 10 --boyut 384
Sonuc: ~/Desktop/kusur_haritalari/<cikti adi>/
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
HAT_RE = re.compile(r"^(\d[A-Z])_", re.I)
ADLAR = {"黑斑": "siyah nokta", "黑边": "siyah kenar", "偏移": "baski kaymasi",
         "不规则": "duzensiz cizik", "气流": "hava akisi", "断栅": "kirik grid",
         "脏污": "kirlenme", "分片": "ayirma cizik", "云雾": "bulutlu", "滚轮": "rulo izi",
         "吸笔": "vakum kalemi", "吸盘": "vakum emici", "舟划": "grafit tekne",
         "舟框印": "tekne cercevesi", "舟印": "kuvars tekne", "皮带": "konveyor",
         "接触": "temas kirlenmesi", "卡点": "nokta baskisi", "其他": "A sinifi",
         "supheli": "C supheli"}


def kategori_adi(klasor):
    for k, v in ADLAR.items():
        if k in klasor:
            return v
    return None


def hizala(im, boyut):
    """hucre sinirlarini bul, kirp, kare boyuta getir, parlakligi dengele"""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    sat, sut = g.mean(1), g.mean(0)
    def sinir(p):
        esik = p.min() + 0.35 * (p.max() - p.min())
        i = np.where(p > esik)[0]
        return (i[0], i[-1] + 1) if len(i) > 10 else (0, len(p))
    y0, y1 = sinir(sat)
    x0, x1 = sinir(sut)
    k = Image.fromarray(g[y0:y1, x0:x1].astype(np.uint8)).resize((boyut, boyut), Image.BILINEAR)
    a = np.asarray(k, dtype=np.float32)
    med = np.median(a)
    return a / max(med, 1.0)              # 1.0 = hucrenin tipik parlakligi


def koyu_maske(a, tipik):
    """hattin tipik hucresine gore belirgin koyu pikseller (parmak hatlari gibi
    her hucrede olan yapilar tipik hucrede de oldugu icin elenir)"""
    fark = a - tipik
    im = Image.fromarray(np.clip(fark * 128 + 128, 0, 255).astype(np.uint8))
    fark = np.asarray(im.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32) / 128 - 1
    return fark < -max(3.0 * fark.std(), 0.08)


def renklendir(d):
    """0-1 yogunlugu siyah->mor->kirmizi->sari renge cevir"""
    d = np.clip(d, 0, 1)
    r = np.clip(3 * d, 0, 1)
    g = np.clip(3 * d - 1.5, 0, 1)
    b = np.clip(1.5 * d, 0, 0.6) * (1 - np.clip(3 * d - 1, 0, 1))
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def kaydet(yol, baslik, ort, yog, n):
    boyut = ort.shape[0]
    o = np.clip(ort / max(np.percentile(ort, 99), 1e-6) * 230, 0, 255).astype(np.uint8)
    o_rgb = np.stack([o] * 3, -1)
    ust = np.percentile(yog, 99.5)
    y_rgb = renklendir(yog / max(ust, 1e-6))
    tuval = Image.new("RGB", (boyut * 2 + 30, boyut + 50), (255, 255, 255))
    tuval.paste(Image.fromarray(o_rgb), (10, 40))
    tuval.paste(Image.fromarray(y_rgb), (boyut + 20, 40))
    c = ImageDraw.Draw(tuval)
    c.text((10, 8), f"{baslik}   (n = {n})", fill=(0, 0, 0))
    c.text((10, 24), "ortalama goruntu", fill=(80, 80, 80))
    c.text((boyut + 20, 24), f"kusur yogunlugu (en yogun bolge: hucrelerin %{100*ust:.0f}'inde)",
           fill=(80, 80, 80))
    tuval.save(yol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cikti", nargs="+", help="ayiklanmis cikti klasor(ler)i")
    ap.add_argument("--min", type=int, default=15, help="bir harita icin en az gorsel")
    ap.add_argument("--boyut", type=int, default=384)
    ap.add_argument("--sinir", type=int, default=1500, help="kategori basina en fazla gorsel")
    a = ap.parse_args()

    gruplar = defaultdict(list)       # (kategori, hat) -> dosyalar
    for kok in a.cikti:
        kok = Path(kok).expanduser()
        for d in kok.iterdir():
            if not d.is_dir():
                continue
            kat = kategori_adi(d.name)
            if not kat:
                continue
            for f in d.iterdir():
                if f.suffix.lower() in UZANTILAR:
                    m = HAT_RE.match(f.name)
                    gruplar[(kat, m.group(1).upper() if m else "?")].append(f)

    # her hattin tipik hucresi: o hattaki gorsellerin (kategori fark etmeksizin) medyani
    rng0 = np.random.RandomState(1)
    hat_dosya = defaultdict(list)
    for (_, h), fs in gruplar.items():
        hat_dosya[h] += fs
    tipik = {}
    for h, fs in hat_dosya.items():
        sec = [fs[i] for i in rng0.permutation(len(fs))[:300]]
        yigin = []
        for f in sec:
            try:
                with Image.open(f) as im:
                    yigin.append(hizala(im, a.boyut))
            except Exception:
                pass
        if yigin:
            tipik[h] = np.median(np.stack(yigin), axis=0)
    genel = np.median(np.stack(list(tipik.values())), axis=0)
    print("tipik hucreler: " + ", ".join(f"{h}" for h in sorted(tipik)))

    hedef = Path.home() / "Desktop" / "kusur_haritalari" / Path(a.cikti[0]).expanduser().name
    hedef.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    kategoriler = sorted({k for k, _ in gruplar})
    print(f"{len(kategoriler)} kategori, hatlar: "
          + ", ".join(sorted({h for _, h in gruplar})))

    for kat in kategoriler:
        hatlar = sorted(h for k, h in gruplar if k == kat)
        toplam = {"ort": None, "yog": None, "n": 0}
        for hat in hatlar:
            fs = gruplar[(kat, hat)]
            if len(fs) > a.sinir:
                fs = [fs[i] for i in rng.permutation(len(fs))[:a.sinir]]
            ort = np.zeros((a.boyut, a.boyut), np.float64)
            yog = np.zeros((a.boyut, a.boyut), np.float64)
            n = 0
            for f in fs:
                try:
                    with Image.open(f) as im:
                        h = hizala(im, a.boyut)
                except Exception:
                    continue
                ort += h; yog += koyu_maske(h, tipik.get(hat, genel)); n += 1
            if n == 0:
                continue
            for k_, v in (("ort", ort), ("yog", yog)):
                toplam[k_] = v.copy() if toplam[k_] is None else toplam[k_] + v
            toplam["n"] += n
            if n >= a.min:
                (hedef / kat).mkdir(exist_ok=True)
                kaydet(hedef / kat / f"{hat}.png", f"{kat} - hat {hat}", ort / n, yog / n, n)
        if toplam["n"] >= a.min:
            (hedef / kat).mkdir(exist_ok=True)
            kaydet(hedef / kat / "tum_hatlar.png", f"{kat} - tum hatlar",
                   toplam["ort"] / toplam["n"], toplam["yog"] / toplam["n"], toplam["n"])
            print(f"  {kat:<18} {toplam['n']:>6} gorsel  hatlar: " +
                  ", ".join(f"{h} {len(gruplar[(kat, h)])}" for h in hatlar))
    print(f"\nHaritalar: {hedef}")
    print(f"  open {hedef}")


if __name__ == "__main__":
    main()
