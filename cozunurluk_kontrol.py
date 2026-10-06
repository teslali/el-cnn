#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COZUNURLUK KONTROLU - baski plakasi izi kucultmede kayboluyor mu?

HIPOTEZ
  Model goruntuyu 512'den 224'e kucultuyor. Orta boy bir nokta
  buyuklugundeki X isareti bu kucultmede tek bir lekeye donusuyor ve
  model onu "tolere edilebilir tek nokta" sanip A diyor.

NE YAPAR
  Her gorsel icin yan yana uc panel uretir:
     512 (orijinal)  |  384  |  224  (model bunu goruyor)
  384 ve 224 panelleri once o boyuta kucultulur, sonra 512'ye geri
  buyutulur (piksel piksel, yumusatmadan) ki kaybolan bilgi gorunsun.
  Kucultme, modelin kullandigi yontemle ayni (bilinear).

  Model KULLANMAZ; saniyeler surer, calisan baska islemleri yavaslatmaz.

NASIL BAKACAKSINIZ
  Her gorselde X isaretini bulun ve uc panelde karsilastirin:
    224'te X hala X ise           -> sorun cozunurluk degil, veri
    224'te nokta, 384'te X ise    -> 384 yeterli
    384'te de nokta ise           -> tam cozunurluk (512) gerekli

Kullanim:
  python cozunurluk_kontrol.py
  python cozunurluk_kontrol.py --klasor baska/klasor
"""

import sys
import argparse
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("HATA: pillow yok (venv aktif mi?)")
    sys.exit(1)

UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
BOYUTLAR = [512, 384, 224]
BASLIK = 28


def panel(im, boyut, hedef=512):
    if boyut == hedef:
        return im.resize((hedef, hedef), Image.BILINEAR)
    kucuk = im.resize((boyut, boyut), Image.BILINEAR)
    return kucuk.resize((hedef, hedef), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasor", default="sizinti_kontrol/1_test_sizan_C")
    ap.add_argument("--cikti", default="sizinti_kontrol/3_cozunurluk")
    a = ap.parse_args()

    kaynak = Path(a.klasor)
    if not kaynak.is_dir():
        print(f"HATA: klasor yok: {kaynak.resolve()}")
        sys.exit(1)
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)

    dosyalar = sorted([f for f in kaynak.iterdir()
                       if f.is_file() and f.suffix.lower() in UZANTILAR])
    if not dosyalar:
        print("HATA: gorsel yok.")
        sys.exit(1)

    for f in dosyalar:
        with Image.open(f) as im:
            im = im.convert("RGB")
            paneller = [panel(im, b) for b in BOYUTLAR]
        W = 512
        tuval = Image.new("RGB", (W * len(BOYUTLAR) + 8 * (len(BOYUTLAR) - 1),
                                  W + BASLIK), "white")
        ciz = ImageDraw.Draw(tuval)
        for k, (b, p) in enumerate(zip(BOYUTLAR, paneller)):
            x = k * (W + 8)
            etiket = f"{b}px" + ("  (orijinal)" if b == 512 else
                                 "  (MODEL BUNU GORUYOR)" if b == 224 else "")
            ciz.text((x + 8, 8), etiket, fill="black")
            tuval.paste(p, (x, BASLIK))
        tuval.save(cikti / f"{f.stem}_karsilastirma.png")

    print(f"{len(dosyalar)} karsilastirma olusturuldu: {cikti.resolve()}")
    print()
    print("Her gorselde X isaretini bulup uc panelde karsilastirin:")
    print("  224'te X hala X        -> sorun veri, cozunurluk degil")
    print("  224'te nokta, 384'te X -> 384 yeterli")
    print("  384'te de nokta        -> tam cozunurluk gerekli")
    print(f"\n  open \"{cikti.resolve()}\"")


if __name__ == "__main__":
    main()
