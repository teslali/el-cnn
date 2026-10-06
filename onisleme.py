#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ON ISLEME: uc kanalli kontrast gorunumu (C dedektoru v3 ve sonrasi)

EL gorselleri siyah-beyaz; modeller uc kanal bekliyor. Uc kanala ayni goruntuyu
koymak yerine:
  kanal 0: ham goruntu               (genis alanli farklar, ör. bulutlu hucre)
  kanal 1: yerel kontrast normalizasyonu (CLAHE benzeri; soluk izleri one cikarir)
  kanal 2: ince cizgi filtresi       (ince, keskin yapilar: catlak, cizik)

Egitimde ve gunluk calismada AYNI fonksiyon kullanilmali. Ek paket gerekmez
(numpy + scipy, sklearn ile zaten kurulu).
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SURUM = "kanal3_v1"
ORT = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def kanal3(im, boyut):
    """PIL goruntu -> (3, boyut, boyut) float32, normalize edilmis."""
    g = np.asarray(im.convert("L").resize((boyut, boyut), Image.BILINEAR), dtype=np.float32)
    # yerel kontrast: (g - yerel ortalama) / yerel sapma
    m = gaussian_filter(g, 12)
    v = gaussian_filter((g - m) ** 2, 12)
    k1 = np.clip(128.0 + 40.0 * (g - m) / np.sqrt(v + 25.0), 0, 255)
    # ince cizgi: ince bulaniklastirmadan fark
    k2 = np.clip(128.0 + 4.0 * (g - gaussian_filter(g, 2)), 0, 255)
    x = np.stack([g, k1, k2]).astype(np.float32) / 255.0
    return (x - ORT) / STD
