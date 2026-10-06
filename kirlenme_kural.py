#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KIRLENME KURALI DENEMESI (kayitli tahminlerle, saniyeler)

Bulgu: kat7 tek basina kirlenmeyi ogrendi (yakalama %34 -> %79), ama model3
ile birlesimde model3'un "cizik" oyu bunu bastiriyor.

Kural: karar su anki sistemle (model3 + kat5) verilir; TEK ISTISNA, kat7'nin
kirlenme olasiligi esigin ustundeyse hucre kirlenmeye gider. A klasoru ve
diger kategoriler etkilenmez.

Esik secimi bu kumeye bakilarak yapilirsa iyimser olur; bu yuzden esas aday
'kat7 en yuksek kirlenme diyorsa' kurali (ayar yok). Esikli satirlar bilgi icin.
Asil sinav yeni gun (9.28 / 9.29).

kat6_karsilastir.py --kume 7 --model model_kat7 bir kez calismis olmali.
  python kirlenme_kural.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

import kategori6 as k6

ONAY = Path("cikti_altin_925k")
HATALAR = Path.home() / "Desktop" / "altin7_hatalar"
ALTIN_KL = Path("altin_kume7")
ONBELLEK = Path("kat6_karsilastir_tahmin_7.npz")
KAT5 = Path("model_kat5/en_iyi_model.pt")
KAT7 = Path("model_kat7/tur2.pt")
KIR = "脏污"


def main():
    z = dict(np.load(ONBELLEK, allow_pickle=True))
    dosyalar = list(z["dosyalar"])
    ck5 = torch.load(KAT5, map_location="cpu")
    ck7 = torch.load(KAT7, map_location="cpu")
    ck3 = torch.load(k6.MODEL3, map_location="cpu")
    k3 = [k6.anahtar(s) for s in ck3["sinif_ad"]]
    k5 = [k6.anahtar(s) for s in ck5["sinif_ad"]]
    k7 = [k6.anahtar(s) for s in ck7["sinif_ad"]]
    hedef = k5
    P3 = k6.hizala(z["P3r"], k3, hedef)
    P5 = k6.ayarla(z["P_SU(m3+kat5)"], k5, ck5.get("egitim_sayi", k6.KAT4_EGITIM),
                   ck5.get("denge_us", 0.5))
    P7 = k6.hizala(k6.ayarla(z["P_kat7tur2"], k7, ck7["egitim_sayi"], ck7["denge_us"]), k7, hedef)

    # etiketler ve agirlik (kat6_karsilastir ile ayni)
    et = {}
    for kok in (ONAY, HATALAR):
        for d in kok.iterdir():
            if d.is_dir() and not d.name.startswith("_") and "emin" not in k6.normal(d.name):
                k = k6.anahtar(d.name)
                if k:
                    for f in d.iterdir():
                        et[f.name] = k
    kayit = {r["dosya"]: r for r in csv.DictReader(open(ONAY / "_kayit.csv", encoding="utf-8-sig"))}
    w, ok_ = 1, None
    oj = ONAY / "_ornekleme.json"
    if oj.exists():
        o = json.loads(oj.read_text(encoding="utf-8"))
        w, ok_ = max(1, int(round(o["agirlik"]))), k6.anahtar(o["klasor"])
    tekrar = np.array([w if k6.anahtar(kayit[d]["klasor"]) == ok_ else 1 for d in dosyalar])
    g = np.array([et[d] for d in dosyalar], dtype=object)
    B = np.array([(gi != k6.A) and (gi in hedef) and (gi not in {"隐裂", "压板"}) for gi in g])

    def gen(x):
        return np.repeat(x[B], tekrar[B], axis=0)

    taban = (P3 + P5) / 2
    a_i, kir_i = hedef.index(k6.A), hedef.index(KIR)
    P7b = P7.copy(); P7b[:, a_i] = -1
    kat7_kir = P7b.argmax(1) == kir_i

    def kural(maske):
        Q = taban.copy()
        Q[maske, :] = 0.0
        Q[maske, kir_i] = 1.0
        return Q

    print("9.25 geri kalani, B hucreleri (isabet/yakalama)\n")
    k6.rapor(print, "SU AN (m3+kat5)", gen(taban), hedef, gen(g))
    k6.rapor(print, "kural: kat7 en yuksek", gen(kural(kat7_kir)), hedef, gen(g))
    print("\n  bilgi icin (esik bu kumeye bakilarak secilirse iyimser olur):")
    for e in (0.3, 0.4, 0.5, 0.6):
        k6.rapor(print, f"kural: kat7 >= {e}", gen(kural(P7[:, kir_i] >= e)), hedef, gen(g))
    print(f"\n  kat7'nin kirlenme dedigi hucre: {int(gen(kat7_kir).sum())} "
          f"(su an kirlenmeye giden: {int(gen(taban.argmax(1) == kir_i).sum())})")


if __name__ == "__main__":
    main()
