#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAT5 mi KAT6 mi? (9.26 altin kume)

9.26 hic bir egitimde yok; dogru cevaplar senin incelemen.
Karsilastirilanlar:
  SU AN      : model3 + kat5            (gunluk.py'de calisan)
  kat6 turN  : tek basina
  m3+kat6 turN: model3 ile birlesim     (gunluk.py'ye girecek aday)

Iki olcum:
  1) Kategori (B hucreleri): genel dogruluk + ana kategorilerde isabet/yakalama
  2) Ikinci gorusun A klasorune etkisi: kategori modelinin A sinifi ikinci
     gorus olarak kullaniliyor; kat6 A konusunda daha katiysa A klasoru
     degisir. A adaylari uzerinde, esik 0.20.

Siyah nokta orneklemesi (_ornekleme.json) hesaba katilir.
kategori6.py ile ayni klasorde calistirin.   Sure: ~25 dakika.
  python kat6_karsilastir.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

import kategori6 as k6

import sys
KUMELER = {
    "5": (Path("cikti_altin_926"), Path.home() / "Desktop" / "altin5_hatalar",
          Path("altin_kume5"), Path("cikti_altin_926b/_kayit.csv")),
    "7": (Path("cikti_altin_925k"), Path.home() / "Desktop" / "altin7_hatalar",
          Path("altin_kume7"), Path("cikti_altin_925k/_kayit.csv")),
}
_k = sys.argv[sys.argv.index("--kume") + 1] if "--kume" in sys.argv else "5"
ADAY_KL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "model_kat6"
ADAY_AD = ADAY_KL.replace("model_", "")
ONAY, HATALAR, ALTIN_KL, YENI_KAYIT = KUMELER[_k]
ONBELLEK = Path(f"kat6_karsilastir_tahmin_{_k}.npz")
TURLAR = [1, 2, 3]
A_ESIK, C_ESIK_ADAY, IKINCI_ESIK = 0.40, 0.05, 0.20


def etiketler():
    et = {}
    for kok in (ONAY, HATALAR):
        if kok.is_dir():
            for d in kok.iterdir():
                if not d.is_dir() or d.name.startswith("_") or "emin" in k6.normal(d.name):
                    continue
                k = k6.anahtar(d.name)          # C supheli -> None (disarida)
                if k:
                    for f in d.iterdir():
                        if f.suffix.lower() in k6.UZANTILAR:
                            et[f.name] = k
    return et


def main():
    et = etiketler()
    kayit = {r["dosya"]: r for r in csv.DictReader(open(ONAY / "_kayit.csv", encoding="utf-8-sig"))}
    aday_kayit = kayit
    if YENI_KAYIT.exists():
        aday_kayit = {r["dosya"]: r for r in csv.DictReader(open(YENI_KAYIT, encoding="utf-8-sig"))}
        print(f"A adaylari ve C dedektoru: {YENI_KAYIT}")
    else:
        print(f"UYARI: {YENI_KAYIT} yok; A adaylari eski ciktidan (C v1 olasiliklari)")
    agirlik_kl, w = None, 1
    oj = ONAY / "_ornekleme.json"
    if oj.exists():
        o = json.loads(oj.read_text(encoding="utf-8"))
        agirlik_kl, w = k6.anahtar(o["klasor"]), max(1, int(round(o["agirlik"])))

    dosyalar = [d for d in et if (ALTIN_KL / d).exists() and d in kayit]
    yollar = [ALTIN_KL / d for d in dosyalar]
    print(f"Etiketli gorsel: {len(dosyalar)}  (siyah nokta klasoru x{w} agirlikli)")

    # modeller
    net3, ck3 = k6.model_yukle(k6.MODEL3)
    k3 = [k6.anahtar(s) for s in ck3["sinif_ad"]]
    onb = dict(np.load(ONBELLEK, allow_pickle=True)) if ONBELLEK.exists() else {}
    if onb and list(onb.get("dosyalar", [])) != dosyalar:
        onb = {}
    if "P3r" in onb:
        P3r = onb["P3r"]; print("model3: kayitli tahminler")
    else:
        print("model3 (224 px)...")
        P3r = k6.olasilik(net3, yollar, 224, len(k3))
        onb["P3r"] = P3r
    adaylar = {"SU AN (m3+kat5)": Path("model_kat5/en_iyi_model.pt")}
    for t in TURLAR:
        if Path(f"{ADAY_KL}/tur{t}.pt").exists():
            adaylar[f"{ADAY_AD} tur{t}"] = Path(f"{ADAY_KL}/tur{t}.pt")

    sonuclar = {}
    for ad, yol in adaylar.items():
        if not yol.exists():
            print(f"atlandi: {yol}"); continue
        net, ck = k6.model_yukle(yol)
        keys = [k6.anahtar(s) for s in ck["sinif_ad"]]
        anah = "P_" + ad.split()[0] + ad.split()[-1]
        if anah in onb:
            Pham = onb[anah]; print(f"{ad}: kayitli tahminler")
        else:
            print(f"{ad} (384 px)...")
            Pham = k6.olasilik(net, yollar, 384, len(keys))
            onb[anah] = Pham
            np.savez(ONBELLEK, dosyalar=np.array(dosyalar, dtype=object), **{k: v for k, v in onb.items() if k != "dosyalar"})
        P = k6.ayarla(Pham, keys, ck.get("egitim_sayi", k6.KAT4_EGITIM), ck.get("denge_us", 0.5))
        sonuclar[ad] = (keys, P)
    np.savez(ONBELLEK, dosyalar=np.array(dosyalar, dtype=object), **{k: v for k, v in onb.items() if k != "dosyalar"})

    # ortak sinif uzayi: kat6'nin siniflari
    hedef = next((v[0] for a, v in sonuclar.items() if a.startswith(ADAY_AD)), sonuclar["SU AN (m3+kat5)"][0])
    P3 = k6.hizala(P3r, k3, hedef)
    g_all = np.array([et[d] for d in dosyalar], dtype=object)
    m_klasor = [k6.anahtar(kayit[d]["klasor"]) for d in dosyalar]
    tekrar = np.array([w if (agirlik_kl and mk == agirlik_kl) else 1 for mk in m_klasor])

    def genislet(x, maske=None):
        return np.repeat(x, tekrar if maske is None else tekrar[maske], axis=0)

    # ---- 1) kategori: B hucreleri, iki modelde de olan siniflar, C haric
    B = np.array([(gi != k6.A) and (gi in hedef) and (gi not in {"隐裂", "压板"}) for gi in g_all])
    print(f"\n{'='*100}\n1) KATEGORI (9.26, B hucreleri)  isabet/yakalama\n{'='*100}")
    for ad, (keys, P) in sonuclar.items():
        Ph = k6.hizala(P, keys, hedef)
        if ad.startswith("SU AN"):
            satirlar = [("SU AN (m3+kat5)", (P3 + Ph) / 2), ("kat5 tek", Ph)]
        else:
            satirlar = [(f"{ad} tek", Ph)]
        if ad.startswith(ADAY_AD):
            satirlar.append((f"m3+{ad}", (P3 + Ph) / 2))
        for baslik, Q in satirlar:
            k6.rapor(print, baslik, genislet(Q[B], B), hedef, genislet(g_all[B], B))

    # ---- 2) ikinci gorusun A klasorune etkisi
    aday = np.array([d in aday_kayit and float(aday_kayit[d]["A_olasiligi"]) >= A_ESIK and
                     float(aday_kayit[d].get("C_dedektor") or 0) < C_ESIK_ADAY for d in dosyalar])
    gA = g_all == k6.A
    top_A = int(genislet(gA).sum())
    a_i = hedef.index(k6.A)
    print(f"\n{'='*100}\n2) A KLASORU (ikinci gorus esik {IKINCI_ESIK})   gercek A: {top_A}\n{'='*100}")
    print(f"  {'':<22} {'A klasoru':>10} {'dogru A':>8} {'isabet':>7} {'A yakalama':>11}")
    sec0 = aday
    d0 = int(genislet(sec0 & gA).sum())
    print(f"  {'ikinci gorus yok':<22} {int(genislet(sec0).sum()):>10} {d0:>8} "
          f"{'%'+str(round(100*d0/max(genislet(sec0).sum(),1))):>7} {'%'+str(round(100*d0/max(top_A,1))):>11}")
    for ad, (keys, P) in sonuclar.items():
        Ph = k6.hizala(P, keys, hedef)
        pA = ((P3 + Ph) / 2)[:, a_i]
        sec = aday & (pA >= IKINCI_ESIK)
        d_ = int(genislet(sec & gA).sum())
        baslik = ad if ad.startswith("SU AN") else f"m3+{ad}"
        print(f"  {baslik:<22} {int(genislet(sec).sum()):>10} {d_:>8} "
              f"{'%'+str(round(100*d_/max(genislet(sec).sum(),1))):>7} {'%'+str(round(100*d_/max(top_A,1))):>11}")


if __name__ == "__main__":
    main()
