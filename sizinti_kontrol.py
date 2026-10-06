#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C SINIFI SIZINTI KONTROLU + VETO KALIBRASYONU

Egitim BITTIKTEN SONRA calistirin (paralel calisirsa ikisi de yavaslar).

BOLUM 1 - TEST KUMESINDE SIZAN C GORSELLERI
  Test kumesinde (10-11 Eylul) %18.7 kesimde 98 C'nin ~28'i A adaylarina
  siziyor; uretimde bu oran ~%1. Celiski gozle cozulecek:
  sizan gorseller  sizinti_kontrol/1_test_sizan_C/  klasorune kopyalanir.
  Soru: bunlar GERCEKTEN catlakli mi, yoksa yanlis etiketli hafif hucreler mi?

BOLUM 2 - VETO ESIGI KALIBRASYONU
  Veri: dogrulanmis gun (cikti_test + hatalar). Etiketler senin kontrolunden.
  Veto kurali: A adayi hucrede kategori modeli (model3) C olasiligi >= esik
  ise A adayligi iptal.
  Her esik icin:
    yakalanan C : o gunun tum C hucrelerinden kacini model3 isaretliyor
    bedel       : A adaylari icindeki GERCEK A hucrelerinden kaci bosa gidiyor

  DURUSTLUK NOTU: A adaylarini secerken model_ab3 kullanilir, model_ab4 degil.
  Cunku ab4 bu gunun verisiyle egitildi; kendi egitim verisinde olcum
  gercekten iyi gorunur ama yaniltir. ab3 ve model3 bu gunu hic gormedi.

Kullanim:
  python sizinti_kontrol.py \\
     --kaynak ~/Desktop/fabrika-fotolar/2B \\
     --dogrulanmis ~/Desktop/CNN/cikti_test \\
     --hatalar ~/Desktop/hatalar
"""

import sys
import re
import time
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
    print(f"HATA: paket eksik ({e.name})")
    sys.exit(1)

GIRDI_BOYUT = 224
BATCH = 24
ISCI = 2
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")

A_ANAHTAR = "其他"
C_ANAHTARLAR = {"隐裂", "压板"}
TEST_GUNLERI = {"20260910", "20260911"}
A_ORANI = 0.187
ESIKLER = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

CIKTI = Path("sizinti_kontrol")
C = "=" * 74
torch.set_num_threads(4)


def normal(s):
    tablo = str.maketrans("ıİşŞçÇğĞöÖüÜ", "iIsScCgGoOuU")
    return " ".join(s.translate(tablo).lower().replace("_", " ").split())


def kisa(ad, n=30):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def gun(p):
    z = ZAMAN.search(Path(p).stem)
    return z.group(1)[:8] if z else ""


class Kume(Dataset):
    def __init__(self, yollar):
        self.yollar = yollar
        self.don = transforms.Compose([
            transforms.Resize((GIRDI_BOYUT, GIRDI_BOYUT)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.don(im.convert("RGB")), i
        except Exception:
            return torch.zeros(3, GIRDI_BOYUT, GIRDI_BOYUT), -1


def model_yukle(yol, cikti_sayisi=None):
    ck = torch.load(yol, map_location="cpu")
    n = cikti_sayisi or len(ck["sinif_ad"])
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, n)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


def tahmin(net, yollar, cikti_sayisi, etiket):
    yl = DataLoader(Kume(yollar), batch_size=BATCH, shuffle=False,
                    num_workers=ISCI)
    P = np.zeros((len(yollar), cikti_sayisi), dtype=np.float32)
    n, t0 = 0, time.time()
    with torch.no_grad():
        for x, idx in yl:
            p = torch.softmax(net(x), 1).numpy()
            for j, i in enumerate(idx.tolist()):
                if i >= 0:
                    P[i] = p[j]
            n += len(idx)
            hz = n / max(time.time() - t0, 1e-9)
            print(f"\r  {etiket}: {n}/{len(yollar)}  "
                  f"kalan ~{(len(yollar)-n)/max(hz,1e-9)/60:.1f} dk",
                  end="", flush=True)
    print(f"   bitti ({(time.time()-t0)/60:.1f} dk)")
    return P


def gorseller(d):
    return sorted([f for f in d.rglob("*")
                   if f.is_file() and f.suffix.lower() in UZANTILAR])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaynak", required=True)
    ap.add_argument("--dogrulanmis", required=True)
    ap.add_argument("--hatalar", required=True)
    ap.add_argument("--ab-test", default=None,
                    help="bolum 1 icin A/B modeli (varsayilan: model_ab4, yoksa ab3)")
    a = ap.parse_args()

    kay = Path(a.kaynak).expanduser().resolve()
    dog = Path(a.dogrulanmis).expanduser().resolve()
    hat = Path(a.hatalar).expanduser().resolve()
    for p in (kay, dog, hat):
        if not p.is_dir():
            print(f"HATA: klasor yok: {p}"); sys.exit(1)

    ab_test_yol = Path(a.ab_test) if a.ab_test else (
        Path("model_ab4/en_iyi_model.pt")
        if Path("model_ab4/en_iyi_model.pt").exists()
        else Path("model_ab3/en_iyi_model.pt"))
    ab_kal_yol = Path("model_ab3/en_iyi_model.pt")
    kat_yol = Path("model3/en_iyi_model.pt")
    for p in (ab_test_yol, ab_kal_yol, kat_yol):
        if not p.exists():
            print(f"HATA: model yok: {p}"); sys.exit(1)

    if CIKTI.exists():
        shutil.rmtree(CIKTI)
    (CIKTI / "1_test_sizan_C").mkdir(parents=True)
    (CIKTI / "2_gunluk_sizan_C").mkdir(parents=True)

    rapor = []

    def yaz(s=""):
        print(s, flush=True)
        rapor.append(s)

    # ---- kategori modeli ve C sinif indeksleri
    net_kat, ck_kat = model_yukle(kat_yol)
    sinif_ad = ck_kat["sinif_ad"]
    c_idx = [i for i, c in enumerate(sinif_ad)
             if any(k in c for k in C_ANAHTARLAR)]
    yaz(f"{C}\nC SINIFI SIZINTI KONTROLU\n{C}")
    yaz(f"Kategori modelinin tanidigi C siniflari: "
        f"{[kisa(sinif_ad[i]) for i in c_idx] or 'YOK'}")
    if len(c_idx) < len(C_ANAHTARLAR):
        yaz("  NOT: baski plakasi kategori modelinde yok (egitimde 150 ornek")
        yaz("  esigi altindaydi). Veto su an sadece gizli catlagi gorebilir.")

    # ==================================================================
    # BOLUM 1 - test kumesi
    # ==================================================================
    yaz(f"\n{C}\nBOLUM 1 - TEST KUMESINDE SIZAN C  (model: {ab_test_yol.parent.name})\n{C}")
    t_yol, t_kat = [], []
    for d in sorted([x for x in kay.iterdir() if x.is_dir()]):
        for f in gorseller(d):
            if gun(f) in TEST_GUNLERI:
                t_yol.append(f); t_kat.append(d.name)
    t_kat = np.array(t_kat)
    t_c = np.array([any(k in k_ for k in C_ANAHTARLAR) for k_ in t_kat])
    yaz(f"Test gorseli: {len(t_yol)}  (C: {int(t_c.sum())})")

    net_ab, _ = model_yukle(ab_test_yol, 2)
    Pt = tahmin(net_ab, t_yol, 2, "A/B test")[:, 0]
    sira = np.argsort(-Pt)
    N = max(1, int(len(t_yol) * A_ORANI))
    ust = set(sira[:N].tolist())
    sizan = [i for i in sira[:N] if t_c[i]]
    rank = {int(i): r for r, i in enumerate(sira, 1)}

    yaz(f"%{100*A_ORANI:.1f} kesim = ilk {N}. Sizan C: {len(sizan)}")
    for k, n in Counter(kisa(t_kat[i]) for i in sizan).most_common():
        yaz(f"   {n:>4}  {k}")
    for i in sizan:
        f = t_yol[i]
        shutil.copy2(f, CIKTI / "1_test_sizan_C" /
                     f"{rank[int(i)]:05d}_a{Pt[i]:.3f}_{f.name}")

    if sizan:
        Pk = tahmin(net_kat, [t_yol[i] for i in sizan], len(sinif_ad),
                    "kategori")
        pc = Pk[:, c_idx].sum(axis=1) if c_idx else np.zeros(len(sizan))
        yaz("  Kategori modeli bu sizan C'lerin kacini veto ederdi:")
        for e in ESIKLER:
            yaz(f"    esik {e:.2f}: {int((pc >= e).sum())}/{len(sizan)}")

    # ==================================================================
    # BOLUM 2 - dogrulanmis gun uzerinde kalibrasyon
    # ==================================================================
    yaz(f"\n{C}\nBOLUM 2 - VETO KALIBRASYONU (dogrulanmis gun, A adaylari: model_ab3)\n{C}")
    g_yol, g_tur = [], []           # tur: A / B / C
    for d in sorted([x for x in dog.iterdir() if x.is_dir()]):
        tur = ("A" if A_ANAHTAR in d.name else
               "C" if any(k in d.name for k in C_ANAHTARLAR) else "B")
        for f in gorseller(d):
            g_yol.append(f); g_tur.append(tur)
    for d in sorted([x for x in hat.iterdir() if x.is_dir()]):
        n_ = normal(d.name)
        tur = ("A" if "a sinif" in n_ else
               "C" if ("gizli" in n_ or "catlak" in n_ or "plaka" in n_)
               else "B")
        for f in gorseller(d):
            g_yol.append(f); g_tur.append(tur)
    g_tur = np.array(g_tur)
    yaz(f"Gorsel: {len(g_yol)}  (A {int((g_tur=='A').sum())}, "
        f"B {int((g_tur=='B').sum())}, C {int((g_tur=='C').sum())})")

    net_ab3, _ = model_yukle(ab_kal_yol, 2)
    Pg = tahmin(net_ab3, g_yol, 2, "A/B gun ")[:, 0]
    siraG = np.argsort(-Pg)
    NG = max(1, int(len(g_yol) * A_ORANI))
    adaylar = siraG[:NG]
    aday_set = set(adaylar.tolist())
    gercek_c = np.where(g_tur == "C")[0]

    ilgili = sorted(aday_set | set(gercek_c.tolist()))
    Pk = tahmin(net_kat, [g_yol[i] for i in ilgili], len(sinif_ad), "kategori")
    pc_map = {i: (float(Pk[j, c_idx].sum()) if c_idx else 0.0)
              for j, i in enumerate(ilgili)}

    aday_A = [i for i in adaylar if g_tur[i] == "A"]
    aday_C = [i for i in adaylar if g_tur[i] == "C"]
    yaz(f"\nA adaylari (ilk %{100*A_ORANI:.1f} = {NG}): "
        f"gercek A {len(aday_A)}, B {int(sum(g_tur[i]=='B' for i in adaylar))}, "
        f"C {len(aday_C)}")
    for i in aday_C:
        f = g_yol[i]
        shutil.copy2(f, CIKTI / "2_gunluk_sizan_C" / f"a{Pg[i]:.3f}_{f.name}")

    yaz(f"\n  {'esik':>6} {'yakalanan C (tum gun)':>22} "
        f"{'sizan C kalan':>14} {'bosa giden A':>13}")
    yaz("  " + "-" * 60)
    for e in ESIKLER:
        yak = sum(pc_map[i] >= e for i in gercek_c)
        kalan = sum(pc_map[i] < e for i in aday_C)
        bosa = sum(pc_map[i] >= e for i in aday_A)
        yaz(f"  {e:>6.2f} {yak:>10}/{len(gercek_c):<11} "
            f"{kalan:>14} {bosa:>13}")

    yaz(f"\nOkuma:")
    yaz("  yakalanan C  : yuksek olsun (veto catlaklari gorebiliyor mu)")
    yaz("  sizan C kalan: vetodan sonra A'ya gecen catlak, 0 olmali")
    yaz("  bosa giden A : gunde kac gercek A hucresi vetoya takiliyor (bedel)")

    (CIKTI / "rapor.txt").write_text("\n".join(rapor), encoding="utf-8")
    yaz(f"\n{C}")
    yaz(f"Klasor: {CIKTI.resolve()}")
    yaz("  1_test_sizan_C   : gozle bakin - gercek catlak mi, yanlis etiket mi?")
    yaz("  2_gunluk_sizan_C : dogrulanmis gunde A adayina giren C'ler")
    yaz(f"  rapor.txt        : bu cikti")
    yaz(C)


if __name__ == "__main__":
    main()
