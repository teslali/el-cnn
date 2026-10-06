#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALTIN KUME OLCUMU

Dogru etiket = senin kararlarin:
  cikti_altin/<kategori>/   -> onayladiklarin (klasor adi = etiket)
  ~/Desktop/altin_hatalar/  -> duzelttiklerin (klasor adi = gercek etiket)
  adinda "emin" gecen klasor -> belirsiz, olcume katilmaz

Karsilastirilanlar:
  model    : cikti_altin/_kayit.csv (gunluk.py'nin ilk karari)
  operator : altin_kume.csv (fabrikadaki etiket)

Olcumler:
  1. genel dogruluk (model / operator)
  2. kategori bazinda isabet ve yakalama
  3. A KESIMI: sabit %18.7 ne yapti, her esikte isabet ve kurtarilan A
  4. gercek C hucreleri nereye gitti
  5. hat bazinda model, vardiya bazinda operator
  6. (--c-dedektor) C dedektoru 512 px'te altin kumede

Kullanim:
  python altin_olcum.py
  python altin_olcum.py --c-dedektor        (+~15 dk)
"""

import sys
import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

CIKTI_KL = Path("cikti_altin")
ALTIN_KL = Path("altin_kume")
ALTIN_CSV = Path("altin_kume.csv")
HATALAR = Path.home() / "Desktop" / "altin_hatalar"
C_MODEL = Path("model_c/en_iyi_model.pt")
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}

A = "其他"
C_KEYS = {"隐裂", "压板"}
ANAHTARLAR = ["舟框印", "舟划", "舟印", "其他", "黑斑", "黑边", "偏移", "隐裂",
              "压板", "气流", "吸笔", "吸盘", "云雾", "断栅", "接触", "脏污",
              "分片", "不规则", "卡点", "皮带", "滚轮", "暗片"]
AD = {"其他": "A sinifi (Diger)", "黑斑": "siyah nokta", "黑边": "siyah kenar",
      "偏移": "baski kaymasi", "隐裂": "gizli catlak", "压板": "baski plakasi",
      "舟划": "grafit tekne", "舟框印": "tekne cercevesi", "舟印": "kuvars tekne",
      "气流": "hava akisi", "吸笔": "vakum kalemi", "吸盘": "vakum emici",
      "云雾": "bulutlu", "断栅": "kirik grid", "接触": "temas kirl.",
      "脏污": "kirlenme", "分片": "ayirma cizik", "不规则": "duzensiz cizik",
      "卡点": "nokta baskisi", "皮带": "konveyor", "滚轮": "rulo izi",
      "暗片": "koyu hucre", "C": "C supheli"}
TURKCE = [("a sinif", "其他"), ("diger", "其他"), ("nokta bask", "卡点"),
          ("siyah nokta", "黑斑"), ("siyah leke", "黑斑"), ("siyah kenar", "黑边"),
          ("siyah kose", "黑边"), ("baski kay", "偏移"), ("baski plaka", "压板"),
          ("gizli", "隐裂"), ("catlak", "隐裂"), ("grafit", "舟划"), ("grafik", "舟划"),
          ("kuvars", "舟印"), ("cerceve", "舟框印"), ("hava", "气流"), ("kalem", "吸笔"),
          ("emici", "吸盘"), ("bulut", "云雾"), ("grid", "断栅"), ("temas", "接触"),
          ("kirlen", "脏污"), ("ayirma", "分片"), ("duzensiz", "不规则"),
          ("cizik", "不规则"), ("konveyor", "皮带"), ("rulo", "滚轮"), ("koyu", "暗片")]
C = "=" * 72


def normal(s):
    return " ".join(s.translate(str.maketrans("ıİşŞçÇğĞöÖüÜ", "iIsScCgGoOuU"))
                    .lower().replace("_", " ").split())


def anahtar(ad):
    if "c supheli" in normal(ad):
        return "C"
    for k in ANAHTARLAR:
        if k in ad:
            return k
    n = normal(ad)
    for ifade, k in TURKCE:
        if ifade in n:
            return k
    return None


def ad(k):
    return AD.get(k, str(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-dedektor", action="store_true")
    ap.add_argument("--kume", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"], default="1",
                    help="1 = 9.22, 2 = 9.24, 3 = 9.23, 4 = 9.25, 5 = 9.26, 6 = 9.27, "
                         "7 = 9.25'in dokunulmamis kismi, 8 = yeni gun, 9 = M vardiyasi orneklemi")
    ap.add_argument("--kayit", default=None,
                    help="model kararlari (farkli bir gunluk.py ciktisi olabilir)")
    a = ap.parse_args()
    global CIKTI_KL, ALTIN_KL, ALTIN_CSV, HATALAR
    if a.kume == "2":
        CIKTI_KL = Path("cikti_altin_924")
        ALTIN_KL = Path("altin_kume2")
        ALTIN_CSV = Path("altin_kume2.csv")
        HATALAR = Path.home() / "Desktop" / "altin2_hatalar"
    if a.kume == "3":
        CIKTI_KL = Path("cikti_altin_923")
        ALTIN_KL = Path("altin_kume3")
        ALTIN_CSV = Path("altin_kume3.csv")
        HATALAR = Path.home() / "Desktop" / "altin3_hatalar"
    if a.kume == "4":
        CIKTI_KL = Path("cikti_altin_925")
        ALTIN_KL = Path("altin_kume4")
        ALTIN_CSV = Path("altin_kume4.csv")
        HATALAR = Path.home() / "Desktop" / "altin4_hatalar"
    if a.kume in ("5", "6"):
        gun_ = {"5": "926", "6": "927"}[a.kume]
        CIKTI_KL = Path(f"cikti_altin_{gun_}")
        ALTIN_KL = Path(f"altin_kume{a.kume}")
        ALTIN_CSV = Path(f"altin_kume{a.kume}.csv")
        HATALAR = Path.home() / "Desktop" / f"altin{a.kume}_hatalar"
    if a.kume == "9":
        CIKTI_KL = Path("cikti_altin_m")
        ALTIN_KL = Path("altin_kume9")
        ALTIN_CSV = Path("altin_kume9.csv")
        HATALAR = Path.home() / "Desktop" / "altin9_hatalar"
    if a.kume == "8":
        CIKTI_KL = Path("cikti_altin_yeni")
        ALTIN_KL = Path("altin_kume8")
        ALTIN_CSV = Path("altin_kume8.csv")
        HATALAR = Path.home() / "Desktop" / "altin8_hatalar"
    if a.kume == "7":
        CIKTI_KL = Path("cikti_altin_925k")
        ALTIN_KL = Path("altin_kume7")
        ALTIN_CSV = Path("altin_kume7.csv")
        HATALAR = Path.home() / "Desktop" / "altin7_hatalar"
    if a.kayit is None:
        a.kayit = str(CIKTI_KL / "_kayit.csv")
    print(f"Altin kume {a.kume}: dogru etiketler {CIKTI_KL} + {HATALAR.name}")
    print(f"Model kararlari   : {a.kayit}")
    for p in (Path(a.kayit), ALTIN_CSV):
        if not p.exists():
            print(f"HATA: bulunamadi: {p}"); sys.exit(1)

    # ---------------- altin etiketler
    altin, belirsiz, taninmayan = {}, set(), Counter()
    kaynaklar = [(CIKTI_KL, False)]
    if HATALAR.is_dir():
        kaynaklar.append((HATALAR, True))
    for kok, duzeltme in kaynaklar:
        for d in kok.iterdir():
            if not d.is_dir() or d.name.startswith("_"):
                continue
            for f in d.rglob("*"):
                if not (f.is_file() and f.suffix.lower() in UZANTILAR):
                    continue
                if "emin" in normal(d.name):
                    belirsiz.add(f.name); continue
                k = anahtar(d.name)
                if k is None:
                    taninmayan[d.name] += 1; continue
                altin[f.name] = k

    model, a_olas, a_aday = {}, {}, {}
    for r in csv.DictReader(open(a.kayit, encoding="utf-8-sig")):
        model[r["dosya"]] = anahtar(r["klasor"])
        a_olas[r["dosya"]] = float(r.get("A_olasiligi") or 0)
        a_aday[r["dosya"]] = r.get("A_adayi_mi") == "evet"
    op, vard, hat = {}, {}, {}
    for r in csv.DictReader(open(ALTIN_CSV, encoding="utf-8-sig")):
        n = Path(r["yol"]).name
        op[n], vard[n], hat[n] = anahtar(r["kategori"]), r["vardiya"], r["hat"]

    ortak = [d for d in altin if d in model and d in op]
    orn = CIKTI_KL / "_ornekleme.json"
    orn_not = ""
    if orn.exists():
        import json
        oj = json.loads(orn.read_text(encoding="utf-8"))
        w = max(1, int(round(oj["agirlik"])))
        ok_ = anahtar(oj["klasor"])
        # orneklem, INCELENEN ciktinin siyah nokta klasorunden yapildi; agirlik
        # --kayit ile baska bir cikti olculse bile o klasore gore verilir
        taban = {r["dosya"]: anahtar(r["klasor"]) for r in
                 csv.DictReader(open(CIKTI_KL / "_kayit.csv", encoding="utf-8-sig"))}
        ortak = [d for d in ortak for _ in range(w if taban.get(d) == ok_ else 1)]
        orn_not = (f"  ORNEKLEME: '{oj['klasor']}' klasorunun {oj['kalan']}/{oj['toplam']} "
                   f"gorseli incelendi, x{w} agirlikla sayildi (sayilar tahmini)")
    print(f"{C}\nALTIN KUME OLCUMU\n{C}")
    print(f"altin etiketli : {len(altin)}   belirsiz: {len(belirsiz)}   "
          f"olcume giren: {len(ortak)}")
    if orn_not:
        print(orn_not)
    if taninmayan:
        print(f"  klasor adi taninmayan (olcum disi): {dict(taninmayan)}")
    if not ortak:
        print("HATA: eslesen gorsel yok."); sys.exit(1)

    g = np.array([altin[d] for d in ortak], dtype=object)
    m = np.array([model[d] for d in ortak], dtype=object)
    # "C supheli" klasoru catlak/plaka ayirmaz: gercek C ise dogru sayilir
    m = np.array([gi if (mi == "C" and gi in C_KEYS) else mi
                  for gi, mi in zip(np.array([altin[d] for d in ortak]), m)],
                 dtype=object)
    o = np.array([op[d] for d in ortak], dtype=object)

    # ---------------- 1. genel
    print(f"\n{C}\n1. GENEL DOGRULUK\n{C}")
    print(f"  model    : %{100*(g == m).mean():.1f}")
    print(f"  operator : %{100*(g == o).mean():.1f}")
    print("  (ayni kategori; B ici karisiklik da hata sayilir)")
    gA, mA, oA = g == A, m == A, o == A
    print(f"\n  A/B karari (asil onemli olan):")
    print(f"    model    : %{100*(gA == mA).mean():.1f}")
    print(f"    operator : %{100*(gA == oA).mean():.1f}")

    # ---------------- 2. kategori
    print(f"\n{C}\n2. KATEGORI BAZINDA  (isabet = o klasore gidenin dogru orani,\n"
          f"   yakalama = gercek orneklerin o klasore giden orani)\n{C}")
    print(f"  {'kategori':<18} {'gercek':>6} | {'model isb':>9} {'yak':>5} | "
          f"{'oper isb':>8} {'yak':>5}")
    for k, n in Counter(g).most_common():
        def pr(t):
            sec = t == k
            isb = (g[sec] == k).mean() if sec.any() else float("nan")
            yak = (t[g == k] == k).mean()
            return isb, yak
        mi, my = pr(m); oi, oy = pr(o)
        f = lambda x: "  -" if np.isnan(x) else f"%{100*x:.0f}"
        print(f"  {ad(k):<18} {n:>6} | {f(mi):>9} {f(my):>5} | {f(oi):>8} {f(oy):>5}")

    # ---------------- 3. A kesimi
    print(f"\n{C}\n3. A KESIMI\n{C}")
    gercek_a = int(gA.sum())
    print(f"  altin kumede gercek A: {gercek_a} (%{100*gercek_a/len(ortak):.1f})")
    print(f"  (ornekleme gunun tum C'lerini icerdigi icin gunluk oran biraz daha yuksek)")
    print(f"  model A klasoru: {int(mA.sum())}, dogru {int((mA & gA).sum())} "
          f"-> isabet %{100*(mA & gA).sum()/max(mA.sum(),1):.0f}, "
          f"A'larin %{100*(mA & gA).sum()/max(gercek_a,1):.0f}'i kurtarildi")
    print(f"  operator      : A dedigi {int(oA.sum())}, dogru {int((oA & gA).sum())} "
          f"-> isabet %{100*(oA & gA).sum()/max(oA.sum(),1):.0f}, "
          f"A'larin %{100*(oA & gA).sum()/max(gercek_a,1):.0f}'i")
    p = np.array([a_olas[d] for d in ortak])
    print(f"\n  Olasilik esigine gore (vetosuz, A/B modelinin ham sirasi):")
    print(f"  {'esik':>6} {'secilen':>8} {'dogru A':>8} {'isabet':>7} {'kurtarilan':>11}")
    for e in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        sec = p >= e
        d_ = int((sec & gA).sum())
        print(f"  {e:>6.2f} {int(sec.sum()):>8} {d_:>8} "
              f"{('%'+str(round(100*d_/sec.sum()))) if sec.sum() else '-':>7} "
              f"{'%'+str(round(100*d_/max(gercek_a,1))):>11}")

    # ---------------- 4. C
    gC = np.isin(g, list(C_KEYS) + ["C"])      # "C" = C_supheli klasorunde onaylananlar
    print(f"\n{C}\n4. GERCEK C HUCRELERI ({int(gC.sum())}) - model nereye koydu\n{C}")
    for k, n in Counter(m[gC]).most_common():
        uyari = "   <-- A'YA SIZINTI" if k == A else ""
        print(f"  {n:>5}  {ad(k)}{uyari}")

    # ---------------- 5. hat / vardiya
    print(f"\n{C}\n5a. HAT BAZINDA MODEL\n{C}")
    h_ = np.array([hat[d] for d in ortak]); v_ = np.array([vard[d] for d in ortak])
    for h in sorted(set(h_)):
        s = h_ == h
        print(f"  {h:<4} n={int(s.sum()):>4}  kategori %{100*(g[s]==m[s]).mean():.0f}  "
              f"A/B %{100*(gA[s]==mA[s]).mean():.0f}")
    print(f"\n{C}\n5b. VARDIYA BAZINDA OPERATOR\n{C}")
    print(f"  {'vardiya':<8} {'n':>5} {'kategori':>9} {'A/B':>6} "
          f"{'kacirilan A':>12} {'yanlis A':>9}")
    for v in "DMN":
        s = v_ == v
        if not s.any():
            continue
        kac = int((s & gA & ~oA).sum()); yan = int((s & ~gA & oA).sum())
        print(f"  {v:<8} {int(s.sum()):>5} {'%'+str(round(100*(g[s]==o[s]).mean())):>9} "
              f"{'%'+str(round(100*(gA[s]==oA[s]).mean())):>6} {kac:>12} {yan:>9}")
    print("  kacirilan A: senin A dedigin, operatorun B dedigi")
    print("  yanlis A   : senin B dedigin, operatorun A dedigi")

    # ---------------- 5c. karisiklik ciftleri
    print(f"\n{C}\n5c. EN SIK KARISIKLIKLAR (gercek -> modelin dedigi)\n{C}")
    hata = Counter((gi, mi) for gi, mi in zip(g, m) if gi != mi)
    top_hata = sum(hata.values())
    for (gi, mi), n in hata.most_common(15):
        print(f"  {n:>5}  %{100*n/max(top_hata,1):>4.1f}   {ad(gi):<18} -> {ad(mi)}")
    print(f"  toplam hata: {top_hata}  (orneklenen klasor varsa agirlikli, tahmini)")

    # ---------------- 6. C dedektoru
    if a.c_dedektor:
        try:
            import torch, torch.nn as nn
            from torch.utils.data import Dataset, DataLoader
            from torchvision import models, transforms
            from PIL import Image
        except ImportError as e:
            print(f"C dedektoru atlandi: {e}"); return
        if not C_MODEL.exists():
            print(f"C dedektoru atlandi: {C_MODEL} yok"); return
        torch.set_num_threads(4)
        ck = torch.load(C_MODEL, map_location="cpu")
        net = models.resnet18(weights=None); net.fc = nn.Linear(512, 2)
        net.load_state_dict(ck["model"]); net.eval()
        don = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        class K(Dataset):
            def __len__(s): return len(ortak)
            def __getitem__(s, i):
                try:
                    with Image.open(ALTIN_KL / ortak[i]) as im:
                        return don(im.convert("RGB")), i
                except Exception:
                    return torch.zeros(3, 512, 512), i
        pc = np.zeros(len(ortak)); n_ = 0
        print(f"\n  C dedektoru calisiyor (512 px)...")
        with torch.no_grad():
            for x, idx in DataLoader(K(), batch_size=8, num_workers=0):
                pc[idx.numpy()] = torch.softmax(net(x), 1)[:, 1].numpy()
                n_ += len(idx); print(f"\r  {n_}/{len(ortak)}", end="", flush=True)
        print()
        aday = np.array([a_aday[d] for d in ortak])
        plaka = g == "压板"
        print(f"\n{C}\n6. C DEDEKTORU (altin kume, hic gormedigi gun)\n{C}")
        print(f"  {'esik':>6} {'C yakalandi':>12} {'plaka':>8} {'A adayinda':>11} "
              f"{'bosa A':>7} {'isaretli B':>11}")
        print(f"  {'':>6} {'':>12} {'':>8} {'kalan C':>11}")
        for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
            s = pc >= e
            print(f"  {e:>6.2f} {int((s&gC).sum()):>6}/{int(gC.sum()):<5} "
                  f"{int((s&plaka).sum()):>4}/{int(plaka.sum()):<3} "
                  f"{int((aday&gC&~s).sum()):>11} {int((aday&gA&s).sum()):>7} "
                  f"{int((~gA&~gC&s).sum()):>11}")
        print("  A adayinda kalan C: vetodan sonra A'ya gecen C (hedef 0)")
        print("  bosa A           : veto edilen gercek A (bedel)")
    print(C)


if __name__ == "__main__":
    main()
