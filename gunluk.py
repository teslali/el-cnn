#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL HUCRE AYIKLAMA - GUNLUK CALISTIRMA (A-oncelikli akis)

YENI AKIS (eskisinden farkli)
  1) model_ab4 (yoksa ab3) TUM gorselleri tarar, A olasiligina gore siralar
  2) En yuksek %N -> "Diger" klasorune gider, biter
  3) Kalanlar model3'ten gecer, kategori klasorlerine ayrilir

  Eski akis once kategori ayirip sadece siyah nokta+kenar'a A/B uyguluyordu
  (5360 gorsel). Yeni akis 7400 gorselin TAMAMINI tariyor.

OLCULEN SONUCLAR
  model_ab3: dogrulama AUC 0.9653, test AUC 0.9600
  C VETOSU: A adaylari kategori modelinden de gecer; gizli catlak
  olasiligi >= 0.05 ise A adayligi iptal. (Baski plakasini henuz goremez.)
  Gozle dogrulama: 1384 gorselde 56 hata -> %96 precision
  Kesim orani: 7400 gorselde 1384 -> %18.7

CIKTI
  Tek klasor, icinde kategori klasorleri. Fabrikadaki 2B yapisiyla ayni.

Kullanim:
    python gunluk.py /gorsellerin/oldugu/klasor
    python gunluk.py /klasor --cikti sonuc
    python gunluk.py /klasor --a-esik 0.5        (daha temkinli: daha az ama daha kesin A)
    python gunluk.py /klasor --tasi              (DIKKAT: kopyalamaz, tasir)
"""

import sys
import csv
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
    print('  pip install torch==2.2.2 torchvision==0.17.2 pillow "numpy<2"')
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
AB_MODEL = (Path("model_ab4") / "en_iyi_model.pt"
            if (Path("model_ab4") / "en_iyi_model.pt").exists()
            else Path("model_ab3") / "en_iyi_model.pt")
KATEGORI_MODEL = Path("model3") / "en_iyi_model.pt"
# 384 px kategori modeli: varsa model3 ile BIRLESTIRILIR (olasilik ortalamasi).
# Altin kume: genel %87.6 -> %88.3; cizik isabeti %50 -> %63,
# kirik grid yakalama %20 -> %40, baski kaymasi yak. %74 -> %85, siyah nokta korunur.
# model_kat5 varsa o kullanilir (altin kume: birlesim %88.3 -> %89.3), yoksa kat4
KAT4_MODEL = next((p for p in [Path("model_kat5") / "en_iyi_model.pt",
                               Path("model_kat4") / "en_iyi_model.pt"] if p.exists()),
                  Path("model_kat4") / "en_iyi_model.pt")
# kat4 egitimi ornekleri 1/sqrt(n) ile dengeledi; tahminde sqrt(n) ile geri alinir
KAT4_EGITIM = {"不规则": 3308, "云雾": 1080, "偏移": 3046, "其他": 4162, "分片": 1425,
               "卡点": 601, "压板": 199, "吸笔": 888, "接触": 461, "断栅": 1239,
               "气流": 2212, "脏污": 1525, "舟划": 1002, "舟框印": 995, "隐裂": 791,
               "黑斑": 6444, "黑边": 3778}
# KIRLENME KURALI: kirlenme karari kat7'ye (kirlenme odakli ince ayar) birakilir.
# 9.25 geri kalani: genel %93.5 -> %94.3, kirlenme yakalama %34 -> %82,
# cizik isabeti %74 -> %86; A klasoru ve diger kategoriler degismez.
KAT7_MODEL = Path("model_kat7") / "tur2.pt"
KIRLENME_ANAHTAR = "脏污"
# UZMAN MODEL (kat8): kat7 + yeni cizik tanimlari ("yeni tanim": eskiden kirik
# grid, "yeni tur": ilk kez gorulen cizik). Grup kurali: birlesim ve kat8 ikisi de
# karisan gruptaysa son sozu kat8 soyler; kat8 kirlenme derse kirlenme.
# Sinav: yeni tipler cizik olarak %6 -> %66; 9.25 ve 10.01 genel ~-0.2 puan.
# kat8 yoksa kat7 ile sadece kirlenme kurali uygulanir.
KAT8_MODEL = Path("model_kat8") / "tur2.pt"
GRUP_ANAHTARLAR = {"不规则", "断栅", "偏移", "脏污", "气流"}
KAT_ANAHTARLAR = ["舟框印", "舟划", "舟印", "其他", "黑斑", "黑边", "偏移", "隐裂",
                  "压板", "气流", "吸笔", "吸盘", "云雾", "断栅", "接触", "脏污",
                  "分片", "不规则", "卡点", "皮带", "滚轮", "暗片"]


def kat_anahtar(ad):
    return next((k for k in KAT_ANAHTARLAR if k in ad), ad)
# 512 px C dedektoru: v2 (tur 3) varsa o, yoksa v1
C_MODEL = next((p for p in [Path("model_c2") / "tur3.pt",
                            Path("model_c") / "en_iyi_model.pt"] if p.exists()),
               Path("model_c") / "en_iyi_model.pt")

GIRDI_BOYUT = 224
BATCH = 24
ISCI = 2
UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}

A_ANAHTAR = "其他"          # Diger = A sinifi klasoru
A_ORANI = 0.187             # ESKI: sabit oran (artik varsayilan degil)
# A kesimi OLASILIK ESIGIYLE: A klasorunun buyuklugu gunun gercek A oranina uyar.
# Altin kume (9.22, vetosuz): esik 0.4 -> isabet %84, A'larin %73'u;
#                             esik 0.3 -> isabet %80, A'larin %84'u.
# Sabit oran, gercek A orani dusuk gunlerde A klasorunu B ile dolduruyordu.
A_ESIK = 0.40

# IKINCI GORUS: A adayini 384 px kategori birlesimi de onaylamali.
# A/B modeli 224 px'te ince cizikleri goremiyor. Altin kumeler (esik 0.20):
#   9.22: A isabeti %87 -> %94 (22 B cikti, 24 A kaybedildi)
#   9.24: A isabeti %34 -> %52 (52 B cikti,  4 A kaybedildi)
# Butun yanlis yerlesimler esit agirlikta (surec takibi): toplam hatayi en aza
# indiren esik. (9.22 + 9.24 toplam hata: 0.05'te 108, 0.20'de 82;
# 9.26'da A klasoru isabeti %77 -> %89.)
IKINCI_ESIK = 0.20

# C VETOSU: A adayinda kategori modelinin C olasiligi bu esigi gecerse
# A adayligi iptal, C klasorune gider.
# Kalibrasyon (dogrulanmis gun): esik 0.05 -> sizan catlak 0, bosa giden A 8.
# NOT: baski plakasi kategori modelinde yok; veto onu GOREMEZ.
C_ANAHTARLAR = {"隐裂", "压板"}
VETO_ESIK = 0.05

# C DEDEKTORU (512 px): tum gorsellere uygulanir, esigi gecen her hucre
# C_KLASOR'e gider (A adayi olsa bile). Altin kume (9.22, hic gormedigi gun):
#   esik 0.30 -> 93 C'nin 92'si, 25 baski plakasinin 25'i yakalandi;
#   A adayinda kalan C 1, bosa giden A 2, isaretlenen B %1.7
# v2, iki kademeli esik (uc altin gun, 366 gercek C):
#   A adaylarinda 0.05 -> A adaylari arasindaki 31 C'nin 29'u, feda edilen A 5/345
#   digerlerinde 0.15 -> 359/366 C, yanlis alarm v1'in ~uctebiri
#   A'ya sizan C: v1'de 6, v2'de 2
# (v1 kullaniliyorsa eski tek esik 0.30 gecerli)
# GUNCELLEME (tatli nokta, 9.26 + 9.27, 312 gercek C): sonuclar paketleme icin
# degil surec takibi icin kullaniliyor; hedef toplam yanlis yerlesimi azaltmak,
# gunde 1-4 kacan C kabul. 0.20 / 0.25: C klasoru 362 -> 242, kacan C ~3.3/gun.
C_ESIK = 0.25
C_ESIK_ADAY = 0.20
C_ESIK_V1 = 0.30
C_BOYUT = 512
C_KLASOR = "C_supheli (gizli catlak - baski plakasi)"   # --c-tek ile tek klasor
# C klasoru uce bolunur (operatorun isi hizlansin diye; hucre sayisi degismez):
#   1_yuksek : dedektor C_YUKSEK_ESIK ustu -> buyuk kismi gercek C, hizli onay
#   2_dusuk  : A adayi degil -> cogu C degil AMA gercek C'lerin ~1/3'u burada, dikkatli bak
#   3_yeniden: A/B modeli A demisti, en ufak catlak suphesiyle C'ye alindi;
#              icinde A, B ve az sayida catlak var -> operator yeniden degerlendirir
C_YUKSEK = "C_supheli_1 隐裂压板疑似-高 (buyuk ihtimalle C)"
C_DUSUK = "C_supheli_2 隐裂压板疑似-低 (dusuk supheli)"
C_ADAY = "C_supheli_3 隐裂压板疑似-需复判 (yeniden degerlendir)"
C_KLASORLER = {C_KLASOR, C_YUKSEK, C_DUSUK, C_ADAY}
# 9.27: 0.90 ustu 84 hucrenin 63'u gercek C (%75); 0.70-0.90 araliginda %30
C_YUKSEK_ESIK = 0.90

C = "=" * 72
torch.set_num_threads(4)


def guvenli(ad):
    for ch in '/\\:*?"<>|':
        ad = ad.replace(ch, "-")
    return ad.strip()[:80] or "bilinmeyen"


def kisa(ad, n=34):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


class Kume(Dataset):
    def __init__(self, yollar, donusum, boyut=GIRDI_BOYUT):
        self.yollar, self.donusum, self.boyut = yollar, donusum, boyut

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        try:
            with Image.open(self.yollar[i]) as im:
                return self.donusum(im.convert("RGB")), i
        except Exception:
            return torch.zeros(3, self.boyut, self.boyut), -1


def donusum(boyut=GIRDI_BOYUT):
    return transforms.Compose([
        transforms.Resize((boyut, boyut)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------- dil (BASLAT.bat icin): en / zh secilirse ekranda sadece
# sade bir ilerleme ve sonda ozet gorunur; ayrintilar cikti klasorundeki
# _islem_gunlugu.txt dosyasina yazilir.
DIL = "tr"
KONSOL = None
ADIM = {
    "A/B": {"en": "Step 1/5  A/B screening", "zh": "第1/5步  A/B判定"},
    "kategori": {"en": "Step 2/5  Defect category (1/2)", "zh": "第2/5步  缺陷分类 (1/2)"},
    "kategori 384": {"en": "Step 3/5  Defect category (2/2)", "zh": "第3/5步  缺陷分类 (2/2)"},
    "kirlenme (kat7)": {"en": "Step 4/5  Scratch / contamination check", "zh": "第4/5步  划痕/脏污检查"},
    "C dedektor": {"en": "Step 5/5  Crack detector", "zh": "第5/5步  隐裂检测"},
}


def konsol(metin, satir_sonu="\n"):
    if KONSOL is not None:
        KONSOL.write(metin + satir_sonu)
        KONSOL.flush()


def tahmin(net, dosyalar, cikti_sayisi, etiket, boyut=GIRDI_BOYUT):
    yl = DataLoader(Kume(dosyalar, donusum(boyut), boyut),
                    batch_size=BATCH if boyut <= 224 else 8,
                    shuffle=False, num_workers=ISCI)
    P = np.zeros((len(dosyalar), cikti_sayisi), dtype=np.float32)
    okunamayan = 0
    n, t0 = 0, time.time()
    with torch.no_grad():
        for x, idx in yl:
            p = torch.softmax(net(x), 1).numpy()
            for j, i in enumerate(idx.tolist()):
                if i >= 0:
                    P[i] = p[j]
                else:
                    okunamayan += 1
            n += len(idx)
            hiz = n / max(time.time() - t0, 1e-9)
            print(f"\r  {etiket}: {n}/{len(dosyalar)}  ({hiz:.1f} g/sn)  "
                  f"kalan ~{(len(dosyalar)-n)/max(hiz,1e-9)/60:.1f} dk",
                  end="", flush=True)
            if DIL != "tr":
                kalan = (len(dosyalar) - n) / max(hiz, 1e-9) / 60
                ad = ADIM.get(etiket.strip(), {}).get(DIL, etiket.strip())
                ek = (f"~{kalan:.0f} min left" if DIL == "en" else f"剩余约 {kalan:.0f} 分钟")
                konsol(f"\r  {ad}: {n}/{len(dosyalar)}   {ek}      ", "")
    if DIL != "tr":
        konsol("   OK" if DIL == "en" else "   完成")
    sure = time.time() - t0
    print(f"   bitti ({sure/60:.1f} dk, {len(dosyalar)/max(sure,1e-9):.1f} g/sn)")
    return P, okunamayan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--cikti", default="cikti")
    ap.add_argument("--dil", choices=["tr", "en", "zh"], default="tr",
                    help="en/zh: sade ekran ciktisi (BASLAT.bat icin)")
    ap.add_argument("--a-esik", type=float, default=A_ESIK,
                    help="A olasiligi bu esigin ustundeyse A adayi")
    ap.add_argument("--a-orani", type=float, default=None,
                    help="ESKI yontem: en ustteki bu orani A say (verilirse esik yerine)")
    ap.add_argument("--tasi", action="store_true")
    ap.add_argument("--veto-esik", type=float, default=VETO_ESIK)
    ap.add_argument("--kirlenme-kurali-kapali", action="store_true",
                    help="kirlenme kararini kat7'ye birakma (eski davranis)")
    ap.add_argument("--c-tek", action="store_true",
                    help="C adaylarini eskisi gibi tek klasore koy")
    ap.add_argument("--c-yuksek", type=float, default=C_YUKSEK_ESIK,
                    help="bu olasiligin ustu 'yuksek' C klasorune")
    ap.add_argument("--c-esik", type=float, default=None,
                    help="A adayi olmayan hucrelerde C esigi")
    ap.add_argument("--c-esik-aday", type=float, default=None,
                    help="A adaylarinda C esigi (daha dusuk, sizmaya karsi)")
    ap.add_argument("--ikinci-esik", type=float, default=IKINCI_ESIK,
                    help="kategori birlesiminin A olasiligi bunun altindaysa A iptal (0 = kapali)")
    ap.add_argument("--c-sadece-aday", action="store_true",
                    help="C dedektorunu sadece A adaylarina uygula (hizli)")
    a = ap.parse_args()
    global DIL, KONSOL
    DIL = a.dil
    gunluk_dosyasi = None
    if DIL != "tr":
        Path(a.cikti).expanduser().mkdir(parents=True, exist_ok=True)
        gunluk_dosyasi = open(Path(a.cikti).expanduser() / "_islem_gunlugu.txt", "w",
                              encoding="utf-8")
        KONSOL = sys.__stdout__
        sys.stdout = gunluk_dosyasi

    kok = Path(a.klasor).expanduser().resolve()
    if not kok.is_dir():
        print(f"HATA: klasor yok: {kok}")
        konsol(f"Folder not found: {kok}" if DIL == "en" else f"找不到文件夹：{kok}")
        sys.exit(1)

    for m in (AB_MODEL, KATEGORI_MODEL):
        if not m.exists():
            print(f"HATA: model yok: {m}")
            konsol(f"Model file missing: {m}" if DIL == "en" else f"缺少模型文件：{m}")
            sys.exit(1)

    konsol("Loading models (the first start can take a few minutes)..."
           if DIL == "en" else "正在加载模型（首次启动可能需要几分钟）...")

    print(f"{C}\nEL HUCRE AYIKLAMA\n{C}")

    # ---------------- modeller
    ck1 = torch.load(AB_MODEL, map_location="cpu")
    net1 = models.resnet18(weights=None)
    net1.fc = nn.Linear(512, 2)
    net1.load_state_dict(ck1["model"])
    net1.eval()

    ck2 = torch.load(KATEGORI_MODEL, map_location="cpu")
    sinif_ad = ck2["sinif_ad"]
    net2 = models.resnet18(weights=None)
    net2.fc = nn.Linear(512, len(sinif_ad))
    net2.load_state_dict(ck2["model"])
    net2.eval()

    a_klasor = next((c for c in sinif_ad if A_ANAHTAR in c), "zz其他Diger")

    print(f"A/B modeli      : AUC {ck1.get('dog_auc', 0):.4f} "
          f"(tur {ck1.get('tur','?')})")
    print(f"Kategori modeli : {len(sinif_ad)} kategori "
          f"(tur {ck2.get('tur','?')})")
    print(f"A sinifi klasoru: {kisa(a_klasor)}")
    if a.a_orani:
        print(f"A kesimi       : sabit oran %{100*a.a_orani:.1f} (eski yontem)")
    else:
        print(f"A kesimi       : olasilik esigi {a.a_esik}")

    # ---------------- dosyalar
    dosyalar = sorted([f for f in kok.rglob("*")
                       if f.is_file() and f.suffix.lower() in UZANTILAR])
    if not dosyalar:
        print("HATA: gorsel bulunamadi.")
        konsol("No images found in this folder." if DIL == "en" else "该文件夹中没有找到图片。")
        sys.exit(1)
    print(f"Gorsel          : {len(dosyalar)}\n")
    konsol(f"Images found: {len(dosyalar)}\n" if DIL == "en" else f"找到图片：{len(dosyalar)} 张\n")

    # ---------------- 1. ADIM: A/B taramasi (tum gorseller)
    P1, hatali1 = tahmin(net1, dosyalar, 2, "A/B      ")
    a_olas = P1[:, 0]
    sira = np.argsort(-a_olas)
    if a.a_orani:
        N = max(1, int(len(dosyalar) * a.a_orani))
        a_secilen = set(sira[:N].tolist())
    else:
        a_secilen = {i for i in range(len(dosyalar)) if a_olas[i] >= a.a_esik}
        N = len(a_secilen)

    print(f"\n  A sinifi adayi : {N}  (%{100*N/len(dosyalar):.1f})")
    if N:
        print(f"  olasilik araligi: {a_olas[sira[0]]:.4f} - {a_olas[sira[N-1]]:.4f}")

    # ---------------- 2. ADIM: kategori modeli (tum gorseller)
    print()
    P2, hatali2 = tahmin(net2, dosyalar, len(sinif_ad), "kategori ")
    c_idx = [i for i, c in enumerate(sinif_ad)
             if any(k in c for k in C_ANAHTARLAR)]
    c_olas = P2[:, c_idx].sum(axis=1) if c_idx else np.zeros(len(dosyalar))

    # ---------------- 2b. ADIM: 384 px kategori modeli ile birlestirme
    karar_ad, PK = sinif_ad, P2
    if KAT4_MODEL.exists():
        ck4 = torch.load(KAT4_MODEL, map_location="cpu")
        ad4 = ck4["sinif_ad"]
        net4 = models.resnet18(weights=None)
        net4.fc = nn.Linear(512, len(ad4))
        net4.load_state_dict(ck4["model"])
        net4.eval()
        print()
        P4, _ = tahmin(net4, dosyalar, len(ad4), "kategori 384",
                       boyut=ck4.get("girdi_boyut", 384))
        k4 = [kat_anahtar(x) for x in ad4]
        k3 = [kat_anahtar(x) for x in sinif_ad]
        sayi = ck4.get("egitim_sayi", KAT4_EGITIM)
        us = ck4.get("denge_us", 0.5)
        carpan = np.array([sayi.get(k, 1) ** us for k in k4])
        P4 = P4 * carpan
        P4 = P4 / P4.sum(axis=1, keepdims=True)
        P3h = np.zeros_like(P4)
        for j, k in enumerate(k4):
            if k in k3:
                P3h[:, j] = P2[:, k3.index(k)]
        PK = (P3h + P4) / 2
        karar_ad = ad4
        print(f"  kategori karari: model3 + {KAT4_MODEL.parent.name} birlesimi")
    else:
        print(f"  UYARI: {KAT4_MODEL} yok, sadece model3 kullaniliyor")
    a_klasor = next((c for c in karar_ad if A_ANAHTAR in c), a_klasor)
    kc_idx = [i for i, c in enumerate(karar_ad) if any(k in c for k in C_ANAHTARLAR)]

    # ---------------- 2c. ADIM: uzman model (kat8, yoksa kat7)
    uzman_yol = KAT8_MODEL if KAT8_MODEL.exists() else KAT7_MODEL
    grup_kurali = uzman_yol == KAT8_MODEL
    uzman_tahmin = [None] * len(dosyalar)          # her hucre icin uzmanin anahtari
    ad_of = {kat_anahtar(c): c for c in karar_ad}
    kir_ad = ad_of.get(KIRLENME_ANAHTAR)
    if uzman_yol.exists() and kir_ad and not a.kirlenme_kurali_kapali:
        ck7 = torch.load(uzman_yol, map_location="cpu")
        ad7 = ck7["sinif_ad"]
        net7 = models.resnet18(weights=None)
        net7.fc = nn.Linear(512, len(ad7))
        net7.load_state_dict(ck7["model"])
        net7.eval()
        print()
        P7, _ = tahmin(net7, dosyalar, len(ad7), "kirlenme (kat7)",
                       boyut=ck7.get("girdi_boyut", 384))
        k7 = [kat_anahtar(x) for x in ad7]
        sayi7, us7 = ck7.get("egitim_sayi", {}), ck7.get("denge_us", 0.3)
        P7 = P7 * np.array([sayi7.get(k, 1) ** us7 for k in k7])
        if A_ANAHTAR in k7:
            P7[:, k7.index(A_ANAHTAR)] = -1.0
        uzman_tahmin = [k7[j] for j in P7.argmax(axis=1)]
        print(f"  uzman model: {uzman_yol.parent.name}/{uzman_yol.name}  "
              f"(grup kurali {'acik' if grup_kurali else 'kapali'}), "
              f"kirlenme dedigi: {sum(u == KIRLENME_ANAHTAR for u in uzman_tahmin)}")
    a_idx = next((i for i, c in enumerate(karar_ad) if A_ANAHTAR in c), None)

    # ---------------- 3. ADIM: C dedektoru (512 px)
    cd_olas = np.zeros(len(dosyalar))
    if C_MODEL.exists():
        ck3 = torch.load(C_MODEL, map_location="cpu")
        net3 = models.resnet18(weights=None)
        net3.fc = nn.Linear(512, 2)
        net3.load_state_dict(ck3["model"])
        net3.eval()
        hedef = (sorted(a_secilen) if a.c_sadece_aday
                 else list(range(len(dosyalar))))
        print()
        P3, _ = tahmin(net3, [dosyalar[i] for i in hedef], 2,
                       "C dedektor", boyut=C_BOYUT)
        for j, i in enumerate(hedef):
            cd_olas[i] = P3[j, 1]
    else:
        print(f"\n  UYARI: {C_MODEL} yok, C dedektoru atlandi.")
    v2 = "model_c2" in str(C_MODEL)
    if a.c_esik is None:
        a.c_esik = C_ESIK if v2 else C_ESIK_V1
    if a.c_esik_aday is None:
        a.c_esik_aday = C_ESIK_ADAY if v2 else a.c_esik
    c_supheli = {i for i in range(len(dosyalar))
                 if cd_olas[i] >= (a.c_esik_aday if i in a_secilen else a.c_esik)}
    print(f"  C dedektoru: {C_MODEL.parent.name}/{C_MODEL.name}  "
          f"esik: A adaylarinda {a.c_esik_aday}, digerlerinde {a.c_esik}")

    # ---------------- 4. ADIM: kararlar
    # oncelik: C dedektoru > kategori modeli catlak vetosu > A > kategori
    veto = {i for i in a_secilen - c_supheli if c_olas[i] >= a.veto_esik}
    ikinci = set()
    if a.ikinci_esik > 0 and a_idx is not None and PK is not P2:
        ikinci = {i for i in a_secilen - c_supheli - veto if PK[i, a_idx] < a.ikinci_esik}
    a_kalan = a_secilen - veto - c_supheli - ikinci

    def c_yeri(i):
        if a.c_tek:
            return C_KLASOR
        if cd_olas[i] >= a.c_yuksek:
            return C_YUKSEK
        return C_ADAY if i in a_secilen else C_DUSUK

    kategoriler = []
    for i in range(len(dosyalar)):
        if i in c_supheli:
            kategoriler.append(c_yeri(i))
        elif i in a_kalan:
            kategoriler.append(a_klasor)
        elif i in veto:
            kategoriler.append(c_yeri(i))
        else:
            # A karari sadece A/B modelinindir: aday olmayan gorsel
            # kategori modelinde "Diger" en yuksek cikarsa bile B'de kalir.
            p = PK[i].copy()
            if a_idx is not None:
                p[a_idx] = -1.0
            kat = karar_ad[int(p.argmax())]
            # tum C adaylari tek klasorde: kategori modeli catlak/plaka derse de
            if any(k in kat for k in C_ANAHTARLAR):
                kat = c_yeri(i)
            elif uzman_tahmin[i] is not None:
                uz = uzman_tahmin[i]
                if uz == KIRLENME_ANAHTAR:
                    kat = kir_ad
                elif grup_kurali and kat_anahtar(kat) in GRUP_ANAHTARLAR \
                        and uz in GRUP_ANAHTARLAR and uz in ad_of:
                    kat = ad_of[uz]
            kategoriler.append(kat)

    supheli = [i for i in range(len(dosyalar))
               if i not in a_secilen and c_olas[i] >= a.veto_esik
               and kategoriler[i] not in C_KLASORLER
               and not any(k in kategoriler[i] for k in C_ANAHTARLAR)]

    print(f"\n  C dedektoru: {len(c_supheli)} hucre isaretledi")
    if not a.c_tek:
        for kl in (C_YUKSEK, C_DUSUK, C_ADAY):
            print(f"    {sum(k == kl for k in kategoriler):>5}  {kl}")
    print(f"    bunlarin {len(c_supheli & a_secilen)} tanesi A adayiydi")
    print(f"  Kategori modeli catlak vetosu: {len(veto)} A adayi daha iptal")
    print(f"  Ikinci gorus (esik {a.ikinci_esik}): {len(ikinci)} A adayi iptal "
          f"-> kendi kategorisine")
    print(f"  A sinifina kalan : {len(a_kalan)}")
    print(f"  Bilgi: A adayi olmayip catlak suphesi tasiyan {len(supheli)} "
          f"gorsel kendi kategorisine gitti (olcum icin)")

    # ---------------- okunamayan uyarisi
    if hatali1 or hatali2:
        print(f"\n  UYARI: {max(hatali1, hatali2)} gorsel okunamadi.")
        print("  Bu gorseller yanlis siniflanmis olabilir.")

    # ---------------- kopyala
    cikti = Path(a.cikti).expanduser().resolve()
    cikti.mkdir(parents=True, exist_ok=True)
    islem = shutil.move if a.tasi else shutil.copy2

    print(f"\n{'TASINIYOR' if a.tasi else 'Kopyalaniyor'}: {cikti}")
    if a.tasi:
        print("  DIKKAT: kaynak dosyalar TASINIYOR.")

    sayac = Counter()
    hata = 0
    kayit = []
    for i, f in enumerate(dosyalar):
        kat = kategoriler[i]
        klasor = cikti / guvenli(kat)
        klasor.mkdir(parents=True, exist_ok=True)
        yeni = klasor / f.name
        if yeni.exists():
            yeni = klasor / f"{f.stem}_{sayac[kat]}{f.suffix}"
        try:
            islem(str(f), str(yeni))
            sayac[kat] += 1
        except Exception as e:
            hata += 1
            if hata <= 3:
                print(f"  hata: {f.name} ({e})")
        kayit.append((f.name, kat, f"{a_olas[i]:.4f}", f"{c_olas[i]:.4f}",
                      "evet" if i in a_secilen else "hayir",
                      "evet" if i in veto else "hayir",
                      f"{cd_olas[i]:.4f}"))

    try:
        with open(cikti / "_kayit.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["dosya", "klasor", "A_olasiligi", "C_olasiligi",
                        "A_adayi_mi", "veto", "C_dedektor"])
            w.writerows(kayit)
    except Exception:
        pass

    # ---------------- ozet
    print(f"\n{C}\nSONUC\n{C}")
    print(f"Toplam gorsel : {len(dosyalar)}")
    print(f"Klasor sayisi : {len(sayac)}")
    if hata:
        print(f"Kopyalanamayan: {hata}")
    print(f"\n{'adet':>7}  {'%':>6}  kategori")
    print("-" * 56)
    for kat, n in sayac.most_common():
        isaret = "   <- A sinifi" if kat == a_klasor else ""
        print(f"{n:>7}  {100*n/len(dosyalar):>5.1f}%  {kisa(kat)}{isaret}")

    print(f"\n{C}")
    print(f"Cikti: {cikti}")
    print(C)
    print("\nOperator icin:")
    print(f"  '{kisa(a_klasor)}' klasoru A sinifi adaylaridir.")
    print("  A klasoru her gun tamamen kontrol edilmeli (icinde ~%20-25 B olabilir;")
    print("  gercek A'larin ~%97'si bu klasore gelir).")
    if a.c_tek:
        print(f"  '{C_KLASOR}': C olmayanlari dogru klasore tasiyin.")
    else:
        print(f"  '{C_YUKSEK}': cogu gercek C, hizlica onaylayin.")
        print(f"  '{C_DUSUK}': cogu C degil ama gercek C'lerin ~1/3'u burada;")
        print("      dikkatli bakin, catlak yoksa dogru klasore tasiyin.")
        print(f"  '{C_ADAY}': A/B modeli A demisti ama catlak suphesi var; icinde A, B")
        print("      ve az sayida catlak olabilir. Her hucreyi yeniden degerlendirip tasiyin.")
    print("  Diger klasorlere goz gezdirin, yanlis yerde duranlari tasiyin.")
    print("\nDaha temkinli istersen (daha az ama daha kesin A):  --a-esik 0.5")

    if DIL != "tr":
        en = DIL == "en"
        konsol("")
        konsol("=" * 60)
        konsol(f"Finished. {len(dosyalar)} images sorted into {len(sayac)} folders."
               if en else f"完成。共 {len(dosyalar)} 张图片，分到 {len(sayac)} 个文件夹。")
        if hata:
            konsol(f"Could not copy: {hata}" if en else f"复制失败：{hata} 张")
        konsol("=" * 60)
        for kat, n in sayac.most_common():
            konsol(f"{n:>7}  {100*n/len(dosyalar):>5.1f}%  {kat}")
        konsol("")
        if en:
            konsol("Please check carefully: the A-class folder (" + kisa(a_klasor) + ")")
            konsol("and the suspected-C folders (C_supheli ...). Glance through the others.")
        else:
            konsol("请重点检查：A级文件夹（" + kisa(a_klasor) + "）")
            konsol("和C类疑似文件夹（C_supheli ...）。其他文件夹请快速浏览。")
        sys.stdout = sys.__stdout__
        gunluk_dosyasi.close()


if __name__ == "__main__":
    main()
