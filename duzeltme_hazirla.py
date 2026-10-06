#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DUZELTME HAZIRLIK - yeniden egitim oncesi veri listesi

Iki kaynagi okur, HICBIR DOSYAYA DOKUNMAZ, sadece bir liste uretir:

  1) hatalar/          "X demis ama Y" klasorleri. Gercek etiket = Y.
                       Egitimde 3 kat agirlik (modelin zorlandigi vakalar).
  2) dogrulanmis/      gunluk.py ciktisinin, hatalar ayiklandiktan sonra
                       kalan kismi. Klasor adi = dogru kategori.
                       Egitimde normal agirlik.

Ayrica 2B klasorunu tarar:
  - Ayni dosya 2B'de de varsa, egitimde 2B'deki kopya yerine bu listedeki
    (duzeltilmis) etiket kullanilir.
  - Test gunlerine (10-11 Eylul) ait dosya varsa uyarir; bunlar egitime
    girerse test olcumu kirlenir.

Cikti:
  duzeltme_listesi.csv   egitim scriptinin okuyacagi liste
  eslesmeyen.txt         klasor adi taninmayanlar (varsa)

Kullanim:
  python duzeltme_hazirla.py \\
      --hatalar ~/Desktop/hatalar \\
      --dogrulanmis ~/Desktop/CNN/cikti_test \\
      --kaynak ~/Desktop/fabrika-fotolar/2B
"""

import sys
import csv
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict

UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp"}
ZAMAN = re.compile(r"(\d{14})")
TEST_GUNLERI = {"20260910", "20260911"}

AGIRLIK_DUZELTME = 3.0
AGIRLIK_DOGRULANMIS = 1.0

A_ANAHTAR = "其他"

# Turkce klasor adindaki ifade -> 2B klasor adinda gecen Cince anahtar.
# SIRA ONEMLI: ozel ifadeler genellerden once.
ESLEME = [
    ("a sinif",        "其他"),
    ("diger",          "其他"),
    ("nokta bask",     "卡点"),
    ("siyah nokta",    "黑斑"),
    ("siyah leke",     "黑斑"),
    ("siyah kenar",    "黑边"),
    ("siyah kose",     "黑边"),
    ("baski kay",      "偏移"),
    ("baski plaka",    "压板"),
    ("gizli",          "隐裂"),
    ("catlak",         "隐裂"),
    ("grafit",         "舟划"),
    ("grafik",         "舟划"),
    ("cerceve",        "舟框印"),
    ("hava",           "气流"),
    ("kalem",          "吸笔"),
    ("emici",          "吸盘"),
    ("bulut",          "云雾"),
    ("grid",           "断栅"),
    ("temas",          "接触"),
    ("kirlen",         "脏污"),
    ("ayirma",         "分片"),
    ("duzensiz",       "不规则"),
    ("cizik",          "不规则"),   # belirsiz: ayirma cizigi de olabilir
    ("konveyor",       "皮带"),
    ("bant",           "皮带"),
]
BELIRSIZ = {"cizik"}

C = "=" * 72


def normal(s):
    tablo = str.maketrans("ıİşŞçÇğĞöÖüÜ", "iIsScCgGoOuU")
    return " ".join(s.translate(tablo).lower().replace("_", " ").split())


def kisa(ad, n=32):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip()


def etiket_bul(metin):
    """Turkce metinden (anahtar, ifade) dondur; bulamazsa (None, None)."""
    for ifade, anahtar in ESLEME:
        if ifade in metin:
            return anahtar, ifade
    return None, None


def gun(f):
    z = ZAMAN.search(f.stem)
    return z.group(1)[:8] if z else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hatalar", required=True)
    ap.add_argument("--dogrulanmis", required=True)
    ap.add_argument("--kaynak", required=True, help="2B klasoru")
    ap.add_argument("--cikti", default="duzeltme_listesi.csv")
    a = ap.parse_args()

    hat = Path(a.hatalar).expanduser().resolve()
    dog = Path(a.dogrulanmis).expanduser().resolve()
    kay = Path(a.kaynak).expanduser().resolve()
    for p, ad in [(hat, "hatalar"), (dog, "dogrulanmis"), (kay, "kaynak")]:
        if not p.is_dir():
            print(f"HATA: {ad} klasoru yok: {p}")
            sys.exit(1)

    # ---- 2B kategori klasorleri: Cince anahtar -> gercek klasor adi
    kat_klasorleri = sorted([d.name for d in kay.iterdir() if d.is_dir()])
    anahtar_klasor = {}
    for _, anahtar in ESLEME:
        eslesen = [k for k in kat_klasorleri if anahtar in k]
        if eslesen:
            anahtar_klasor[anahtar] = eslesen[0]

    print(f"{C}\nDUZELTME HAZIRLIK\n{C}")
    print(f"hatalar     : {hat}")
    print(f"dogrulanmis : {dog}")
    print(f"kaynak (2B) : {kay}  ({len(kat_klasorleri)} kategori)")

    kayitlar = {}          # dosya adi -> dict
    eslesmeyen = []
    belirsiz = []
    ab_hatalari = Counter()

    # ---- 1) hatalar
    print(f"\n{C}\n1) HATALAR KLASORU\n{C}")
    for d in sorted([x for x in hat.iterdir() if x.is_dir()]):
        gorseller = [f for f in d.rglob("*")
                     if f.is_file() and f.suffix.lower() in UZANTILAR]
        if not gorseller:
            continue
        ad = normal(d.name)
        if " ama " in ad:
            tahmin_metin, gercek_metin = ad.split(" ama ", 1)
        else:
            tahmin_metin, gercek_metin = "", ad
        gercek_anahtar, ifade = etiket_bul(gercek_metin)
        tahmin_anahtar, _ = etiket_bul(tahmin_metin)

        if gercek_anahtar is None or gercek_anahtar not in anahtar_klasor:
            eslesmeyen.append((d.name, len(gorseller)))
            print(f"  ??  {len(gorseller):>4}  {d.name}")
            continue

        hedef = anahtar_klasor[gercek_anahtar]
        not_ = ""
        if ifade in BELIRSIZ:
            belirsiz.append((d.name, kisa(hedef)))
            not_ = "   (belirsiz: kontrol edin)"
        print(f"  ok  {len(gorseller):>4}  {d.name:<44} -> {kisa(hedef)}{not_}")

        a_gercek = gercek_anahtar == A_ANAHTAR
        a_tahmin = tahmin_anahtar == A_ANAHTAR
        if a_gercek != a_tahmin:
            ab_hatalari["B dedi, aslinda A" if a_gercek
                        else "A dedi, aslinda B"] += len(gorseller)

        for f in gorseller:
            kayitlar[f.name] = {"yol": f, "etiket": hedef,
                                "kaynak": "duzeltme",
                                "agirlik": AGIRLIK_DUZELTME}

    # ---- 2) dogrulanmis
    print(f"\n{C}\n2) DOGRULANMIS CIKTI\n{C}")
    dog_say = Counter()
    for d in sorted([x for x in dog.iterdir() if x.is_dir()]):
        hedef = d.name if d.name in kat_klasorleri else None
        if hedef is None:
            # klasor adi 2B ile birebir degilse Cince anahtarla dene
            for _, anahtar in ESLEME:
                if anahtar in d.name and anahtar in anahtar_klasor:
                    hedef = anahtar_klasor[anahtar]
                    break
        gorseller = [f for f in d.rglob("*")
                     if f.is_file() and f.suffix.lower() in UZANTILAR]
        if not gorseller:
            continue
        if hedef is None:
            eslesmeyen.append((f"[dogrulanmis] {d.name}", len(gorseller)))
            print(f"  ??  {len(gorseller):>4}  {d.name}")
            continue
        eklenen = 0
        for f in gorseller:
            if f.name in kayitlar:        # hatalar oncelikli
                continue
            kayitlar[f.name] = {"yol": f, "etiket": hedef,
                                "kaynak": "dogrulanmis",
                                "agirlik": AGIRLIK_DOGRULANMIS}
            eklenen += 1
        dog_say[hedef] += eklenen
        print(f"  ok  {eklenen:>4}  {kisa(d.name)}")

    if not kayitlar:
        print("\nHATA: hic gorsel bulunamadi.")
        sys.exit(1)

    # ---- 3) 2B ile cakisma ve test gunu kontrolu
    print(f"\n{C}\n3) 2B ILE KARSILASTIRMA\n{C}")
    kaynak_harita = {}
    for d in kay.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() in UZANTILAR:
                    kaynak_harita[f.name] = d.name
    cakisan = [k for k in kayitlar if k in kaynak_harita]
    etiket_farkli = [k for k in cakisan
                     if kaynak_harita[k] != kayitlar[k]["etiket"]]
    test_gunu = [k for k, v in kayitlar.items() if gun(v["yol"]) in TEST_GUNLERI]
    gunler = Counter(gun(v["yol"]) for v in kayitlar.values())

    print(f"  2B'de de bulunan      : {len(cakisan)}")
    print(f"    etiketi farkli olan : {len(etiket_farkli)}  "
          f"(egitimde duzeltilmis etiket kullanilacak)")
    print(f"  gun dagilimi          : " +
          ", ".join(f"{g or '?'}:{n}" for g, n in sorted(gunler.items())))
    if test_gunu:
        print(f"\n  UYARI: {len(test_gunu)} gorsel TEST gunlerine ait "
              f"({sorted(TEST_GUNLERI)}).")
        print("  Bunlar egitim listesinden cikarildi, test temiz kalsin.")
        for k in test_gunu:
            kayitlar.pop(k)

    # ---- yaz
    cikti = Path(a.cikti)
    with open(cikti, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["yol", "etiket", "kaynak", "agirlik", "2B_eski_etiket"])
        for k, v in sorted(kayitlar.items()):
            w.writerow([str(v["yol"]), v["etiket"], v["kaynak"],
                        v["agirlik"], kaynak_harita.get(k, "")])

    if eslesmeyen:
        with open("eslesmeyen.txt", "w", encoding="utf-8") as fh:
            for ad, n in eslesmeyen:
                fh.write(f"{n:>5}  {ad}\n")

    # ---- ozet
    duz = [v for v in kayitlar.values() if v["kaynak"] == "duzeltme"]
    dgr = [v for v in kayitlar.values() if v["kaynak"] == "dogrulanmis"]
    print(f"\n{C}\nOZET\n{C}")
    print(f"  duzeltme (x{AGIRLIK_DUZELTME:.0f} agirlik) : {len(duz)}")
    print(f"  dogrulanmis (x1)       : {len(dgr)}")
    print(f"  toplam                 : {len(kayitlar)}")

    if ab_hatalari:
        print(f"\n  A/B hatalari (en degerli duzeltmeler):")
        for k, n in ab_hatalari.items():
            print(f"    {n:>4}  {k}")

    print(f"\n  Duzeltmelerin gercek kategori dagilimi:")
    for kat, n in Counter(v["etiket"] for v in duz).most_common():
        print(f"    {n:>4}  {kisa(kat)}")

    if belirsiz:
        print(f"\n  BELIRSIZ ESLESMELER (sadece 'cizik' yazanlar):")
        for ad, hedef in belirsiz:
            print(f"    {ad}  -> {hedef}")
        print("  Bunlar 'duzensiz cizik' olarak alindi. Ayirma kaynakli")
        print("  cizikse klasor adina 'ayirma' ekleyip tekrar calistirin.")

    if eslesmeyen:
        print(f"\n  ESLESMEYEN KLASORLER ({len(eslesmeyen)}) - listeye ALINMADI:")
        for ad, n in eslesmeyen:
            print(f"    {n:>4}  {ad}")
        print("  Klasor adini duzeltip scripti tekrar calistirin.")

    print(f"\n  Liste: {cikti.resolve()}")
    print(C)


if __name__ == "__main__":
    main()
