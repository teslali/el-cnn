#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EL Kusur Siniflandirma - Hata Inceleme

Modelin yanildigi gorselleri, gozle bakilabilsin diye ayri klasorlere kopyalar.
Guven skoruna gore siralar: en tepedeki, modelin EN EMIN oldugu hatadir.

Amac: modelin "kusurlu" dedigi ama "Diger" (iyi) etiketli gorselleri incelemek.
Sorulacak soru: bu gorsellerde gercekten soluk bir kusur var mi?
  - VARSA  -> model calisiyor, sorun gorev taniminda (siddet esigi)
  - YOKSA  -> model gercekten yaniliyor

Kullanim:
    python hata_incele.py                 (varsayilan: ikili asama)
    python hata_incele.py cok             (cok sinifli asama)
"""

import sys
import csv
import shutil
from pathlib import Path
from collections import Counter

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RAPOR_DIZIN = Path("raporlar")
CIKTI_DIZIN = Path("inceleme")
HER_GRUPTAN = 40          # her hata turunden kac ornek kopyalansin
C = "=" * 68


def kisa(ad, n=30):
    t = ad[1:] if ad[:1].isalpha() else ad
    return t[:n].strip().replace("/", "-")


def guvenli_ad(s):
    """Klasor adi icin guvenli hale getir."""
    for ch in '/\\:*?"<>|':
        s = s.replace(ch, "-")
    return s.strip()[:60] or "bilinmeyen"


def main():
    hangi = sys.argv[1] if len(sys.argv) > 1 else "ikili"

    # siniflandir.py her iki asamayi da ayni dosyaya yaziyor olabilir;
    # mevcut csv dosyalarini bul
    csvler = sorted(RAPOR_DIZIN.glob("hatalar_*.csv"))
    if not csvler:
        print(f"HATA: {RAPOR_DIZIN}/hatalar_*.csv bulunamadi.")
        print("Once siniflandir.py calistirin.")
        sys.exit(1)

    print(f"Bulunan rapor dosyalari:")
    for c in csvler:
        print(f"   {c}")

    # meta'dan gorsel kokunu al
    import json
    meta_yol = Path("ozellikler") / "meta_224.json"
    if not meta_yol.exists():
        print(f"HATA: {meta_yol} yok.")
        sys.exit(1)
    kok = Path(json.loads(meta_yol.read_text(encoding="utf-8"))["kok"])
    print(f"\nGorsel koku: {kok}")

    # tum hatalari oku
    satirlar = []
    for c in csvler:
        with open(c, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    r["guven"] = float(r["guven"])
                except (ValueError, KeyError):
                    r["guven"] = 0.0
                satirlar.append(r)

    if not satirlar:
        print("Hata kaydi yok.")
        return

    print(f"Toplam hatali tahmin: {len(satirlar)}")

    # hata turlerine gore grupla
    gruplar = {}
    for r in satirlar:
        anahtar = (r["gercek"], r["tahmin"])
        gruplar.setdefault(anahtar, []).append(r)

    print(f"\n{C}\nHATA TURLERI (cok olandan aza)\n{C}")
    sirali = sorted(gruplar.items(), key=lambda kv: -len(kv[1]))
    for (g, t), lst in sirali:
        ort = sum(x["guven"] for x in lst) / len(lst)
        print(f"  {len(lst):>5}x  {kisa(g,26):<28} -> {kisa(t,26):<28} "
              f"(ort.guven {ort:.2f})")

    # kopyala
    if CIKTI_DIZIN.exists():
        shutil.rmtree(CIKTI_DIZIN)
    CIKTI_DIZIN.mkdir()

    print(f"\n{C}\nKOPYALANIYOR\n{C}")
    toplam_kopya = 0
    for (g, t), lst in sirali:
        # modelin EN EMIN oldugu hatalar en ustte
        lst.sort(key=lambda x: -x["guven"])
        secilen = lst[:HER_GRUPTAN]

        klasor_ad = f"{guvenli_ad(kisa(g,22))}__MODEL_DEDI__{guvenli_ad(kisa(t,22))}"
        hedef = CIKTI_DIZIN / klasor_ad
        hedef.mkdir(parents=True, exist_ok=True)

        kopyalanan = 0
        for sira, r in enumerate(secilen, 1):
            kaynak = kok / r["yol"]
            if not kaynak.exists():
                continue
            # dosya adinin basina sira ve guven -> siralamayi Finder'da gorursun
            yeni_ad = f"{sira:03d}_guven{r['guven']:.2f}_{kaynak.name}"
            try:
                shutil.copy2(kaynak, hedef / yeni_ad)
                kopyalanan += 1
            except Exception as e:
                print(f"   kopyalanamadi: {kaynak.name} ({e})")
        toplam_kopya += kopyalanan
        print(f"  {kopyalanan:>4} gorsel -> {klasor_ad}")

    # ozet dosyasi
    ozet = CIKTI_DIZIN / "OKU.txt"
    with open(ozet, "w", encoding="utf-8") as fh:
        fh.write("HATA INCELEME\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("Klasor adlari su bicimde:\n")
        fh.write("   <GERCEK ETIKET>__MODEL_DEDI__<MODELIN TAHMINI>\n\n")
        fh.write("Dosya adlarindaki sayi guven skorudur.\n")
        fh.write("001 ile baslayan dosya, modelin EN EMIN oldugu hatadir.\n\n")
        fh.write("BAKARKEN SORULACAK SORU:\n")
        fh.write("  'Diger' (iyi) etiketli ama model 'kusurlu' demis gorsellerde\n")
        fh.write("  gercekten soluk bir kusur var mi?\n\n")
        fh.write("  VARSA -> model calisiyor, sorun siddet esiginde\n")
        fh.write("  YOKSA -> model gercekten yaniliyor\n\n")
        fh.write("=" * 60 + "\n")
        for (g, t), lst in sirali:
            fh.write(f"\n{len(lst):>5}x  {g}\n         -> model dedi: {t}\n")

    print(f"\n{C}")
    print(f"Toplam kopyalanan : {toplam_kopya}")
    print(f"Klasor            : {CIKTI_DIZIN.resolve()}")
    print(C)
    print("\nSimdi Finder'da su klasoru ac ve gozle bak:")
    print(f"   open {CIKTI_DIZIN.resolve()}")
    print("\nOncelikle 'Diger' ile baslayan ve model 'kusurlu' demis klasore bak.")
    print("Dosyalar guven sirasina gore: 001 en emin oldugu hata.")


if __name__ == "__main__":
    main()
