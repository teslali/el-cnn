# EL Hücre Görsel Ayıklama

Güneş hücresi üretim hattındaki EL (elektrolüminesans) test görsellerini otomatik olarak fabrikanın kusur kategorilerine ayıran ve A sınıfı hücreleri belirleyen bir derin öğrenme sistemi.

Ayrıntılı durum, ölçüm sonuçları, kararlar ve sıradaki işler için: **[PROJE_DEVIR.md](PROJE_DEVIR.md)**

> **Not:** Bu depoda kod, belgeler ve her model klasörünün eğitim günlüğü (`model_*/gunluk.txt`) var. Görseller, model ağırlıkları (`.pt`) ve veri listeleri (`*.csv`) fabrikanın verisi olduğu için ve boyutları nedeniyle depoya dahil değildir; çalıştırmak için model dosyaları ayrıca paylaşılır.

## İncelemeye nereden başlanır

1. **[PROJE_DEVIR.md](PROJE_DEVIR.md)**: sistemin tamamı, kararlar, ölçüm yöntemi ve sonuçlar
2. **`gunluk.py`**: modellerin nasıl birleştirildiği ve karar kuralları (A/B, kategori, uzman model, ikinci görüş, C dedektörü)
3. **`kategori8.py`** (+ yardımcıları `kategori6.py`): bir eğitimin baştan sona akışı: veri kaynakları ve ağırlıkları, dengeleme, ince ayar, her turdan sonra altın kümelerde sınav
4. **`c_dedektor2.py`**: 512 px çatlak / baskı plakası dedektörü
5. **`altin_olcum.py`**: doğruluğun nasıl ölçüldüğü
6. **`model_*/gunluk.txt`**: her eğitimin tur tur sonuçları

## Sistem özeti

`gunluk.py` bir görsel klasörünü şu adımlarla ayıklar:

1. **A/B modeli** (`model_ab4`, 224 px): A sınıfı adaylarını belirler
2. **Kategori modelleri** (`model3` 224 px + `model_kat5` 384 px): kusur kategorisi
3. **Uzman model** (`model_kat8`, 384 px): birbiriyle karışan grupta (düzensiz çizik, kırık grid, baskı kayması, kirlenme, hava akışı) son kararı verir
4. **İkinci görüş:** kategori modelleri onaylamazsa A adaylığı iptal edilir
5. **C dedektörü** (`model_c2`, 512 px): gizli çatlak ve baskı plakası şüphelileri, güven düzeyine göre üç klasöre

Çıktı, fabrikanın ayıklama programıyla aynı adlarda kategori klasörleri ve bir `_kayit.csv` dosyasıdır. Görseller kopyalanır, kaynağa dokunulmaz.

## Kurulum

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Geliştirme ortamı: macOS, Python 3, PyTorch 2.2.2 (CPU). Model dosyaları ayrıca kopyalanmalıdır (bkz. PROJE_DEVIR.md, bölüm 4).

## Kullanım

```bash
# Bir klasörü ayıklamak
python gunluk.py <gorsel_klasoru> --cikti <cikti_klasoru>

# İngilizce / Çince sade ekran çıktısıyla
python gunluk.py <gorsel_klasoru> --cikti <cikti_klasoru> --dil zh

# Etiketli bir altın kümede doğruluk ölçümü
python altin_olcum.py --kume 9
```

## Scriptler

### Günlük kullanım ve ölçüm
| Script | Ne yapar |
|---|---|
| `gunluk.py` | Ayıklama sisteminin tamamı |
| `altin_olcum.py` | Altın kümelerde doğruluk ölçümü (`--kume 1-9`, `--kayit`) |
| `yeni_envanter.py` | Gün klasörü envanteri ve altın örneklem seçimi |
| `klasor_ornek.py` | Herhangi bir klasörden rastgele örneklem |
| `kalan_ornek.py` | Bir günün incelenmemiş kısmından örneklem |
| `siyah_ornek.py` | Siyah nokta klasörünün 1/4 örneklemesi (inceleme yükünü azaltmak için) |
| `c_bol.py` | Tek C klasörlü eski bir çıktıyı üç klasöre böler |
| `benzer_bul.py` | Benzer görsel arama: birkaç örnekten yola çıkıp veride benzerlerini bulur |
| `paket_hazirla.sh` | Windows için kurulum gerektirmeyen demo paketi üretir |

### Eğitim
| Script | Ne yapar |
|---|---|
| `ab_model4.py` | A/B modeli (kullanılan: `model_ab4`) |
| `kategori5.py` | Kategori modeli 384 px (kullanılan: `model_kat5`) |
| `kategori6.py` | Altın günlerle kategori modeli; ayrıca kat7/kat8'in kullandığı yardımcı fonksiyonlar |
| `kategori7.py` | Kirlenmenin yeni tanımı için ince ayar (`model_kat7`) |
| `kategori8.py` | Yeni çizik tanımları için ince ayar (kullanılan: `model_kat8`) |
| `c_dedektor2.py` | C dedektörü v2, 512 px (kullanılan: `model_c2`) |
| `c_dedektor3.py`, `onisleme.py` | C dedektörü v3, kontrast kanallarıyla (kazanç getirmedi) |

### Analiz ve karşılaştırma
| Script | Ne yapar |
|---|---|
| `kat6_karsilastir.py` | Kategori modellerini altın kümede yan yana karşılaştırır |
| `kirlenme_kural.py` | Kayıtlı tahminlerle karar kuralı denemesi |
| `ikinci_gorus.py` | İkinci görüş eşiği analizi |
| `c_tatli_nokta.py` | C eşikleri için en iyi denge noktası araması |
| `c2_esik.py`, `c2_aday.py`, `c_esik_deneme.py`, `c_kaynak.py`, `c3_karsilastir.py` | C dedektörü eşik ve sürüm analizleri |
| `kusur_haritasi.py` | Kategori ve hat bazında ortalama görüntü ve kusur yoğunluk haritası (prototip) |

### Eski ve deneysel
Projenin önceki aşamalarına ait; bilgi için saklanıyor.

| Script | Ne yapar |
|---|---|
| `ab_model2.py`, `ab_model3.py`, `ab_iki_kademe.py`, `ab384.py` | Önceki A/B modeli denemeleri |
| `ince_ayar3.py`, `kategori4.py`, `kat_karsilastir.py` | Önceki kategori modelleri ve karşılaştırmaları |
| `c_dedektor.py` | C dedektörü v1 |
| `a_dogrulayici.py` | A doğrulayıcı modeli denemesi (başarısız) |
| `ayrim.py`, `cift_inceleme.py` | Operatör etiketleriyle model tahminlerinin karşılaştırılması ve inceleme listeleri |
| `kirlenme_benzer.py` | Kirlenme için benzer arama (yerini `benzer_bul.py` aldı) |
| `otomasyon.py`, `duzeltme_hazirla.py`, `sinir_bul.py`, `sizinti_kontrol.py`, `tani_cizik.py`, `cozunurluk_kontrol.py` | Erken dönem otomasyon, düzeltme hazırlama ve tanı araçları |

## Beklenen klasör yapısı

Scriptler model ve veri klasörlerini çalışma klasörüne göre arar (`model_kat8/tur2.pt`, `altin_kume9/`, `cikti_altin_m/` gibi). Bu yüzden scriptlerin yeri değiştirilmemeli; hepsi aynı klasörde çalışır. Bazı scriptler birbirini içe aktarır (ör. `kategori7.py` ve `kategori8.py` → `kategori6.py`, `c_dedektor3.py` → `c_dedektor2.py`).
