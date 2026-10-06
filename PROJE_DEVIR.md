# EL Hücre Görsel Ayıklama — Proje Devir Belgesi

*Son güncelleme: 6 Ekim 2026 (ikinci sürüm: günlerin tarafsızlık tablosu ve kat8 doğrulama planı eklendi). Uzun bir sohbetten yeni sohbete geçiş için hazırlandı. Bu belge, sistemin güncel hâlini, dosyaları, kuralları ve sıradaki işleri anlatır.*

---

## 1. Proje özeti

- **Amaç:** Fabrikada EL test makinesinden gelen hücre görsellerini otomatik olarak fabrikanın kusur kategorilerine ayırmak ve A sınıfı hücreleri belirlemek.
- **Kullanım amacı:** Hücreler bu sonuçlara göre paketlenmiyor. Sonuçlar üretimdeki kusurları görmek ve proses iyileştirmesi için kullanılıyor. Excel'de her kusur türü için beklenen bir hedef yüzde var; vardiyada üretilen 70-80 bin hücreden sadece **bin 64 ve 84** inceleniyor. EL sayımları **4 saatte bir** Excel'e giriliyor.
- **Hedef:** Her hücreyi doğru kategoriye koymak (birden çok kusur varsa baskın olan; %100 kesinlik şart değil). Ayarlar **toplam yanlış yerleşimi en aza indirecek** şekilde seçiliyor; C hücrelerinin başka klasörlere kaçması olabildiğince az olmalı (kabul: günde ~100 C'den 1-4'ü).
- **Operatör iş akışı (planlanan):** Bütün klasörlere göz gezdirilir; A ve C klasörleri kritik olduğu için daha dikkatli kontrol edilir. Vardiya sonunda C hücreleri mevcut ayıklama programında gizli çatlak / baskı plakası olarak ayrılır. Kontrol süresi bütçesi 30-60 dk/gün, ne kadar az o kadar iyi.
- **Berkay** projeyi büyük ölçüde mesai dışında geliştiriyor. İlk eğitim verisinin (~40 bin) çoğunu elle kendisi ayırdı.
- **İletişim:** Berkay Türkçe yazıyor. Yöneticiler Çince konuşuyor; yöneticiye giden mesajlar Türkçe + Çince (bazen İngilizce) hazırlanıyor. Notlarda operatörleri kötüleyen karşılaştırmalar ("model operatörlerden iyi" gibi) kullanılmıyor; odak modelin gücü ve faydası.

## 2. Donanım ve ortam

- MacBook Pro Early 2013, i7 4 çekirdek, 8 GB RAM, ekran kartı yok, macOS Catalina.
- PyTorch 2.2.2 (CPU). Proje klasörü: `~/Desktop/CNN`, sanal ortam: `venv`.
- Her yeni terminalde:
  ```bash
  cd ~/Desktop/CNN
  source venv/bin/activate
  ```
- Eğitim süreleri uzun: 384 px kategori modeli tur başına ~65-90 dk, 512 px C dedektörü tur başına ~95 dk. Uzun işler `caffeinate -i` ile çalıştırılıyor; scriptler kesilirse aynı komutla kaldığı turdan devam ediyor.
- Ayıklama hızı (bütün modellerle) bu Mac'te ~1,4-1,7 görsel/sn; 2.000 görsel ~25 dk. Darboğaz 512 px C dedektörü.

## 3. Sistemin güncel hâli (`gunluk.py`)

Sırasıyla:

1. **A/B modeli:** `model_ab4/en_iyi_model.pt` (224 px). A olasılığı **≥ 0,40** ise A adayı (`A_ESIK`). Eski sabit %18,7 oran yöntemi `--a-orani` ile hâlâ seçilebilir ama kullanılmıyor.
2. **Kategori:** `model3` (224 px) + `model_kat5` (384 px) olasılıklarının ortalaması. kat5'in olasılıkları eğitimdeki dengelemeye göre geri ayarlanıyor (`egitim_sayi ** denge_us`, checkpoint'te kayıtlı).
3. **Uzman model kuralı:** `model_kat8/tur2.pt` (384 px).
   - kat8 "kirlenme" derse → kirlenme.
   - Birleşimin kararı ve kat8'in kararı ikisi de **karışan grupta** ise (düzensiz çizik, kırık grid, baskı kayması, kirlenme, hava akışı) → kat8'in kararı.
   - kat8 yoksa kat7 ile sadece kirlenme kuralı uygulanır. Kapatmak için `--kirlenme-kurali-kapali` (bu seçeneğin karışan grup kuralını da kapatıp kapatmadığı `gunluk.py`'den doğrulanacak).
4. **İkinci görüş:** A adayı, birleşimin A olasılığı **< 0,20** ise iptal edilir ve kategorisine gider (`IKINCI_ESIK`).
5. **C dedektörü v2:** `model_c2/tur3.pt` (512 px), bütün görsellere. Eşik: A adaylarında **0,20**, diğerlerinde **0,25**. Ayrıca model3'ün çatlak vetosu (0,05) ve kategori modelinin çatlak/baskı plakası kararları da C klasörüne gönderir.
6. **C klasörü üçe bölünür** (Çince adlı):
   - `C_supheli_1 隐裂压板疑似-高 (buyuk ihtimalle C)`: dedektör ≥ 0,90
   - `C_supheli_2 隐裂压板疑似-低 (dusuk supheli)`: A adayı olmayanlar; gerçek C'lerin ~1/3'ü burada, dikkatli bakılmalı
   - `C_supheli_3 隐裂压板疑似-需复判 (yeniden degerlendir)`: A adayıyken çatlak şüphesiyle alınanlar; içinde A, B ve az sayıda çatlak olabilir
   - Tek klasör için `--c-tek`.

**Çıktı:** Fabrika ayıklama programındaki adlarla kategori klasörleri (`f黑斑Siyah leke + siyah nokta`, `zz其他Diger` vb.) + `_kayit.csv` (dosya, klasor, A_olasiligi, C_olasiligi, A_adayi_mi, veto, C_dedektor). Görseller **kopyalanır**, kaynağa dokunulmaz.

**Önemli seçenekler:** `--cikti`, `--a-esik`, `--ikinci-esik`, `--c-esik`, `--c-esik-aday`, `--c-yuksek`, `--c-tek`, `--c-sadece-aday`, `--kirlenme-kurali-kapali`, `--dil tr|en|zh` (en/zh: ekranda sade ilerleme ve özet, ayrıntılar çıktı klasöründe `_islem_gunlugu.txt`).

**Kategori modelindeki sınıflar (kat5/kat7/kat8):** siyah nokta, siyah kenar, düzensiz çizik, baskı kayması, hava akışı, kırık grid, kirlenme, ayırma çiziği, bulutlu, vakum kalemi, temas kirlenmesi, grafit tekne, tekne çerçevesi, rulo izi, baskı plakası, gizli çatlak, nokta baskısı, A sınıfı. Konveyör bant izi ve vakum emici yok ya da çok zayıf.

## 4. Model dosyaları

| Model | Dosya | Durum |
|---|---|---|
| A/B (224) | `model_ab4/en_iyi_model.pt` | Kullanılıyor |
| Kategori (224) | `model3/en_iyi_model.pt` | Kullanılıyor |
| Kategori (384) | `model_kat5/en_iyi_model.pt` | Kullanılıyor |
| Uzman: kirlenme | `model_kat7/tur2.pt` | Yedek (kat8 yoksa) |
| Uzman: kirlenme + yeni çizik tanımları | `model_kat8/tur2.pt` | Kullanılıyor |
| C dedektörü v2 (512) | `model_c2/tur3.pt` | Kullanılıyor |
| C v1 | `model_c/` | Eski |
| C v3 (kontrast kanalları) | `model_c3/` | Kazanç yok, kullanılmıyor |
| kat6 (altın günlerle) | `model_kat6/` | Kat5'ten iyi değil |
| A/B 384 | `model_ab384/` | Kazanç yok |
| A doğrulayıcısı | `model_a/` | Başarısız |

## 5. Ölçüm: altın kümeler

**Yöntem:** Modelin görmediği bir günden ~2.000 görsellik dengeli örneklem (+ günün bütün C'leri). Berkay her sonucu kontrol eder: **modelin dediği kusur hücrede hiç yoksa** görseli gerçek kategorisiyle `altinN_hatalar` klasörüne taşır; doğruysa yerinde bırakır. Emin olmadıkları `emin degilim` klasörüne (ölçüme girmez). Siyah nokta en kalabalık kategori olduğu için `siyah_ornek.py` klasörün rastgele 1/4'ünü bırakıp gerisini `_incelenmeyen_siyah_nokta`'ya taşır; ölçüm incelenenleri 4 kat sayar. `_` ile başlayan klasörler ölçüme girmez.

| `--kume` | Gün | Onay klasörü | Düzeltmeler | Görseller |
|---|---|---|---|---|
| 1 | 9.22 | `cikti_altin` | `altin_hatalar` | `altin_kume` |
| 2 | 9.24 | `cikti_altin_924` | `altin2_hatalar` | `altin_kume2` |
| 3 | 9.23 | `cikti_altin_923` | `altin3_hatalar` | `altin_kume3` |
| 4 | 9.25 | `cikti_altin_925` | `altin4_hatalar` | `altin_kume4` |
| 5 | 9.26 | `cikti_altin_926` | `altin5_hatalar` | `altin_kume5` |
| 6 | 9.27 | `cikti_altin_927` | `altin6_hatalar` | `altin_kume6` |
| 7 | 9.25'in incelenmemiş kısmı | `cikti_altin_925k` | `altin7_hatalar` | `altin_kume7` |
| 8 | 9.30 | `cikti_altin_yeni` | `altin8_hatalar` | `altin_kume8` |
| 9 | 10.01 M vardiyası | `cikti_altin_m` | `altin9_hatalar` | `altin_kume9` |

Düzeltme klasörleri masaüstünde, diğerleri `~/Desktop/CNN` içinde. Başka bir çıktıyı aynı etiketlerle ölçmek için: `python altin_olcum.py --kume N --kayit <cikti>/_kayit.csv`.

**Günlerin kullanımı ve tarafsızlık (kat8'e göre):**

| Gün | Kullanım | kat8 için tarafsız mı? |
|---|---|---|
| 4-21 Eylül | Eğitim verisi; eski benzer aramaların havuzu (kirlenme, konveyör, kırık grid ilk deneme) | Hayır |
| 9.22-9.24, 9.26, 9.27 | Altın küme; eşik seçimi; kat6-8 eğitiminde sınır örnekleri | Hayır |
| 9.25 (küme 4 ve 7) | Altın küme; kat7/kat8 sınavı; kurallar bu güne bakılarak seçildi | Kısmen |
| 9.30 | Altın küme; benzer arama örnek kaynağı ve havuzu (altın örneklem hariç); kat8 eğitimi | Hayır |
| 10.01 M | Altın küme; benzer arama havuzu (altın örneklem hariç); kat8 sınavı, eğitime girmedi | Kısmen |
| 10.01 D/N, 9.28, 9.29, 10.02+ | Hiç kullanılmadı (Mac'e getirilmedi) | **Evet** |

- Eski veri benzer aramaları, 9.22 ve sonrasındaki bütün altın günleri bilerek dışarıda bıraktı.
- kat8 eğitimi 10.01 tarihli bütün görselleri dışarıda bıraktı. Ama 10.01 M'den benzer aramayla bulunan örnekler `yeni_duzeltmeler`'de duruyor. İleride bu filtre kaldırılırsa 10.01 o modelin sınavı olmaktan çıkar.
- 10.01 D/N görülmemiş olsa da havuz olarak kullanılan M vardiyasıyla aynı günün üretimi; sonuç biraz iyimser olabilir. En bağımsız seçenek 10.02 ve sonrası.
- Yeni bir gün bir kez eşik ya da kural seçiminde kullanıldıktan sonra o ayar için artık tarafsız sayılmaz; bu tabloya eklenmeli.

**Son sonuçlar (güncel sisteme yakın ayarlarla):**
- Kategori doğruluğu: ~%92-95 (9.30'da %90,4: o gün tanım değişikliğinden etkilenen tip çok fazlaydı; o olmasa ~%93,5)
- A/B kararı: %98-99,4
- A klasörü isabeti güne göre %76-96, gerçek A'ların %94-100'ü A klasöründe
- C yakalama ~%96-99; A'ya sızan C günde 0-1
- Kirlenme kuralı (kat7/kat8): hiç görülmemiş günlerde yakalama %34 → %88-90
- kat8: yeni çizik tipleri çizik olarak %6 → %66; diğer günlerde genel ~-0,2 puan; gerçek kırık grid yakalaması birkaç puan düştü

## 6. Veri ve klasörler

- `~/Desktop/fabrika-fotolar/2B`: ~43 bin görsel (4-13 Eylül), çoğunu Berkay elle ayırdı
- `~/Desktop/fabrika-fotolar/yeni`: ~122 bin (15-22 Eylül), operatör etiketli
- `yeni2` (9.24), `yeni3` (9.23), `yeni4` (9.25), `yeni5` (9.26), `yeni6` (9.27), `yeni8` (9.30); `~/Desktop/M(16-00)` (10.01 M vardiyası)
- **Fabrika klasör düzeni:** gün / vardiya `D(08-16)`, `M(16-00)`, `N(00-08)` / 2 saatlik dilim (`8-10`...) / bin (64, 84) / hat (1B, 2A, 2B, 3A) / kategori. Dosya adı hat koduyla başlar ve 14 haneli zaman damgası içerir. `OverTime` / `Class400` dosyaları bozuk sayılıp dışarıda bırakılıyor.
- `~/Desktop/CNN` içinde: `yeni_envanter.csv` (9.15-9.22 envanteri), `ayrim_tahmin.csv` ve `ayrim_gruplar.csv` (105 bin görselin operatör/model karşılaştırması: anlasan_A, anlasan_B vb.), `duzeltme_listesi.csv` (automation-test günü, doğrulanmış), `altin_kume*.csv`.
- **Masaüstündeki inceleme klasörleri (eğitim için çok değerli, silinmemeli):**
  - `yeni_duzeltmeler`: Berkay'ın incelediği etiketler; alt klasörler arasında `kirlenme yeni tanim` (218), `duzensiz cizik yeni tanim` (386), `duzensiz cizik yeni tur` (159)
  - `cift_inceleme_hatalar` (~630), `altin*_hatalar`
  - `rulo_izi_ornekleri`: konveyör aramasından çıkan, çoğu rulo izi; ileride rulo izi eğitimi için
  - `9.27_AI`: yöneticiye gösterilen tam gün çıktısı
- **Klasör adı kuralları:** Scriptler Türkçe klasör adlarını anahtar kelimelerle tanıyor ("cizik" → düzensiz çizik, "grid" → kırık grid, "hava" → hava akışı, "kirlen" → kirlenme, "a sinif" → A, "c supheli" → C...). Bir klasör adında başka bir kategorinin anahtar kelimesi geçmemeli (ör. kirlenme klasörünün adında "hava", çizik klasörünün adında "grid" olmamalı). Yeni tanımlar için `... yeni tanim`, yeni türler için `... yeni tur` alt klasörleri kullanılıyor.

## 7. Scriptler (`~/Desktop/CNN`)

**Not:** Yeni sohbet bu dosyaları göremez. Bir script değişecekse Berkay Mac'teki güncel hâlini yükler.

| Script | Ne yapar |
|---|---|
| `gunluk.py` | Günlük ayıklama, bütün sistem (bölüm 3) |
| `altin_olcum.py` | Altın küme ölçümü (`--kume 1-9`, `--kayit`); kategori tablosu, A kesimi, C, hat/vardiya, en sık karışıklıklar |
| `yeni_envanter.py` | Gün klasörü envanteri + altın örneklem (`--altin-gun`, `--altin-adet`, `--altin-klasor`, `--altin-csv`, `--cikti`) |
| `klasor_ornek.py` | Herhangi bir klasörden rastgele örneklem (`--hedef`, `--haric`) |
| `kalan_ornek.py` | 9.25'in incelenmemiş kısmından örneklem |
| `siyah_ornek.py` | Siyah nokta klasörünün 1/4 örneklemesi |
| `c_bol.py` | Eski tek C klasörlü bir çıktıyı üç klasöre böler |
| `benzer_bul.py` | **Benzer görsel arama:** örnek klasörlerden yola çıkıp eski veride ya da bir klasörde benzerleri bulur (`--ornek`, `--havuz`, `--havuz-klasor`, `--haric`, `--cikti`, `--hedef-klasor`, `--adet`). Kirlenme ve yeni çizik tiplerini öğretmede çok etkili oldu |
| `kategori5.py` … `kategori8.py` | Kategori modeli eğitimleri (kat8: kat7 + yeni çizik tanımları, grup kuralını sınavda karşılaştırır) |
| `kat6_karsilastir.py` | Kategori modellerini altın kümede karşılaştırır (`--kume`, `--model`), tahminleri önbelleğe alır |
| `kirlenme_kural.py` | Kayıtlı tahminlerle kural denemesi |
| `c_dedektor2.py`, `c_dedektor3.py`, `onisleme.py` | C dedektörü eğitimleri (v3 kontrast kanalları, kazanç yok) |
| `c_tatli_nokta.py`, `c_kaynak.py`, `c_esik_deneme.py`, `c2_esik.py`, `c2_aday.py`, `c3_karsilastir.py` | C eşik ve dedektör analizleri |
| `ikinci_gorus.py` | İkinci görüş eşik analizi |
| `kusur_haritasi.py` | Kategori ve hat bazında ortalama görüntü + kusur yoğunluk haritası (prototip, kenarda) |
| `paket_hazirla.sh`, `BASLAT.bat` | Windows demo paketi (bölüm 9) |
| `ayrim.py`, `cift_inceleme.py`, `kirlenme_benzer.py` | Eski veri ayrımı ve inceleme araçları |

## 8. Çalışma alışkanlıkları

- **İndirme düzeni:** İndirilenler'deki eski kopyalar silinmeden yeni dosya indirilirse Mac `gunluk (1).py` gibi ad veriyor ve `mv` başarısız oluyor; eski sürüm yerinde kalıyor. Sıra: önce `find ~/Downloads -maxdepth 1 -name 'dosya*.py' -delete`, **sonra** indir, sonra `mv`, sonra `grep` ile doğru sürüm olduğunu kontrol et. Bir ölçümü yorumlamadan önce çıktıda beklenen satırların (ör. `uzman model: model_kat8/tur2.pt`) göründüğünü kontrol et.
- **Terminal çıktıları:** Berkay uzun çıktıları "Terminal Saved Output" dosyası olarak yüklüyor (yapıştırılan metin bazen boş geliyor).
- **Eğitim sürerken** veri ve inceleme klasörlerine dokunulmamalı; scriptler veri değişirse baştan başlıyor.
- **Altın küme akışı:** `yeni_envanter.py` (ya da `klasor_ornek.py`) → `gunluk.py` → `siyah_ornek.py` → inceleme → `altin_olcum.py --kume N`.
- **Benzer arama akışı:** örnekler bir klasörde → `benzer_bul.py` → Berkay aranan tipte olanları `yeni_duzeltmeler/<hedef>` klasörüne taşır, diğerlerine dokunmaz → ince ayar eğitimi.
- **Kategori modeli kararları** her zaman altın kümede mevcut sistemle yan yana ölçülerek veriliyor; yeni bir model sadece daha iyiyse devreye giriyor.
- **Devir belgesi:** Berkay proje dosyasındaki md'yi düzenleyemiyor; sistem değiştiğinde belgenin tam güncel hâli dosya olarak verilir.

## 9. Windows demo paketi

- `paket_hazirla.sh` Mac'te `~/Desktop/ELAI_paket.zip` üretir (~450 MB): gömülü Python 3.11 + PyTorch 2.2.2 CPU + modeller + `BASLAT.bat` + `OKU_README.txt` + `PROJECT_SUMMARY.txt`.
- `BASLAT.bat`: dil seçimi (English / 中文) → klasör seçme penceresi → `gunluk.py --dil` → sonuç `<klasör>_AI`.
- Sorunlar ve çözümleri: hata mesajları dil modunda kayda gidiyordu (düzeltildi); klasör seçicinin yolu karakter kodlaması yüzünden bozuluyordu (PowerShell UTF-8 ile düzeltildi). İş bilgisayarında test edildi, çalışıyor; yöneticiye gönderildi.
- **Yöneticideki paket kat8'den önceki sürüm.** Güncellemek için `paket_hazirla.sh`'e `model_kat8/tur2.pt` eklenmeli.
- Yönetici için hazırlanan proje notu (İngilizce) paketin içinde `PROJECT_SUMMARY.txt` olarak duruyor.

## 10. Alan bilgisi ve kararlar

- **Tanım değişiklikleri:** nokta baskısının çoğu artık hava akışı; bir hava akışı tipi artık kirlenme; bir kırık grid tipi artık düzensiz çizik. Eski etiketler eski tanıma göre. Yeni tanımlar benzer arama + ince ayarla öğretiliyor.
- **Yeni çizik türü** 9.30 ve 10.01'de görüldü; model önceden baskı kaymasıyla karıştırıyordu.
- **Konveyör bant izi** rulo iziyle karışıyor ve veride çok az örneği var. 9.30'daki 10 konveyörün hepsi rulo izine gitti.
- **Birden çok kusur:** baskın olana göre dosyalanıyor; kural esnek.
- **A/B sınırı** kişiden kişiye değişebiliyor; sınırdaki kararlar ikisi de doğru olabilir.
- **Operatör etiketleri** tutarsız (ör. operatörlerin kırık grid dediklerinin ~%24'ü gerçekten kırık grid); bu yüzden eğitimde Berkay'ın incelediği ve operatörle modelin anlaştığı etiketler tercih ediliyor.
- **Boğum** bölgeleri her hücrede aynı yerde ama hatlar hücreleri birebir aynı konumda fotoğraflamıyor; bizim görsellerde boğum görünmüyor. Konum bilgisi için hizalama + boğum haritası fikri var (ileride).
- **Öğrenilen dersler:** Çözünürlük küçük detayda işe yarıyor (C 512, kategori 384); A/B modelini 384'e taşımak etiket sorunu yüzünden işe yaramadı; kontrast kanalları C'de kazanç getirmedi; birleşimde model3 uzman modelin doğru kararını bastırabiliyor, bu yüzden kurallar kullanılıyor; benzer arama yöntemi yeni tanımları öğretmede çok etkili.
- **Çatlak / çizik ayrımı** doğası gereği zor (hattaki başka bir yapay zekâ yazılımı da ayıramıyor); bu yüzden şüpheli C'ler her zaman insan kontrolüne gidiyor.

## 11. Masaüstü programı (planlanan)

- Test bilgisayarlarından paylaşılan klasör üzerinden görselleri otomatik almak; geldikçe ayıklamak
- Hat bazında durdur / başlat (CC, ELBL testleri için): karar **dosya adındaki çekim saatine** göre; durdurulan aralıkta çekilen görseller hiçbir zaman işlenmez ve Excel'e girmez; sebep ve kişi kaydı; uzun süre durdurulmuş hat için uyarı
- Fabrika klasör düzeninde çıktı; 4 saatlik rapor otomatik (EL_Rapor)
- İnceleme için ELSort; düzeltmeler, programın kendi kararıyla son klasörün karşılaştırılmasından otomatik çıkarılır
- Eşikler için slider (tahmini etkiyi göstererek, C için alt sınır, yetki ve değişiklik kaydı, geriye dönük değil)
- Düzeltmelerle düzenli yeniden eğitim; yeni model sadece sabit sınavı geçerse ve onaylanırsa devreye girer; sürüm saklama ve geri dönüş
- İleride: isteğe bağlı kutu işaretleme, "buna benzeyenleri bul", ısı haritası, kusur haritası raporu
- Güçlü bilgisayarda modelleri tek bir modelde birleştirmek (damıtma) ve ONNX ile hızlandırmak

## 12. Sıradaki işler (sırasıyla)

1. **kat8 kuralını hiç görülmemiş yeni bir günde kontrol etmek** (devam ediyor; gün henüz seçilmedi)
   - Gün: bölüm 5'teki tabloda "Evet" olan günlerden biri; tercihen 10.02 ve sonrası.
   - Küme 10 olarak eklenecek (`altin_kume10`, `altin10_hatalar`, onay klasörü `cikti_altin_<gün>`). `altin_olcum.py`'ye küme 10 eklenmesi gerekiyor (Berkay güncel hâlini yükleyecek).
   - Akış: örneklem → `gunluk.py` (kat8 açık; çıktıda `uzman model: model_kat8/tur2.pt` kontrolü) → `siyah_ornek.py` → inceleme → `altin_olcum.py --kume 10` → aynı görseller kural kapalı ve kat7 ile ayıklanıp `--kayit` ile ölçülür.
   - **Yanlılık önlemi:** İnceleme kat8 açık sistemin çıktısı üzerinden yapıldığı için, birden çok kusurlu hücrelerde incelenen sistem avantajlı olur. İki sürümün ayrıştığı hücreler ayrı bir klasöre çıkarılıp iki karar birlikte görülerek incelenecek (yeni küçük karşılaştırma scripti; mevcut scriptlere dokunulmaz).
   - Dengeli 2.000'lik örneklemde kirlenme ve yeni çizik tipinden birkaç düzine çıkabilir; yakalama yüzdeleri ±10 puan oynayabilir. Kırık grid yakalaması aynı tabloda izlenecek.
2. **Konveyör yoğun gün:** Berkay o günün görsellerini getirecek (hatırlatılmalı). Konveyörü ayrı kategori olarak eklemek; rulo iziyle birlikte, `rulo_izi_ornekleri` klasörünü de kullanarak
3. **"Belirsiz" klasörü:** modelin emin olmadığı hücreleri ayrı bir kontrol klasörüne almak; eşiği altın günlerde ölçmek (yeni, hiç görülmemiş kusur türleri için)
4. **Kırık grid dedektörü** (512 px, C dedektörü gibi); kat8'den sonra kırık grid yakalaması biraz düştü
5. **Daha sonra:** kusur haritaları (prototip hazır), masaüstü programı, güçlü bilgisayarla damıtma/ONNX, overlay kutularının güvenilirlik kontrolü, boğum haritası, bulutlu hücre analizi, pilot geri bildirimiyle C v4, çatlak/baskı plakası ayrım modeli, hava akışı → siyah nokta ve siyah kenar → siyah nokta karışıklıkları için benzer arama
