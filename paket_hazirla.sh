#!/bin/bash
# =====================================================================
# WINDOWS PAKETI HAZIRLA (Mac'te calisir)
#
# Kurulum gerektirmeyen tasinabilir bir klasor uretir:
#   ELAI_paket/
#     BASLAT.bat          <- cift tikla, klasor sec, sonuc "<klasor>_AI"
#     OKU_README.txt
#     PROJECT_SUMMARY.txt <- proje notu
#     gunluk.py
#     python/             <- gomulu Python 3.11 + PyTorch (CPU) + kutuphaneler
#     model_ab4/ model3/ model_kat5/ model_kat7/ model_c2/
#
# Calistirma (CNN klasorunde, venv acik):
#   bash paket_hazirla.sh
# Sonuc: masaustunde ELAI_paket.zip
# =====================================================================
set -e
CNN="$HOME/Desktop/CNN"
P="$HOME/Desktop/ELAI_paket"
SP="$P/python/Lib/site-packages"
cd "$CNN"

echo "== 1/5 Klasor hazirlaniyor"
rm -rf "$P" "$HOME/Desktop/ELAI_paket.zip"
mkdir -p "$P/python"

echo "== 2/5 Windows icin gomulu Python 3.11 indiriliyor"
curl -L --fail -o /tmp/py_embed.zip \
  https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
unzip -q /tmp/py_embed.zip -d "$P/python"
# site-packages'i gorunur yap
printf 'python311.zip\r\n.\r\nLib\\site-packages\r\nimport site\r\n' > "$P/python/python311._pth"
mkdir -p "$SP"

echo "== 3/5 Windows kutuphaneleri indiriliyor (PyTorch CPU 2.2.2, ~250 MB)"
python -m pip install --quiet --target "$SP" \
  --platform win_amd64 --python-version 3.11 --implementation cp \
  --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  "torch==2.2.2" "torchvision==0.17.2" "numpy<2" "pillow"

echo "== 4/5 Modeller ve script kopyalaniyor"
grep -q -- "--dil" gunluk.py || { echo "HATA: gunluk.py eski surum (dil secenegi yok)"; exit 1; }
cp gunluk.py "$P/"
for m in "model_ab4/en_iyi_model.pt" "model3/en_iyi_model.pt" \
         "model_kat5/en_iyi_model.pt" "model_kat7/tur2.pt" "model_c2/tur3.pt"; do
  if [ ! -f "$m" ]; then echo "HATA: $m bulunamadi"; exit 1; fi
  mkdir -p "$P/$(dirname "$m")"
  cp "$m" "$P/$m"
done

# --- BASLAT.bat (Windows satir sonlariyla)
cat > /tmp/baslat.txt << 'EOF'
@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ============================================================
echo   AI Cell Image Sorting  /  AI 电池片图片分拣
echo ============================================================
echo.
echo   1 = English
echo   2 = 中文
echo.
choice /c 12 /n /m "Language / 语言 [1/2]: "
if errorlevel 2 goto ZH

:EN
set "DIL=en"
set "M_SEC=Please select the folder that contains the cell images..."
set "M_DESC=Select the image folder"
set "M_YOK=No folder selected."
set "M_GIRIS=Input"
set "M_CIKIS=Output"
set "M_BASLA=Sorting has started. The original images will not be changed."
set "M_HATA=An error occurred. Please send a screenshot of this window."
set "M_BITTI=Done. The result folder will now open."
goto DEVAM

:ZH
set "DIL=zh"
set "M_SEC=请选择包含电池片图片的文件夹..."
set "M_DESC=请选择图片文件夹"
set "M_YOK=未选择文件夹。"
set "M_GIRIS=输入"
set "M_CIKIS=输出"
set "M_BASLA=开始分拣，原图片不会被改动。"
set "M_HATA=出错了，请截图此窗口并发送。"
set "M_BITTI=完成，结果文件夹即将打开。"

:DEVAM
echo.
echo %M_SEC%
set "KLASOR="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -STA -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.Description = $env:M_DESC; if ($f.ShowDialog() -eq 'OK') { $f.SelectedPath }"`) do set "KLASOR=%%I"
if "%KLASOR%"=="" (
  echo %M_YOK%
  pause
  exit /b
)
set "CIKTI=%KLASOR%_AI"
echo.
echo %M_GIRIS%: %KLASOR%
echo %M_CIKIS%: %CIKTI%
echo.
echo %M_BASLA%
echo.
"%~dp0python\python.exe" "%~dp0gunluk.py" "%KLASOR%" --cikti "%CIKTI%" --dil %DIL%
if errorlevel 1 (
  echo.
  echo %M_HATA%
  pause
  exit /b
)
echo.
echo %M_BITTI%
explorer "%CIKTI%"
pause
EOF
sed 's/$/\r/' /tmp/baslat.txt > "$P/BASLAT.bat"

cat > /tmp/oku.txt << 'EOF'
AI Cell Image Sorting - how to use / 使用方法
=============================================

1. Unzip this folder anywhere (e.g. Desktop). Do not run it from inside the zip.
   把文件夹解压到任意位置（例如桌面），不要在压缩包里直接运行。

2. Double-click BASLAT.bat, choose the language (1 = English, 2 = 中文),
   then select a folder that contains cell images (sub-folders are fine).
   The original images are NOT changed.
   双击 BASLAT.bat，选择语言（1 = English，2 = 中文），然后选择包含电池片
   图片的文件夹（可以有子文件夹）。原图片不会被改动。

3. The result is created next to the selected folder, with "_AI" added
   to its name, and opens automatically when finished.
   结果会在所选文件夹旁边生成，名称后加 "_AI"，完成后自动打开。

Speed: roughly a few minutes per 1,000 images on a normal office computer.
速度：普通办公电脑上大约每1000张需要几分钟。

If Windows shows a security warning, choose "More info" -> "Run anyway".
如果Windows出现安全提示，请选择"更多信息" -> "仍要运行"。

If an error mentions a missing DLL, install the Microsoft Visual C++
Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
如果提示缺少DLL，请安装 Microsoft Visual C++ 运行库：
https://aka.ms/vs/17/release/vc_redist.x64.exe
EOF
sed 's/$/\r/' /tmp/oku.txt > "$P/OKU_README.txt"

cat > /tmp/ozet.txt << 'OZETEOF'
AI-Based Cell Image Sorting: Project Summary
============================================

This package contains the current version of the system. See OKU_README.txt
for how to run it.

1. What the system does

The system automatically sorts cell images into the factory's defect categories,
using the same folder structure as our current sorting process, and identifies
A-class cells. Cracks and pressing-plate marks are collected in "suspected C"
folders, split into three groups by confidence (high probability, low suspicion,
re-evaluate) so they can be checked efficiently.

The goal is to reduce the manual sorting workload, make defect statistics
consistent from shift to shift, and give earlier visibility of process issues on
the production lines.

2. How it was trained

- Over the course of development, the system has worked with close to 200,000
  cell images: a pool of about 165,000 labeled images used to build the models,
  plus additional days used only for testing.
- The initial training set of about 40,000 images was mostly sorted by hand, one
  by one, by me; the rest of the pool came from daily sorting.
- All images in the pool were processed by the system, and about 60,000 were
  selected for training based on label reliability and category balance.
- Around 25,000 images were reviewed individually by me to provide reliable
  labels and to measure accuracy.
- The system combines several specialized models: one for the A/B decision, two
  category models working together, a dedicated higher-resolution detector for
  cracks and pressing-plate marks, and a model specifically trained for the newly
  defined contamination type.

How accuracy is measured: On days the models have never seen, I select a balanced
sample of about 2,000 images and check every result individually. Siyah nokta
(black spot) is the largest category and has consistently shown ~95-96% accuracy,
so only a random quarter of that folder is checked, and each checked image counts
as four. Because the selection is random, this gives the same statistical result
as checking the whole folder.

3. Current results

Measured on seven separate days the models had never seen:
- Defect category accuracy: about 92-94%
- A/B decision accuracy: about 98-99%
- Cracks and pressing-plate marks: nearly all are placed in the suspected-C
  folders. By design, about 3-4% may land in other folders; this keeps the C
  folders a manageable size.

4. Challenges

Hardware: Training runs on a 13-year-old laptop without a graphics card. Each
training run takes a full night (6-10 hours), so only one experiment can be done
per day, and the size and resolution of the models are limited. This has been the
biggest constraint on development speed.

Changing defect definitions: During the project, three category definitions
changed: dot pressing marks were reclassified as airflow marks, one type of
airflow mark as contamination, and one type of broken grid as irregular scratch.
The model keeps applying the old definition until it is retrained with examples
of the new one. For the contamination change, after targeted retraining with
about 200 newly labeled examples, detection of this type rose from 34% to 82%.

Rare and unseen defect types: The model can only choose among the categories it
has learned. On 9.30, conveyor belt marks, which appeared only a handful of times
in the training data, were all sorted as roller marks, and a type of scratch that
had almost never been seen before (20-30 cells) was spread across several
categories.

Crack vs. scratch: Separating fine cracks from scratches is inherently difficult,
which is why suspected cracks are always routed to a folder for human review.

AI tool limits: Development relies on a personal AI assistant subscription
($20/month) with usage limits, which occasionally interrupts the work.

5. Next steps

- Teach the model the latest definition change (broken grid -> irregular scratch)
  and add conveyor belt marks as a separate category, using a similar-image search
  tool that finds related examples in the existing data.
- A dedicated model for broken grid, currently the most frequently missed category.
- An "uncertain" folder for cells the model is not confident about, so that new or
  unusual defect types are flagged for review instead of being guessed.
- Testing larger and higher-resolution models.

6. How it will work in daily use

A desktop application will:
- automatically collect images from the test computers through the shared folder
  and sort them,
- allow image collection to be paused per line (e.g., during CC or ELBL testing),
- allow decision thresholds to be adjusted,
- generate the 4-hour report automatically.

Operators will look through all folders and move any misplaced images, checking
the A and C folders more carefully because they are the most critical. At the end
of the shift, C-class cells are separated into cracks and pressing-plate marks in
the current sorting program.

Every correction is recorded automatically and used in regular retraining. A new
model is put into use only after it passes a fixed test set, so performance cannot
silently get worse. Over time we expect the system to improve, especially for new
definitions and rare defect types, as operator corrections accumulate.

A note on the work: Most of this project was developed outside working hours.
During working hours, I made sure none of my main duties were left undone.
OZETEOF
sed 's/$/\r/' /tmp/ozet.txt > "$P/PROJECT_SUMMARY.txt"

echo "== 5/5 Zip olusturuluyor"
cd "$HOME/Desktop"
find ELAI_paket -name "__pycache__" -type d -prune -exec rm -rf {} +
zip -qr ELAI_paket.zip ELAI_paket
echo
echo "TAMAM: $HOME/Desktop/ELAI_paket.zip  ($(du -h ELAI_paket.zip | cut -f1))"
