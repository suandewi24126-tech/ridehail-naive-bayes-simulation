# Klasifikasi Subjektivitas Ulasan Ride-Hailing dengan Multinomial Naive Bayes

Repository proyek Pemodelan dan Simulasi: klasifikasi ulasan aplikasi Gojek, Grab, dan Maxim ke kelas **Objektif** dan **Subjektif** menggunakan pipeline **TF-IDF + Multinomial Naive Bayes**.

> **Penting:** `data/demo_reviews.csv` berisi contoh sintetis untuk memeriksa apakah aplikasi berjalan. Jangan gunakan data demo, metrik demo, atau tangkapan layarnya sebagai hasil penelitian. Ganti dengan dataset SubRide berlabel asli sebelum melakukan eksperimen dan menulis laporan.

## 1. Kesesuaian dengan proyek sebelumnya dan UAS

Laporan kelompok sebelumnya menggunakan dataset SubRide dengan 1.338 ulasan, atribut `score`, `app`, `review`, dan `label`, serta konvensi label `0 = objektif` dan `1 = subjektif`. Laporan tersebut menguraikan 960 ulasan subjektif dan 378 objektif. Repository ini mempertahankan studi kasus dan skema data tersebut, tetapi mengganti algoritma utama menjadi Multinomial Naive Bayes agar sesuai dengan salah satu metode yang disebut pada lembar UAS.

## 2. Struktur repository

```text
ridehail-nb-simulation/
├── app.py
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── data/
│   ├── demo_reviews.csv        # sintetis, hanya untuk uji fungsi
│   ├── raw/                    # letakkan dataset asli di sini (tidak diunggah Git)
│   └── processed/
├── ridehail_nb/
│   ├── data.py                 # baca dan validasi CSV
│   ├── preprocessing.py        # pembersihan teks
│   ├── modeling.py             # TF-IDF + MultinomialNB
│   ├── simulation.py           # eksperimen parameter dan skenario data
│   └── reporting.py            # penyimpanan model dan metrik
├── scripts/
│   ├── train_model.py
│   └── run_simulation.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── models/                     # model hasil pelatihan (lokal)
└── reports/                    # metrik dan hasil simulasi (lokal)
```

## 3. Dataset asli yang diperlukan

CSV minimal harus memiliki dua kolom:

| Kolom | Keterangan |
|---|---|
| `review` | Teks ulasan berbahasa Indonesia |
| `label` | `0` untuk objektif, `1` untuk subjektif |

Kolom tambahan `score` dan `app` boleh disertakan. Nama kolom tidak peka huruf besar/kecil. Nilai label yang diterima adalah `0/1`, `objektif/subjektif`, atau `objective/subjective`.

Simpan dataset asli secara lokal sebagai:

```text
data/raw/subride.csv
```

Jangan unggah dataset ke GitHub sebelum memeriksa lisensi, izin redistribusi, dan kemungkinan adanya informasi pribadi. `.gitignore` mengabaikan CSV pada `data/raw/`.

## 4. Menjalankan di Visual Studio Code

Disarankan menggunakan Python 3.11 atau 3.12.

### Windows PowerShell

```powershell
# Buka folder repository di VS Code, lalu buka Terminal
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Jika PowerShell menolak aktivasi environment, jalankan sekali pada terminal pengguna:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Lalu aktifkan kembali `.venv`.

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Di VS Code pilih interpreter dari `.venv` melalui **Python: Select Interpreter** pada Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).

## 5. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka pada alamat lokal yang dicetak Streamlit. Saat pertama kali dijalankan, aplikasi memakai dataset demo sintetis. Untuk analisis sebenarnya, unggah dataset asli menggunakan tombol **Unggah dataset CSV** pada sidebar.

Halaman aplikasi:

1. **Ringkasan data**: jumlah baris, distribusi kelas, dan pratinjau dataset.
2. **Pelatihan & prediksi**: melatih model, menampilkan metrik, confusion matrix, classification report, dan memprediksi teks baru.
3. **Simulasi eksperimen**: menguji variasi `alpha`, jumlah data latih, serta distribusi kelas pada data training.

## 6. Jalankan pelatihan dari terminal

Setelah menyimpan dataset asli di `data/raw/subride.csv`:

```bash
python scripts/train_model.py --data data/raw/subride.csv --alpha 1.0 --test-size 0.2 --seed 42
```

Hasil model disimpan lokal di `models/naive_bayes_pipeline.joblib`; metrik disimpan di `reports/metrics.json`.

## 7. Jalankan simulasi eksperimen

```bash
python scripts/run_simulation.py --data data/raw/subride.csv --seeds 5
```

Eksperimen yang dijalankan:

- **Variasi alpha**: 0.1, 0.5, 1.0, 2.0, dan 5.0.
- **Variasi jumlah data latih**: 60%, 70%, dan 80% dari seluruh dataset. Data uji 20% ditahan tetap untuk setiap seed; subset yang lebih kecil diambil dari pool training 80% yang sama.
- **Distribusi kelas training**: distribusi asli dibandingkan dengan random undersampling. Undersampling hanya dilakukan pada data training; data uji tetap mengikuti distribusi alami.
- Setiap skenario diulang dengan lima random seed (42–46) secara default.

Hasil run dan ringkasannya disimpan di `reports/`. File hasil tersebut diabaikan Git secara default.

## 8. Pengujian kode

```bash
pytest -q
```

## 9. Catatan metodologis penting

- TF-IDF berada di dalam `Pipeline`, sehingga vectorizer hanya di-fit pada data training. Ini menghindari kebocoran informasi dari data uji.
- Pembagian data memakai stratifikasi agar proporsi kelas dipertahankan pada holdout.
- Untuk simulasi undersampling, data uji tidak ikut diseimbangkan.
- F1 yang digunakan dalam tabel utama adalah **macro F1**, sehingga performa kedua kelas diperhitungkan secara setara.
- Confidence yang ditampilkan Naive Bayes adalah probabilitas keluaran model, bukan jaminan kebenaran prediksi.
- Periksa kembali definisi label subjektif/objektif dan konsistensi anotasi sebelum pelatihan.
- Simulasi di repository ini adalah rancangan eksperimen yang perlu dijalankan memakai dataset asli. Hasil demo sintetis tidak boleh dikutip sebagai hasil penelitian.

## 10. Membuat repository GitHub

1. Buat repository baru di GitHub, misalnya `ridehail-naive-bayes-simulation`.
2. Jangan pilih opsi menambahkan README atau `.gitignore` jika folder lokal ini sudah berisi keduanya.
3. Dari terminal pada folder proyek jalankan:

```bash
git init
git add .
git status
git commit -m "Initial project: TF-IDF Multinomial Naive Bayes simulation"
git branch -M main
git remote add origin https://github.com/USERNAME/ridehail-naive-bayes-simulation.git
git push -u origin main
```

Ganti `USERNAME` dengan username GitHub milik kelompok. Periksa `git status` sebelum commit untuk memastikan dataset asli dan file rahasia tidak ikut terunggah.
