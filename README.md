# Sistem Klasifikasi Penyakit Daun Tomat (MKNN + HSV + LBP)

Implementasi sesuai proposal *"Implementasi Modified K-Nearest Neighbor (MKNN) untuk
Klasifikasi Penyakit Daun Tomat"*. Kelas: **Bercak Kering**, **Busuk Daun Fitoftora**, **Sehat**.

Nilai **K ditentukan otomatis** (bukan manual) melalui grid-search terhadap akurasi data
validasi pada saat training di Google Colab.

## Struktur Folder

```
TomatoDiseaseMKNN/
├── training/
│   └── TomatoDiseaseMKNN_Training.ipynb   <- dijalankan di Google Colab
├── backend/
│   ├── app.py                             <- Flask backend
│   ├── requirements.txt
│   ├── model/
│   │   └── model_mknn.pkl                 <- hasil training, ditaruh di sini
│   ├── static/
│   │   ├── css/style.css
│   │   └── uploads/                       <- gambar yang diunggah user (otomatis)
│   └── templates/
│       ├── base.html
│       ├── index.html
│       ├── klasifikasi.html
│       └── hasil.html
└── README.md
```

## 1. Training Model (Google Colab)

1. Pastikan folder dataset di Google Drive Anda bernama **`TomatoPlants`** dengan struktur:
   ```
   MyDrive/TomatoPlants/
   ├── Bercak Kering/            (1000 gambar)
   ├── Busuk Daun Fitoftora/     (1000 gambar)
   └── Sehat/                    (1000 gambar)
   ```
2. Buka **Google Colab**, upload/`File > Open notebook` file
   `training/TomatoDiseaseMKNN_Training.ipynb`.
3. Jalankan seluruh cell secara berurutan (`Runtime > Run all`).
   - Cell akan meminta izin akses Google Drive (mount drive).
   - Sistem melakukan ekstraksi fitur HSV Color Moment (9 fitur) + LBP (5 fitur) = 14 fitur.
   - Sistem mencoba beberapa nilai K (1, 3, 5, ..., 25) secara otomatis dan memilih K dengan
     akurasi validasi terbaik.
   - Confusion matrix dan classification report akan ditampilkan.
4. Di cell terakhir, file **`model_mknn.pkl`** akan otomatis terunduh ke komputer Anda
   (lewat `files.download`).
5. Pindahkan file `model_mknn.pkl` tersebut ke folder:
   ```
   backend/model/model_mknn.pkl
   ```

## 2. Menyiapkan Kode di VS Code

1. Buka folder `TomatoDiseaseMKNN/backend` di **Visual Studio Code**.
2. Anda bisa mengedit tampilan web melalui:
   - `templates/index.html`, `templates/klasifikasi.html`, `templates/hasil.html`
   - `static/css/style.css`
   - Logika backend ada di `app.py`.

## 3. Menjalankan Server Lokal (Anaconda Prompt)

Buka **Anaconda Prompt**, lalu jalankan:

```bat
cd path\ke\TomatoDiseaseMKNN\backend

conda create -n tomat-mknn python=3.10 -y
conda activate tomat-mknn

pip install -r requirements.txt

python app.py
```

Setelah server berjalan, akan muncul log seperti:

```
[INFO] Model MKNN berhasil dimuat. K optimal = 7
 * Running on http://127.0.0.1:5000
```

Buka browser dan akses **http://127.0.0.1:5000**

## 4. Alur Penggunaan Web

1. **Halaman Home** — menampilkan penjelasan sistem dan objek penelitian (3 kelas penyakit).
2. **Halaman Klasifikasi** — unggah gambar daun tomat (drag & drop atau pilih file), lalu
   klik **Predict**.
3. **Halaman Hasil** — menampilkan gambar yang diunggah, label hasil klasifikasi, persentase
   keyakinan tiap kelas, serta informasi Gejala / Penyebab / Penanganan.

## Catatan Penting

- Fungsi ekstraksi fitur pada `backend/app.py` **harus identik** dengan fungsi ekstraksi fitur
  pada notebook Colab (resize 128x128, HSV color moment 9 fitur, LBP statistik 5 fitur).
  Jika Anda mengubah salah satunya, ubah juga yang lain agar hasil prediksi tetap konsisten.
- MKNN adalah algoritma *lazy learner*: seluruh data latih beserta nilai validitasnya ikut
  disimpan di dalam `model_mknn.pkl`, sehingga ukuran file model relatif lebih besar
  dibanding model machine learning umumnya — ini adalah hal yang wajar.
- Jika ingin mengganti/menambah dataset, cukup ulangi langkah **Training Model** di Colab,
  lalu ganti file `model_mknn.pkl` di folder `backend/model/`.
