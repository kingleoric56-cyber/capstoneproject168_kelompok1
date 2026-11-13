"""
Sistem Klasifikasi Penyakit Daun Tomat
Backend: Flask
Model: Modified K-Nearest Neighbor (MKNN) dengan fitur HSV Color Moment + Local Binary Pattern (LBP)

Cara menjalankan (lewat Anaconda Prompt):
    conda create -n tomat-mknn python=3.10 -y
    conda activate tomat-mknn
    pip install -r requirements.txt
    python app.py

Lalu buka browser ke http://127.0.0.1:5000
"""

import os
import pickle
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from scipy.spatial.distance import cdist
from scipy.stats import skew
from skimage.feature import local_binary_pattern
from werkzeug.utils import secure_filename

# --------------------------------------------------------------------------
# Konfigurasi Aplikasi
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_mknn.pkl")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
app.secret_key = "ganti-dengan-secret-key-anda"

NAMA_MAHASISWA = "CapstoneProject 168 - Kelompok 1"

# Label yang dipakai saat gambar yang diunggah dianggap terlalu jauh/berbeda
# dari seluruh data latih (kemungkinan bukan daun tomat / bukan salah satu dari 3 kelas)
UNKNOWN_LABEL = "Tidak Dikenali"

# Semakin besar nilai ini (mendekati 100), semakin "longgar" sistem menerima gambar
# sebagai salah satu dari 3 kelas. Semakin kecil, semakin ketat/mudah menolak gambar asing.
OOD_PERCENTILE = 98.5

# --------------------------------------------------------------------------
# Informasi edukasi tiap kelas (ditampilkan di halaman hasil & halaman utama)
# --------------------------------------------------------------------------
INFO_PENYAKIT = {
    "Bercak Kering": {
        "slug": "bercak-kering",
        "gejala": (
            "Muncul bercak-bercak kecil berwarna coklat tua hingga kehitaman pada daun bagian "
            "bawah/tua terlebih dahulu, dengan pola cincin konsentris (target spot) di tengah "
            "bercak. Bercak dapat melebar dan menyatu, dikelilingi area menguning (klorosis), "
            "sehingga daun mengering dan rontok dari bagian bawah tanaman ke atas."
        ),
        "penyebab": "Cendawan/jamur Alternaria solani (penyakit bercak kering / early blight).",
        "penanganan": (
            "(1) Menjaga jarak tanam agar sirkulasi udara baik dan kelembaban daun tidak terlalu "
            "tinggi. (2) Membuang dan memusnahkan daun yang terinfeksi agar tidak menjadi sumber "
            "penularan. (3) Melakukan rotasi tanaman, tidak menanam tomat/famili Solanaceae secara "
            "berturut-turut di lahan yang sama. (4) Aplikasi fungisida berbahan aktif klorotalonil "
            "atau mankozeb sesuai dosis anjuran, terutama pada musim dengan kelembaban tinggi."
        ),
    },
    "Busuk Daun Fitoftora": {
        "slug": "busuk-daun-fitoftora",
        "gejala": (
            "Gejala awal berupa bercak kebasah-basahan berwarna hijau pucat hingga coklat pada "
            "tepi atau ujung daun, kemudian melebar dengan cepat dan membentuk area nekrotik "
            "berwarna coklat gelap. Pada kondisi lembab, di bagian bawah daun sering terlihat "
            "lapisan spora berwarna putih keabu-abuan di sekitar tepi bercak."
        ),
        "penyebab": "Cendawan/jamur Phytophthora infestans (penyakit busuk daun / late blight).",
        "penanganan": (
            "(1) Memasang turus/ajir sebagai penegak tanaman agar sirkulasi udara lebih baik dan "
            "daun tidak lembab berkepanjangan. (2) Pengendalian secara preventif menggunakan "
            "fungisida sistemik/protektif dengan interval 3-4 hari pada musim hujan dan 5-7 hari "
            "pada musim kemarau. (3) Menghindari penyiraman yang membasahi daun, sebaiknya "
            "penyiraman dilakukan pada pangkal batang. (4) Segera memusnahkan tanaman yang "
            "terserang parah untuk mencegah penyebaran spora."
        ),
    },
    "Sehat": {
        "slug": "sehat",
        "gejala": (
            "Daun tomat sehat memiliki warna hijau segar merata, permukaan mulus tanpa bercak, "
            "layu, maupun perubahan warna, serta pertumbuhan daun yang normal."
        ),
        "penyebab": "Tidak ada infeksi patogen — kondisi tanaman dalam keadaan baik.",
        "penanganan": (
            "(1) Lakukan penyiraman secara rutin sesuai kebutuhan dan pastikan drainase lahan "
            "baik. (2) Lakukan pemupukan berimbang dan pemeriksaan tanaman secara berkala untuk "
            "deteksi dini apabila muncul gejala penyakit. (3) Jaga kebersihan lahan dari gulma dan "
            "sisa tanaman sakit."
        ),
    },
    UNKNOWN_LABEL: {
        "slug": "tidak-dikenali",
        "gejala": (
            "Karakteristik warna dan tekstur pada gambar ini berbeda cukup jauh dari seluruh data "
            "latih daun tomat yang dipelajari sistem, sehingga tidak dapat digolongkan sebagai "
            "salah satu dari tiga kelas yang dikenali (Bercak Kering, Busuk Daun Fitoftora, atau Sehat)."
        ),
        "penyebab": (
            "Kemungkinan gambar yang diunggah bukan citra daun tomat, atau kualitas/pencahayaan "
            "gambar terlalu berbeda dari data latih (misalnya terlalu gelap, buram, atau bukan close-up daun)."
        ),
        "penanganan": (
            "Silakan unggah ulang dengan foto close-up satu helai daun tomat, pencahayaan cukup, "
            "dan latar belakang polos, agar sistem dapat mengenali fitur warna dan teksturnya dengan baik."
        ),
    },
}

# --------------------------------------------------------------------------
# Ekstraksi Fitur — HARUS SAMA PERSIS dengan yang dipakai saat training di Colab
# --------------------------------------------------------------------------
def extract_hsv_color_moment(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        channel = hsv[:, :, ch].astype(np.float64).flatten()
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        skew_val = skew(channel)
        feats.extend([mean_val, std_val, skew_val])
    return feats


def extract_lbp_features(img_bgr, P=8, R=1):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=P, R=R, method="default")

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-9)

    idx = np.arange(256)
    mean_val = np.sum(idx * prob)
    variance = np.sum(((idx - mean_val) ** 2) * prob)
    std_val = np.sqrt(variance) if variance > 0 else 1e-9
    skewness = np.sum(((idx - mean_val) ** 3) * prob) / (std_val ** 3 + 1e-9)
    kurtosis_val = (np.sum(((idx - mean_val) ** 4) * prob) / (std_val ** 4 + 1e-9)) - 3
    nz = prob[prob > 0]
    entropy_val = -np.sum(nz * np.log2(nz))

    return [mean_val, variance, skewness, kurtosis_val, entropy_val]


def extract_features(img_bgr, img_size):
    img_resized = cv2.resize(img_bgr, (img_size, img_size))
    hsv_feats = extract_hsv_color_moment(img_resized)
    lbp_feats = extract_lbp_features(img_resized)
    return np.array(hsv_feats + lbp_feats).reshape(1, -1)


# --------------------------------------------------------------------------
# Modified K-Nearest Neighbor — prediksi memakai artefak yang sudah dilatih
# --------------------------------------------------------------------------
class ModifiedKNNPredictor:
    def __init__(self, artifact, ood_percentile=OOD_PERCENTILE):
        self.X_train = artifact["X_train"]
        self.y_train = artifact["y_train"]
        self.validity = artifact["validity"]
        self.k = artifact["best_k"]
        self.a = artifact.get("a", 0.5)
        self.scaler = artifact["scaler"]
        self.class_names = artifact["class_names"]
        self.img_size = artifact["img_size"]

        # Ambang jarak untuk mendeteksi gambar yang "asing" (out-of-distribution),
        # dihitung langsung dari data latih yang sudah tersimpan di model_mknn.pkl.
        # Tidak perlu training ulang di Colab.
        self.distance_threshold = self._compute_distance_threshold(ood_percentile)

    @staticmethod
    def _euclidean(X, x):
        return np.sqrt(np.sum((X - x) ** 2, axis=1))

    def _compute_distance_threshold(self, percentile):
        """Untuk tiap data latih, hitung jarak ke tetangga ke-k terdekatnya sendiri
        (leave-one-out). Ambang batas = persentil ke-`percentile` dari jarak-jarak itu.
        Gambar baru yang jaraknya ke tetangga ke-k lebih jauh dari ambang ini dianggap
        tidak mirip data latih manapun -> ditolak sebagai 'Tidak Dikenali'."""
        dist_matrix = cdist(self.X_train, self.X_train)
        sorted_dist = np.sort(dist_matrix, axis=1)
        # kolom index 0 = jarak ke diri sendiri (0), kolom index k = tetangga ke-k
        kth_distances = sorted_dist[:, self.k]
        return float(np.percentile(kth_distances, percentile))

    def predict(self, feature_vector):
        x_scaled = self.scaler.transform(feature_vector)[0]
        dists = self._euclidean(self.X_train, x_scaled)
        sorted_idx = np.argsort(dists)
        k_idx = sorted_idx[: self.k]

        # --- Deteksi out-of-distribution ---
        kth_distance = dists[sorted_idx[self.k - 1]]
        if kth_distance > self.distance_threshold:
            return UNKNOWN_LABEL, {UNKNOWN_LABEL: 100.0}

        weight_per_class = {}
        for i in k_idx:
            w = self.validity[i] * (1.0 / (dists[i] + self.a))
            label = self.y_train[i]
            weight_per_class[label] = weight_per_class.get(label, 0.0) + w

        total_weight = sum(weight_per_class.values()) + 1e-9
        probabilities = {
            label: round((w / total_weight) * 100, 2)
            for label, w in weight_per_class.items()
        }
        pred_label = max(weight_per_class, key=weight_per_class.get)
        return pred_label, probabilities


# --------------------------------------------------------------------------
# Memuat model saat aplikasi start
# --------------------------------------------------------------------------
mknn_model = None
model_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    mknn_model = ModifiedKNNPredictor(artifact)
    print(f"[INFO] Model MKNN berhasil dimuat. K optimal = {mknn_model.k}")
    print(f"[INFO] Ambang deteksi gambar asing (distance_threshold) = {mknn_model.distance_threshold:.4f}")
except FileNotFoundError:
    model_load_error = (
        f"File model tidak ditemukan di '{MODEL_PATH}'. "
        "Jalankan notebook training di Google Colab terlebih dahulu, lalu salin "
        "'model_mknn.pkl' ke folder backend/model/."
    )
    print(f"[WARNING] {model_load_error}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    # "Tidak Dikenali" hanya dipakai di halaman hasil klasifikasi, bukan untuk
    # ditampilkan sebagai salah satu objek penelitian di halaman utama.
    info_penyakit_utama = {
        label: info for label, info in INFO_PENYAKIT.items() if label != UNKNOWN_LABEL
    }
    return render_template(
        "index.html",
        nama_mahasiswa=NAMA_MAHASISWA,
        info_penyakit=info_penyakit_utama,
    )


@app.route("/klasifikasi", methods=["GET"])
def klasifikasi():
    return render_template(
        "klasifikasi.html",
        nama_mahasiswa=NAMA_MAHASISWA,
        model_error=model_load_error,
        best_k=mknn_model.k if mknn_model else None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if mknn_model is None:
        flash("Model belum tersedia. Silakan jalankan training terlebih dahulu.")
        return redirect(url_for("klasifikasi"))

    if "file" not in request.files or request.files["file"].filename == "":
        flash("Silakan pilih file gambar terlebih dahulu.")
        return redirect(url_for("klasifikasi"))

    file = request.files["file"]

    if not allowed_file(file.filename):
        flash("Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.")
        return redirect(url_for("klasifikasi"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        flash("Gagal membaca file gambar. Coba unggah gambar lain.")
        return redirect(url_for("klasifikasi"))

    feature_vector = extract_features(img_bgr, mknn_model.img_size)
    pred_label, probabilities = mknn_model.predict(feature_vector)

    info = INFO_PENYAKIT.get(pred_label, {})

    return render_template(
        "hasil.html",
        nama_mahasiswa=NAMA_MAHASISWA,
        image_path=url_for("static", filename=f"uploads/{unique_name}"),
        image_filename=unique_name,
        pred_label=pred_label,
        probabilities=probabilities,
        gejala=info.get("gejala", "-"),
        penyebab=info.get("penyebab", "-"),
        penanganan=info.get("penanganan", "-"),
        best_k=mknn_model.k,
    )


@app.route("/proses/<path:filename>")
def proses_klasifikasi(filename):
    """Menampilkan detail tahapan klasifikasi: preprocessing, ekstraksi fitur
    HSV & LBP, hingga perhitungan Modified K-Nearest Neighbor untuk gambar
    yang sudah pernah diunggah (dipanggil ulang dari halaman hasil)."""
    if mknn_model is None:
        flash("Model belum tersedia. Silakan jalankan training terlebih dahulu.")
        return redirect(url_for("klasifikasi"))

    safe_name = secure_filename(filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    if not os.path.isfile(filepath):
        flash("Gambar tidak ditemukan. Silakan unggah ulang.")
        return redirect(url_for("klasifikasi"))

    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        flash("Gagal membaca file gambar.")
        return redirect(url_for("klasifikasi"))

    img_size = mknn_model.img_size
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized_bgr = cv2.resize(img_bgr, (img_size, img_size))
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # Simpan citra hasil preprocessing (resize) supaya bisa ditampilkan di halaman
    resized_filename = f"resized_{safe_name}"
    resized_path = os.path.join(app.config["UPLOAD_FOLDER"], resized_filename)
    cv2.imwrite(resized_path, resized_bgr)

    # Cuplikan matriks citra (format mirip tampilan numpy array, ada elipsis "...")
    np.set_printoptions(precision=4, suppress=True, threshold=60, edgeitems=3)
    matriks_citra = str(img_rgb.astype(float))
    matriks_preprocessing = str(resized_rgb.astype(float) / 255.0)
    np.set_printoptions()  # kembalikan ke pengaturan default

    # Ekstraksi fitur (identik dengan pipeline training/predict)
    hsv_feats = extract_hsv_color_moment(resized_bgr)
    lbp_feats = extract_lbp_features(resized_bgr)
    feature_vector = np.array(hsv_feats + lbp_feats).reshape(1, -1)

    hsv_labels = [
        "H - Mean", "H - Std Deviasi", "H - Skewness",
        "S - Mean", "S - Std Deviasi", "S - Skewness",
        "V - Mean", "V - Std Deviasi", "V - Skewness",
    ]
    lbp_labels = ["Mean", "Variance", "Skewness", "Kurtosis", "Entropy"]

    hsv_table = list(zip(hsv_labels, [round(v, 4) for v in hsv_feats]))
    lbp_table = list(zip(lbp_labels, [round(v, 4) for v in lbp_feats]))

    # Detail perhitungan MKNN
    x_scaled = mknn_model.scaler.transform(feature_vector)[0]
    dists = mknn_model._euclidean(mknn_model.X_train, x_scaled)
    sorted_idx = np.argsort(dists)
    k = mknn_model.k
    k_idx = sorted_idx[:k]

    neighbor_rows = []
    weight_per_class = {}
    for rank, i in enumerate(k_idx, start=1):
        d = float(dists[i])
        v = float(mknn_model.validity[i])
        w = v * (1.0 / (d + mknn_model.a))
        label = mknn_model.y_train[i]
        weight_per_class[label] = weight_per_class.get(label, 0.0) + w
        neighbor_rows.append({
            "rank": rank,
            "label": label,
            "distance": round(d, 4),
            "validity": round(v, 4),
            "weight": round(w, 4),
        })

    kth_distance = float(dists[sorted_idx[k - 1]])
    is_unknown = kth_distance > mknn_model.distance_threshold

    total_weight = sum(weight_per_class.values()) + 1e-9
    voting_table = [
        {"label": label, "weight": round(w, 4), "percent": round((w / total_weight) * 100, 2)}
        for label, w in sorted(weight_per_class.items(), key=lambda item: item[1], reverse=True)
    ]

    if is_unknown:
        final_label = UNKNOWN_LABEL
        final_percent = 100.0
    else:
        final_label = max(weight_per_class, key=weight_per_class.get)
        final_percent = round((weight_per_class[final_label] / total_weight) * 100, 2)

    return render_template(
        "proses.html",
        nama_mahasiswa=NAMA_MAHASISWA,
        original_image=url_for("static", filename=f"uploads/{safe_name}"),
        resized_image=url_for("static", filename=f"uploads/{resized_filename}"),
        matriks_citra=matriks_citra,
        matriks_preprocessing=matriks_preprocessing,
        hsv_table=hsv_table,
        lbp_table=lbp_table,
        neighbor_rows=neighbor_rows,
        voting_table=voting_table,
        final_label=final_label,
        final_percent=final_percent,
        k=k,
        distance_threshold=round(mknn_model.distance_threshold, 4),
        kth_distance=round(kth_distance, 4),
        is_unknown=is_unknown,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
