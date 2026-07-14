# Kacamata Pintar SIGMA 2.0 👓

Repositori ini berisi kode sumber untuk proyek **Kacamata Pintar SIGMA 2.0**, yang berfokus pada fitur komunikasi dua arah bagi penyandang tunarungu.

### 📁 Struktur Modul
1. **`Proyek OPSI/`**: Inti kecerdasan visual (*Sign-to-Text*) menggunakan YOLOv8 dan MobileSAM untuk mendeteksi bahasa isyarat (BISINDO) secara *real-time*.
2. **`STT/`**: Modul *Speech-to-Text* (pendengaran) offline menggunakan OpenAI Whisper untuk mendengarkan ucapan lawan bicara dan menampilkannya sebagai *subtitle* di layar kacamata.

### 📥 Tautan Dataset (12 GB)
Karena ukuran dataset sangat besar, dataset gambar tidak disimpan di Github. Silakan unduh dataset melalui link Google Drive berikut:
*(Tempelkan Link Google Drive Bersama Anda di sini)*

### 🚀 Cara Menjalankan Program (Instalasi)
Untuk menjalankan sistem tanpa error, buat *virtual environment* lokal Anda sendiri dengan 3 langkah mudah ini:

```bash
# 1. Buat virtual environment baru
python -m venv .venv

# 2. Aktifkan environment (Khusus Windows)
.venv\Scripts\activate

# 3. Instal seluruh library AI yang dibutuhkan
pip install -r STT/requirements.txt
```
