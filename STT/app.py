import queue
import threading
from collections import deque
import pyaudio
import numpy as np
import whisper
import customtkinter as ctk

# Konfigurasi Audio
RATE = 16000
CHANNELS = 1
CHUNK = 1024

# Konfigurasi VAD (Voice Activity Detection) menggunakan Energy Threshold
ENERGY_THRESHOLD = 0.02  # Ambang batas volume (sesuaikan jika kurang peka/terlalu peka)
SILENCE_LIMIT = 1.0      # Batas waktu hening (dalam detik) sebelum mengirim kalimat

# Konfigurasi Tema UI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class STTApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Konfigurasi Window Utama
        self.title("SIGMA: Whisper Speech-to-Text")
        self.geometry("800x600")
        self.minsize(600, 400)

        # Header Title
        self.header_label = ctk.CTkLabel(
            self, 
            text="SIGMA: Whisper Offline STT (Base)", 
            font=ctk.CTkFont(family="Inter", size=24, weight="bold")
        )
        self.header_label.pack(pady=(20, 10))

        # Status Label
        self.status_label = ctk.CTkLabel(
            self, 
            text="Status: Mengunduh/Memuat model Whisper 'base'...", 
            font=ctk.CTkFont(family="Inter", size=14),
            text_color="orange"
        )
        self.status_label.pack(pady=(0, 10))

        # Text Box untuk menampilkan transkripsi
        self.textbox = ctk.CTkTextbox(
            self, 
            width=700, 
            height=400, 
            font=ctk.CTkFont(family="Inter", size=28),
            wrap="word",
            corner_radius=10
        )
        self.textbox.pack(pady=10, padx=20, expand=True, fill="both")
        
        self.is_running = True
        self.audio_queue = queue.Queue()

        # Mulai inisialisasi di thread terpisah
        threading.Thread(target=self.init_system, daemon=True).start()

    def init_system(self):
        try:
            # Menggunakan model 'base'
            self.model = whisper.load_model("base")
            self.after(0, self.set_status, "Status: Model 'base' dimuat. Menyiapkan mikrofon...", "orange")

            # Inisialisasi PyAudio
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16, 
                channels=CHANNELS, 
                rate=RATE, 
                input=True, 
                frames_per_buffer=CHUNK
            )

            self.after(0, self.set_status, "Status: Mendengarkan suara...", "green")

            # Jalankan Thread Perekaman (VAD) & Thread Transkripsi
            threading.Thread(target=self.record_audio, daemon=True).start()
            threading.Thread(target=self.transcribe_audio, daemon=True).start()

        except Exception as e:
            self.after(0, self.set_status, "Error: Inisialisasi gagal.", "red")
            self.after(0, self.update_text, f"[ERROR]: {e}")

    def record_audio(self):
        """Merekam menggunakan VAD berbasis Energy Threshold (Deteksi Volume)."""
        silence_chunks_limit = int(SILENCE_LIMIT * RATE / CHUNK)
        # Menyimpan sekitar 5 chunk (~300ms) sebelum suara terdeteksi agar suku kata pertama tidak terpotong
        pre_speech_chunks = 5 
        
        try:
            while self.is_running:
                frames = []
                silence_chunks = 0
                is_speaking = False
                ring_buffer = deque(maxlen=pre_speech_chunks)

                while self.is_running:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    
                    # Konversi audio ke tipe float32 dan hitung Root Mean Square (RMS)
                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = np.sqrt(np.mean(audio_np**2))

                    if not is_speaking:
                        # Terus simpan potongan audio terakhir di ring buffer saat masih hening
                        ring_buffer.append(data)
                        
                        # Jika volume suara melampaui ambang batas, tandai mulai bicara
                        if rms > ENERGY_THRESHOLD:
                            is_speaking = True
                            frames.extend(ring_buffer) # Masukkan potongan awal suara
                            ring_buffer.clear()
                    else:
                        # Sedang bicara, terus rekam
                        frames.append(data)
                        
                        # Jika suara mulai pelan/hening, tambah counter hening
                        if rms < ENERGY_THRESHOLD:
                            silence_chunks += 1
                        else:
                            silence_chunks = 0 # Reset hening jika suara kembali keras

                        # Jika durasi hening sudah menyentuh batas (1 detik), potong rekaman kalimat ini
                        if silence_chunks >= silence_chunks_limit:
                            break

                # Setelah kalimat terpotong, kirim ke antrean untuk ditranskripsi
                if frames and self.is_running:
                    audio_data = b''.join(frames)
                    self.audio_queue.put(audio_data)

        except Exception as e:
            self.after(0, self.set_status, "Error: Perekaman terhenti.", "red")
            self.after(0, self.update_text, f"[ERROR Record]: {e}")

    def transcribe_audio(self):
        """Mengambil data dari antrean audio, memprosesnya dengan Whisper 'base', lalu update UI."""
        try:
            while self.is_running:
                # Blokir antrean sampai ada rekaman 1 kalimat utuh yang dikirim dari record_audio()
                audio_data = self.audio_queue.get()
                if audio_data is None:
                    break
                
                self.after(0, self.set_status, "Status: Memproses suara...", "orange")
                
                # Konversi data audio raw ke format Numpy Array Float32
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Lakukan inferensi
                result = self.model.transcribe(audio_np, language='id', fp16=False)
                text = result.get('text', '').strip()
                
                if text:
                    teks_koreksi = self.koreksi_teks(text)
                    self.after(0, self.update_text, teks_koreksi)
                
                self.after(0, self.set_status, "Status: Mendengarkan suara...", "green")

        except Exception as e:
            self.after(0, self.set_status, "Error: Transkripsi gagal.", "red")
            self.after(0, self.update_text, f"[ERROR Transcribe]: {e}")

    def koreksi_teks(self, teks):
        """Fungsi Post-Processing untuk mengoreksi kesalahan ejaan dari Whisper."""
        # Ubah ke huruf kecil untuk memudahkan pencocokan kata
        teks_lower = teks.lower()
        
        # Kamus sederhana untuk perbaikan kata (Anda bisa menambahkan kata lain di sini)
        kamus_perbaikan = {
            "namasaya": "nama saya",
            "sayah": "saya",
            "namah": "nama",
            "biyar": "biar",
            "karna": "karena",
            "kesinin": "kesini",
            "naikapa": "naik apa",
            "apah": "apa"
        }
        
        for salah, benar in kamus_perbaikan.items():
            teks_lower = teks_lower.replace(salah, benar)
            
        # Kembalikan dengan format kapital di awal kalimat
        if teks_lower:
            return teks_lower[0].upper() + teks_lower[1:]
        return teks_lower

    def update_text(self, text):
        """Memperbarui teks di dalam textbox dari main thread."""
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", ctk.END)
        self.textbox.insert(ctk.END, text)
        self.textbox.configure(state="disabled")
        self.textbox.see(ctk.END)

    def set_status(self, text, color="gray"):
        """Memperbarui label status dari main thread."""
        self.status_label.configure(text=text, text_color=color)

    def on_closing(self):
        """Membersihkan resource saat ditutup."""
        self.is_running = False
        self.audio_queue.put(None)
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        self.destroy()

if __name__ == "__main__":
    app = STTApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
