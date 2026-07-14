import cv2
import torch
import torch.nn as nn
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import sys
from collections import deque

# Import Ultralytics & MobileSAM
from ultralytics import YOLO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MobileSAM')))
try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError:
    print("Error: Pastikan folder MobileSAM ada di root proyek.")
    sys.exit(1)

# --- Konfigurasi CustomTkinter ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- Arsitektur Model LSTM (Sesuai Mode Darurat) ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=2, num_classes=76, dropout_rate=0.2):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return out

class SIGMAApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SIGMA 2.0 - BISINDO Translator")
        self.geometry("1000x600")
        
        # Inisialisasi Device & Model
        self.device = torch.device("cpu") # Pakai CPU lokal
        print("Memuat model YOLOv8 & MobileSAM...")
        
        # 1. Load YOLOv8 (Untuk deteksi tangan)
        # Pastikan file yolov8n.pt ada, kalau tidak akan didownload otomatis
        self.yolo_model = YOLO('yolov8n.pt') 
        
        # 2. Load MobileSAM
        sam_checkpoint = os.path.join('MobileSAM', 'weights', 'mobile_sam.pt')
        self.sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint).to(self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # 3. Load LSTM
        self.classes = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "Beli", "Jual", "Bawa", "Lihat", "Ambil", "Buang", "Cari",
            "Tas", "Mobil", "Pensil", "Baju", "Buku", "Meja", "Kursi", "Lemari",
            "Mahal", "Murah", "Bagus", "Jelek", "Besar", "Kecil", "Panjang", "Tinggi", "Berat",
            "Sekarang", "Kemarin", "Sering", "Jarang", "Tidak",
            "Aku", "Kamu", "Kita", "Kami", "Mereka", "Ini", "Itu",
            "Atas", "Bawah", "Depan", "Belakang", "Luar", "Samping",
            "Halo", "Iya", "Maaf", "Tolong", "Terima kasih", "Sama-sama",
            "Apa", "Kenapa"
        ]
        self.lstm_model = SignLanguageLSTM(num_classes=len(self.classes)).to(self.device)
        
        # Load bobot yang baru di-training!
        model_path = os.path.join('models', 'best_model_darurat.pth')
        if os.path.exists(model_path):
            self.lstm_model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Model LSTM berhasil dimuat!")
        else:
            print(f"File {model_path} tidak ditemukan! Taruh file hasil training di folder models.")
            
        self.lstm_model.eval()
        
        # Buffer untuk 30 frame
        self.sequence_buffer = deque(maxlen=30)
        
        # Layout UI (Kiri Kamera, Kanan Panel)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Frame Kamera (Kiri)
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.camera_frame, text="")
        self.video_label.pack(expand=True, fill="both")
        
        # Frame Kontrol (Kanan)
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.title_label = ctk.CTkLabel(self.control_frame, text="Current Sign", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=(20, 5))
        
        self.sign_display = ctk.CTkLabel(self.control_frame, text="-", font=("Arial", 36, "bold"), text_color="#1E90FF")
        self.sign_display.pack(pady=(0, 20))
        
        self.btn_start = ctk.CTkButton(self.control_frame, text="Start Camera", command=self.start_camera)
        self.btn_start.pack(pady=10, fill="x", padx=20)
        
        self.btn_stop = ctk.CTkButton(self.control_frame, text="Stop Camera", command=self.stop_camera, fg_color="red")
        self.btn_stop.pack(pady=10, fill="x", padx=20)
        
        # Kamera Setup
        self.cap = None
        self.is_running = False

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.update_frame()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.video_label.configure(image="")
        self.sign_display.configure(text="-")

    def update_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Flip biar kayak cermin
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 1. Deteksi Tangan pakai YOLO (Anggap tangan adalah objek 'person' atau kita bypass crop ke tengah)
                # Untuk prototype lokal agar tidak lag parah, kita deteksi langsung seluruh frame ke MobileSAM
                # *Idealnya YOLO memotong tangan (crop) sesuai PRD*
                
                # 2. Ekstraksi Fitur MobileSAM
                with torch.no_grad():
                    self.predictor.set_image(rgb_frame)
                    feature = self.pool(self.predictor.features).view(-1).cpu()
                    self.sequence_buffer.append(feature)
                
                # 3. Prediksi LSTM kalau buffer udah penuh (30 frame)
                if len(self.sequence_buffer) == 30:
                    seq_tensor = torch.stack(list(self.sequence_buffer)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = self.lstm_model(seq_tensor)
                        probs = torch.softmax(output, dim=1)
                        max_prob, predicted = torch.max(probs, 1)
                        
                        # Threshold biar gak nebak ngasal
                        if max_prob.item() > 0.6:
                            pred_class = self.classes[predicted.item()]
                            self.sign_display.configure(text=pred_class)
                
                # Konversi ke gambar UI
                img = Image.fromarray(rgb_frame)
                imgtk = ctk.CTkImage(light_image=img, size=(640, 480))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
            self.after(30, self.update_frame) # Update tiap ~30ms

if __name__ == "__main__":
    app = SIGMAApp()
    app.mainloop()