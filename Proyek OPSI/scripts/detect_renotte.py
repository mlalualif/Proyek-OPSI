import cv2
import torch
import torch.nn as nn
import os
import sys
import threading
import time
from collections import deque
from ultralytics import YOLO

# Mundur satu folder untuk cari MobileSAM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MobileSAM')))
try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError:
    print("Error: MobileSAM tidak ditemukan.")
    sys.exit(1)

device = torch.device("cpu")

# --- Arsitektur Model LSTM (Sesuai PRD) ---
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

# Variabel Global untuk Sinkronisasi Thread
raw_frames_buffer = deque(maxlen=30)
current_prediction = "Menunggu Gerakan..."
current_confidence = 0.0
is_processing = False

# --- PEKERJA AI (Berjalan di Background) ---
def ai_worker(model, predictor, pool, classes):
    global current_prediction, current_confidence, is_processing, raw_frames_buffer
    
    while True:
        if len(raw_frames_buffer) == 30 and not is_processing:
            is_processing = True
            current_prediction = "Memproses AI..."
            current_confidence = 0.0
            
            frames_to_process = list(raw_frames_buffer)
            features_list = []
            
            # Proses Segmentasi MobileSAM
            with torch.no_grad():
                for f in frames_to_process:
                    rgb_f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    predictor.set_image(rgb_f)
                    feat = pool(predictor.features).view(-1).cpu()
                    features_list.append(feat)
            
            # Prediksi Temporal LSTM
            seq_tensor = torch.stack(features_list).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(seq_tensor)
                probs = torch.softmax(output, dim=1)
                max_prob, predicted = torch.max(probs, 1)
                
                if max_prob.item() > 0.5:
                    current_prediction = classes[predicted.item()]
                    current_confidence = max_prob.item() * 100
                else:
                    current_prediction = "Ulangi Gerakan"
                    current_confidence = 0.0
                    
            time.sleep(2) # Tahan teks hasil agar sempat terbaca
            is_processing = False
            raw_frames_buffer.clear()
            
        time.sleep(0.1)

def main():
    print("Memuat Pipeline SIGMA 2.0 (YOLOv8 + MobileSAM + LSTM)...")
    
    # Load YOLOv8 Sesuai PRD
    print("Memuat YOLOv8...")
    yolo_model = YOLO('yolov8n.pt') 
    
    # Load MobileSAM Sesuai PRD
    print("Memuat MobileSAM...")
    sam_checkpoint = os.path.join('MobileSAM', 'weights', 'mobile_sam.pt')
    sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint).to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    pool = nn.AdaptiveAvgPool2d((2, 2))
    
    classes = [
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
    
    lstm_model = SignLanguageLSTM(num_classes=len(classes)).to(device)
    model_path = os.path.join('models', 'best_model_darurat.pth')
    if os.path.exists(model_path):
        lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.eval()

    # Jalankan Pekerja AI
    worker = threading.Thread(target=ai_worker, args=(lstm_model, predictor, pool, classes))
    worker.daemon = True
    worker.start()

    cap = cv2.VideoCapture(0)
    print("Kamera Aktif. Mulai peragakan isyarat (Sistem berjalan Real-Time).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Simpan frame untuk dianalisis MobileSAM + LSTM
        if not is_processing:
            raw_frames_buffer.append(frame.copy())
            
        # --- Eksekusi YOLOv8 (Tracking Visual) ---
        results = yolo_model(frame, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Deteksi objek (Bawaannya akan mendeteksi orang/badan)
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                label_text = current_prediction
                if current_confidence > 0:
                    label_text += f" {current_confidence:.1f}%"
                    
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 255, 255), -1)
                cv2.putText(frame, label_text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                break 

        cv2.imshow("SIGMA 2.0 - PRD Compliant", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()