import cv2
import torch
import torch.nn as nn
import os
import sys

# Mundur satu folder untuk cari MobileSAM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MobileSAM')))
try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError:
    print("Error: MobileSAM tidak ditemukan.")
    sys.exit(1)

device = torch.device("cpu")

# --- Arsitektur Model LSTM ---
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

def main():
    print("Memuat model MobileSAM & LSTM...")
    
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
        print("Model LSTM siap!")
    else:
        print(f"Error: {model_path} tidak ditemukan!")
        sys.exit(1)
        
    lstm_model.eval()

    cap = cv2.VideoCapture(0)
    current_prediction = "Menunggu gerakan..."
    frames_buffer = []

    print("\n--- KAMERA CONTINUOUS SIAP ---")
    print("Sistem akan otomatis menerjemahkan setiap 30 frame gerakan.")
    print("Tekan 'Q' pada keyboard untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        frames_buffer.append(frame)
        
        # Tampilkan status buffer di layar
        cv2.putText(display_frame, f"Merekam: {len(frames_buffer)}/30", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Jika memori buffer penuh 30 frame (Sesuai PRD)
        if len(frames_buffer) == 30:
            cv2.putText(display_frame, "MEMPROSES...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("SIGMA 2.0 - Continuous", display_frame)
            cv2.waitKey(1)
            
            # Ekstraksi MobileSAM
            features_list = []
            with torch.no_grad():
                for f in frames_buffer:
                    rgb_f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    predictor.set_image(rgb_f)
                    feat = pool(predictor.features).view(-1).cpu()
                    features_list.append(feat)
            
            # Prediksi AI
            seq_tensor = torch.stack(features_list).unsqueeze(0).to(device)
            with torch.no_grad():
                output = lstm_model(seq_tensor)
                probs = torch.softmax(output, dim=1)
                max_prob, predicted = torch.max(probs, 1)
                
                # Biar ga nebak ngasal saat lu diem, kita kasih Threshold 50%
                if max_prob.item() > 0.5:
                    current_prediction = classes[predicted.item()]
                else:
                    current_prediction = "..." # Tidak terdeteksi jelas
            
            # KOSONGKAN BUFFER OTOMATIS (Mulai rekam kata selanjutnya)
            frames_buffer = [] 

        # Tampilkan Teks Bawah
        cv2.rectangle(display_frame, (0, 400), (640, 480), (0, 0, 0), -1)
        cv2.putText(display_frame, f"Hasil: {current_prediction}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("SIGMA 2.0 - Continuous", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()