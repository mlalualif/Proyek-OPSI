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

# --- Arsitektur Model LSTM Darurat ---
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
    
    # Load MobileSAM
    sam_checkpoint = os.path.join('MobileSAM', 'weights', 'mobile_sam.pt')
    sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint).to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    pool = nn.AdaptiveAvgPool2d((2, 2))
    
    # Load LSTM
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
    current_prediction = "Tekan 'R' untuk menebak"
    is_recording = False
    frames_buffer = []

    print("\n--- KAMERA SIAP ---")
    print("Tekan 'R' untuk mulai merekam 30 frame (1 gerakan).")
    print("Tekan 'Q' untuk keluar dari program.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Kalau lagi mode rekam
        if is_recording:
            cv2.putText(display_frame, f"Merekam... {len(frames_buffer)}/30", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frames_buffer.append(frame)
            
            # Kalau udah ngumpul 30 frame, langsung proses!
            if len(frames_buffer) == 30:
                is_recording = False
                cv2.putText(display_frame, "Memproses Data...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("SIGMA 2.0 - Ringan", display_frame)
                cv2.waitKey(1)
                
                # Ekstraksi fitur pakai MobileSAM
                features_list = []
                with torch.no_grad():
                    for f in frames_buffer:
                        rgb_f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        predictor.set_image(rgb_f)
                        feat = pool(predictor.features).view(-1).cpu()
                        features_list.append(feat)
                
                # Prediksi LSTM
                seq_tensor = torch.stack(features_list).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = lstm_model(seq_tensor)
                    _, predicted = torch.max(output.data, 1)
                    current_prediction = classes[predicted.item()]
                
                frames_buffer = [] # Reset buffer untuk gerakan selanjutnya

        # Tampilkan hasil prediksi
        cv2.rectangle(display_frame, (0, 400), (640, 480), (0, 0, 0), -1)
        cv2.putText(display_frame, f"Hasil: {current_prediction}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("SIGMA 2.0 - Ringan", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Tekan Q untuk keluar
            break
        elif key == ord('r') and not is_recording: # Tekan R untuk rekam
            is_recording = True
            frames_buffer = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()