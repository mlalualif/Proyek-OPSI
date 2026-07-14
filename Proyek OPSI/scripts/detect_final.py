import cv2
import torch
import torch.nn as nn
import os
import sys
import threading
from collections import deque

# Mundur satu folder untuk cari MobileSAM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MobileSAM')))
try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError:
    print("Error: MobileSAM tidak ditemukan.")
    sys.exit(1)

device = torch.device("cpu")

# --- Thread Kamera (Biar video tetep mulus) ---
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
                
    def read(self):
        return self.ret, self.frame
        
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# --- Arsitektur Model ---
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
    print("Menyiapkan Sistem SIGMA 2.0...")
    
    # Setup MobileSAM
    sam_checkpoint = os.path.join('MobileSAM', 'weights', 'mobile_sam.pt')
    sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint).to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    pool = nn.AdaptiveAvgPool2d((2, 2))
    
    # Setup Class & LSTM
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
    else:
        print("Model tidak ditemukan!")
        sys.exit(1)
        
    lstm_model.eval()

    cap = CameraStream(0)
    
    # Deque akan otomatis menjaga maksimal 30 frame di belakang layar
    sequence = deque(maxlen=30)
    current_prediction = "Mendeteksi..."

    print("Kamera Nyala. Mulai peragakan isyarat.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # --- UI Nicholas Renotte Style (Sangat Bersih) ---
        # Kotak warna oren/biru di bagian atas layar
        cv2.rectangle(display_frame, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(display_frame, current_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("SIGMA 2.0 - BISINDO Translator", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # --- Urusan Dapur AI (Silent Processing) ---
        rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            predictor.set_image(rgb_f)
            feat = pool(predictor.features).view(-1).cpu()
            sequence.append(feat)

        if len(sequence) == 30:
            seq_tensor = torch.stack(list(sequence)).unsqueeze(0).to(device)
            with torch.no_grad():
                output = lstm_model(seq_tensor)
                probs = torch.softmax(output, dim=1)
                max_prob, predicted = torch.max(probs, 1)
                
                if max_prob.item() > 0.6:  # Threshold biar ga ngasal nebak
                    current_prediction = classes[predicted.item()]

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()