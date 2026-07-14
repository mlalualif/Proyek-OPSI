import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import time
from tqdm import tqdm

# --- Import MobileSAM ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MobileSAM')))
try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError as e:
    print(f"Error: MobileSAM tidak ditemukan. Pastikan folder MobileSAM ada di root proyek.")
    print(f"Error detail: {e}")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Dataset Class untuk LSTM (Membaca Feature yang sudah diekstrak) ---
class BISINDOFeatureDataset(Dataset):
    def __init__(self, features_dir, classes):
        self.features_dir = features_dir
        self.classes = classes
        self.samples = [] # List of (feature_path, class_idx)
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(features_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for feat_file in os.listdir(class_dir):
                if feat_file.endswith('.pt'):
                    feat_path = os.path.join(class_dir, feat_file)
                    self.samples.append((feat_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, class_idx = self.samples[idx]
        features = torch.load(feat_path) # (frames_per_seq, 1024)
        return features, class_idx

# --- 2. LSTM Model ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=2, num_classes=76):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Ambil output dari time step terakhir
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# --- 3. Feature Extractor (Caching Strategy) ---
def extract_and_cache_features(data_dir, features_dir, sam_checkpoint, classes, frames_per_seq=45):
    """
    Ekstrak fitur menggunakan MobileSAM (statis) sekali saja dan simpan ke disk (.pt).
    Ini menghemat waktu drastis karena MobileSAM tidak perlu dieksekusi setiap epoch.
    Data Augmentation (Horizontal Flip) juga di-generate dan disimpan.
    """
    if not os.path.exists(sam_checkpoint):
        print(f"Error: Weights MobileSAM tidak ditemukan di {sam_checkpoint}")
        sys.exit(1)
        
    print("Memuat model MobileSAM untuk Feature Extraction...")
    sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    pool = nn.AdaptiveAvgPool2d((2, 2)) # Pooling 64x64 -> 2x2 (1024 features)
    
    os.makedirs(features_dir, exist_ok=True)
    
    for class_name in classes:
        class_data_dir = os.path.join(data_dir, class_name)
        class_feat_dir = os.path.join(features_dir, class_name)
        
        if not os.path.isdir(class_data_dir):
            continue
            
        os.makedirs(class_feat_dir, exist_ok=True)
        
        sequences = os.listdir(class_data_dir)
        print(f"Mengekstrak fitur untuk kelas '{class_name}' ({len(sequences)} sequences)...")
        
        for seq_folder in tqdm(sequences):
            seq_path = os.path.join(class_data_dir, seq_folder)
            feat_path = os.path.join(class_feat_dir, f"{seq_folder}.pt")
            feat_flip_path = os.path.join(class_feat_dir, f"{seq_folder}_flip.pt")
            
            # Skip jika sudah pernah diekstrak (resume capability)
            if os.path.exists(feat_path) and os.path.exists(feat_flip_path):
                continue
                
            if not os.path.isdir(seq_path):
                continue
                
            frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg')])
            
            features_normal = []
            features_flipped = []
            
            with torch.no_grad():
                for i in range(frames_per_seq):
                    if i < len(frame_files):
                        img_path = os.path.join(seq_path, frame_files[i])
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        pass # Pad dengan gambar terakhir jika kurang
                    
                    # 1. Ekstrak Normal
                    predictor.set_image(img)
                    feat = predictor.features # (1, 256, 64, 64)
                    feat = pool(feat).view(-1) # -> 1024
                    features_normal.append(feat.cpu())
                    
                    # 2. Ekstrak Flipped (Data Augmentation Horizontal Flip)
                    img_flip = cv2.flip(img, 1)
                    predictor.set_image(img_flip)
                    feat_flip = predictor.features
                    feat_flip = pool(feat_flip).view(-1)
                    features_flipped.append(feat_flip.cpu())
            
            # Simpan ke .pt file
            torch.save(torch.stack(features_normal), feat_path)
            torch.save(torch.stack(features_flipped), feat_flip_path)

# --- 4. Main Loop ---
def main():
    print("=== Training SIGMA 2.0 Hybrid Model (MobileSAM + LSTM) ===")
    
    data_dir = os.path.join('dataset', 'videos')
    features_dir = os.path.join('dataset', 'features_cache')
    models_dir = 'models'
    sam_checkpoint = os.path.join('MobileSAM', 'weights', 'mobile_sam.pt')
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 76 Kelas sesuai PRD
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
    
    frames_per_seq = 45
    epochs = 30
    batch_size = 32
    
    print(f"Device yang digunakan: {device}")
    
    # TAHAP 1: Ekstrak & Cache Fitur (termasuk Data Augmentation)
    extract_and_cache_features(data_dir, features_dir, sam_checkpoint, classes, frames_per_seq)
    
    # TAHAP 2: Training LSTM
    print("\n--- Memulai Training LSTM ---")
    dataset = BISINDOFeatureDataset(features_dir, classes)
    
    if len(dataset) == 0:
        print("Error: Dataset kosong. Pastikan Anda sudah merekam data di dataset/videos/")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SignLanguageLSTM(input_size=1024, hidden_size=256, num_layers=2, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Total sequences untuk training (termasuk augmentasi flip): {len(dataset)}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {accuracy:.2f}%")
        
        # Simpan checkpoint secara berkala
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(models_dir, f'lstm_epoch_{epoch+1}.pth'))
            
    # Simpan model final
    final_model_path = os.path.join(models_dir, 'lstm_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining selesai 100%! Model final tersimpan di {final_model_path}")

if __name__ == "__main__":
    main()
