import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# --- Import MobileSAM ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MobileSAM')))
try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError as e:
    print(f"Error: MobileSAM tidak ditemukan. Pastikan folder MobileSAM ada di root proyek.")
    print(f"Error detail: {e}")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Dataset Class dengan Temporal Jittering ---
class BISINDOFeatureDataset(Dataset):
    def __init__(self, features_dir, classes, is_train=True, jitter_std=0.01):
        self.features_dir = features_dir
        self.classes = classes
        self.is_train = is_train
        self.jitter_std = jitter_std
        self.samples = [] 
        
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
        
        # Data Augmentation: Temporal Jittering (Gaussian Noise)
        if self.is_train:
            noise = torch.randn_like(features) * self.jitter_std
            features = features + noise
            
        return features, class_idx

# --- 2. Optimized LSTM Model ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_layers=3, num_classes=76, dropout_rate=0.4):
        super(SignLanguageLSTM, self).__init__()
        # Tambahan hidden_size 512, num_layers 3, dan dropout 0.4
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Ambil output timestep terakhir
        out = self.dropout(out)
        out = self.fc(out)
        return out

# --- 3. Confusion Matrix Generator ---
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(24, 20)) # Ukuran besar agar muat 76 kelas
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - BISINDO')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix berhasil disimpan di {save_path}")

# --- 4. Feature Extractor (Sama seperti hybrid) ---
def extract_and_cache_features(data_dir, features_dir, sam_checkpoint, classes, frames_per_seq=45):
    if not os.path.exists(sam_checkpoint):
        print(f"Warning: Weights MobileSAM tidak ditemukan di {sam_checkpoint}. Jika cache sudah ada, proses bisa lanjut.")
        return
        
    print("Memeriksa cache fitur MobileSAM...")
    sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    pool = nn.AdaptiveAvgPool2d((2, 2))
    
    os.makedirs(features_dir, exist_ok=True)
    
    for class_name in classes:
        class_data_dir = os.path.join(data_dir, class_name)
        class_feat_dir = os.path.join(features_dir, class_name)
        if not os.path.isdir(class_data_dir): continue
        os.makedirs(class_feat_dir, exist_ok=True)
        
        sequences = os.listdir(class_data_dir)
        # Hitung sequence yang belum diproses
        unprocessed = [s for s in sequences if not os.path.exists(os.path.join(class_feat_dir, f"{s}.pt"))]
        
        if len(unprocessed) > 0:
            for seq_folder in tqdm(unprocessed, desc=f"Ekstraksi {class_name}"):
                seq_path = os.path.join(class_data_dir, seq_folder)
                feat_path = os.path.join(class_feat_dir, f"{seq_folder}.pt")
                feat_flip_path = os.path.join(class_feat_dir, f"{seq_folder}_flip.pt")
                
                if not os.path.isdir(seq_path): continue
                
                frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg')])
                features_normal, features_flipped = [], []
                
                with torch.no_grad():
                    for i in range(frames_per_seq):
                        if i < len(frame_files):
                            img_path = os.path.join(seq_path, frame_files[i])
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        else:
                            pass # Pad
                        
                        predictor.set_image(img)
                        features_normal.append(pool(predictor.features).view(-1).cpu())
                        
                        img_flip = cv2.flip(img, 1)
                        predictor.set_image(img_flip)
                        features_flipped.append(pool(predictor.features).view(-1).cpu())
                
                torch.save(torch.stack(features_normal), feat_path)
                torch.save(torch.stack(features_flipped), feat_flip_path)

# --- 5. Main Training Loop ---
def main():
    print("=== Training SIGMA 2.0 OPTIMIZED (LSTM 3-Layer 512, Early Stopping, LR Scheduler) ===")
    
    data_dir = os.path.join('dataset', 'videos')
    features_dir = os.path.join('dataset', 'features_cache')
    models_dir = 'models'
    logs_dir = 'logs'
    sam_checkpoint = os.path.join('MobileSAM', 'weights', 'mobile_sam.pt')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
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
    max_epochs = 100
    batch_size = 64
    early_stop_patience = 8
    
    print(f"Device yang digunakan: {device}")
    
    # TAHAP 1: Cache Fitur
    extract_and_cache_features(data_dir, features_dir, sam_checkpoint, classes, frames_per_seq)
    
    # TAHAP 2: Siapkan Dataset & Split (80% Train, 20% Val)
    print("\n--- Menyiapkan Data Split ---")
    full_dataset = BISINDOFeatureDataset(features_dir, classes, is_train=False) 
    
    if len(full_dataset) == 0:
        print("Error: Dataset kosong. Ekstraksi fitur mungkin gagal.")
        return
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # Aktifkan flag is_train di dataset training agar Temporal Jittering berjalan
    train_subset.dataset.is_train = True
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    print(f"Total Data: {len(full_dataset)} | Train: {train_size} | Val: {val_size}")
    
    # Inisialisasi Model, Loss, Optimizer, Scheduler
    model = SignLanguageLSTM(input_size=1024, hidden_size=512, num_layers=3, num_classes=len(classes), dropout_rate=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Scheduler: Kurangi LR sebanyak 0.5x jika Validation Loss tidak turun selama 5 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n--- Memulai Training Teroptimasi ---")
    
    for epoch in range(max_epochs):
        # -- Fase Training --
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # -- Fase Validasi --
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1:02d}/{max_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Step LR Scheduler berdasarkan Validation Loss
        scheduler.step(avg_val_loss)
        
        # -- Early Stopping & Model Checkpointing --
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(models_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Model terbaik baru tersimpan! (Val Acc: {best_val_acc:.2f}%)")
            patience_counter = 0
            
            # Generate Confusion Matrix terbaru untuk model terbaik
            plot_confusion_matrix(all_targets, all_preds, classes, os.path.join(logs_dir, 'confusion_matrix_best.png'))
        else:
            patience_counter += 1
            print(f"  -> Tidak ada peningkatan performa (Patience: {patience_counter}/{early_stop_patience})")
            if patience_counter >= early_stop_patience:
                print("\n[!] EARLY STOPPING DIPICU. Training dihentikan karena stagnasi.")
                break
                
    print(f"\nProses selesai. Akurasi Validasi Terbaik: {best_val_acc:.2f}%")
    print(f"Bobot terbaik disimpan di: {os.path.join(models_dir, 'best_model.pth')}")
    print(f"Cek hasil Confusion Matrix di folder {logs_dir}/")

if __name__ == "__main__":
    main()
