import cv2
import os
import time

def main():
    print("=== SIGMA 2.0 Dataset Recorder ===")
    gesture_name = input("Masukkan nama gerakan (contoh: Terima Kasih): ").strip()
    
    if not gesture_name:
        print("Nama gerakan tidak boleh kosong!")
        return

    # Buat struktur direktori
    base_dir = os.path.join("dataset", "videos", gesture_name)
    os.makedirs(base_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses webcam.")
        return

    # Set frame rate as requested in PRD if possible (15-30 FPS)
    # Most webcams default to 30, so we just read as fast as possible or enforce a delay if needed.
    
    total_sequences = 35
    frames_per_sequence = 45
    
    print(f"\nPerekaman untuk '{gesture_name}' akan segera dimulai.")
    print(f"Target: {total_sequences} sequence, {frames_per_sequence} frame per sequence.")
    print("SOP: Tangan di bawah -> Peragakan -> Tangan kembali ke bawah dalam 45 frame.")
    print("Tekan 'q' pada jendela video kapan saja untuk membatalkan.")
    
    time.sleep(3) # Persiapan awal
    
    for seq in range(total_sequences):
        seq_dir = os.path.join(base_dir, f"seq_{seq+1:02d}")
        os.makedirs(seq_dir, exist_ok=True)
        
        print(f"\n--- Sequence {seq+1}/{total_sequences} ---")
        print("Bersiap... (5 detik cooldown)")
        
        # Cooldown 5 detik sambil membaca frame agar buffer webcam tidak basi
        start_cooldown = time.time()
        while time.time() - start_cooldown < 5.0:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Hitung sisa waktu mundur
            sisa_waktu = int(5.0 - (time.time() - start_cooldown)) + 1
            
            # Tampilkan frame dengan teks cooldown dan timer
            display_frame = frame.copy()
            cv2.putText(display_frame, f"COOLDOWN: {sisa_waktu}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(display_frame, "Posisi Istirahat (Tangan di Bawah)", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('SIGMA 2.0 Recorder', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        print("MULAI MEREKAM!")
        
        frame_count = 0
        while frame_count < frames_per_sequence:
            ret, frame = cap.read()
            if not ret:
                print("Error membaca frame dari webcam.")
                break
            
            # Simpan mentahan secara rapi
            frame_path = os.path.join(seq_dir, f"frame_{frame_count:02d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Tampilkan ke layar
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Merekam Seq: {seq+1}/{total_sequences}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame: {frame_count+1}/{frames_per_sequence}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('SIGMA 2.0 Recorder', display_frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nPerekaman dibatalkan oleh pengguna.")
                cap.release()
                cv2.destroyAllWindows()
                return

    print(f"\nSelesai! Berhasil merekam {total_sequences} sequence untuk gerakan '{gesture_name}'.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
