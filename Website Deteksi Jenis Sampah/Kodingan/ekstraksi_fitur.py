import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import traceback

# --- KONFIGURASI ---
# Ganti dengan path di komputermu. Gunakan 'r' di depan string untuk menangani backslash.
INPUT_FOLDER = r"C:\Users\user\Downloads\Pola Fauzan\augmentasi"
OUTPUT_CSV_FILE = r"C:\Users\user\Downloads\Pola Fauzan\sampah_features1.csv"
# ------------------

def calculate_color_features(image):
    """
    Menghitung fitur warna dari gambar dalam ruang warna HSV.
    HSV lebih baik dalam memisahkan intensitas warna (Value) dari warna itu sendiri (Hue)
    dan kemurniannya (Saturation).
    """
    try:
        # Konversi gambar ke HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        features = {
            'color_hue_mean': np.mean(h),
            'color_hue_std': np.std(h),
            'color_saturation_mean': np.mean(s),
            'color_saturation_std': np.std(s),
            'color_value_mean': np.mean(v),
            'color_value_std': np.std(v)
        }
        return features
    except cv2.error as e:
        print(f"  Error saat kalkulasi fitur warna: {e}")
        return {}

def calculate_texture_features(gray_image):
    """
    Menghitung fitur tekstur menggunakan Gray-Level Co-occurrence Matrix (GLCM).
    Ini mengukur hubungan spasial antara piksel-piksel.
    """
    try:
        # Pastikan gambar dalam format 8-bit
        if gray_image.dtype != np.uint8:
            gray_image = cv2.convertScaleAbs(gray_image)

        # Hitung GLCM. distances=[5], angles=[0] berarti membandingkan piksel
        # yang berjarak 5 piksel secara horizontal.
        glcm = graycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        
        features = {
            'texture_contrast': graycoprops(glcm, 'contrast')[0, 0],
            'texture_dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
            'texture_homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'texture_energy': graycoprops(glcm, 'energy')[0, 0],
            'texture_correlation': graycoprops(glcm, 'correlation')[0, 0],
            'texture_asm': graycoprops(glcm, 'ASM')[0, 0]
        }
        return features
    except Exception as e:
        print(f"  Error saat kalkulasi fitur tekstur: {e}")
        return {}

def extract_features_from_image(image_path):
    """Fungsi utama untuk mengekstrak semua fitur dari satu gambar."""
    try:
        # Baca gambar menggunakan OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Gagal membaca gambar: {image_path}")
            return None
            
        # Konversi ke Grayscale untuk fitur tekstur
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ekstraksi Fitur
        color_features = calculate_color_features(image)
        texture_features = calculate_texture_features(gray_image)
        
        # Gabungkan semua fitur menjadi satu dictionary
        all_features = {**color_features, **texture_features}
        
        return all_features

    except Exception as e:
        print(f"  Error saat memproses file {os.path.basename(image_path)}: {e}")
        traceback.print_exc()
        return None

# --- Main Script ---
if __name__ == "__main__":
    all_extracted_data = []
    
    # Cek apakah folder input ada
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Folder input tidak ditemukan di '{INPUT_FOLDER}'")
        exit()

    # Iterasi melalui setiap folder kelas (plastik, kertas, dll.)
    for class_label in os.listdir(INPUT_FOLDER):
        class_path = os.path.join(INPUT_FOLDER, class_label)
        if os.path.isdir(class_path):
            print(f"\nMemproses kelas: {class_label}")
            for image_name in os.listdir(class_path):
                # Hanya proses file gambar
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_full_path = os.path.join(class_path, image_name)
                    print(f"  Ekstraksi fitur dari: {image_name}")
                    
                    features = extract_features_from_image(image_full_path)
                    
                    if features:
                        features['filename'] = image_name
                        features['class'] = class_label
                        all_extracted_data.append(features)
    
    if not all_extracted_data:
        print("\nTidak ada data fitur yang berhasil diekstrak.")
    else:
        # Buat DataFrame dari list of dictionaries
        features_df = pd.DataFrame(all_extracted_data)
        
        # Susun ulang kolom agar 'filename' dan 'class' di depan
        cols = ['filename', 'class'] + [col for col in features_df.columns if col not in ['filename', 'class']]
        features_df = features_df[cols]
        
        # Simpan ke CSV
        try:
            # Pastikan direktori output ada
            output_dir = os.path.dirname(OUTPUT_CSV_FILE)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            features_df.to_csv(OUTPUT_CSV_FILE, index=False)
            print(f"\nEkstraksi fitur selesai. Data disimpan ke: {OUTPUT_CSV_FILE}")
            print(f"Total {len(features_df)} sampel dengan {len(features_df.columns)-2} fitur per sampel.")
        except Exception as e:
            print(f"\nError saat menyimpan file CSV: {e}")