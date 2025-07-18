import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# --- KONFIGURASI ---
# Pastikan path ini sesuai dengan lokasi file di komputermu.
INPUT_CSV_FILE = r"C:\Users\user\Downloads\Pola Fauzan\sampah_features1.csv"
OUTPUT_NORMALIZED_CSV_FILE = r"C:\Users\user\Downloads\Pola Fauzan\sampah_normalized1.csv"
# ------------------

# 1. Muat dataset dari file CSV
try:
    df = pd.read_csv(INPUT_CSV_FILE)
    print(f"Dataset '{os.path.basename(INPUT_CSV_FILE)}' berhasil dimuat.")
    print(f"Jumlah baris: {len(df)}, Jumlah kolom: {len(df.columns)}")
except FileNotFoundError:
    print(f"Error: File '{INPUT_CSV_FILE}' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan skrip ekstraksi fitur terlebih dahulu.")
    exit()
except Exception as e:
    print(f"Error saat memuat CSV: {e}")
    exit()

# 2. Pisahkan kolom fitur dari kolom identifikasi (filename) dan label (class)
if 'filename' not in df.columns or 'class' not in df.columns:
    print("Error: Kolom 'filename' atau 'class' tidak ditemukan di file CSV.")
    exit()

identifier_columns = ['filename']
label_column = ['class']
feature_columns = [col for col in df.columns if col not in identifier_columns + label_column]

print(f"\nKolom yang akan dinormalisasi ({len(feature_columns)} fitur):")
print(feature_columns)

features_to_normalize = df[feature_columns]

# 3. Buat objek scaler (MinMaxScaler) dan lakukan normalisasi
# MinMaxScaler mengubah setiap fitur ke rentang [0, 1]
scaler = MinMaxScaler()
normalized_features_array = scaler.fit_transform(features_to_normalize)

# Ubah kembali hasil array menjadi DataFrame dengan nama kolom yang benar
normalized_features_df = pd.DataFrame(normalized_features_array, columns=feature_columns)

# 4. Gabungkan kembali kolom identifier, label, dan fitur yang sudah dinormalisasi
# Pastikan indexnya sesuai untuk penggabungan yang benar
df_normalized = pd.concat([df[identifier_columns].reset_index(drop=True),
                           df[label_column].reset_index(drop=True),
                           normalized_features_df.reset_index(drop=True)], axis=1)

print("\nBeberapa baris pertama dataset setelah normalisasi:")
print(df_normalized.head())

# 5. Simpan DataFrame yang sudah dinormalisasi ke file CSV baru
try:
    # Cek jika direktori output ada, jika tidak maka buat baru
    output_dir = os.path.dirname(OUTPUT_NORMALIZED_CSV_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Direktori output '{output_dir}' dibuat.")

    df_normalized.to_csv(OUTPUT_NORMALIZED_CSV_FILE, index=False)
    print(f"\n✅ Normalisasi selesai. Data disimpan ke file: {OUTPUT_NORMALIZED_CSV_FILE}")

except Exception as e:
    print(f"\nError saat menyimpan file CSV yang sudah dinormalisasi: {e}")