import pandas as pd
import numpy as np

# Load data dari file Excel
# SOLUSI: Gunakan 'raw string' dengan menambahkan huruf 'r' di depan.
# Ini memberitahu Python untuk mengabaikan karakter backslash.
# Jangan lupa tambahkan nama file lengkapnya.
file_path = r'C:\Users\user\Downloads\Pola Fauzan\Perhitungan Manual.xlsx'
data = pd.read_excel(file_path)

# Asumsi data latih sudah diberi label (class)
# Tambahkan kolom label pada data latih (label misalnya: paper, plastic, metal, glass)
# Misalnya, kita tambahkan secara manual di sini:
# Pastikan jumlah label sesuai dengan jumlah baris di file Excel Anda
# Contoh ini untuk 10 baris data
data['class'] = ['paper', 'plastic', 'metal', 'glass', 'paper', 'plastic', 'metal', 'glass', 'paper', 'plastic'] # contoh label

# Data uji (data yang akan diuji)
test_data = {
    'color_hue_mean': 0.229119433,
    'color_hue_std': 0.35490973,
    'color_saturation_mean': 0.259370123,
    'color_saturation_std': 0.589894243,
    'color_value_mean': 0.397305526,
    'color_value_std': 0.779350359,
    'texture_contrast': 0.222525399,
    'texture_dissimilarity': 0.26904165,
    'texture_homogeneity': 0.56766754,
    'texture_energy': 0.552967977,
    'texture_correlation': 0.909638097,
    'texture_asm': 0.34051125
}

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(data1, data2):
    # Pastikan kedua array memiliki panjang yang sama
    return np.sqrt(np.sum((data1 - data2) ** 2))

# Mengambil nama kolom fitur (semua kolom kecuali 'class')
feature_columns = data.columns[:-1]

# Menyiapkan data latih dan data uji sebagai array numpy
train_data_features = data[feature_columns].to_numpy()
test_data_array = np.array([test_data[col] for col in feature_columns])

# Menyusun jarak untuk setiap data latih terhadap data uji
distances = []
for index, row_features in enumerate(train_data_features):
    dist = euclidean_distance(row_features, test_data_array)
    distances.append((dist, data['class'][index]))

# Mengurutkan jarak dan memilih 3 tetangga terdekat (K=3)
distances.sort(key=lambda x: x[0])
nearest_neighbors = distances[:3]

# Menentukan kelas berdasarkan suara mayoritas (majority vote)
classes = [neighbor[1] for neighbor in nearest_neighbors]
predicted_class = max(set(classes), key=classes.count)

# Menampilkan hasil
print(f"Predicted Class: {predicted_class}")
print(f"3 Nearest Neighbors: {nearest_neighbors}")