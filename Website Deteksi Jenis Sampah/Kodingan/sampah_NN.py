# ==============================================================================
# APLIKASI KLASIFIKASI SAMPAH (NEURAL NETWORK - MLPClassifier)
#
# Deskripsi:
# Script ini menggabungkan model Neural Network (MLPClassifier dari scikit-learn)
# untuk klasifikasi jenis sampah dengan antarmuka pengguna grafis (GUI).
# Versi ini disesuaikan untuk menggunakan fitur dari 'sampah_features1.csv'.
#
# Diperlukan file:
# - sampah_features1.csv (data fitur asli yang belum dinormalisasi)
# ==============================================================================


# --- Impor Semua Library yang Dibutuhkan ---
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # Menggunakan MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from skimage.feature import graycomatrix, graycoprops
import os

# ==============================================================================
# BAGIAN 1: LOGIKA MODEL (TRAINING DAN PRA-PEMROSESAN)
# ==============================================================================

# --- Konfigurasi dan Variabel Global ---
CSV_PATH = 'sampah_features1.csv'

MODEL_NN = None
SCALER = None
LABEL_ENCODER = None
MODEL_ACCURACY = 0.0
FEATURE_COLUMNS = []

# Kamus pemetaan dari jenis sampah (untuk tampilan GUI)
TRASH_TYPE_MAPPING = {
    "paper": "Kertas",
    "plastic": "Plastik",
    "metal": "Logam",
    "glass": "Kaca"
}

# --- Fungsi Ekstraksi Fitur (Disesuaikan dengan file sampah_features1.csv) ---
def calculate_texture_features(gray_image):
    """
    Menghitung fitur tekstur menggunakan Gray-Level Co-occurrence Matrix (GLCM).
    Ini mengukur hubungan spasial antara piksel-piksel.
    """
    try:
        if gray_image.dtype != np.uint8:
            gray_image = cv2.convertScaleAbs(gray_image)
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

def calculate_color_features(image_bgr):
    """
    Menghitung fitur warna dari gambar dalam ruang warna HSV.
    HSV lebih baik dalam memisahkan intensitas warna (Value) dari warna itu sendiri (Hue)
    dan kemurniannya (Saturation).
    """
    try:
        hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
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

def extract_features_from_image(image_path):
    try:
        image_orig = cv2.imread(image_path)
        if image_orig is None: return None
        
        image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        texture_feats = calculate_texture_features(gray_image)
        color_feats = calculate_color_features(image_resized)
        
        all_features = {**texture_feats, **color_feats}
        return all_features
    except Exception as e:
        print(f"Error saat mengekstrak fitur dari gambar baru: {e}")
        return None

# --- Fungsi Training Model ---
def train_nn_model():
    """Melatih model Neural Network (MLPClassifier) dari file CSV."""
    global MODEL_NN, SCALER, LABEL_ENCODER, MODEL_ACCURACY, FEATURE_COLUMNS
    
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File '{CSV_PATH}' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return False
        
    df_labels = df['class']
    df_features = df.drop(columns=['filename', 'class'])
    FEATURE_COLUMNS = df_features.columns.tolist()
    
    X = df_features.values
    y_text = df_labels.values
    
    LABEL_ENCODER = LabelEncoder()
    y = LABEL_ENCODER.fit_transform(y_text)
    
    # Pembagian data dengan strategi anti-kebocoran data
    def get_base_filename(filename):
        base = filename.split('_rot')[0]
        base = base.replace('_nobg', '')
        return base
    
    df['base_filename'] = df['filename'].apply(get_base_filename)
    unique_base_files = df['base_filename'].unique()
    base_train_files, base_test_files = train_test_split(unique_base_files, test_size=0.2, random_state=42)
    
    train_df = df[df['base_filename'].isin(base_train_files)]
    test_df_all_aug = df[df['base_filename'].isin(base_test_files)]
    test_df = test_df_all_aug[~test_df_all_aug['filename'].str.contains('_rot')]

    X_train = train_df[FEATURE_COLUMNS]
    y_train_text = train_df['class']
    
    X_test = test_df[FEATURE_COLUMNS]
    y_test_text = test_df['class']

    y_train = LABEL_ENCODER.transform(y_train_text)
    y_test = LABEL_ENCODER.transform(y_test_text)

    SCALER = MinMaxScaler()
    X_train_scaled = SCALER.fit_transform(X_train)
    X_test_scaled = SCALER.transform(X_test)
    
    # Inisialisasi MLPClassifier
    # Parameter dapat disesuaikan untuk performa yang lebih baik
    MODEL_NN = MLPClassifier(
        hidden_layer_sizes=(100, 50), # Dua hidden layer dengan 100 dan 50 neuron
        max_iter=500,                 # Jumlah iterasi maksimum
        activation='relu',            # Fungsi aktivasi ReLU
        solver='adam',                # Optimizer Adam
        random_state=42,              # Untuk reproduktifitas
        verbose=True                  # Menampilkan progress training
    )
    MODEL_NN.fit(X_train_scaled, y_train)
    
    y_pred = MODEL_NN.predict(X_test_scaled)
    MODEL_ACCURACY = accuracy_score(y_test, y_pred)
    
    print("Model Neural Network (MLPClassifier) untuk klasifikasi sampah berhasil dilatih.")
    print(f"Akurasi Model pada Test Set: {MODEL_ACCURACY:.2%}")
    return True

# ==============================================================================
# BAGIAN 2: ANTARMUKA PENGGUNA (GUI) - DISEDERHANAKAN
# ==============================================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Klasifikasi Jenis Sampah (Neural Network)")
        self.root.geometry("700x700")
        self.root.configure(bg="#f0f0f0")

        # --- Konfigurasi Gaya ---
        self.font_title = ("Helvetica", 18, "bold")
        self.font_subtitle = ("Helvetica", 12, "bold")
        self.font_normal = ("Helvetica", 11)
        self.color_bg = "#f0f0f0"
        self.color_frame = "#ffffff"
        self.color_text = "#333333"
        self.color_button = "#3498db"
        self.color_button_hover = "#2980b9"
        self.color_button_text = "#ffffff"

        # --- Main Frame ---
        main_frame = Frame(root, bg=self.color_bg, padx=20, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Tombol Upload ---
        self.upload_btn = Button(
            main_frame, text="⬆️  Pilih Gambar Sampah", command=self.upload_and_predict,
            font=("Helvetica", 12, "bold"), bg=self.color_button, fg=self.color_button_text,
            relief=tk.FLAT, padx=20, pady=10, cursor="hand2"
        )
        self.upload_btn.pack(pady=(0, 20))
        self.upload_btn.bind("<Enter>", lambda e: self.upload_btn.config(bg=self.color_button_hover))
        self.upload_btn.bind("<Leave>", lambda e: self.upload_btn.config(bg=self.color_button))

        # --- Frame untuk Gambar ---
        image_frame = Frame(main_frame, bg=self.color_frame, bd=1, relief=tk.SOLID, padx=10, pady=10)
        image_frame.pack(pady=10, fill=tk.X)
        self.image_label = Label(image_frame, bg=self.color_frame)
        self.image_label.pack(pady=10)
        self.display_placeholder_image()

        # --- Frame untuk Hasil ---
        result_frame = Frame(main_frame, bg=self.color_frame, bd=1, relief=tk.SOLID)
        result_frame.pack(pady=20, padx=10, fill=tk.BOTH, expand=True)
        Label(result_frame, text="🔍 Hasil Prediksi", font=self.font_title, bg=self.color_frame, fg=self.color_text).pack(pady=(15, 10))
        ttk.Separator(result_frame, orient='horizontal').pack(fill='x', padx=20, pady=5)
        
        details_frame = Frame(result_frame, bg=self.color_frame)
        details_frame.pack(pady=15, padx=20, fill=tk.X)
        details_frame.grid_columnconfigure(1, weight=1)

        Label(details_frame, text="Jenis Sampah:", font=self.font_subtitle, bg=self.color_frame, fg=self.color_text).grid(row=0, column=0, sticky="nw", pady=5)
        self.jenis_sampah_label = Label(details_frame, text="--", font=self.font_normal, bg=self.color_frame, fg=self.color_text, wraplength=400, justify=tk.LEFT)
        self.jenis_sampah_label.grid(row=0, column=1, sticky="w", pady=5)

        Label(details_frame, text="Tingkat Keyakinan:", font=self.font_subtitle, bg=self.color_frame, fg=self.color_text).grid(row=1, column=0, sticky="nw", pady=5)
        self.confidence_label = Label(details_frame, text="--", font=self.font_normal, bg=self.color_frame, fg=self.color_text)
        self.confidence_label.grid(row=1, column=1, sticky="w", pady=5)

        ttk.Separator(result_frame, orient='horizontal').pack(fill='x', padx=20, pady=10)
        self.akurasi_label = Label(result_frame, text=f"Akurasi Model (Neural Network): {MODEL_ACCURACY:.2%}", font=("Helvetica", 10, "italic"), bg=self.color_frame, fg="#888888")
        self.akurasi_label.pack(pady=(0, 15))

    def display_placeholder_image(self):
        try:
            placeholder = Image.new('RGB', (300, 300), color='#e0e0e0')
            from PIL import ImageDraw, ImageFont
            d = ImageDraw.Draw(placeholder)
            try:
                fnt = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                fnt = ImageFont.load_default()
            text = "Gambar akan tampil di sini"
            text_bbox = d.textbbox((0, 0), text, font=fnt)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            d.text(((300-text_width)/2, (300-text_height)/2), text, font=fnt, fill='#888888')
            photo = ImageTk.PhotoImage(placeholder)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception:
            self.image_label.config(text="Silakan pilih gambar untuk memulai.")

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(
            title="Pilih sebuah gambar sampah",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if not file_path:
            return
        self.display_image(file_path)
        self.run_prediction(file_path)

    def display_image(self, file_path):
        try:
            img = Image.open(file_path)
            img.thumbnail((350, 350))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error Gambar", f"Gagal memuat gambar: {e}")
            self.display_placeholder_image()

    def run_prediction(self, file_path):
        self.jenis_sampah_label.config(text="Menganalisis...")
        self.confidence_label.config(text="Mohon tunggu...")
        self.root.update_idletasks()

        new_features = extract_features_from_image(file_path)
        if new_features is None:
            self.jenis_sampah_label.config(text="Gagal memproses gambar.")
            self.confidence_label.config(text="--")
            return

        new_features_df = pd.DataFrame([new_features])
        new_features_df = new_features_df[FEATURE_COLUMNS] 
        
        new_features_values = new_features_df.values
        new_features_scaled = SCALER.transform(new_features_values)
        
        prediction_encoded = MODEL_NN.predict(new_features_scaled)
        prediction_proba = MODEL_NN.predict_proba(new_features_scaled)
        
        predicted_trash_type = LABEL_ENCODER.inverse_transform(prediction_encoded)[0]
        confidence = np.max(prediction_proba) * 100
        
        # Menggunakan pemetaan untuk tampilan yang lebih baik
        display_trash_type = TRASH_TYPE_MAPPING.get(predicted_trash_type, predicted_trash_type)

        self.jenis_sampah_label.config(text=f"{display_trash_type.capitalize()}")
        self.confidence_label.config(text=f"{confidence:.2f}%")

# ==============================================================================
# --- FUNGSI MAIN UNTUK MENJALANKAN APLIKASI ---
# ==============================================================================
if __name__ == "__main__":
    if train_nn_model():
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Startup Error",
            "Gagal memulai aplikasi karena model tidak dapat dilatih.\n\n"
            f"Pastikan file '{CSV_PATH}' "
            "berada di folder yang sama dengan skrip ini."
        )
        print("Gagal memulai aplikasi karena model tidak dapat dilatih.")

