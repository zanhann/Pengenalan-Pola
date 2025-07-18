# Mengimpor library 'os' yang digunakan untuk berinteraksi dengan sistem operasi,
# seperti membuat folder dan membaca nama file.
import os
# Mengimpor class 'Image' dari library 'Pillow' (PIL) untuk membuka, memanipulasi, dan menyimpan file gambar.
from PIL import Image

# --- KONFIGURASI ---
# Variabel ini menyimpan path (lokasi) folder tempat gambar-gambar asli Anda berada.
# 'r' di depan string berarti "raw string", ini mencegah karakter backslash (\) dianggap sebagai escape character.
INPUT_FOLDER = r"C:\Users\user\Downloads\Pola Fauzan\kibulbg"

# Variabel ini menyimpan path (lokasi) folder di mana gambar-gambar baru (hasil augmentasi) akan disimpan.
OUTPUT_FOLDER = r"C:\Users\user\Downloads\Pola Fauzan\kibulaugmentasi"
# -------------------

def augment_and_save_images(root_input_folder, root_output_folder):
    """
    Fungsi utama untuk melakukan augmentasi (rotasi 90, 180, 270 derajat).
    Fungsi ini akan membaca semua gambar dari struktur folder input, melakukan augmentasi,
    dan menyimpannya ke struktur folder output yang serupa. Gambar asli juga ikut disalin.
    """
    # Mencetak pesan ke terminal untuk memberitahu pengguna bahwa proses telah dimulai.
    print(f"Memulai augmentasi dari: {root_input_folder}")
    # Memberitahu pengguna di mana hasil akan disimpan.
    print(f"Hasil augmentasi akan disimpan di: {root_output_folder}")

    # Mengecek apakah folder untuk menampung semua hasil sudah ada atau belum.
    if not os.path.exists(root_output_folder):
        # Jika folder belum ada, maka buat folder tersebut.
        os.makedirs(root_output_folder)
        # Memberi konfirmasi bahwa folder utama telah berhasil dibuat.
        print(f"Folder output utama '{root_output_folder}' berhasil dibuat.")

    # Membuat variabel untuk menghitung jumlah gambar asli yang berhasil diproses.
    processed_images_count = 0
    # Membuat variabel untuk menghitung total file yang dibuat di folder output.
    augmented_files_count = 0

    # Memulai perulangan (loop) untuk setiap item di dalam folder input utama.
    # Setiap item ini diharapkan adalah sebuah folder yang berisi gambar satu jenis daun.
    for jenis_daun_folder_name in os.listdir(root_input_folder):
        # Menggabungkan path folder utama dengan nama subfolder untuk mendapatkan path lengkap.
        path_jenis_daun_input = os.path.join(root_input_folder, jenis_daun_folder_name)
        # Melakukan hal yang sama untuk path output, agar strukturnya sama.
        path_jenis_daun_output = os.path.join(root_output_folder, jenis_daun_folder_name)

        # Memastikan bahwa item yang sedang diproses adalah sebuah direktori/folder, bukan file.
        if os.path.isdir(path_jenis_daun_input):
            # Di dalam folder output, cek apakah subfolder untuk jenis daun ini sudah ada.
            if not os.path.exists(path_jenis_daun_output):
                # Jika belum, buat subfolder tersebut.
                os.makedirs(path_jenis_daun_output)
                # Memberi konfirmasi bahwa subfolder output telah dibuat.
                print(f"  Folder '{path_jenis_daun_output}' dibuat.")

            # Mencetak nama folder yang sedang diproses untuk memberi tahu progres ke pengguna.
            print(f"\n  Memproses folder: {jenis_daun_folder_name}")
            # Memulai perulangan kedua untuk setiap file gambar di dalam subfolder jenis daun.
            for image_filename in os.listdir(path_jenis_daun_input):
                # Memanggil fungsi lain (helper function) untuk memproses satu per satu gambar.
                # Fungsi ini akan mengembalikan jumlah augmentasi yang berhasil dibuat untuk gambar ini.
                num_augmented = process_single_image(image_filename, path_jenis_daun_input, path_jenis_daun_output)
                
                # Mengecek apakah gambar berhasil diproses (setidaknya menghasilkan 1 augmentasi).
                if num_augmented > 0:
                    # Jika berhasil, tambah hitungan gambar asli yang diproses.
                    processed_images_count += 1
                    # Tambah hitungan total file (jumlah augmentasi + 1 untuk file asli).
                    augmented_files_count += num_augmented + 1 

    # Setelah semua folder dan gambar selesai diproses, cetak ringkasan hasilnya.
    print("\nProses augmentasi selesai.")
    # Mencetak total gambar asli yang berhasil diolah.
    print(f"Total gambar asli diproses: {processed_images_count}")
    # Mencetak total file yang seharusnya ada di folder output.
    print(f"Total file di folder output (termasuk asli dan augmentasi): {augmented_files_count}")


def process_single_image(image_filename, current_input_folder_path, current_output_folder_path):
    """
    Fungsi ini bertugas memproses satu file gambar: menyimpan versi asli
    dan versi rotasinya (90, 180, 270).
    Fungsi akan mengembalikan jumlah file augmentasi (baru) yang berhasil dibuat.
    """
    # Menentukan ekstensi file apa saja yang dianggap sebagai gambar yang valid.
    valid_image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    # Membuat variabel lokal untuk menghitung augmentasi yang berhasil dibuat untuk gambar ini.
    num_augmented_created = 0

    # Mengecek apakah nama file (dijadikan huruf kecil) diakhiri dengan salah satu ekstensi yang valid.
    if image_filename.lower().endswith(valid_image_extensions):
        # Menggunakan blok 'try-except' untuk menangani error. Jika ada error pada satu gambar,
        # program tidak akan berhenti total, tapi akan lanjut ke gambar berikutnya.
        try:
            # Membuat path lengkap ke file gambar input yang akan dibuka.
            img_path_input = os.path.join(current_input_folder_path, image_filename)
            # Membuka file gambar menggunakan library Pillow dan menyimpannya ke variabel.
            img_pil = Image.open(img_path_input)

            # Pengecekan penting untuk gambar PNG yang transparan (hasil remove background).
            # Mengecek apakah mode gambar adalah 'RGBA' (Red, Green, Blue, Alpha/Transparansi).
            if img_pil.mode == 'RGBA' or 'A' in img_pil.info.get('transparency', ()):
                # Jika ya, pastikan gambar dikonversi ke mode RGBA untuk menjaga transparansi.
                img_pil = img_pil.convert('RGBA')

            # Memisahkan nama file dari ekstensinya (contoh: 'daun1.png' -> 'daun1' dan '.png').
            base_name, ext = os.path.splitext(image_filename)

            # --- TAHAP 1: Simpan gambar asli ---
            # Membuat path lengkap untuk menyimpan gambar asli di folder output.
            path_output_asli = os.path.join(current_output_folder_path, image_filename)
            # Menyimpan salinan gambar asli ke path tersebut.
            img_pil.save(path_output_asli)

            # --- TAHAP 2: Rotasi 90 derajat ---
            # Memutar gambar 90 derajat. 'expand=True' memastikan ukuran kanvas disesuaikan agar gambar tidak terpotong.
            img_90 = img_pil.rotate(90, expand=True)
            # Membuat nama file baru untuk gambar rotasi 90 derajat (contoh: 'daun1_rot90.png').
            path_output_90 = os.path.join(current_output_folder_path, f"{base_name}_rot90{ext}")
            # Menyimpan gambar yang sudah dirotasi.
            img_90.save(path_output_90)
            # Menambah hitungan file augmentasi yang berhasil dibuat.
            num_augmented_created += 1

            # --- TAHAP 3: Rotasi 180 derajat ---
            # Memutar gambar asli 180 derajat.
            img_180 = img_pil.rotate(180, expand=True)
            # Membuat nama file baru untuk gambar rotasi 180 derajat.
            path_output_180 = os.path.join(current_output_folder_path, f"{base_name}_rot180{ext}")
            # Menyimpan gambar yang sudah dirotasi.
            img_180.save(path_output_180)
            # Menambah hitungan file augmentasi.
            num_augmented_created += 1

            # --- TAHAP 4: Rotasi 270 derajat ---
            # Memutar gambar asli 270 derajat.
            img_270 = img_pil.rotate(270, expand=True)
            # Membuat nama file baru untuk gambar rotasi 270 derajat.
            path_output_270 = os.path.join(current_output_folder_path, f"{base_name}_rot270{ext}")
            # Menyimpan gambar yang sudah dirotasi.
            img_270.save(path_output_270)
            # Menambah hitungan file augmentasi.
            num_augmented_created += 1

        # Jika terjadi error apapun di dalam blok 'try' (misal file gambar rusak),
        # blok 'except' ini akan dijalankan.
        except Exception as e:
            # Mencetak pesan error ke terminal tanpa menghentikan seluruh program.
            print(f"    Error saat augmentasi '{image_filename}': {e}")
    
    # Setelah semua proses selesai (atau gagal), kembalikan jumlah augmentasi yang berhasil dibuat.
    # Jika gambar tidak valid atau error, nilai yang dikembalikan adalah 0.
    return num_augmented_created

# --- Jalankan Fungsi Augmentasi ---
# Blok ini adalah titik awal eksekusi program saat file ini dijalankan langsung.
if __name__ == '__main__':
    # Melakukan pengecekan terakhir untuk memastikan folder input yang dikonfigurasi benar-benar ada.
    if not os.path.isdir(INPUT_FOLDER):
        # Jika tidak ada, cetak pesan error yang jelas kepada pengguna.
        print(f"ERROR: Folder input utama '{INPUT_FOLDER}' tidak ditemukan!")
        print("Pastikan variabel INPUT_FOLDER sudah benar dan folder tersebut ada.")
    else:
        # Jika folder input ada, panggil fungsi utama untuk memulai seluruh proses augmentasi.
        augment_and_save_images(INPUT_FOLDER, OUTPUT_FOLDER)