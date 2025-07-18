import os
from rembg import remove
from PIL import Image

def process_images_in_folders():
    """
    Menemukan semua folder di dalam path input, menghapus background dari setiap gambar,
    dan menyimpannya ke path output dengan struktur folder yang sama.
    """
    # 1. Tentukan path input dan output
    # Gunakan 'r' di depan string untuk menangani backslashes di Windows
    input_base_path = r"C:\Users\user\Downloads\Pola Fauzan\kibul"
    output_base_path = r"C:\Users\user\Downloads\Pola Fauzan\kibulbg"

    # 2. Buat folder output utama jika belum ada
    os.makedirs(output_base_path, exist_ok=True)
    print(f"Folder output utama '{output_base_path}' siap digunakan.")

    try:
        # 3. Dapatkan daftar semua folder di dalam direktori input
        list_of_folders = [f for f in os.listdir(input_base_path) if os.path.isdir(os.path.join(input_base_path, f))]
        
        if not list_of_folders:
            print(f"Tidak ada folder yang ditemukan di '{input_base_path}'.")
            return

        print(f"Ditemukan folder: {list_of_folders}")

    except FileNotFoundError:
        print(f"ERROR: Path input tidak ditemukan: '{input_base_path}'")
        return

    # 4. Loop melalui setiap folder yang ditemukan
    for folder_name in list_of_folders:
        input_folder = os.path.join(input_base_path, folder_name)
        output_folder = os.path.join(output_base_path, folder_name)

        # Buat subfolder di direktori output
        os.makedirs(output_folder, exist_ok=True)
        print(f"\n📂 Memproses folder: {folder_name}")

        # 5. Loop melalui setiap file di dalam folder
        for filename in os.listdir(input_folder):
            # Periksa apakah file adalah gambar
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_folder, filename)
                
                # Ubah ekstensi file output menjadi .png untuk mendukung transparansi
                base_name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{base_name}.png")

                try:
                    # Buka gambar input
                    with Image.open(input_path) as img:
                        # Hapus background
                        output_image = remove(img)
                        # Simpan gambar hasil
                        output_image.save(output_path)
                        print(f"  ✅ Berhasil: {filename} -> {os.path.basename(output_path)}")
                
                except Exception as e:
                    print(f"  ❌ Gagal memproses {filename}. Error: {e}")

    print("\n🎉 Semua proses telah selesai!")

# Jalankan fungsi
if __name__ == "__main__":
    process_images_in_folders()