
# Deteksi Buah

Projek Akhir Pengolahan Citra ini membahas tentang deteksi buah.Buah yang digunakan yaitu lemon, alpukat, dan apel.

## Teori 

PAda program nantinya ada beberapa source code yang akan sering digunakan seperti OpenCV.

OpenCV adalah perpustakaan(Library) open-source yang
dikembangkan oleh Intel pada tahun 2000. Hal ini sebagian besar digunakan dalam tugas-tugas computer visi seperti deteksi objek, deteksi wajah, pengenalan wajah, segmentasi gambar, dll tetapi juga berisi banyak fungsi berguna yang mungkin perlu dalam Pengolahan citra digital.
OpenCV (opencv.org) merupakan sebuah framework
yang banyak digunakan untuk keperluan pemrosesan image
dan video. OpenCV bersifat gratis dan tersedia untuk
berbagai bahasa pemrograman, seperti C++, Python, Java dan lain-lain pada platform Windows, Linux, iOS, Android dan seterusnya. OpenCV dapat digunakan untuk melakukan berbagai operasi terhadap citra dan video dengan berbagai algoritma terbaru serta banyak digunakan pada berbagai
sistem pengolahan citra yang populer.

Numpy
Gambar pada dasarnya adalah array nilai piksel di
mana setiap piksel diwakili oleh nilai 1 (skala abu-abu) atau 3 (RGB). Oleh karena itu, NumPy dapat dengan mudah melakukan tugas seperti pemotongan gambar,
penyembunyian, atau manipulasi nilai piksel.

Deteksi objek:
Konsep dasar deteksi objek
Metode deteksi objek tradisional seperti metode berbasis fitur (misalnya, metode dengan menggunakan Histogram of Oriented Gradients, Haar-like features, atau Local Binary Patterns)
Pendekatan deteksi objek berbasis deep learning menggunakan arsitektur seperti Faster R-CNN, YOLO, atau SSD.

Deteksi Objek:
Konsep dasar deteksi objek
Metode deteksi objek tradisional seperti metode berbasis fitur (misalnya, metode dengan menggunakan Histogram of Oriented Gradients, Haar-like features, atau Local Binary Patterns)
Pendekatan deteksi objek berbasis deep learning menggunakan arsitektur seperti Faster R-CNN, YOLO, atau SSD.

Preprocessing dan Augmentasi Data:
Preprocessing gambar untuk mempersiapkan data pelatihan
Augmentasi data untuk meningkatkan keberagaman dan jumlah data pelatihan

Dataset:
Pengumpulan dataset gambar buah yang berkualitas
Anotasi data untuk menandai lokasi dan batas objek buah dalam gambar

Evaluasi dan Metrik:
Mean Average Precision (mAP)
Intersection over Union (IoU)
Precision, Recall, dan F1-score

Transfer Learning:
Menggunakan model yang telah dilatih sebelumnya (pretrained model) untuk deteksi buah
Fine-tuning model untuk tugas deteksi buah

Implementasi menggunakan Library atau Framework:
Menggunakan library atau framework seperti OpenCV, TensorFlow, Keras, PyTorch, atau Caffe untuk implementasi deteksi buah

Teknik Lanjutan:
Deteksi multi-objek dan pelacakan objek dalam video
Deteksi buah dengan latar belakang yang kompleks atau terjal.

## Praktikum
Untuk menampilkan output buah dengan tampilan seperti yang ada di program yaitu background gambar diubah menjadi hitam dan gambar buahnya tetap dengan warna aslinya.

-Langkah 1:
Ambil 3 buah dengan warna yang berbeda dan posisikan buah tersebut secara berdekatan. 
Kemudian foto buah tersebut menggunakan handphone dengan jarak yang tidak begitu jauh. 
Perlu diperhatikan bahwa pada saat akan mengambil gambar buah menggunakan handphone perhatikan bahwa background foto atau tempat meletakkan buah tidak sama dengan warna buah tersebut. 
Gunakan background dengan tampilan polos dan warna yang tidak begitu terang dan tdk sama dengan warna buah aslinya agar hasil foto yang digunakan untuk menghasilkan output seperti yang diinginkan dapat sesuai.

langkah 2:
Download foto tersebut atau pindahkan foto tersebut ke laptop agar dapat dipindahkan ke file jupyter notebook atau google collab melalui google drive.
Selanjutnya foto tersebut saya beri nama buah3.

Langkah 3:
Mencari materi terkait deteksi buah dan mencari source code yang benar agar menampilkan output yang diinginkan. Adapun beberapa kali percobaan menggunakan source code yang bermacam-macam agar tampilan deteksi buah tersebut sesuai. Ada banyak kendala yang dihadaoi diantaranya background gambar tidak berubah menjadi warna hitam melainkan tetap seperti gambar aslinya. Kemudian buah apel dan buah alpukat berubah menjadi warna hitam seperti warna background yang diinginkan yaitu hitam, dan hany buah lemon yang muncul seperti warna aslinya pada saat output muncul.
Dan terakhir semua gambar tampil sesuai warna aslinya tanpa berubah menjadi warna hitam, walaupun tampilan buahnya tidak begitu sempurna tetapi tampilan yang dihasilkan menggunakan program tersebut jauh lebih baik daripada yang sebelumnya.

Berikut dibawah ini program dan penjelasan dari tiap source codenya.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter_fruit_by_color(image, lower_color, upper_color):

###  Konversi gambar ke mode HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

### Buat mask berdasarkan rentang warna yang ditentukan
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

### Terapkan mask ke gambar asli
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    return filtered_image

### Membaca gambar buah
image = cv2.imread('/content/drive/MyDrive/Colab Notebooks/buah3.jpg')

### Konversi gambar ke mode HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

### Menentukan rentang warna kuning, hijau, dan merah
lower_green = np.array([30, 50, 50])
upper_green = np.array([70, 255, 255])

lower_yellow = np.array([10, 100, 100])
upper_yellow = np.array([130, 255, 255])

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

### Membuat mask untuk buah dengan warna kuning, hijau, dan merah
mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

### Menggabungkan semua mask
combined_mask = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_green, mask_red))

### Menampilkan gambar asli dan hasil filter
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(cv2.bitwise_and(image, image, mask=combined_mask), cv2.COLOR_BGR2RGB))
ax[1].set_title('Filtered')
ax[1].axis('off')

plt.show()

### Daftar buah dan rentang warnanya
fruits = {
    'Kuning': combined_mask & mask_yellow,
    'Hijau': combined_mask & mask_green,
    'Merah': combined_mask & mask_red
}

### Perulangan untuk menampilkan buah satu per satu
    for fruit, mask in fruits.items():

### Menerapkan mask ke gambar asli
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

### Menampilkan gambar buah
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    ax.set_title(fruit)
    ax.axis('off')
    plt.show()



