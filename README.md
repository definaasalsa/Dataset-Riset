## 1. Rencana Topik
   Perbandingan CNN Sederhana dan MobileNetV2 pada Klasifikasi Jenis Buah dengan Uji Ketahanan terhadap Gangguan Pencahayaan dan Latar.
## 2. Formulasi Masalah
- Bagaimana kecenderungan pendekatan dan hasil penelitian klasifikasi jenis buah berdasarkan citra pada dataset Fruits-360?
- Apa saja celah penelitian (research gap) yang masih terbuka dari penelitian terdahulu?
- Topik pengembangan apa yang layak diusulkan berdasarkan celah tersebut?
## 3. Riset Gap Penelitian Terdahulu
Gap 1: Model berat versus model ringan. Shahi et al. (2022) menyatakan bahwa metode DL terbaru untuk klasifikasi buah umumnya berarsitektur berat sehingga membutuhkan penyimpanan besar dan pelatihan mahal, dan karena itu model ringan perlu dieksplorasi tanpa mengorbankan akurasi. Namun Sajid et al. (2025) menemukan bahwa model ringan seperti MobileNetV3-Small dan EfficientNet-Lite mengalami penurunan akurasi pada dataset yang lebih kompleks, sehingga keseimbangan antara akurasi dan efisiensi belum tuntas.
Gap 2: Data terkontrol versus kondisi dunia nyata. Dataset Fruits-360 dibuat dengan latar putih (Oltean, 2017–), dan Sajid et al. (2025) sendiri mengakui bahwa ketergantungan pada dataset terkontrol mungkin belum menangkap variasi dunia nyata. Mereka merencanakan pengujian dengan variasi pencahayaan dan latar, pipeline deteksi objek, serta penerapan pada perangkat edge.
Gap 3: Hasil antarstudi sulit dibandingkan langsung (analisis penulis). Tabel perbandingan pada Sajid et al. (2025) memperlihatkan bahwa studi-studi Fruits-360 memakai jumlah kelas dan rasio pembagian data yang berbeda-beda, dan hasil satu arsitektur dapat sangat berbeda antarstudi, misalnya MobileNet-V2 yang mencapai 65% pada Salem et al. (2024). Hal ini menunjukkan perlunya protokol evaluasi yang seragam.
## 4. Peluang Pengembangan
- Menguji model pada foto buah yang diambil sendiri (kamera ponsel, latar dan cahaya beragam) sebagai data uji eksternal.
- Membandingkan CNN sederhana dan model ringan (misalnya MobileNetV2) dengan protokol pembagian data dan metrik yang seragam.
- Menambah kelas buah lokal Indonesia. Catatan: sebagian buah tropis seperti salak, rambutan, dan manggis sudah ada dalam daftar kelas Fruits-360, sehingga daftar kelas versi yang dipakai harus diperiksa terlebih dahulu sebelum menentukan buah yang benar-benar belum tercakup.
- Mengonversi model ke format ringan (misalnya TensorFlow Lite) untuk aplikasi mobile atau web.
- Memperluas ke klasifikasi tingkat kematangan.
## 5. Sumber Data dan Referensi
**a. Nama Repository dataset**
- Fruits-360

**b. Alamat Github/Kaggle**
- https://www.kaggle.com/datasets/moltean/fruits

**c. Referensi Jurnal**
1. Mureşan, H., & Oltean, M. (2018). Fruit recognition from images using deep learning. Acta Universitatis Sapientiae, Informatica, 10(1), 26–42. https://doi.org/10.2478/ausi-2018-0002
2. Oltean, M. (2017–). Fruits-360 dataset [Dataset]. Lisensi CC BY-SA 4.0. https://github.com/fruits-360
3. Sajid, et al. (2025). Leveraging VGG-19 for automated fruit classification in smart agriculture. PeerJ Computer Science, 11, e3391. https://doi.org/10.7717/peerj-cs.3391 (daftar penulis lengkap perlu dilengkapi dari halaman artikel)
4. Salem, D. A., Hassan, N. A., & Hamdy, R. M. (2024). Impact of transfer learning compared to convolutional neural networks on fruit detection. Journal of Intelligent & Fuzzy Systems, 46(4), 7791–7803. https://doi.org/10.3233/JIFS-233514
5. Shahi, T. B., Sitaula, C., Neupane, A., & Guo, W. (2022). Fruit classification using attention-based MobileNetV2 for industrial applications. PLOS ONE, 17(2), e0264586. https://doi.org/10.1371/journal.pone.0264586
6. Ukwuoma, C. C., Zhiguang, Q., Heyat, M. B. B., Ali, L., Almaspoor, Z., Monday, H. N., & Khan, D. M. (2022). Recent advancements in fruit detection and classification using deep learning techniques. Mathematical Problems in Engineering, 2022, 1–29. https://doi.org/10.1155/2022/9210947
