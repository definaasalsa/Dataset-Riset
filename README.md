## 1. Rencana Topik
   Pengelolahan citra dan pembelajaran mendalam (Deep learning) untuk pertanian cerdas: estimasi tingkat keparahan penyakit tanaman dengan pendekatan klasifikasi ordinal.
## 2. Formulasi Masalah
Penilaian keparahan karat daun kopi Robusta secara visual subjektif. Penelitian berbasis citra yang ada umumnya hanya mengklasifikasi jenis penyakit, atau memperlakukan tingkat keparahan sebagai kelas nominal. 
## 3. Riset Gap Penelitian Terdahulu
## Research Gap
| Gap | Bukti dari penelitian lain | Posisi penelitian ini |
|---|---|---|
| Keparahan karat kopi dimodelkan sebagai kelas biasa | Esgario dkk. [13] (cross-entropy nominal), Zuñiga Cajas dkk. [15], Ardah dkk. [16] (RL1-4 sebagai kelas terpisah) | Memakai klasifikasi ordinal (CORAL) |
| Kinerja pada kopi Robusta masih lemah | Zuñiga Cajas dkk. [15]: F1 sekitar 0,80 pada Caturra turun menjadi sekitar 0,52 pada citra Robusta RoCoLe | Fokus langsung pada Robusta/RoCoLe |
| Kesalahan terkonsentrasi pada keparahan tinggi | Esgario dkk. [13]: kelas keparahan tinggi dan sangat tinggi paling sering salah diklasifikasi | Menganalisis level 3-4 dan penanganan data timpang |
| Penelitian citra kopi di Indonesia berhenti di jenis penyakit | Puspita dkk. [19], Mutmainah & Alamin [20] | Menilai tingkat keparahan |
| Ordinal terbukti di domain lain, belum di karat kopi | Verbesselt dkk. [18] (gulma), Cao dkk. [11] (CORAL, estimasi usia) | Menerapkan CORAL pada karat kopi RoCoLe |
## 4. Peluang Pengembangan
Menguji model pada citra kebun kopi Indonesia (batasan 3 melarangnya sekarang).
Mengembangkan alat bantu berbasis HP.
Membandingkan CORN, regresi, atau loss berbasis jarak.
Menguji backbone lain.
Menambah data keparahan tinggi dari sumber lain (batasan 8 melarangnya sekarang).
Menghitung persen luas daun lewat segmentasi (batasan 6 melarangnya sekarang).

## 5. Sumber Data dan Referensi
**a. Nama Repository dataset**
- RoCoLe: A robusta coffee leaf images dataset

**b. Alamat Github/Kaggle**
- https://data.mendeley.com/datasets/c5yvn32dzg/2

**c. Referensi Jurnal**
1. Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. Pattern Recognition Letters, 140, 325–331. https://doi.org/10.1016/j.patrec.2020.11.008
2. Esgario, J. G. M., Krohling, R. A., & Ventura, J. A. (2020). Deep learning for classification and severity estimation of coffee leaf biotic stress. Computers and Electronics in Agriculture, 169, 105162. https://doi.org/10.1016/j.compag.2019.105162
3. Gutiérrez, P. A., Pérez-Ortiz, M., Sánchez-Monedero, J., Fernández-Navarro, F., & Hervás-Martínez, C. (2016). Ordinal regression methods: Survey and experimental study. IEEE Transactions on Knowledge and Data Engineering, 28(1), 127–146. https://doi.org/10.1109/TKDE.2015.2457911
4. Mutmainah, S., & Alamin, Z. (2026). Detection of coffee leaf diseases using deep learning to support digitalization and smart agriculture. Jurnal Teknik Informatika (Jutif), 7(3), 2517–2533. https://doi.org/10.52436/1.jutif.2026.7.3.5338
5. Parraga-Alava, J., Cusme, K., Loor, A., & Santander, E. (2019). RoCoLe: A robusta coffee leaf images dataset for evaluation of machine learning based methods in plant diseases recognition. Data in Brief, 25, 104414. https://doi.org/10.1016/j.dib.2019.104414
6. Puspita, D., Padya, I. R., Efan, & Putrawansyah, F. (2025). Robusta coffee disease classification using optimized MobileNetV3 and DenseNet169. Ingénierie des Systèmes d'Information, 30(11), 2917–2928. https://doi.org/10.18280/isi.301110
7. Verbesselt, S., Daems, R., Willekens, A., & Van Beek, J. (2025). UAV based weed pressure detection through relative labelling. Remote Sensing, 17(20), 3434. https://doi.org/10.3390/rs17203434
8. Zuñiga Cajas, J. J., Peña Ramos, O. D., Lasso, E., Avelino, J., Corrales, J. C., & Figueroa Martínez, C. N. (2025). Estimating the severity of coffee leaf rust using deep learning and image processing. Inteligencia Artificial, 28(76), 200–222. https://doi.org/10.4114/intartif.vol28iss76pp200-222
