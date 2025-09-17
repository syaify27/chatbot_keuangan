# Chatbot Keuangan dengan RAG dan Streamlit

Ini adalah chatbot keuangan yang dibangun dengan Python, Streamlit, Langchain, dan Google Gemini API. Chatbot ini dapat menjawab pertanyaan berdasarkan dokumen yang diunggah (kemampuan RAG) dan juga dapat mencatat serta melaporkan data keuangan.

## Fitur

- **Kemampuan RAG:** Chatbot dapat memahami dan menjawab pertanyaan berdasarkan konten dari file PDF yang Anda unggah.
- **Pencatatan Keuangan:** Catat transaksi keuangan (pemasukan dan pengeluaran) dengan mudah.
- **Laporan Keuangan:** Hasilkan laporan keuangan sederhana yang mencakup total pemasukan, total pengeluaran, dan keuntungan, beserta visualisasi dasar.
- **Antarmuka Streamlit:** Antarmuka pengguna yang ramah dan interaktif yang dibuat dengan Streamlit.

## Cara Menjalankan

1. **Instal dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Atur kunci API Anda:**
   - Buat file `.env` di direktori root proyek.
   - Tambahkan kunci API Google Gemini Anda ke file `.env` seperti berikut:
     ```
     GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
     ```

3. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run main.py
   ```
