
import streamlit as st
import google.generativeai as genai
import os

# --- Konfigurasi Halaman dan Judul ---
st.set_page_config(page_title="Tes Chatbot Gemini Paling Dasar", page_icon="🔥")
st.title("🔥 Tes Chatbot Paling Dasar")
st.caption("Menghilangkan semua kerumitan, hanya tes koneksi murni ke Google API.")

# --- Langkah 1: Konfigurasi Kunci API ---
st.write("### Langkah 1: Membaca Kunci API")
api_key = None
try:
    # Cara yang benar untuk mengakses secrets di Streamlit Cloud
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.success("Kunci API (GOOGLE_API_KEY) berhasil dibaca dari secrets.")
    st.info(f"Kunci API dimulai dengan: `{api_key[:4]}...`")
except (KeyError, FileNotFoundError):
    st.error("Kunci API (GOOGLE_API_KEY) tidak ditemukan di Streamlit Secrets.")
    st.warning("Mohon periksa kembali [Manage app] > [Settings] > [Secrets].")
    st.stop()

# --- Langkah 2: Mengkonfigurasi Library Google AI ---
st.write("### Langkah 2: Mengkonfigurasi Library Google AI")
try:
    genai.configure(api_key=api_key)
    st.success("Library Google AI berhasil dikonfigurasi.")
except Exception as e:
    st.error("Terjadi error saat mengkonfigurasi library Google AI.")
    st.exception(e)
    st.stop()

# --- Langkah 3: Inisialisasi Model ---
st.write("### Langkah 3: Inisialisasi Model `gemini-1.5-flash`")
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    st.success("Model `gemini-1.5-flash` berhasil diinisialisasi.")
except Exception as e:
    st.error("Gagal menginisialisasi model. Ini bisa jadi masalah dengan kunci API atau izin. Atau nama model salah.")
    st.exception(e)
    st.stop()

# --- Langkah 4: Antarmuka Chat ---
st.write("### Langkah 4: Mulai Chat!")
st.info("Jika semua langkah di atas berhasil, chatbot di bawah ini seharusnya berfungsi.")

# Inisialisasi riwayat chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan pesan lama
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Terima input pengguna
if prompt := st.chat_input("Kirim pesan ke Gemini..."):    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner("Memanggil API Google..."):               
                response = model.generate_content(prompt)
                full_response = response.text
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error("GAGAL TOTAL saat memanggil API. Ini adalah akar masalahnya.")
            st.exception(e)

