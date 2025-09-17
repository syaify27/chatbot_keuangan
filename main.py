
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os

st.set_page_config(page_title="Tes Koneksi API Google", page_icon="🔬")
st.header("Tes Koneksi API Google 🔬")

st.info("Skrip ini mencoba melakukan satu panggilan ke API Google Gemini menggunakan kunci yang dikonfigurasi di Streamlit Secrets.")

api_key = None

# 1. Mencoba membaca kunci API dari Streamlit Secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.success("Berhasil membaca GOOGLE_API_KEY dari Streamlit Secrets.")
    st.code(f"Kunci API yang ditemukan (beberapa karakter pertama): {api_key[:4]}...{api_key[-4:]}", language="text")
except (KeyError, FileNotFoundError):
    st.error("Gagal menemukan GOOGLE_API_KEY di Streamlit Secrets.")
    st.warning("Pastikan Anda telah mengatur secret dengan benar di menu [Manage app] > [Settings] > [Secrets].")
    st.stop()

# 2. Mencoba menginisialisasi dan memanggil model
if api_key:
    st.write("---")
    st.write("Mencoba menginisialisasi model `gemini-pro`...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.1
        )
        st.success("Berhasil menginisialisasi model ChatGoogleGenerativeAI.")

        st.write("---")
        st.write("Mengirim pesan tes: 'Halo, apakah kamu aktif?'")
        with st.spinner("Menunggu respons dari API... (Ini mungkin memakan waktu beberapa detik)"):
            try:
                response = llm.predict("Halo, apakah kamu aktif?")
                st.success("TES BERHASIL! API merespons dengan sukses.")
                st.balloons()
                st.subheader("Respons dari API:")
                st.write(response)
            except Exception as e:
                st.error("GAGAL saat memanggil API. Ini adalah akar masalahnya.")
                st.subheader("Detail Error:")
                st.exception(e)
                st.warning("Error ini biasanya berarti Penagihan (Billing) atau Vertex AI API belum diaktifkan di Proyek Google Cloud Anda.")

    except Exception as e:
        st.error("GAGAL saat menginisialisasi model. Error ini bisa terjadi jika kunci API tidak valid.")
        st.subheader("Detail Error:")
        st.exception(e)
