
import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def initialize_session_state():
    if "api_key_configured" not in st.session_state:
        try:
            st.secrets["GOOGLE_API_KEY"]
            st.session_state.api_key_configured = True
        except (KeyError, FileNotFoundError):
            st.session_state.api_key_configured = False

    if 'transactions' not in st.session_state:
        st.session_state.transactions = pd.DataFrame(columns=['Date', 'Description', 'Type', 'Amount'])
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_rag' not in st.session_state:
        st.session_state.conversation_rag = None
    if 'llm' not in st.session_state and st.session_state.api_key_configured:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.4,
            convert_system_message_to_human=True
        )

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        except Exception:
            pass
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    if not st.session_state.api_key_configured: return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_rag_conversation_chain(vectorstore):
    if not st.session_state.get('llm'): return None
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )

def record_transaction(description, trans_type, amount):
    if not all([description, trans_type, amount is not None]):
        st.toast("Gagal mencatat: informasi tidak lengkap.")
        return
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    new_transaction = pd.DataFrame([[today, description, trans_type, float(amount)]], columns=['Date', 'Description', 'Type', 'Amount'])
    st.session_state.transactions = pd.concat([st.session_state.transactions, new_transaction], ignore_index=True)
    st.toast(f"Transaksi dicatat: {description}")

def generate_financial_report():
    if not st.session_state.transactions.empty:
        st.subheader("Laporan Keuangan Terkini")
        st.dataframe(st.session_state.transactions, use_container_width=True)
        
        total_income = st.session_state.transactions[st.session_state.transactions['Type'] == 'Pemasukan']['Amount'].sum()
        total_expense = st.session_state.transactions[st.session_state.transactions['Type'] == 'Pengeluaran']['Amount'].sum()
        profit = total_income - total_expense

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pemasukan", f"Rp{total_income:,.0f}")
        col2.metric("Total Pengeluaran", f"Rp{total_expense:,.0f}")
        col3.metric("Keuntungan", f"Rp{profit:,.0f}")

        try:
            fig, ax = plt.subplots()
            st.session_state.transactions.groupby('Type')['Amount'].sum().plot(kind='bar', ax=ax, color=['#28a745', '#dc3545'])
            ax.set_title('Pemasukan vs Pengeluaran')
            ax.set_ylabel('Jumlah (Rp)')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal membuat grafik: {e}")
    else:
        st.info("Belum ada data keuangan untuk ditampilkan.")

FINANCIAL_AGENT_PROMPT = '''
Anda adalah asisten keuangan AI. Analisis teks pengguna untuk niat mereka.
- Untuk mencatat transaksi, ekstrak: `description`, `type`("Pemasukan"/"Pengeluaran"), dan `amount`.
- Untuk melihat laporan, set `intent` ke "report".
- Jika tidak relevan, set `intent` ke "general_conversation".

Contoh:
- "kemarin saya jual 2 baju seharga 100 ribu" -> {{"intent": "record_transaction", "details": {{"description": "Jual 2 baju", "type": "Pemasukan", "amount": 100000}}}}
- "tolong catat pengeluaran untuk bensin 50rb" -> {{"intent": "record_transaction", "details": {{"description": "Bensin", "type": "Pengeluaran", "amount": 50000}}}}
- "laporan keuangan bulan ini dong" -> {{"intent": "report"}}

Balas HANYA dengan JSON.

Teks Pengguna: "{user_question}"
JSON Output:
'''

def handle_userinput(user_question):
    if not st.session_state.get('llm'):
        st.error("Model LLM tidak terinisialisasi.")
        return

    llm = st.session_state.llm
    prompt = FINANCIAL_AGENT_PROMPT.format(user_question=user_question)
    response_text = llm.predict(prompt)
    
    intent = "general_conversation"
    analysis = {}
    try:
        cleaned_text = response_text.strip().replace('```json', '').replace('```', '')
        if cleaned_text:
            analysis = json.loads(cleaned_text)
            if isinstance(analysis, dict):
                intent = analysis.get("intent", "general_conversation")
    except (json.JSONDecodeError, AttributeError):
        pass

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    bot_response_content = ""

    if intent == "record_transaction":
        details = analysis.get("details", {})
        record_transaction(details.get("description"), details.get("type"), details.get("amount"))
        bot_response_content = f"Baik, saya sudah mencatat: {details.get('description', '')} sejumlah Rp{details.get('amount', 0):,.0f} sebagai {details.get('type', '')}."
    elif intent == "report":
        bot_response_content = "Tentu, ini laporan keuangan Anda saat ini."
    else: # general_conversation
        if st.session_state.conversation_rag:
            response = st.session_state.conversation_rag({'question': user_question})
            bot_response_content = response['answer']
        else:
            bot_response_content = llm.predict(f"Human: {user_question}\nAI Assistant:")
            
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response_content})

def main():
    st.set_page_config(page_title="Asisten Keuangan AI", page_icon="💡", layout="wide")
    st.title("Asisten Keuangan AI 💡")
    
    initialize_session_state()

    if not st.session_state.api_key_configured:
        st.error("Kunci API Google (GOOGLE_API_KEY) tidak dikonfigurasi. Harap atur di [Secrets] aplikasi Anda.")
        st.stop()

    with st.sidebar:
        st.header("Opsi")
        st.subheader("Analisis Dokumen (RAG)")
        pdf_docs = st.file_uploader("Unggah PDF & klik 'Proses'", accept_multiple_files=True, type="pdf")
        if st.button("Proses Dokumen"):
            if pdf_docs:
                with st.spinner("Memproses dokumen..."):                    
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation_rag = get_rag_conversation_chain(vectorstore)
                        st.success("Dokumen berhasil diproses!")
                    else:
                        st.error("Gagal mengekstrak teks dari PDF.")
            else:
                st.warning("Harap unggah file PDF terlebih dahulu.")
        
        st.subheader("Input Manual")
        with st.form("manual_transaction_form", clear_on_submit=True):
            desc = st.text_input("Deskripsi")
            typ = st.selectbox("Tipe", ["Pemasukan", "Pengeluaran"])
            amt = st.number_input("Jumlah", min_value=0.0, format="%.0f")
            submitted = st.form_submit_button("Catat Manual")
            if submitted:
                record_transaction(desc, typ, amt)

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message["content"] == "Tentu, ini laporan keuangan Anda saat ini.":
                    generate_financial_report()

    if user_question := st.chat_input("Tanya apa saja atau catat transaksi..."):
        handle_userinput(user_question)
        st.rerun()

if __name__ == '__main__':
    main()
