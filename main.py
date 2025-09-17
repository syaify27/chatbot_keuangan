
import streamlit as st
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# --- Initialization ---
def initialize_session_state():
    if 'transactions' not in st.session_state:
        st.session_state.transactions = pd.DataFrame(columns=['Date', 'Description', 'Type', 'Amount'])
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_rag' not in st.session_state:
        st.session_state.conversation_rag = None
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)

# --- RAG (PDF Processing) Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_rag_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )

# --- Financial Agent Functions ---
def record_transaction(description, trans_type, amount):
    if not description or not trans_type or amount is None:
        return
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    new_transaction = pd.DataFrame([[today, description, trans_type, amount]], columns=['Date', 'Description', 'Type', 'Amount'])
    st.session_state.transactions = pd.concat([st.session_state.transactions, new_transaction], ignore_index=True)
    st.toast(f"Transaksi dicatat: {description} ({trans_type}) - Rp{amount:,.0f}")

def generate_financial_report():
    if not st.session_state.transactions.empty:
        st.subheader("Laporan Keuangan Terkini")
        st.dataframe(st.session_state.transactions)
        
        total_income = st.session_state.transactions[st.session_state.transactions['Type'] == 'Pemasukan']['Amount'].sum()
        total_expense = st.session_state.transactions[st.session_state.transactions['Type'] == 'Pengeluaran']['Amount'].sum()
        profit = total_income - total_expense

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pemasukan", f"Rp{total_income:,.0f}")
        col2.metric("Total Pengeluaran", f"Rp{total_expense:,.0f}")
        col3.metric("Keuntungan", f"Rp{profit:,.0f}")

        fig, ax = plt.subplots()
        st.session_state.transactions.groupby('Type')['Amount'].sum().plot(kind='bar', ax=ax, color=['#d4edda', '#f8d7da'])
        ax.set_title('Pemasukan vs Pengeluaran')
        ax.set_ylabel('Jumlah (Rp)')
        st.pyplot(fig)
    else:
        st.info("Belum ada data keuangan untuk ditampilkan. Silakan catat transaksi pertama Anda.")


FINANCIAL_AGENT_PROMPT = """
Anda adalah asisten keuangan AI yang sangat cerdas. Tugas Anda adalah menganalisis teks dari pengguna untuk mengidentifikasi niat mereka, apakah itu untuk mencatat transaksi keuangan atau meminta laporan.

Jika pengguna ingin **mencatat transaksi**, ekstrak informasi berikut:
- `description`: Deskripsi singkat dan jelas tentang transaksi.
- `type`: Tipe transaksi, harus salah satu dari "Pemasukan" atau "Pengeluaran".
- `amount`: Jumlah transaksi dalam bentuk angka, tanpa titik atau koma.

Jika pengguna ingin **melihat laporan**, set `intent` ke "report".

Jika teks pengguna tidak terkait dengan kedua hal tersebut, set `intent` ke "general_conversation".

Contoh:
- "kemarin saya jual 2 baju seharga 100 ribu" -> `{"intent": "record_transaction", "details": {"description": "Jual 2 baju", "type": "Pemasukan", "amount": 100000}}`
- "tolong catat pengeluaran untuk bensin 50rb" -> `{"intent": "record_transaction", "details": {"description": "Bensin", "type": "Pengeluaran", "amount": 50000}}`
- "laporan keuangan bulan ini dong" -> `{"intent": "report"}`
- "bagaimana cuaca hari ini?" -> `{"intent": "general_conversation"}`

Selalu balas HANYA dengan format JSON yang valid.

Teks Pengguna: "{user_question}"
JSON Output:
"""

def handle_userinput(user_question):
    llm = st.session_state.llm
    
    # 1. Analyze intent
    prompt = FINANCIAL_AGENT_PROMPT.format(user_question=user_question)
    response_text = llm.predict(prompt)
    
    try:
        analysis = json.loads(response_text)
        intent = analysis.get("intent")
    except (json.JSONDecodeError, AttributeError):
        intent = "general_conversation"

    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    bot_response = ""
    # 2. Execute action based on intent
    if intent == "record_transaction":
        details = analysis.get("details", {})
        record_transaction(details.get("description"), details.get("type"), details.get("amount"))
        bot_response = f"Baik, saya sudah mencatat: {details.get('description', '')} sejumlah Rp{details.get('amount', 0):,.0f} sebagai {details.get('type', '')}."
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    
    elif intent == "report":
        bot_response = "Tentu, ini laporan keuangan Anda saat ini."
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        generate_financial_report() # This function will use st.write, etc.

    else: # general_conversation or RAG
        if st.session_state.conversation_rag:
            response = st.session_state.conversation_rag({'question': user_question})
            bot_response = response['answer']
        else:
            bot_response = llm.predict(f"User: {user_question}\nAI:")
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})


def main():
    st.set_page_config(page_title="Asisten Keuangan AI", page_icon="💡")
    st.header("Asisten Keuangan AI 💡")
    initialize_session_state()

    # Sidebar for PDF processing
    with st.sidebar:
        st.subheader("Analisis Dokumen (Opsional)")
        pdf_docs = st.file_uploader("Unggah PDF & klik 'Proses'", accept_multiple_files=True)
        if st.button("Proses Dokumen"):
            if not pdf_docs:
                st.warning("Harap unggah file PDF terlebih dahulu.")
            else:
                with st.spinner("Memproses Dokumen..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("Gagal mengekstrak teks dari PDF. Coba file lain.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation_rag = get_rag_conversation_chain(vectorstore)
                        st.success("Dokumen berhasil diproses! Anda kini bisa bertanya tentang isinya.")
        
        st.subheader("Input Manual (Fallback)")
        desc = st.text_input("Deskripsi Manual")
        typ = st.selectbox("Tipe Manual", ["Pemasukan", "Pengeluaran"])
        amt = st.number_input("Jumlah Manual", min_value=0.0, format="%.2f")
        if st.button("Catat Manual"):
            record_transaction(desc, typ, amt)

    # Main chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Tanya apa saja atau catat transaksi..."):
        handle_userinput(user_question)
        st.experimental_rerun()

if __name__ == '__main__':
    main()
