
import streamlit as st
import os
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

# --- Financial Data Storage ---
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=['Date', 'Description', 'Type', 'Amount'])

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"**You:** {message.content}")
            else:
                st.write(f"**Bot:** {message.content}")
    else:
        st.write("Bot: Harap upload dokumen terlebih dahulu.")

# --- Financial Functions ---
def record_transaction(description, trans_type, amount):
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    new_transaction = pd.DataFrame([[today, description, trans_type, amount]], columns=['Date', 'Description', 'Type', 'Amount'])
    st.session_state.transactions = pd.concat([st.session_state.transactions, new_transaction], ignore_index=True)
    st.success("Transaksi berhasil dicatat!")

def generate_financial_report():
    if not st.session_state.transactions.empty:
        st.subheader("Laporan Keuangan")
        
        # Display raw data
        st.dataframe(st.session_state.transactions)

        # Summary
        total_income = st.session_state.transactions[st.session_state.transactions['Type'] == 'Pemasukan']['Amount'].sum()
        total_expense = st.session_state.transactions[st.session_state.transactions['Type'] == 'Pengeluaran']['Amount'].sum()
        profit = total_income - total_expense

        st.write(f"**Total Pemasukan:** Rp{total_income:,.2f}")
        st.write(f"**Total Pengeluaran:** Rp{total_expense:,.2f}")
        st.write(f"**Keuntungan:** Rp{profit:,.2f}")
        
        # Visualization
        fig, ax = plt.subplots()
        st.session_state.transactions.groupby('Type')['Amount'].sum().plot(kind='bar', ax=ax)
        ax.set_title('Pemasukan vs Pengeluaran')
        ax.set_ylabel('Jumlah (Rp)')
        st.pyplot(fig)
    else:
        st.info("Belum ada data keuangan untuk ditampilkan.")


def main():
    st.set_page_config(page_title="Chatbot Keuangan",
                       page_icon=":books:")
    st.header("Chatbot Keuangan :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Dokumen Anda")
        pdf_docs = st.file_uploader(
            "Unggah PDF Anda di sini dan klik 'Proses'", accept_multiple_files=True)
        if st.button("Proses"):
            with st.spinner("Memproses"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
        
        st.subheader("Pencatatan Keuangan")
        description = st.text_input("Deskripsi")
        trans_type = st.selectbox("Tipe", ["Pemasukan", "Pengeluaran"])
        amount = st.number_input("Jumlah", min_value=0.0, format="%.2f")
        if st.button("Catat Transaksi"):
            record_transaction(description, trans_type, amount)
            
        if st.button("Tampilkan Laporan Keuangan"):
            generate_financial_report()

    user_question = st.text_input("Ajukan pertanyaan tentang dokumen Anda:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
