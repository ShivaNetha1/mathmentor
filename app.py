import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load FAISS vectorstore from disk
def load_vectorstore(path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# Setup local LLM from Ollama
def get_ollama_llm():
    return Ollama(model="gemma3:4b")  # Make sure this is the correct tag you downloaded

# Streamlit UI
st.set_page_config(page_title="📘 Math Mentor (RAG + Gemma)", layout="wide")
st.title("🧠 Math Mentor (Local RAG with Gemma via Ollama)")

query = st.text_input("🔎 Ask a math question:")
if query:
    with st.spinner("Thinking..."):
        # Load retriever
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever()

        # Load LLM
        llm = get_ollama_llm()

        # RAG pipeline
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # Get result
        result = qa_chain({"query": query})

        # Show answer
        st.subheader("📌 Answer:")
        st.write(result["result"])

        # Show source chunks
        with st.expander("📄 Retrieved Source Chunks"):
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"**Chunk {i}:**")
                st.code(doc.page_content[:1000])
