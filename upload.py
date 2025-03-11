import os
import torch
import faiss
import pdfplumber
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import streamlit as st

# -------------------------------------------------------------------------
# Set device: use MPS on macOS if available, otherwise CPU
# -------------------------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
st.write("Using device:", device)

# -------------------------------------------------------------------------
# Function to extract text from a bank statement PDF file
# -------------------------------------------------------------------------
def process_bank_statement(file_path: str) -> str:
    """
    Extract text from a PDF bank statement.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
    return text.strip()

# -------------------------------------------------------------------------
# Function to build a FAISS index given a list of documents
# -------------------------------------------------------------------------
def build_rag_database(documents: list[str], embedding_model: SentenceTransformer):
    """
    Given a list of documents, encode them and build a FAISS index.
    Returns the FAISS index object.
    """
    if not documents or len(documents) == 0:
        st.warning("⚠️ No documents found. Using an empty FAISS index.")
        dimension = 384  # Default embedding dimension for MiniLM
        index = faiss.IndexFlatL2(dimension)
        return index  # 返回一个空的 FAISS 索引

    documents = [str(doc) for doc in documents]  # Ensure all docs are strings
    try:
        doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True).cpu().numpy()
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings)
        st.write(f"✅ FAISS index built with {index.ntotal} documents (dimension={dimension}).")
    except Exception as e:
        st.error(f"❌ Error encoding documents: {e}")
        index = faiss.IndexFlatL2(384)
    return index

# -------------------------------------------------------------------------
# Sidebar: File upload and document addition
# -------------------------------------------------------------------------
st.sidebar.header("Upload Bank Statement PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if "documents" not in st.session_state:
    st.session_state.documents = []

if uploaded_file is not None:
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("File uploaded successfully!")

    extracted_text = process_bank_statement(file_path)
    st.sidebar.subheader("Extracted Text")
    st.sidebar.text_area("Text", extracted_text, height=200)

    if st.sidebar.button("Add Document to Database"):
        st.session_state.documents.append(extracted_text)
        st.sidebar.success("Document added!")

# -------------------------------------------------------------------------
# Main: Initialize FAISS index only when documents exist
# -------------------------------------------------------------------------
faiss_index = None
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

if st.session_state.documents:
    faiss_index = build_rag_database(st.session_state.documents, embedding_model)

# -------------------------------------------------------------------------
# Load the Llama model and create a text-generation pipeline
# -------------------------------------------------------------------------
@st.cache_resource
def load_llama_pipeline():
    model_name = "meta-llama/Llama-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llama_pipeline = load_llama_pipeline()

# -------------------------------------------------------------------------
# Function to generate a response using the Llama pipeline and retrieved context
# -------------------------------------------------------------------------
def generate_response(query: str, faiss_index, documents: list[str], embedding_model: SentenceTransformer, llama_pipeline) -> str:
    """
    Retrieve context documents and generate a response based on a custom prompt.
    """
    retrieved_docs = documents if documents else ["No relevant bank transaction data found."]
    context = "\n".join(retrieved_docs)
    prompt = (
        "Use the following context from bank transaction documents to answer the question. "
        "Keep the answer concise (no more than three sentences) and end with 'thanks for asking!'.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    st.write("### Formatted Prompt for Generation:")
    st.code(prompt, language="text")
    inputs = llama_pipeline.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = llama_pipeline.model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.4,
        top_p=0.7,
        repetition_penalty=1.5,
        num_return_sequences=1
    )
    return llama_pipeline.tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------------------------------------------------
# Main chat interface
# -------------------------------------------------------------------------
st.header("Chat with the Bank Transaction RAG Chatbot")
user_query = st.text_input("Enter your question:")

if st.button("Send"):
    if user_query:
        if faiss_index is not None:
            response = generate_response(user_query, faiss_index, st.session_state.documents, embedding_model, llama_pipeline)
            st.write("**Chatbot:**", response)
        else:
            st.warning("Please upload a document first to enable the chatbot.")
    else:
        st.warning("Please enter a question.")
