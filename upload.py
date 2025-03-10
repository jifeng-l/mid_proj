import os
import torch
import faiss
import pdfplumber
import random
from datasets import load_dataset
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
    doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True).cpu().numpy()
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    st.write(f"✅ FAISS index built with {index.ntotal} documents (dimension={dimension}).")
    return index

# -------------------------------------------------------------------------
# Function to retrieve top-k documents from the FAISS index
# -------------------------------------------------------------------------
def retrieve_top_k_docs(query: str, index, documents: list[str], embedding_model: SentenceTransformer, k=3, threshold=0.5):
    """
    Search the FAISS index for the top-k relevant documents for the given query.
    """
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = []
    for i, dist in zip(indices[0], distances[0]):
        # Check that the index is valid and the similarity meets the threshold
        if i < len(documents) and dist >= threshold:
            retrieved_docs.append(documents[i])
    return retrieved_docs

# -------------------------------------------------------------------------
# Function to generate a response using the Llama pipeline and retrieved context
# -------------------------------------------------------------------------
def generate_response(query: str, faiss_index, documents: list[str], embedding_model: SentenceTransformer, llama_pipeline) -> str:
    """
    Retrieve context documents and generate a response based on a custom prompt.
    """
    # Retrieve the top relevant documents
    retrieved_docs = retrieve_top_k_docs(query, faiss_index, documents, embedding_model, k=3, threshold=0.5)
    context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant bank transaction data found."
    # Create the prompt using the context
    prompt = (
        "Use the following context from bank transaction documents to answer the question. "
        "Keep the answer concise (no more than three sentences) and end with 'thanks for asking!'.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    st.write("### Formatted Prompt for Generation:")
    st.code(prompt, language="text")
    # Tokenize the prompt and move tensors to the target device
    inputs = llama_pipeline.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Generate the answer
    outputs = llama_pipeline.model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.4,
        top_p=0.7,
        repetition_penalty=1.5,
        num_return_sequences=1
    )
    response = llama_pipeline.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# -------------------------------------------------------------------------
# Sidebar: File upload and document addition
# -------------------------------------------------------------------------
st.sidebar.header("Upload Bank Statement PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary folder
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("File uploaded successfully!")
    
    # Process the PDF to extract text
    extracted_text = process_bank_statement(file_path)
    st.sidebar.subheader("Extracted Text")
    st.sidebar.text_area("Text", extracted_text, height=200)
    
    # Add the extracted text to the document database if the button is clicked
    if st.sidebar.button("Add Document to Database"):
        if "documents" not in st.session_state:
            st.session_state.documents = []
        st.session_state.documents.append(extracted_text)
        st.sidebar.success("Document added!")

# -------------------------------------------------------------------------
# Main: Initialize document database and FAISS index
# -------------------------------------------------------------------------
# If no documents exist in session state, load a sample dataset (cc_news) as a starting point.
if "documents" not in st.session_state:
    dataset = load_dataset("cc_news", split="train[:1000]")
    docs = [row["text"] for row in dataset if isinstance(row["text"], str) and row["text"].strip() != ""]
    st.session_state.documents = docs

st.write(f"✅ Number of documents in the database: {len(st.session_state.documents)}")

# Build the FAISS index and load the embedding model
@st.cache_data
def load_index(docs):
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    index = build_rag_database(docs, embedding_model)
    return index, embedding_model

faiss_index, embedding_model = load_index(st.session_state.documents)

# -------------------------------------------------------------------------
# Load the Llama model and create a text-generation pipeline
# -------------------------------------------------------------------------
@st.cache_resource
def load_llama_pipeline():
    model_name = "meta-llama/Llama-3.1-8B"
    # Load the full-precision model (bitsandbytes is not used on MPS)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Create the text-generation pipeline
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llama_pipeline = load_llama_pipeline()

# -------------------------------------------------------------------------
# Main chat interface
# -------------------------------------------------------------------------
st.header("Chat with the Bank Transaction RAG Chatbot")
user_query = st.text_input("Enter your question:")

if st.button("Send"):
    if user_query:
        response = generate_response(user_query, faiss_index, st.session_state.documents, embedding_model, llama_pipeline)
        st.write("**Chatbot:**", response)
    else:
        st.warning("Please enter a question.")