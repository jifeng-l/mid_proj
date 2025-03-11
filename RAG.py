import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from datasets import load_dataset

# -------------------------------------------------------------------------
# 1. Load Llama-3 Model & Tokenizer
# -------------------------------------------------------------------------
llama_model_name = "meta-llama/Llama-3-8B"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16, device_map="auto")

def get_llama_pipeline():
    """Creates a HuggingFace pipeline for Llama-3 text generation."""
    return pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer, max_new_tokens=300)

llama_pipeline = get_llama_pipeline()
llm = HuggingFacePipeline(pipeline=llama_pipeline)

# -------------------------------------------------------------------------
# 2. Initialize FAISS Index & Embeddings
# -------------------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS(embedding_function=embedding_model)

def build_rag_database():
    """Fetches financial data and builds FAISS index."""
    print("📥 Loading financial dataset...")
    dataset = load_dataset("zeroshot/financial-news", split="train")  # Financial news dataset
    financial_texts = [item["text"] for item in dataset if isinstance(item["text"], str)]
    vector_store.add_texts(financial_texts)
    print(f"✅ FAISS index built with {len(financial_texts)} financial documents.")

# Load and index financial dataset
build_rag_database()

# -------------------------------------------------------------------------
# 3. Define RAG-based RetrievalQA Chain
# -------------------------------------------------------------------------
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA(llm=llm, retriever=retriever, return_source_documents=True)

def generate_rag_response(query: str):
    """Retrieves documents and generates an answer using Llama-3."""
    response = qa_chain.run(query)
    return response
