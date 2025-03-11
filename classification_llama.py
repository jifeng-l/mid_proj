import torch
import faiss
import random
import pdfplumber
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# -------------------------------------------------------------------------
# 1. Set the device: use MPS if available (Apple Silicon), otherwise CPU
# -------------------------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------------------------------
# 2. Function to process a bank statement PDF
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
        print(f"Error processing {file_path}: {e}")
    return text.strip()

# -------------------------------------------------------------------------
# 3. Classification Module: Detect Inquiry Type
# -------------------------------------------------------------------------
from transformers import pipeline as hf_pipeline

classifier = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_request(query: str, file_path: str = None) -> bool:
    """
    If a file (document/image) is provided, return True.
    Otherwise, classify if the request is finance/property-related.
    """
    if file_path:
        return True
    
    candidate_labels = ["financial inquiry", "property-related question", "general inquiry"]
    result = classifier(query, candidate_labels=candidate_labels)
    
    return result["labels"][0] in ["financial inquiry", "property-related question"]

# -------------------------------------------------------------------------
# 4. Llama-3 Based Text Generation
# -------------------------------------------------------------------------
PROMPT_TEMPLATE = (
    "Use the following context to answer the question. Keep it concise.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

def generate_response(query: str) -> str:
    """
    Generate a response for the given query using Llama-3.
    """
    formatted_prompt = PROMPT_TEMPLATE.format(context="No specific context found.", question=query)
    
    inputs = llama_tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = llama_pipeline.model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.4,
        top_p=0.7,
        repetition_penalty=1.5,
        num_return_sequences=1
    )
    
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# -------------------------------------------------------------------------
# 5. Initialize Llama model
# -------------------------------------------------------------------------
model_name = "meta-llama/Llama-3.1-8B"
# peft_model_name = "FinGPT/fingpt-mt_llama3-8b_lora"

llama_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
llama_model.to(device)

llama_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token

# peft_model = PeftModel.from_pretrained(llama_model, peft_model_name)
llama_model.eval()


llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer)

# -------------------------------------------------------------------------
# 6. API Interfaces for Frontend
# -------------------------------------------------------------------------
def classification_api(query: str, file_path: str = None) -> bool:
    """ API interface for classification module. """
    return classify_request(query, file_path)

def llama_generation_api(query: str) -> str:
    """ API interface for Llama-3 response generation. """
    return generate_response(query)

# Example usage
if __name__ == "__main__":
    sample_query = "What was my last bank transaction?"
    print("Classification Result:", classification_api(sample_query))
    print("Generated Response:", llama_generation_api(sample_query))
