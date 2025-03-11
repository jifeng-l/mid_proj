import torch
import hashlib
import re
from transformers import pipeline

# -------------------------------------------------------------------------
# 1. Sensitive Information Detection with Improved Model & Prompt
# -------------------------------------------------------------------------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def contains_sensitive_info(query: str) -> bool:
    """
    Uses a more reliable model and refined prompt to determine if the query contains sensitive information.
    """
    candidate_labels = ["ssn or credit card number or PASSPORT number or BANK_ACCOUNT", "transaction ID and others"]
    hypothesis_template = "This text {}."
    
    result = classifier(query, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template)
    
    return result["labels"][0] == "ssn or credit card number or PASSPORT number or BANK_ACCOUNT"

# -------------------------------------------------------------------------
# 2. Selective Encryption of Sensitive Information using Regex
# -------------------------------------------------------------------------
def encrypt_sensitive_info(query: str) -> str:
    """
    Uses regex to detect and encrypt only sensitive parts of a query while keeping non-sensitive text intact.
    """
    patterns = {
        r"\b\d{16}\b": "CREDIT_CARD",  # Credit Card Number
        r"\b\d{9}\b": "SSN",  # Social Security Number
        r"\b[A-Z]{2}\d{6}\b": "PASSPORT",  # Passport Number
        r"\b\d{10,12}\b": "BANK_ACCOUNT",  # Bank Account Number
        r"\b(?:\d{3}-\d{2}-\d{4})\b": "SSN_FORMAT"  # SSN (Formatted)
    }
    
    def hash_match(match):
        return hashlib.sha256(match.group(0).encode()).hexdigest()[:10]  # Shortened hash for readability
    
    encrypted_text = query
    for pattern in patterns:
        encrypted_text = re.sub(pattern, hash_match, encrypted_text)
    
    return encrypted_text

# -------------------------------------------------------------------------
# 3. API Interfaces for Frontend
# -------------------------------------------------------------------------
def sensitive_info_api(query: str) -> bool:
    """ API interface for detecting sensitive information. """
    return contains_sensitive_info(query)

def encrypt_api(query: str) -> str:
    """ API interface for encrypting only sensitive parts of the query. """
    return encrypt_sensitive_info(query)

# Example usage
if __name__ == "__main__":
    sample_query = "On March 5th, I transferred $500 to John Doe via Chase Bank. The transaction ID is 1234567890"
    print("Contains Sensitive Info:", sensitive_info_api(sample_query))
    print("Encrypted Query:", encrypt_api(sample_query))
