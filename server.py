from flask import Flask, request, jsonify
import torch
import hashlib
import re
from transformers import pipeline
from private import contains_sensitive_info, encrypt_sensitive_info
from classification_llama import classify_request, generate_response

app = Flask(__name__)

# -------------------------------------------------------------------------
# 1. API Routes with Root Endpoint for Health Check
# -------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({"message": "Server is running. Available endpoints: /classify, /detect, /encrypt, /generate"})

@app.route("/classify", methods=["POST"])
def classify_api():
    """API endpoint for classification."""
    data = request.get_json()
    query = data.get("query", "")
    file_path = data.get("file_path", None)
    result = classify_request(query, file_path)
    return jsonify({"classified_as_sensitive": result})

@app.route("/detect", methods=["POST"])
def detect_sensitive_api():
    """API endpoint for detecting sensitive information."""
    data = request.get_json()
    query = data.get("query", "")
    result = contains_sensitive_info(query)
    return jsonify({"contains_sensitive_info": result})

@app.route("/encrypt", methods=["POST"])
def encrypt_api():
    """API endpoint for encrypting sensitive information."""
    data = request.get_json()
    query = data.get("query", "")
    encrypted_text = encrypt_sensitive_info(query)
    return jsonify({"encrypted_query": encrypted_text})

@app.route("/generate", methods=["POST"])
def generate_api():
    """API endpoint for generating text responses using Llama."""
    data = request.get_json()
    query = data.get("query", "")
    response = generate_response(query)
    return jsonify({"generated_response": response})

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)