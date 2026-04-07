from flask import Flask, request, jsonify
from flask_cors import CORS
import re, os
import torch
import numpy as np
from urllib.parse import urlparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

app = Flask(__name__)
CORS(app)

# -----------------------------
# GLOBAL MODEL (lazy loaded)
# -----------------------------
bert_pipe = None

def load_model():
    global bert_pipe

    if bert_pipe is None:
        print("Loading model from Hugging Face...")

        MODEL_NAME = "kanikathakur/phishing-bert-model"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        device = 0 if torch.cuda.is_available() else -1

        bert_pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=128,
            device=device
        )

        print("Model loaded successfully.")

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
SUSPICIOUS_WORDS = [
    'login', 'verify', 'secure', 'account', 'update', 'confirm',
    'bank', 'paypal', 'signin', 'password', 'credential', 'billing',
    'suspended', 'unusual', 'click', 'free', 'winner', 'prize'
]

def extract_features(url: str) -> dict:
    try:
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        domain = parsed.netloc.lower()
        path   = parsed.path.lower()
        full   = url.lower()

        suspicious_found = [w for w in SUSPICIOUS_WORDS if w in full]

        return {
            "url_length": len(url),
            "has_https": parsed.scheme == 'https',
            "num_dots": url.count('.'),
            "num_hyphens": url.count('-'),
            "num_digits": sum(c.isdigit() for c in url),
            "num_special_chars": sum(c in '@?=&%#_~' for c in url),
            "has_ip_address": bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain)),
            "subdomain_depth": max(0, len(domain.split('.')) - 2),
            "path_length": len(path),
            "suspicious_words": suspicious_found,
            "has_suspicious_words": len(suspicious_found) > 0,
            "domain": domain,
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": "bert-phishing-url"
    })


@app.route('/predict', methods=['POST'])
def predict():
    load_model()  # 🔥 lazy load here

    data = request.get_json()
    url = data.get('url', '').strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Feature extraction
    features = extract_features(url)

    try:
        result = bert_pipe(url)[0]
        label = result['label'].lower()
        score = result['score']

        # Normalize → probability of phishing
        prob = score if label == 'phishing' else 1.0 - score
        pred = 1 if prob >= 0.5 else 0

    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500

    # Risk classification
    if prob >= 0.8:
        risk = "HIGH"
    elif prob >= 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return jsonify({
        "url": url,
        "prediction": "phishing" if pred == 1 else "benign",
        "confidence": round(prob * 100, 1),
        "risk_level": risk,
        "features": features,
        "model": "bert-base-uncased (fine-tuned)"
    })


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)