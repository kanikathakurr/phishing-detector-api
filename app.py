
from flask import Flask, request, jsonify
from flask_cors import CORS
import re, os
import torch
import numpy as np
from urllib.parse import urlparse
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          pipeline)

app = Flask(__name__)
CORS(app)  # Allow Chrome extension to call this

# ── Load BERT model 
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'bert-phishing-url')

print(f"Loading BERT model from: {MODEL_DIR}")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
DEVICE     = 0 if torch.cuda.is_available() else -1

bert_pipe = pipeline(
    'text-classification',
    model=bert_model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=128,
    device=DEVICE
)

LABEL2ID = {'benign': 0, 'phishing': 1}
print("BERT model loaded successfully.")

# ── Feature extraction 
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
            "url_length":           len(url),
            "has_https":            parsed.scheme == 'https',
            "num_dots":             url.count('.'),
            "num_hyphens":          url.count('-'),
            "num_digits":           sum(c.isdigit() for c in url),
            "num_special_chars":    sum(c in '@?=&%#_~' for c in url),
            "has_ip_address":       bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain)),
            "subdomain_depth":      max(0, len(domain.split('.')) - 2),
            "path_length":          len(path),
            "suspicious_words":     suspicious_found,
            "has_suspicious_words": len(suspicious_found) > 0,
            "domain":               domain,
        }
    except Exception as e:
        return {"error": str(e)}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url  = data.get('url', '').strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Feature breakdown
    features = extract_features(url)

    # BERT prediction
    try:
        result = bert_pipe(url)[0]
        label  = result['label'].lower()   # 'phishing' or 'benign'
        score  = result['score']           # confidence in the predicted label

        # Normalise so prob always = P(phishing)
        prob = score if label == 'phishing' else 1.0 - score
        pred = 1 if prob >= 0.5 else 0

    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500

    # Risk level
    if prob >= 0.8:
        risk = "HIGH"
    elif prob >= 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return jsonify({
        "url":        url,
        "prediction": "phishing" if pred == 1 else "benign",
        "confidence": round(prob * 100, 1),
        "risk_level": risk,
        "features":   features,
        "model":      "bert-base-uncased (fine-tuned)"
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "bert-phishing-url"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
