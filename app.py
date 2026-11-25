import os
import joblib
import requests
from flask import Flask, render_template, request, jsonify
from utils import clean_text

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
feature_names = vectorizer.get_feature_names_out()

FACTCHECK_KEY = os.environ.get("FACTCHECK_API_KEY", "")


def search_fact_checks(query):
    if not FACTCHECK_KEY:
        return []
    try:
        r = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": query[:200], "key": FACTCHECK_KEY, "languageCode": "en"},
            timeout=8,
        )
        results = []
        for claim in r.json().get("claims", [])[:5]:
            for review in claim.get("claimReview", [])[:1]:
                results.append({
                    "claim": claim.get("text", ""),
                    "publisher": review.get("publisher", {}).get("name", ""),
                    "rating": review.get("textualRating", ""),
                    "url": review.get("url", ""),
                })
        return results
    except Exception:
        return []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data.get("text") or "").strip()

    if len(text.split()) < 20:
        return jsonify({"error": "Please paste at least 20 words of the news article."}), 400

    vec = vectorizer.transform([clean_text(text)])
    probs = model.predict_proba(vec)[0]          # [fake_prob, real_prob]
    label = "Real" if probs[1] >= 0.5 else "Fake"
    confidence = round(float(max(probs)) * 100, 1)

    # words that pushed the prediction the most
    coo = vec.tocoo()
    coefs = model.coef_[0]
    contrib = sorted(zip(coo.col, coo.data), key=lambda x: abs(x[1] * coefs[x[0]]), reverse=True)[:5]
    key_words = [feature_names[i] for i, _ in contrib]

    return jsonify({
        "label": label,
        "confidence": confidence,
        "key_words": key_words,
        "fact_checks": search_fact_checks(text),
        "factcheck_enabled": bool(FACTCHECK_KEY),
    })


if __name__ == "__main__":
    app.run(debug=True)