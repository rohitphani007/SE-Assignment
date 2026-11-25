# Web-Based Fake News Detection and Verification System

Detects whether a news article looks Real or Fake (TF-IDF + Logistic Regression) and shows existing fact-checks (Google Fact Check Tools API, optional).

## How to run
1. pip install flask scikit-learn pandas joblib requests
2. Download Fake and Real News Dataset from Kaggle and put Fake.csv and True.csv in this folder
3. python train.py
4. python app.py
5. Open http://127.0.0.1:5000

Optional: set the environment variable FACTCHECK_API_KEY to enable fact-check search.
