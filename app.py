from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import re

model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")
le_employment = joblib.load("encoder_employment.pkl")
le_approval = joblib.load("encoder_approval.pkl")

app = Flask(__name__)

custom_stopwords = {
    'the', 'and', 'is', 'in', 'to', 'of', 'for', 'on', 'a', 'with', 'i', 'need', 'want', 'this', 'it'
}

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = [word for word in text.split() if word not in custom_stopwords and len(word) > 2]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            text = request.form["text"]
            income = float(request.form["income"])
            credit_score = float(request.form["credit_score"])
            loan_amount = float(request.form["loan_amount"])
            dti = float(request.form["dti"])
            employment = request.form["employment"]

            text_clean = clean_text(text)
            text_vec = tfidf.transform([text_clean]).toarray()

            emp_encoded = le_employment.transform([employment])[0]

            input_data = np.hstack((
                text_vec,
                [[income, credit_score, loan_amount, dti, emp_encoded]]
            ))

            input_scaled = scaler.transform(input_data)

            prediction_encoded = model.predict(input_scaled)[0]
            prediction_label = le_approval.inverse_transform([prediction_encoded])[0]

            return render_template("predict.html", prediction=prediction_label)

        except Exception as e:
            return render_template("predict.html", prediction=None, error=str(e))

    return render_template("predict.html", prediction=None)

@app.route('/detail')
def detail():
    df = pd.read_csv("loan_data.csv")
    return render_template("detail.html", tables=[df.head().to_html(classes='data', header="true")])

if __name__ == "__main__":
    app.run(debug=True)
