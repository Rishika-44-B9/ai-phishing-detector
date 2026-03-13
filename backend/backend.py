from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("dataset/phishing_dataset.csv")

X = data['text']
y = data['label']

vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vector, y)

app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def detect():
    message = request.json["message"]

    msg_vector = vectorizer.transform([message])
    prediction = model.predict(msg_vector)[0]

    return jsonify({"result": prediction})

app.run()
