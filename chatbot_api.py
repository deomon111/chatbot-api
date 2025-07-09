# chatbot_api.py
from flask import Flask, request, jsonify
import pickle, random, json
import numpy as np

# Load artifacts
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
lbl_encoder = pickle.load(open("label_encoder.pkl","rb"))
clf         = pickle.load(open("classifier.pkl","rb"))
with open("intents.json") as f:
    intents = json.load(f)["intents"]

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg  = data.get("message", "")
    X    = vectorizer.transform([msg])
    pred = clf.predict(X)[0]
    # pick a random response
    for intent in intents:
        if intent["tag"] == pred:
            return jsonify({"response": random.choice(intent["responses"])})
    return jsonify({"response": "Sorry, I don't understand."})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
