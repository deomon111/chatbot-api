from flask import Flask, request, jsonify
import pickle, json, random, os

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
lbl_encoder = pickle.load(open("label_encoder.pkl", "rb"))
clf = pickle.load(open("classifier.pkl", "rb"))
with open("intents.json") as f:
    intents = json.load(f)["intents"]

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.get_json(force=True).get("message", "")
    X = vectorizer.transform([msg])

    # Get probabilities & predicted tag
    probas = clf.predict_proba(X)[0]
    pred_idx = probas.argmax()
    pred_tag = lbl_encoder.inverse_transform([pred_idx])[0]

    # DEBUG LINE (remove in production)
    print(f"[DEBUG] input={msg!r}, probs={probas}, pred_tag={pred_tag}")

    # Optional: low‐confidence fallback
    if probas.max() < 0.2:
        return jsonify({"response": "I’m not sure I understand. Could you rephrase?"})

    # Lookup intent
    for intent in intents:
        if intent["tag"] == pred_tag:
            return jsonify({"response": random.choice(intent["responses"])})

    # Fallback
    return jsonify({"response": "Sorry, I don't understand."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
