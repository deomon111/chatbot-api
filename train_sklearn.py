import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1) Load intents file
with open("intents.json") as f:
    data = json.load(f)["intents"]

# 2) Prepare training data
sentences = []
labels = []
for intent in data:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# 3) Encode labels as integers
lbl_encoder = LabelEncoder()
y = lbl_encoder.fit_transform(labels)

# 4) Convert text to numeric features
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(sentences)

# 5) Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# 6) Save artifacts for later use
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(lbl_encoder, open("label_encoder.pkl", "wb"))
pickle.dump(clf, open("classifier.pkl", "wb"))

print("Training complete. Artifacts saved to vectorizer.pkl, label_encoder.pkl, classifier.pkl.")
