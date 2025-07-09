# train_sklearn.py
import json
import random
import pickle

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1) Load intents
with open("intents.json") as f:
    data = json.load(f)["intents"]

sentences, labels = [], []
for intent in data:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# 2) Encode labels
lbl_encoder = LabelEncoder()
y = lbl_encoder.fit_transform(labels)

# 3) Vectorize text
vectorizer = TfidfVectorizer(max_features=500)  # you can tune max_features
X = vectorizer.fit_transform(sentences)

# 4) Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# 5) Save everything
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(lbl_encoder, open("label_encoder.pkl", "wb"))
pickle.dump(clf, open("classifier.pkl", "wb"))
