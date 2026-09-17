import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("spam.tsv", sep="\t", header=None, names=["label", "message"])

print(f"Total messages: {len(df)}")
print(f"\nLabel distribution:")
print(df["label"].value_counts())
print(f"\nExample spam message:")
print(df[df["label"] == "spam"]["message"].iloc[0])
print(f"\nExample legitimate message:")
print(df[df["label"] == "ham"]["message"].iloc[0])

# Convert labels to numbers
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"],
    test_size=0.2,
    random_state=42
)

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel accuracy: {accuracy:.2%}")
print("\nDetailed report:")
print(classification_report(y_test, predictions, target_names=["Legitimate", "Spam"]))

# Save model and vectorizer
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved!")