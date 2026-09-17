from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load saved model and vectorizer
print("Loading spam detection model...")
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Model ready!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    data = request.get_json()
    message = data["message"]

    if not message.strip():
        return jsonify({"error": "Please enter a message"})

    # Convert text to numbers and predict
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0]

    is_spam = bool(prediction == 1)
    confidence = round(float(probability[1] if is_spam else probability[0]) * 100, 2)

    return jsonify({
        "is_spam": is_spam,
        "label": "SPAM" if is_spam else "LEGITIMATE",
        "confidence": confidence,
        "emoji": "🚨" if is_spam else "✅"
    })

if __name__ == "__main__":
    app.run(debug=True)