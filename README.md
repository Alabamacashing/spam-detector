# Spam Detector

A web app that classifies text messages/emails as spam or legitimate using a Naive Bayes model trained with TF-IDF features.

## Live Demo
https://spam-detector-31bi.onrender.com

## Accuracy
97.94% on held-out test data

## Tech Stack
- Python, Flask
- scikit-learn (Naive Bayes + TF-IDF)
- Deployed on Render

## How it works
1. Enter a message in the text box
2. The app converts the message into TF-IDF features
3. The trained Naive Bayes model predicts spam or legitimate with a confidence score

## Running locally
\`\`\`bash
pip install -r requirements.txt
python app.py
\`\`\`
Then open http://127.0.0.1:5000
