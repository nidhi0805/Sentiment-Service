from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
model_file = "sentiment_model.pkl"
with open(model_file, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return "Sentiment Analysis Microservice is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Please provide input text for sentiment analysis."}), 400

    # Prepare data
    text = [data["text"]]  # Convert input to list for model prediction
    # Apply preprocessing if necessary
    df = pd.DataFrame({"text": text})  # Example, modify as needed
    predictions = model.predict(df["text"])

    return jsonify({"text": text[0], "sentiment": predictions[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
