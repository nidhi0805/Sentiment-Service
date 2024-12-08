from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
sentiment_model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text data provided"}), 400

    text_data = data['text']
    # Transform text data to the same form used during training
    text_vector = vectorizer.transform([text_data])
    # Predict sentiment
    prediction = sentiment_model.predict(text_vector)

    return jsonify({"sentiment": prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Use a specific port
