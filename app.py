from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load pre-trained model and vectorizer
sentiment_model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract text data from the request
    data = request.json.get('text')
    # Preprocess and predict
    vectorized = vectorizer.transform([data])
    prediction = sentiment_model.predict(vectorized)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
