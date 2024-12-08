from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model
sentiment_model = joblib.load('sentiment_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return '''
    <html>
    <head>
        <title>Sentiment Service</title>
    </head>
    <body>
        <h1>Sentiment Analysis Service</h1>
        <p>Created by Nidhi Patel</p>
        <p>Use the <a href="/predict">/predict</a> endpoint to get sentiment analysis predictions.</p>
        <form action="/submit" method="post">
            <label for="text">Enter text for sentiment analysis:</label><br>
            <textarea id="text" name="text" rows="4" cols="50"></textarea><br><br>
            <input type="submit" value="Submit">
        </form>
    </body>
    </html>
    '''

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        text = request.form['text']
        # Vectorize the text
        text_vector = tfidf_vectorizer.transform([text])
        # Predict sentiment
        prediction = sentiment_model.predict(text_vector)[0]
        return render_template('result.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    # Vectorize the text
    text_vector = tfidf_vectorizer.transform([text])
    # Predict sentiment
    prediction = sentiment_model.predict(text_vector)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
