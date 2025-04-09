from flask import Flask, request, jsonify, render_template
import pickle
import re
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Workaround for TensorFlow issue
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

#--------------------------------------

# Import the ABSA and summary functions

#--------------------------------------
from absa import aspect_based_sentiment_improved  # Import your main ABSA function
from summary import generate_summary  # Import your summary function

app = Flask(__name__)

# ----------------------------

# Load Models and Tokenizers

# ----------------------------


# Load the sentiment analysis BiLSTM model
model = tf.keras.models.load_model('best_model.keras')

# Load the saved tokenizer for sentiment analysis
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Maximum sequence length for sentiment analysis
max_length = 150  # Adjust as needed


# ----------------------------

# Helper Functions

# ----------------------------

def clean_text(text):
    """Clean the text for sentiment analysis."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ----------------------------
### Complete Pipeline
# ----------------------------

def complete_pipeline(review, max_length=100):
    # Overall sentiment prediction using the cleaned review
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)[0][0]
    if pred > 0.5:
        overall_sentiment = "Positive"
    elif pred >= 0.15:
        overall_sentiment = "Neutral"
    else:
        overall_sentiment = "Negative"
    
    # Generate summary using T5
    summary = generate_summary(review)
    
    # Get aspect-based sentiment analysis results using the improved ABSA function
    # We'll use the full review for better context, but could also use the summary
    aspects = aspect_based_sentiment_improved(summary)
    
    return {
        "Overall Sentiment": overall_sentiment,
        "Summary": summary,
        "Aspect-based Sentiments": aspects
    }

# ----------------------------
# Flask Routes
# ----------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    review = data.get('review', '')
    if not review:
        return jsonify({"error": "No review text provided"}), 400
    try:
        result = complete_pipeline(review)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sentiment', methods=['POST'])
def sentiment_api():
    data = request.get_json()
    review = data.get('review', '')
    
    if not review:
        return jsonify({"error": "No review text provided"}), 400
    try:
        result = complete_pipeline(review)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

