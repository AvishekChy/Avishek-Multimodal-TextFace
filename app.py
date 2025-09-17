import os

from flask import Flask, jsonify, render_template, request

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError as e:
    print(
        f"Error: TensorFlow or Keras not found. Install with 'pip install tensorflow==2.17.0 keras==3.3.3'. Error: {e}"
    )
    exit(1)

import pickle

import numpy as np

# # Demo mode: Set to True if models aren't ready (no loading, use placeholder predictions)
# DEMO_MODE = True  # Change to False once models are downloaded

# Minimal stopword list (no nltk)
stop_words = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

app = Flask(__name__)

# Load models and tokenizer
try:
    text_model = load_model("text_emotion_model.h5")
    facial_model = load_model("facial_emotion_model.h5")
    with open("text_tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)


def preprocess_text(text):
    tokens = [
        word.lower()
        for word in text.split()
        if word.lower() not in stop_words
        or word.lower() in {"sadness", "joy", "love", "anger", "fear", "surprise"}
    ]
    return " ".join(tokens)


# Mappings
text_label_map = {0: 3, 1: 2, 2: 2, 3: 0, 4: 1, 5: 4}
facial_class_map = {0: 0, 1: None, 2: 1, 3: 2, 4: None, 5: 3, 6: 4}
common_emotions = ["angry", "fear", "happy", "sad", "surprise"]


def predict_combined_emotion(image_path, text):
    try:
        # Text prediction
        clean_text = preprocess_text(text)
        text_seq = tokenizer.texts_to_sequences([clean_text])
        text_pad = pad_sequences(text_seq, maxlen=200, padding="post")
        text_probs_raw = text_model.predict(text_pad, verbose=0)[0]
        text_probs = np.zeros(5)
        for i, prob in enumerate(text_probs_raw):
            mapped_idx = text_label_map.get(i)
            if mapped_idx is not None:
                text_probs[mapped_idx] += prob

        # Image prediction
        img = load_img(image_path, target_size=(48, 48), color_mode="grayscale")
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        facial_probs_raw = facial_model.predict(img_array, verbose=0)[0]
        facial_probs = np.zeros(5)
        for i, prob in enumerate(facial_probs_raw):
            mapped_idx = facial_class_map.get(i)
            if mapped_idx is not None:
                facial_probs[mapped_idx] = prob

        # Fuse predictions
        fused_probs = (text_probs + facial_probs) / 2
        predicted_idx = np.argmax(fused_probs)
        predicted_emotion = common_emotions[predicted_idx]
        confidence = fused_probs[predicted_idx] * 100

        return predicted_emotion, confidence
    except Exception as e:
        return None, str(e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "text" not in request.form:
        return jsonify({"error": "Please upload an image and enter text"}), 400

    image = request.files["image"]
    text = request.form["text"]

    if image.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # Save uploaded image temporarily
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Get prediction
    emotion, confidence = predict_combined_emotion(image_path, text)

    # Clean up
    os.remove(image_path)

    if emotion is None:
        return jsonify({"error": f"Prediction failed: {confidence}"}), 500

    return jsonify({"emotion": emotion, "confidence": f"{confidence:.2f}%"})


if __name__ == "__main__":
    app.run(debug=True)
