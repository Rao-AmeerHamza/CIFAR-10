from flask import Flask, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

app = Flask(__name__)
model = MobileNetV2(weights='imagenet')

@app.route("/", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img = Image.open(file).convert("RGB").resize((224, 224))
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file"}), 400

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction_scores = model.predict(img_array)
    decoded = decode_predictions(prediction_scores, top=1)[0][0]
    prediction = decoded[1]
    confidence = f"{decoded[2] * 100:.2f}%"

    return jsonify({"prediction": prediction, "confidence": confidence})

# Required for Vercel
def handler(request):
    return app(request.environ, start_response=request.start_response)
