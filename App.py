from flask import Flask, render_template, request
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import uuid

app = Flask(__name__)

# Load MobileNetV2 model pretrained on ImageNet
model = MobileNetV2(weights='imagenet')

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="No file selected")

        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type. Please upload an image (png, jpg, jpeg, gif).")

        try:
            # Open and resize the image to 224x224 for MobileNetV2
            img = Image.open(file).convert("RGB").resize((224, 224))
        except UnidentifiedImageError:
            return render_template("index.html", error="Invalid image file")

        # Preprocess the image for MobileNetV2
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict using MobileNetV2
        prediction_scores = model.predict(img_array)
        decoded = decode_predictions(prediction_scores, top=1)[0][0]
        prediction = decoded[1]  # class name
        confidence = f"{decoded[2] * 100:.2f}%"

        # Save uploaded image for display
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        img.save(filepath)
        image_url = filepath

    return render_template("index.html", prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
