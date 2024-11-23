import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pandas as pd

app = Flask(__name__)

# Folder for temporary uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
MODEL_PATH = "model/traffic_sign_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Load class labels from meta.csv
meta_csv_path = "dataset/meta/Meta.csv"
if not os.path.exists(meta_csv_path):
    raise FileNotFoundError(f"Metadata file not found at {meta_csv_path}")

meta_data = pd.read_csv(meta_csv_path)

# Verify the column name for class labels (e.g., 'ClassId')
if 'ClassId' not in meta_data.columns:
    raise KeyError("Column 'ClassId' not found in metadata file. Check the file structure.")

# Assuming 'ClassId' is your class label column (adjust if needed)
classes = meta_data['ClassId'].tolist()

# Example mapping of ClassId to Sign Name
class_id_to_name = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "Speed Limit 60",
    4: "Speed Limit 70",
    5: "Speed Limit 80",
    6: "End of Speed Limit 80",
    7: "Speed Limit 100",
    8: "Speed Limit 120",
    9: "No Overtaking",
    10: "No Overtaking Trucks",
    11: "Right of Way",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Vehicles",
    16: "No Trucks",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve Left",
    20: "Dangerous Curve Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows",
    25: "Construction",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Snow or Ice",
    31: "Wild Animals Crossing",
    32: "End of Restrictions",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout Mandatory",
    41: "End of No Overtaking",
    42: "End of No Overtaking Trucks",
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if an image file is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image
    try:
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError("Uploaded file is not a valid image.")
        image = cv2.resize(image, (32, 32))
        image = np.expand_dims(image / 255.0, axis=0)

        # Predict the class
        predictions = model.predict(image)
        class_idx = np.argmax(predictions)
        
        # Get the sign name using the mapping
        class_name = class_id_to_name.get(class_idx, "Unknown Sign")

        # Clean up the uploaded file after prediction
        os.remove(filepath)

        return jsonify({"prediction": class_name})

    except Exception as e:
        os.remove(filepath)  # Clean up in case of an error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
