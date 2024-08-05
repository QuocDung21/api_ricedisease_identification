from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

# Load the ONNX model
ort_session = ort.InferenceSession('../model.onnx')


# Function to load and preprocess a single image
def load_and_preprocess_image(img_bytes, img_size=(224, 224)):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = img_array.astype(np.float32)
    img_array = np.transpose(img_array, (0, 3, 1, 2))  # Change to (1, 3, 224, 224)
    return img_array


# Function to make prediction
def predict_image(ort_session, img_array, label_names):
    # Make prediction with ONNX model
    inputs = {ort_session.get_inputs()[0].name: img_array}
    predictions = ort_session.run(None, inputs)[0]
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_label = label_names[predicted_class_idx]
    confidence_score = np.max(predictions, axis=1)[0]
    return predicted_label, confidence_score


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        img_bytes = file.read()
        img_array = load_and_preprocess_image(img_bytes)

        # Assuming label_names contains the class names in the correct order
        label_names = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast',
                       'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

        predicted_label, confidence_score = predict_image(ort_session, img_array, label_names)
        return jsonify({
            'predicted_label': predicted_label,
            'confidence_score': float(confidence_score)
        })
if __name__ == '__main__':
    app.run(debug=True)
