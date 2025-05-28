from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
model = tf.keras.models.load_model('./model/model1.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    img = load_img(file_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('home.html', background_image_url=url_for('static', filename='images/bg.jpg'))

@app.route('/about')
def about():
    return render_template('about.html', background_image_url=url_for('static', filename='images/bg.jpg'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'fingerprint' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['fingerprint']
        if file.filename == '':
            return jsonify({'error': 'No file provided'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg, bmp'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        try:
            img = preprocess_image(file_path)
            predictions = model.predict(img)
            predicted_class = int(np.argmax(predictions[0]))

            class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
            predicted_label = class_names[predicted_class]

            result = {
                'name': request.form.get('name'),
                'mobile': request.form.get('mobile'),
                'gender': request.form.get('gender'),
                'age': request.form.get('age'),
                'bloodgroup': predicted_label,
                'confidence': float(np.max(predictions[0]))
            }

            # Returning JSON response instead of rendering template
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return render_template('predict.html', background_image_url=url_for('static', filename='images/bg.jpg'))

# ---------------------------- Blood Group Image Serving ----------------------------
@app.route('/images/<group>')
def get_images(group):
    dataset_path = os.path.join(app.static_folder, 'bloodgroup_dataset', 'dataset_blood_group')
    group_path = os.path.join(dataset_path, group)

    if not os.path.exists(group_path):
        return jsonify([])

    image_urls = [
        f"/static/bloodgroup_dataset/dataset_blood_group/{group}/{fname}"
        for fname in os.listdir(group_path)
        if fname.lower().endswith(".bmp")
    ][:20]  # Limit to first 20 images to avoid overloading browser

    return jsonify(image_urls)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
