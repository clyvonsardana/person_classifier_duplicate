# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables
MODEL = None
CLASS_NAMES = []

@app.route('/')
def home():
    # Get existing classes and photo counts
    classes = []
    photo_counts = {}
    
    if os.path.exists(UPLOAD_FOLDER):
        classes = [f for f in os.listdir(UPLOAD_FOLDER) 
                  if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
        
        for class_name in classes:
            class_path = os.path.join(UPLOAD_FOLDER, class_name)
            photo_counts[class_name] = len([f for f in os.listdir(class_path) 
                                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    return render_template('index.html', 
                         classes=classes, 
                         photo_counts=photo_counts,
                         model_trained=(MODEL is not None))

@app.route('/add_class', methods=['POST'])
def add_class():
    class_name = request.form.get('class_name')
    if class_name:
        # Create folder for the class
        class_path = os.path.join(UPLOAD_FOLDER, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            global CLASS_NAMES
            CLASS_NAMES.append(class_name)
    return redirect(url_for('home'))

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    class_name = request.form['class_name']
    class_folder = os.path.join(UPLOAD_FOLDER, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    photos = request.files.getlist('photos')
    for photo in photos:
        if photo and photo.filename:
            photo.save(os.path.join(class_folder, photo.filename))

    return redirect(url_for('home'))

@app.route('/train', methods=['POST'])
def train_model():
    global MODEL, CLASS_NAMES
    
    # Reset CLASS_NAMES to match current folders
    CLASS_NAMES = [f for f in os.listdir(UPLOAD_FOLDER) 
                   if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
    
    # Prepare data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    # Flow training images in batches
    train_generator = train_datagen.flow_from_directory(
        UPLOAD_FOLDER,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        UPLOAD_FOLDER,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Use transfer learning with MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    MODEL = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    # Compile model
    MODEL.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = MODEL.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )
    
    return jsonify({
        'status': 'success', 
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1]
    })

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'photo' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Convert FileStorage to BytesIO
    img = image.load_img(BytesIO(photo.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = MODEL.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]) * 100)  # Convert to standard Python float

    return jsonify({'class': predicted_class, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)