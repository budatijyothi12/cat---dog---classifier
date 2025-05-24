import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

def predict_cat_or_dog(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict using the model
    predictions = model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=3)[0]

    # Check if 'cat' or 'dog' is in predictions
    for _, label, confidence in decoded_preds:
        if 'cat' in label.lower():
            return "Cat"
        elif 'dog' in label.lower():
            return "Dog"
    return "Not sure – try another image"

# Example usage
if __name__ == "__main__":
    img_path = input("Enter the path of a cat or dog image: ")
    if os.path.exists(img_path):
        result = predict_cat_or_dog(img_path)
        print("Prediction:", result)
    else:
        print("Invalid image path.")
