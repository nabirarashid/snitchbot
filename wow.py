import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

model_path = "/Users/user/Downloads/converted_keras/keras_model.h5"
label_path = "/Users/user/Downloads/converted_keras/labels.txt"

# Load the model with the custom object
def load_keras_model(path):
    return load_model(path, compile=False)

# Load the labels
def load_class_names(path2):
    with open(path2, "r") as file:
        return file.readlines()

model = load_keras_model(model_path)
class_names = load_class_names(label_path)

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image_path = "/Users/user/Downloads/test image.avif"
image = Image.open(image_path).convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image2 = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image2)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name, end="")
print("Confidence Score:", confidence_score)

