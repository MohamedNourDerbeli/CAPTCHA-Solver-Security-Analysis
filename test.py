import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Function to preprocess image (adjusted to match the model's expected input)
def preprocess_image(image_path, target_size=(128, 50)):  # Correct target size (200, 50)
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize(target_size)  # Resize to match the model input (200, 50)
    img = np.array(img)
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (200, 50, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 200, 50, 1)
    return img

# Convert integer predictions back to text
def int_to_text(prediction, int_to_char):
    return ''.join([int_to_char[int(idx)] for idx in prediction])

# Load the trained model
model = load_model('captcha_solver_model.h5')

# Create the reverse character-to-integer mapping for decoding
chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'  # Character set
int_to_char = {i: char for i, char in enumerate(chars)}

# Load a test image
test_image_path = 'CTD/2b827.png'  # Replace with your test image path
test_image = preprocess_image(test_image_path)

# Make a prediction
prediction = model.predict(test_image)

# Convert the predicted output to a readable text (inverse of text_to_int)
predicted_text = int_to_text(np.argmax(prediction, axis=-1), int_to_char)
print(f"Predicted text: {predicted_text}")
