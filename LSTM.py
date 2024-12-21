import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Example: Generate some random data for demonstration
# Replace this with your actual dataset
# X_train shape: (num_samples, timesteps, features)
# Y_train shape: (num_samples,) where the labels are integers for classification

num_samples = 912  # Adjust to match your data
timesteps = 50  # Adjust based on your input data
features = 128  # Adjust based on your input data
num_classes = 13  # Adjust based on your classification task

# Generating dummy data as an example
X_train = np.random.randn(num_samples, timesteps, features)
Y_train = np.random.randint(0, num_classes, num_samples)

# One-hot encode the labels
Y_train = to_categorical(Y_train, num_classes=num_classes)

# Check if the number of samples in X_train and Y_train match
assert X_train.shape[0] == Y_train.shape[0], "Number of samples in X_train and Y_train must be the same"

# Define the model
model = Sequential()

# Add an LSTM layer with 128 units and the input shape based on X_train
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))

# Add a Dense layer for classification with 13 output units (since we have 13 classes)
model.add(Dense(num_classes, activation='softmax'))  # For multi-class classification

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split data into training and validation sets (example: 80% training, 20% validation)
# Replace this with your actual validation data
X_val = X_train[-int(num_samples*0.2):]  # Last 20% for validation
Y_val = Y_train[-int(num_samples*0.2):]
X_train = X_train[:-int(num_samples*0.2)]  # First 80% for training
Y_train = Y_train[:-int(num_samples*0.2)]

# Train the model
epochs = 10  # Number of epochs for training
batch_size = 32  # Batch size for training

# Train the model with validation data
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))

# Optionally, you can save the model after training
model.save("captcha_solver_model.h5")
