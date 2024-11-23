import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Paths
train_data_path = "dataset/train"  # Path to training data
meta_csv_path = "dataset/meta/meta.csv"  # Path to metadata
model_save_path = "model/traffic_sign_model.h5"  # Path to save the model

# Load dataset
def load_data(data_path):
    images, labels = [], []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (32, 32))  # Resize to 32x32
                images.append(img)
                labels.append(int(label))
    return np.array(images), np.array(labels)

print("Loading data...")
X, y = load_data(train_data_path)
X = X / 255.0  # Normalize pixel values
y = to_categorical(y)  # One-hot encode labels
print(f"Data loaded. Total samples: {X.shape[0]}")

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape())

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
if not os.path.exists("model"):
    os.makedirs("model")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
