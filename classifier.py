import tensorflow as tf
from keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.reshape(-1,28,28,1) / 255.0
x_test = x_test.reshape(-1,28,28,1) / 255.0

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Data loaded successfully!")
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model built successfully!")
model.summary()

# Step 4: Train the model
import os
from keras.models import load_model

# Check if model is already saved
if os.path.exists("mnist_model.h5"):
    model = load_model("mnist_model.h5")
    print("Model loaded from mnist_model.h5")
else:
    # Train the model only if not saved
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2
    )
    model.save("mnist_model.h5")
    print("Model trained and saved")

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")

import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

import numpy as np

# Show 5 test images with their predictions
for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    pred = np.argmax(model.predict(x_test[i:i+1]))
    plt.title(f"Predicted: {pred}")
    plt.show()

