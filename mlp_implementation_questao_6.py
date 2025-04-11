import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the breast cancer dataset from scikit-learn
# Site was not working on the time that I was coding it, so I got the dataset from sklearn
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and validation sets (70-30)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Create the model
model = Sequential()

# Add layers
# First layer with input shape
model.add(Dense(8, input_shape=(X_train.shape[1],), kernel_initializer='random_uniform'))
model.add(Activation('relu'))

# Hidden layer
model.add(Dense(4, kernel_initializer='random_uniform'))
model.add(Activation('relu'))

# Output layer
model.add(Dense(1, kernel_initializer='random_uniform'))
model.add(Activation('sigmoid'))

# Create optimizer with specific learning rate
sgd = SGD(learning_rate=0.01, momentum=0.9)

# Compile the model
model.compile(optimizer=sgd,
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Print model summary
print("\nModel Summary:")
model.summary()

# Train the model with 5 epochs
print("\nTraining model...")
history = model.fit(X_train_scaled, y_train,
                   epochs=5,
                   batch_size=32,
                   validation_data=(X_val_scaled, y_val),
                   verbose=1)

# Plot the cost curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Cost Curves (5 Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Print final metrics
print("\nFinal Metrics:")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}") 