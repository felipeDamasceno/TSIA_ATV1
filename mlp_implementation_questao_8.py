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
import time
import psutil
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

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

# Different learning rates to test
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
results = []

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    
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
    sgd = SGD(learning_rate=lr, momentum=0.9)
    
    # Compile the model
    model.compile(optimizer=sgd,
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Measure initial memory
    initial_memory = get_memory_usage()
    
    # Train the model and measure time
    start_time = time.time()
    history = model.fit(X_train_scaled, y_train,
                       epochs=5,
                       batch_size=32,
                       validation_data=(X_val_scaled, y_val),
                       verbose=0)
    training_time = time.time() - start_time
    
    # Measure final memory
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory
    
    # Store results
    results.append({
        'learning_rate': lr,
        'training_time': training_time,
        'memory_used': memory_used,
        'train_accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1]
    })

# Create a DataFrame with results
results_df = pd.DataFrame(results)
print("\nResults:")
print(results_df.to_string(float_format=lambda x: '{:.4f}'.format(x)))

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy plot
ax1.plot(results_df['learning_rate'], results_df['train_accuracy'], 'b-', label='Training')
ax1.plot(results_df['learning_rate'], results_df['val_accuracy'], 'r--', label='Validation')
ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs Learning Rate')
ax1.set_xscale('log')  # Use log scale for learning rate
ax1.legend()
ax1.grid(True)

# Training time plot
ax2.plot(results_df['learning_rate'], results_df['training_time'], 'g-')
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Training Time (s)')
ax2.set_title('Training Time vs Learning Rate')
ax2.set_xscale('log')  # Use log scale for learning rate
ax2.grid(True)

# Memory usage plot
ax3.plot(results_df['learning_rate'], results_df['memory_used'], 'm-')
ax3.set_xlabel('Learning Rate')
ax3.set_ylabel('Memory Used (MB)')
ax3.set_title('Memory Usage vs Learning Rate')
ax3.set_xscale('log')  # Use log scale for learning rate
ax3.grid(True)

plt.tight_layout()
plt.show() 