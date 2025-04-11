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
import gc

class MemoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MemoryCallback, self).__init__()
        self.memory_usage = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Force garbage collection
        gc.collect()
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == '__main__':
    # Load and prepare data
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Different batch sizes to test
    batch_sizes = [8, 16, 32, 64, 128]
    results = []

    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")
        
        # Clear memory and garbage collection
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Create memory callback
        memory_callback = MemoryCallback()
        
        # Create the model
        model = Sequential()
        model.add(Dense(8, input_shape=(X_train_scaled.shape[1],), kernel_initializer='random_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(4, kernel_initializer='random_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(1, kernel_initializer='random_uniform'))
        model.add(Activation('sigmoid'))
        
        sgd = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train and measure time
        start_time = time.time()
        history = model.fit(X_train_scaled, y_train,
                          epochs=5,
                          batch_size=batch_size,
                          validation_data=(X_val_scaled, y_val),
                          verbose=0,
                          callbacks=[memory_callback])
        training_time = time.time() - start_time
        
        # Calculate average memory usage during training
        avg_memory = np.mean(memory_callback.memory_usage)
        
        results.append({
            'batch_size': batch_size,
            'training_time': training_time,
            'memory_used': avg_memory,
            'train_accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        })
        
        # Force cleanup
        del model
        gc.collect()
        tf.keras.backend.clear_session()

    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df.to_string(float_format=lambda x: '{:.4f}'.format(x)))

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(results_df['batch_size'], results_df['train_accuracy'], 'b-', label='Training')
    ax1.plot(results_df['batch_size'], results_df['val_accuracy'], 'r--', label='Validation')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Batch Size')
    ax1.legend()
    ax1.grid(True)

    # Training time plot
    ax2.plot(results_df['batch_size'], results_df['training_time'], 'g-')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time vs Batch Size')
    ax2.grid(True)

    # Memory usage plot
    ax3.plot(results_df['batch_size'], results_df['memory_used'], 'm-')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Memory Used (MB)')
    ax3.set_title('Memory Usage vs Batch Size')
    ax3.grid(True)

    plt.tight_layout()
    plt.show() 