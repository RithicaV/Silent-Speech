import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. Load the prepared data
X = np.load("X.npy")
y = np.load("y.npy")
labels = np.load("labels.npy")

# Reshape X to (samples, length, 1)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 2. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- DATA AUGMENTATION ---
# This creates new variations of your data so the model learns better
def augment_data(X_data, y_data):
    X_aug, y_aug = [], []
    for i in range(len(X_data)):
        # Original
        X_aug.append(X_data[i])
        y_aug.append(y_data[i])
        
        # Add slight noise
        noise = np.random.normal(0, 0.02, X_data[i].shape)
        X_aug.append(X_data[i] + noise)
        y_aug.append(y_data[i])
        
        # Slight time shift
        shifted = np.roll(X_data[i], np.random.randint(-5, 5))
        X_aug.append(shifted)
        y_aug.append(y_data[i])
        
    return np.array(X_aug), np.array(y_aug)

X_train_aug, y_train_aug = augment_data(X_train, y_train)
# -------------------------

# 3. Define a Slimmer Model (To prevent overfitting)
model = models.Sequential([
    layers.Input(shape=(400, 1)), 
    layers.Conv1D(32, kernel_size=7, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3), # Forcing the model to not rely on specific points
    
    layers.Conv1D(64, kernel_size=5, activation='relu'),
    layers.GlobalAveragePooling1D(), # Simplifies the features
    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Train
print(f"Original samples: {len(X_train)}")
print(f"Augmented samples: {len(X_train_aug)}")
print("Training started...")

# Increased epochs but added EarlyStopping to stop when it stops improving
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train_aug, y_train_aug, 
          epochs=100, 
          validation_data=(X_test, y_test), 
          batch_size=16,
          callbacks=[early_stop])

# 5. Save the model
model.save("silent_speech_model.h5")
print("Model saved successfully!")
