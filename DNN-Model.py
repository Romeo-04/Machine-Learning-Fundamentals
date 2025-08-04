from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd
import numpy as np

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

train_y = train.pop('Species')
test_y = test.pop('Species')

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)
print("Species classes:", SPECIES)
print("First 5 training samples:")
print(train.head())

# Modern TensorFlow 2.x approach - no need for feature columns or input functions
# Prepare the data directly
X_train = train.values  # Convert to numpy array
X_test = test.values
y_train = train_y.values
y_test = test_y.values

print(f"\nFeature names: {train.columns.tolist()}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation='relu', input_shape=(4,)),  # First hidden layer: 30 nodes
    tf.keras.layers.Dense(10, activation='relu'),                    # Second hidden layer: 10 nodes
    tf.keras.layers.Dense(3, activation='softmax')                   # Output layer: 3 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # For integer labels (0, 1, 2)
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=100,  # Equivalent to steps but more intuitive
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model
print("\nEvaluating model...")
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

print(f"First 10 predictions: {predicted_classes[:10]}")
print(f"First 10 actual values: {y_test[:10]}")
print(f"Predicted species: {[SPECIES[i] for i in predicted_classes[:10]]}")
print(f"Actual species: {[SPECIES[i] for i in y_test[:10]]}")

# Display prediction probabilities for first 5 samples
print("\nPrediction probabilities for first 5 test samples:")
for i in range(5):
    print(f"Sample {i+1}:")
    for j, species in enumerate(SPECIES):
        print(f"  {species}: {predictions[i][j]:.4f}")
    print(f"  Predicted: {SPECIES[predicted_classes[i]]}, Actual: {SPECIES[y_test[i]]}")
    print()

print("DNN model training and evaluation completed!")