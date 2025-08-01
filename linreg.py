from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import clear_output
except ImportError:
    def clear_output():
        pass  # Fallback function if IPython is not available
from six.moves import urllib

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())
print(dftrain.shape)
print(dfeval.shape)

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Modern TensorFlow 2.x approach using Keras preprocessing
def preprocess_data(train_df, eval_df):
    # Handle missing values
    train_df = train_df.copy()
    eval_df = eval_df.copy()
    
    train_df['age'].fillna(train_df['age'].median(), inplace=True)
    train_df['fare'].fillna(train_df['fare'].median(), inplace=True)
    eval_df['age'].fillna(train_df['age'].median(), inplace=True)  # Use training median for consistency
    eval_df['fare'].fillna(train_df['fare'].median(), inplace=True)
    
    # Combine both datasets to ensure consistent one-hot encoding
    combined_categorical = pd.concat([train_df[CATEGORICAL_COLUMNS], eval_df[CATEGORICAL_COLUMNS]])
    categorical_encoded = pd.get_dummies(combined_categorical)
    
    # Split back into training and evaluation sets
    train_categorical = categorical_encoded.iloc[:len(train_df)]
    eval_categorical = categorical_encoded.iloc[len(train_df):]
    
    # Add numeric columns
    train_numeric = train_df[NUMERIC_COLUMNS]
    eval_numeric = eval_df[NUMERIC_COLUMNS]
    
    # Combine features
    X_train = pd.concat([train_categorical, train_numeric], axis=1)
    X_eval = pd.concat([eval_categorical, eval_numeric], axis=1)
    
    return X_train, X_eval

# Preprocess the data
X_train, X_eval = preprocess_data(dftrain, dfeval)

print("Processed training data shape:", X_train.shape)
print("Features:", list(X_train.columns))

# Create a simple linear model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel summary:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_eval, y_eval),
    verbose=1
)

# Evaluate the model
print("\nEvaluating model...")
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
eval_loss, eval_accuracy = model.evaluate(X_eval, y_eval, verbose=0)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Evaluation Accuracy: {eval_accuracy:.4f}")

# Make predictions
predictions = model.predict(X_eval)
predicted_classes = (predictions > 0.5).astype(int).flatten()

print(f"\nFirst 10 predictions: {predicted_classes[:10]}")
print(f"First 10 actual values: {y_eval.iloc[:10].values}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

print("Linear regression model training completed!")
