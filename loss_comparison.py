# loss_comparison.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define true labels and predictions

# True labels for 3 samples, 3 classes (One-hot encoded)
y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)

# Model's first prediction (somewhat correct)
y_pred_1 = tf.constant([[0.8, 0.1, 0.1],
                        [0.2, 0.7, 0.1],
                        [0.1, 0.2, 0.7]], dtype=tf.float32)

# Modified prediction (less confident)
y_pred_2 = tf.constant([[0.6, 0.2, 0.2],
                        [0.3, 0.4, 0.3],
                        [0.2, 0.3, 0.5]], dtype=tf.float32)

# Step 2: Compute Losses

# Mean Squared Error (MSE)
mse_1 = tf.keras.losses.MeanSquaredError()(y_true, y_pred_1).numpy()
mse_2 = tf.keras.losses.MeanSquaredError()(y_true, y_pred_2).numpy()

# Categorical Cross-Entropy (CCE)
cce_1 = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred_1).numpy()
cce_2 = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred_2).numpy()

# Step 3: Print the loss values
print("Loss Values:")
print("MSE (Prediction 1):", mse_1)
print("MSE (Prediction 2):", mse_2)
print("CCE (Prediction 1):", cce_1)
print("CCE (Prediction 2):", cce_2)

# Step 4: Plot bar chart comparison

loss_names = ["MSE_1", "MSE_2", "CCE_1", "CCE_2"]
loss_values = [mse_1, mse_2, cce_1, cce_2]

plt.figure(figsize=(8, 5))
plt.bar(loss_names, loss_values, color='skyblue')
plt.title("Comparison of Loss Function Values")
plt.ylabel("Loss Value")
plt.xlabel("Loss Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# ✅ Save plot as image file
plt.savefig("loss_comparison_plot.png")
print("✅ Bar chart saved as 'loss_comparison_plot.png'")

