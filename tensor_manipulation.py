# tensor_manipulation.py

import tensorflow as tf

# 1. Create a random tensor of shape (4, 6)
tensor = tf.random.uniform(shape=(4, 6), minval=0, maxval=10, dtype=tf.float32)
print("Original Tensor:\n", tensor)

# 2. Find its rank and shape
rank = tf.rank(tensor)             # Rank: number of dimensions
shape = tf.shape(tensor)           # Shape: size of each dimension

print("\nRank of the original tensor:", rank.numpy())
print("Shape of the original tensor:", shape.numpy())

# 3. Reshape it to (2, 3, 4)
reshaped_tensor = tf.reshape(tensor, shape=(2, 3, 4))
print("\nReshaped Tensor to (2, 3, 4):\n", reshaped_tensor)

# Transpose it to (3, 2, 4)
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print("\nTransposed Tensor to (3, 2, 4):\n", transposed_tensor)

# 4. Broadcast a smaller tensor (1, 4) to match shape (3, 2, 4)
# First, create smaller tensor of shape (1, 4)
small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])  # Shape (1, 4)

# Broadcasting will automatically expand dimensions to match (3, 2, 4)
broadcasted_sum = transposed_tensor + small_tensor
print("\nResult after broadcasting and adding:\n", broadcasted_sum)

# 5. Explanation of broadcasting:
print("""
Explanation:
Broadcasting automatically expands the smaller tensor (1, 4) to match the larger tensor (3, 2, 4).
This works because:
- TensorFlow first expands (1, 4) to (1, 1, 4)
- Then it replicates across the first two dimensions to become (3, 2, 4)
- The shapes become compatible and element-wise addition is performed.
""")
