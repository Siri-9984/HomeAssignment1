Home_Assignment1_700752884

Name: Anaka Siri Reddy
ID: 700752884

1.Tensor Manipulations & Reshaping

Step 1: Create a random tensor of shape (4, 6)
Creates a 2D tensor with 4 rows and 6 columns filled with random float values between 0 and 10.
Example: tf.random.uniform(shape=(4, 6), minval=0, maxval=10)

Step 2: Find its rank and shape
- tf.rank(tensor): returns the number of dimensions (should be 2)
- tf.shape(tensor): returns the shape as [4, 6]

Step 3: Reshape and transpose the tensor
- tf.reshape(tensor, (2, 3, 4)): reshapes it into shape (2, 3, 4)
- tf.transpose(reshaped_tensor, perm=[1, 0, 2]): reorders dimensions to (3, 2, 4)

Step 4: Broadcasting and addition
- A smaller tensor of shape (1, 4) is created using tf.constant
- TensorFlow automatically broadcasts it to (3, 2, 4)
- Performs element-wise addition with the transposed tensor

Step 5: Broadcasting Explanation
Broadcasting Explanation:
- The smaller tensor has shape (1, 4), which is treated as (1, 1, 4)
- TensorFlow automatically broadcasts it to (3, 2, 4) to match the larger tensor
- Then element-wise addition is performed across all matching dimensions

Expected Output:
- Printed rank and shape before and after reshaping/transposing
- Final tensor values after broadcasting and addition (random values)

2.Loss Functions & Hyperparameter Tuning 

Step 1: Define true values (y_true) and predictions (y_pred)
- Use one-hot encoded labels for y_true: shape (3, 3)
- Define two prediction tensors: one accurate (y_pred_1) and one slightly off (y_pred_2)

Step 2: Compute Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE)
- Use tf.keras.losses.MeanSquaredError() to calculate MSE
- Use tf.keras.losses.CategoricalCrossentropy() to calculate CCE
- Compare how each loss function evaluates the predictions

Step 3: Modify predictions and compare loss values
- Modify y_pred slightly (make it less confident)
- Recalculate MSE and CCE
- Print and compare loss values for both prediction sets

Step 4: Plot loss values using Matplotlib
  -Create a bar chart to compare:
  - MSE for y_pred_1 and y_pred_2
  - CCE for y_pred_1 and y_pred_2
- Label axes and add a title

Expected Output:
- Printed loss values:
    MSE (prediction 1), MSE (prediction 2)
    CCE (prediction 1), CCE (prediction 2)
- A bar chart comparing all four values

3. Train a Neural Network and Log to TensorBoard 

Step 1: Load and preprocess the MNIST dataset
- Use tf.keras.datasets.mnist.load_data() to load handwritten digit data
- Normalize pixel values to the range [0, 1] by dividing by 255.0

Step 2: Define and compile a simple neural network
- Use tf.keras.models.Sequential() to build a model
  Layers:
    - Flatten layer to convert 28x28 images to a 1D vector
    - Dense layer with 128 units and ReLU activation
    - Dropout layer to prevent overfitting
    - Output Dense layer with 10 units and softmax activation
- Compile the model with:
    - Optimizer: Adam
    - Loss function: sparse_categorical_crossentropy
    - Metric: accuracy

Step 3: Set up TensorBoard callback
- Create a log directory using datetime formatting: logs/fit/YYYYMMDD-HHMMSS
- Use tf.keras.callbacks.TensorBoard(log_dir=...) to set up logging

Step 4: Train the model for 5 epochs with TensorBoard enabled
- Use model.fit() with:
    - training data (x_train, y_train)
    - validation data (x_test, y_test)
    - epochs=5
    - callbacks=[tensorboard_callback]

Step 5: Launch TensorBoard and analyze metrics
- In a Jupyter Notebook, use:
    %tensorboard --logdir logs/fit
- Or from terminal:
    tensorboard --logdir=logs/fit

Expected Output:
- Model trains for 5 epochs
- TensorBoard logs are saved in logs/fit/
- Training and validation accuracy/loss graphs visible in TensorBoard


      

 







