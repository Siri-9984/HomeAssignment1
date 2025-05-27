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

4. Questions to Answer:

1. What patterns do you observe in the training and validation accuracy curves?
Ans: In the training and validation accuracy curves, the following patterns are commonly observed:

      - Training Accuracy:
          - Increases consistently over epochs.
          - Indicates the model is learning the training data well.

      - Validation Accuracy:
          - Initially rises along with training accuracy.
          - May eventually plateau or decrease, depending on how well the model generalizes.

     Key points:
        - If both training and validation accuracy increase and stay close, it shows the model is generalizing effectively.
        - If training accuracy keeps improving but validation accuracy stagnates or declines, it suggests overfitting.
        - This pattern highlights the importance of using validation data to evaluate real-world performance.

2. How can you use TensorBoard to detect overfitting?
Ans: TensorBoard provides visual insights into model performance during training and validation.

How to detect overfitting:
- Monitor the training and validation loss curves:
  - If "training loss decreases" while "validation loss increases", it's a sign of overfitting.
- Look at the accuracy curves:
  - If "training accuracy keeps increasing" but "validation accuracy stops improving or drops", overfitting is likely occurring.
- Overfitting means the model is memorizing training data and failing to generalize to unseen data.
- TensorBoard allows early detection so you can apply solutions like:
  - Dropout layers
  - Early stopping
  - Data augmentation
  - Reduced model complexity
TensorBoard is an essential tool to visually catch overfitting by comparing training vs. validation performance in real time.

3. What happens when you increase the number of epochs?
Ans: 
Training a model for more epochs allows it to learn from the data over a longer period of time.
Effects of increasing epochs:
- Initially:
  - Both training and validation accuracy may improve.
  - The model gets better at identifying patterns in the training data.

- Eventually:
  - Training accuracy continues to increase.
  - Validation accuracy may stop improving or even decrease.
This behavior indicates overfitting:
- The model starts memorizing training data instead of learning general features.
- Validation loss increases while training loss keeps decreasing.
Finally, While more epochs can improve learning, too many can lead to overfitting and reduced generalization.








