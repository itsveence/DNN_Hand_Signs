# DNN_Hand_Signs

# Multilayer Neural Network with TensorFlow

This repository contains a Python implementation of a 3-layer neural network using TensorFlow. 

This Neural Network was trained on a dataset to classify samples into classes. The architecture of the network includes three layers with ReLU activation functions for the first two layers and a softmax function for the output layer. The weights and biases of the network are trained using gradient descent and mini-batch processing.

## Dependencies

The code in this repository requires the following Python libraries:

- TensorFlow
- NumPy
- Matplotlib

## Model

The model architecture is a fully connected 3-layer neural network with the following structure:

1. Input Layer: Takes in the flattened data samples.
2. First Hidden Layer: 25 neurons with ReLU activation.
3. Second Hidden Layer: 12 neurons with ReLU activation.
4. Output Layer: 6 neurons (representing 6 classes) with Softmax activation.

The number of neurons in the hidden layers and the output layer can be adjusted based on the number of features in the data and the number of output classes, respectively.

## Training

The model is trained using mini-batch gradient descent. The batch size and number of epochs are hyperparameters that can be tuned. During training, the model calculates the loss and accuracy for both the training and testing sets. These values are printed out at regular intervals to allow for tracking of the model's progress.

The model training process involves the following steps:
1. Forward propagation: The model makes a prediction based on the current weights and biases.
2. Loss computation: The model computes the loss by comparing the predicted output to the actual output.
3. Backpropagation: The model computes the gradient of the loss with respect to each of the weights and biases.
4. Parameter update: The model updates the weights and biases using the computed gradients.

## Evaluation

The performance of the model is evaluated in terms of accuracy. The accuracy of the model on the training and testing sets is computed after each epoch and stored for later analysis. This accuracy information can be used to track the learning progress of the model and identify issues such as overfitting or underfitting.

## Visualization

The cost and accuracy values are plotted against the number of epochs to visualize the model's learning progress. The decreasing cost and increasing accuracy over time indicate that the model is learning to classify the samples correctly.

## Usage

1. Clone the repository to your local machine.
2. Ensure that you have the necessary Python libraries installed.
3. Run the Python script to train the model and visualize the results.

```python
parameters, costs, train_acc, test_acc = model(new_train, new_y_train, new_test, new_y_test, nn_shape, num_epochs=300, minibatch_size=64)
```

4. Plot the cost and accuracy graphs.

```python
# Plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))
plt.show()

# Plot the train accuracy
plt.plot(np.squeeze(train_acc))
plt.ylabel('Train Accuracy')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))

# Plot the test accuracy
plt.plot(np.squeeze(test_acc))
plt.ylabel('Test Accuracy')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))
plt.show()
```

This will display two plots: one for the cost and another for the accuracy during the training process.

## Note

The code in this repository is meant to serve as a simple example of how to implement and train a multilayer neural network using TensorFlow. The structure of the network and the training process can be adjusted based on your specific requirements.
