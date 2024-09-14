# PyTorch Softmax Classification

This project demonstrates how to implement a softmax classification model using PyTorch on the Iris dataset. The model is built to classify Iris flowers into one of three species based on four features.

## Overview

The project includes:

- Loading and preprocessing the Iris dataset.
- Building a neural network model with PyTorch.
- Training the model and evaluating its performance.

## Requirements

Ensure you have the following libraries installed:

- `torch`
- `scikit-learn`
- `numpy`
- `matplotlib`

You can install the required libraries using pip:

```bash
pip install torch scikit-learn numpy matplotlib
```

## Dataset

The Iris dataset is used, which contains measurements for iris flowers. The dataset features four attributes per flower and includes three classes.

## Steps

### 1. Data Preparation

1. **Load and Preprocess Data**
   - Load the Iris dataset.
   - Standardize the feature values.
   - One-hot encode the labels.

2. **Convert to PyTorch Tensors**
   - Convert the features and labels to PyTorch tensors and move them to the GPU.

3. **Train-Test Split**
   - Split the data into training and testing sets.

### 2. Model Definition

Define a neural network model using PyTorch:

- **Model Architecture**
  - `Linear1`: Fully connected layer with input dimension to hidden layer (10 neurons).
  - `Linear2`: Fully connected layer from hidden layer to output dimension (3 neurons for the classes).
  - Activation functions: Sigmoid for hidden layer and Softmax for the output layer.

### 3. Training

1. **Initialize the Model**
   - Create an instance of the `SoftmaxClassification` model.

2. **Define Loss and Optimizer**
   - Use `CrossEntropyLoss` for classification.
   - Use `AdamW` optimizer for training.

3. **Training Loop**
   - Iterate over epochs, compute the loss, and update model parameters.

4. **Plot Training Loss**
   - Plot the loss over epochs using Matplotlib.

### 4. Code

Here is a snippet of the code used in this project:

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

class SoftmaxClassification(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxClassification, self).__init__()
        self.linear1 = nn.Linear(input_dim, 10)
        self.linear2 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.Sigmoid()(x)
        x = self.linear2(x)
        x = nn.Softmax(dim=1)(x)
        return x

# Load and preprocess data
x, y = load_iris(return_X_y=True)
y = y.reshape(-1, 1)
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = OneHotEncoder(sparse=False).fit_transform(y)

x = torch.tensor(x, dtype=torch.float32).to('cuda')
y = torch.tensor(y, dtype=torch.float32).to('cuda')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SoftmaxClassification(4, 3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# Training loop
losses = []
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plotting loss
plt.plot(range(epochs), losses, 'g')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()
```

## Results

- **Model Training**: The model is trained for 100 epochs.
- **Loss Visualization**: A plot of training loss over epochs is generated to visualize the learning process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The Iris dataset is provided by the UCI Machine Learning Repository.
- PyTorch for the deep learning framework.
- Scikit-learn for data preprocessing utilities.
- Matplotlib for plotting.

```

This `README.md` file provides a structured overview of your project, making it easier for others to understand and use your code.
