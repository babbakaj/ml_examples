# example following Fig. 6.10 in Bishop's book Deep learning
# neural net with 1 hidden layer of three units is considered 'universal approximator'
# different functions are considered: sin, abs, quadratic
# contribution of each neuron to the output is illustrated

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

    
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.fc1  = nn.Linear(n_input, n_hidden, bias=True)
        self.fc2  = nn.Linear(n_hidden, n_output, bias=True)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def gen_data(num, func):
    # Generate random numbers between -1 and 1
    x = (2 * torch.rand(num) - 1).reshape(-1,1) 
    t = func(x)

    return x, t

def sine_func(x):
    return torch.sin(x*torch.pi)

def quadratic(x):
    return x**2

def abs_func(x):
    return torch.abs(x)


# Initialize the model and optimizer
n_input  = 1
n_hidden = 3
n_output = 1

model     = MLP(n_input,n_hidden,n_output)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)  # Adam optimizer

# Training loop
num_epochs = 2000
batch_size = 64

# function to approximate
fnc = sine_func

for epoch in range(num_epochs):
    # Minibatch construct
    xb, yb = gen_data(batch_size, fnc)
    # Forward pass
    outputs = model(xb)
    loss    = F.mse_loss(outputs, yb)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ((epoch + 1) % 200 == 0) or (epoch == 0):
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
x_test = torch.linspace(-1, 1, 101).reshape(-1, 1)
y_test = fnc(x_test).reshape(-1,1)
y_pred = model(x_test).detach().numpy()

# Extract weights and biases
fc1_weights = model.fc1.weight.data.numpy()  # Weights of the hidden layer (3x1 matrix)
fc1_biases  = model.fc1.bias.data.numpy()    # Biases of the hidden layer (3x1 vector)
fc2_weights = model.fc2.weight.data.numpy()  # Weights of the output layer (1x3 matrix)
fc2_bias    = model.fc2.bias.data.numpy()      # Bias of the output layer (scalar)

# print("Hidden Layer Weights (fc1):\n", fc1_weights)
# print("Hidden Layer Biases (fc1):\n", fc1_biases)
# print("Output Layer Weights (fc2):\n", fc2_weights)
# print("Hidden Layer Biases (fc2):\n", fc2_bias)

h1 = torch.tanh(x_test * fc1_weights[0, 0] + fc1_biases[0]) * fc2_weights[0,0]
h2 = torch.tanh(x_test * fc1_weights[1, 0] + fc1_biases[1]) * fc2_weights[0,1]
h3 = torch.tanh(x_test * fc1_weights[2, 0] + fc1_biases[2]) * fc2_weights[0,2]

htot = (h1 + h2 + h3) + fc2_bias[0] # should be the same as the prediction

plt.figure(figsize=(10, 6))
plt.plot(x_test, h1, label='Neuron 1 output', linestyle='--')
plt.plot(x_test, h2, label='Neuron 2 output', linestyle='--')
plt.plot(x_test, h3, label='Neuron 3 output', linestyle='--')
# plt.plot(x_test, htot, label='sum Neuron 1,2,3', linestyle='--', c='r')
plt.plot(x_test, y_pred, label='NN prediction', color='r', linewidth=1.5)
plt.plot(x_test, y_test, label='True function', color='blue')
plt.xlabel('x')
plt.ylabel('Output')
plt.title(f'Contribution of each hidden neuron to the final prediction of {fnc.__name__} function')
plt.legend()
plt.show()
