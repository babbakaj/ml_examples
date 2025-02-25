# simple example implementing a classification task in pytorch
# clasify two dimensional input in two possible classes (0,1)
# plot training loss and validation loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt # for making figures

# Generate some synthetic data
np   .random.seed(42)
torch.manual_seed(42)

# Create a dataset with 2 features and 2 classes
x = np.random.randn(10000, 2)
y = (x[:, 0] * x[:, 1] > 0).astype(int)  # Simple non-linear decision boundary

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

n1 = int(0.8*len(x))
# manual split
Xtr, Ytr     = x_tensor[:n1], y_tensor[:n1]
Xval, Yval   = x_tensor[n1:], y_tensor[n1:]

batch_size = 32

# Create a TensorDataset and DataLoader
dataset    = TensorDataset(x_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # TODO: Define the layers of the network
        # Use nn.Linear for the fully connected layers
        # Use nn.ReLU for the activation function
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_size, hidden_size)
        self.fc3  = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # TODO: Implement the forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

# Hyperparameters
input_size    = 2
hidden_size   = 10
output_size   = 2
learning_rate = 0.001
num_epochs    = 20

# Initialize the model, loss function, and optimizer
model     = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

lossi    = []
loss_val = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss    = criterion(outputs, batch_y)

        loss     .backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)

    if (epoch) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_x, batch_y in val_loader:
            outputs   = model(batch_x)
            val_loss += criterion(outputs, batch_y).item()
        avg_val_loss  = val_loss / len(val_loader)
        # print(f'Validation Loss: {avg_val_loss:.4f}')
    
    lossi   .append(avg_train_loss)
    loss_val.append(avg_val_loss)

model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total   += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    # print (outputs)
    # print (batch_y.shape)

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

plt.figure(figsize=(8, 6))
plt.plot(lossi,    label='Training Loss')
plt.plot(loss_val, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()
