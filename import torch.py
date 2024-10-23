import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

# Neural NetWork Class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)  #  hidden layer 1
        self.fc2 = nn.Linear(3, 4)  #  from Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(5, 1)  # from Hidden layer 2 to the output layer
        nn.init.rand_(self.fc1.weight)
        nn.init.rand_(self.fc1.bias)
        nn.init.rand_(self.fc2.weight)
        nn.init.rand_(self.fc2.bias)
        nn.init.rand_(self.fc3.weight)
        nn.init.rand_(self.fc3.bias)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

# Get random data
X = torch.rand(100, 3)  #  3 features and 100 samples
y = torch.rand(100, 1)  #  1 target  and 100 samples

model = SimpleNN()

# Get predictions before training
with torch.no_grad():
    predictions_before_training = model(X)

# Plot predictions before training
sns.scatterplot(x=X[:, 0], y=predictions_before_training.squeeze(), label='Predictions Before Training')
plt.title("Predictions Before Training")
plt.show()

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get predictions after training
with torch.no_grad():
    predictions_after_training = model(X)

# Plot predictions before and after training
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X[:, 0], y=predictions_before_training.squeeze(), label='Before Training')
plt.title("Predictions Before Training")

plt.subplot(1, 2, 2)
sns.scatterplot(x=X[:, 0], y=predictions_after_training.squeeze(), label='After Training')
plt.title("Predictions After Training")
plt.show()
