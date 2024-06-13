# -*- coding: utf-8 -*-
"""dropout-sampling.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DiQ6dSJLv8W-l-QrKJU99RiUABvAKp2u
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# fix random seeds
np.random.seed(1337)
torch.manual_seed(1337)

n_samples = 1000
x = np.random.uniform(low=-5, high=5, size=(n_samples))

# split the data to within distribution segment
x1 = x[(x >= -3) & (x <= 3)]
y1 = x1 * np.sin(x1)

# out of distribution data segment
x2 = x[(x < -3) | (x > 3)]
y2 = x2 * np.sin(x2)

print(x1.shape, y1.shape)
print(x2.shape, y2.shape)
sns.scatterplot(x=x1, y=y1, label="in-distribution data")
sns.scatterplot(x=x2, y=y2, label="out of distribution data")

# create a simple neural network and train it on a subset of the in-distribution data
# create in-distribution dataset and ood dataset
def get_dataset(x, y):
  x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
  y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
  return TensorDataset(x_tensor, y_tensor)

in_distribution_dataset = get_dataset(x1, y1)
ood_dataset = get_dataset(x2, y2)

n_val = int(0.3 * len(in_distribution_dataset))
n_train = len(in_distribution_dataset) - n_val
train_dataset, val_dataset = torch.utils.data.random_split(in_distribution_dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
ood_dataloader = DataLoader(ood_dataset, batch_size=1, shuffle=True)

class SimpleRegressionModel(nn.Module):
    def __init__(self, p=0.3):
        super(SimpleRegressionModel, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.dropout1 = nn.Dropout(p)
        self.fc2 = nn.Linear(50, 100)
        self.dropout2 = nn.Dropout(p)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

model = SimpleRegressionModel(p=0.3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = nn.MSELoss()

epochs = 1000
model.train()
for epoch in range(epochs):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

# predict with uncertainty
def predict_with_uncertainty(model, input_tensor, num_samples=100):
    model.eval()
    model.dropout1.train() # IMPORTANT enable dropout during inference
    model.dropout2.train() # IMPORTANT enable dropout during inference
    input_tensor = input_tensor.repeat(num_samples, 1 , 1)
    with torch.no_grad():
        predictions = model(input_tensor)
    mean_predictions = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)

    return predictions.numpy().reshape(-1, 1), mean_predictions, uncertainty

# get a sample from in-distribution validation set
xb = torch.tensor([0.2], dtype=torch.float32).view(-1, 1)
preds, mean, std_dev = predict_with_uncertainty(model, xb)
print(mean.item(), std_dev.item())

# get a sample from ood data
xb = torch.tensor([-4.3], dtype=torch.float32).view(-1, 1)
preds, mean, std_dev = predict_with_uncertainty(model, xb)
print(mean.item(), std_dev.item())

full_domain = np.linspace(-5, 5, 1000)
full_y = full_domain * np.sin(full_domain)
full_tensor = torch.tensor(full_domain, dtype=torch.float32).view(-1, 1)
full_y_tensor = torch.tensor(full_y, dtype=torch.float32).view(-1, 1)
full_dataset = TensorDataset(full_tensor, full_y_tensor)
full_loader = DataLoader(full_dataset, batch_size=1)

min_results = []
max_results = []
mean_results = []

for xb, yb in full_loader:
  preds, mean, std_dev = predict_with_uncertainty(model, xb)
  min_results.append(np.min(preds))
  max_results.append(np.max(preds))
  mean_results.append(mean.item())

full_domain.shape, mean.shape

sns.scatterplot(x=x1, y=y1, label="in-distribution data")
sns.scatterplot(x=x2, y=y2, label="out of distribution data")
plt.axvline(x=-3, color='r', label='start of training range', ls="--")
plt.axvline(x=3, color='r', label='end of training range', ls="--")

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')

a=sns.lineplot(x=full_domain, y=mean_results, label = 'predictions_mean')
a=sns.lineplot(x=full_domain, y=full_y, label = 'target_y')
b=sns.lineplot(x=full_domain, y=max_results, label='predictions_max', )
c=sns.lineplot(x=full_domain, y=min_results, label='predictions_min', )

line = c.get_lines()
plt.fill_between(line[1].get_xdata(), line[2].get_ydata(), line[3].get_ydata(), alpha=.5)
plt.axvline(x=-3, color='r', label='start of training range', ls="--")
plt.axvline(x=3, color='r', label='end of training range', ls="--")

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')

model.dropout1.training

model.eval()
