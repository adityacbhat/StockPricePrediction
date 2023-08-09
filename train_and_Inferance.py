import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as pltdf = pd.read_csv('AAPL.csv')
#df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

df.set_index('date', inplace=True)
features = df.drop(['symbol', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor'], axis=1)

# Scale the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['close'], test_size=0.2, shuffle=False)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
batch_size = 64
train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

model = StockPredictor(input_dim=X_train.shape[1])
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 200
model.train()
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_function(predictions, target.view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs} loss: {loss.item()}')
model.eval()
predictions = []
with torch.no_grad():
    for data, _ in test_loader:
        output = model(data)
        predictions.extend(output.squeeze().tolist())

plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()


scaled_last_row = scaler.transform(last_row)

# Convert to a PyTorch tensor
input_tensor = torch.tensor(scaled_last_row, dtype=torch.float)

# Make a prediction using the trained model
model.eval() # Set the model to evaluation mode
with torch.no_grad():
    prediction = model(input_tensor)

print(f"The predicted stock price for the next day is: {prediction.item()}")