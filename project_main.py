import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from google.colab import drive



# Mount Google Drive
drive.mount('/content/drive')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CNN_reg(nn.Module):
    def __init__(self, dropout_prob):
        super(CNN_reg, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_prob)

        # Calculate the output dimension after the convolutions and pooling
        self.fc1_in_features = 21760
        self.fc1 = nn.Linear(in_features=self.fc1_in_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Second Convolutional Block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Third Convolutional Block
        x = self.conv3(x)
        x = self.relu(x)
        # Flatten for Fully Connected Layers
        x = x.view(-1, self.fc1_in_features)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(-1, 1)
        return x

# Load the dataset from the .pth file
dataset = torch.load('/content/drive/MyDrive/audio_dataset.pth')

# Define train, validation, and test split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the number of samples for each split
total_samples = len(dataset)
train_samples = int(train_ratio * total_samples)
val_samples = int(val_ratio * total_samples)
test_samples = total_samples - train_samples - val_samples

# Split the dataset into train, validation, and test sets
train_set, val_set, test_set = random_split(dataset, [train_samples, val_samples, test_samples])

# Create DataLoader for train, validation, and test sets
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)

# Define the model and move it to the appropriate device
model = CNN_reg(dropout_prob=0.5).to(device)

# Define loss function
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        labels = labels.view_as(outputs)
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item() * inputs.size(0)  # Accumulate the loss

    # Calculate the average loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            # Move inputs and labels to device
            outputs = model(inputs)  # Forward pass
            labels = labels.view_as(outputs)
            loss = criterion(outputs, labels)  # Calculate the loss
            val_loss += loss.item() * inputs.size(0)  # Accumulate the loss

    # Calculate the average validation loss
    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Validation Loss: {avg_val_loss:.4f}")



# Test loop
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
        outputs = model(inputs)  # Forward pass
        labels = labels.view_as(outputs)
        print(outputs, labels)
        loss = criterion(outputs, labels)  # Calculate the loss
        test_loss += loss.item() * inputs.size(0)  # Accumulate the loss

# Calculate the average test loss
avg_test_loss = test_loss / len(test_loader.dataset)
print(f"Test Loss: {avg_test_loss:.4f}")
