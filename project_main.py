import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from google.colab import drive
from itertools import product
import random
import numpy as np
from sklearn.model_selection import train_test_split


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Mount Google Drive
drive.mount('/content/drive')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set predict to True to get the results on test
predict = True

def create_df_from_pkl_filenames(directory):
    files = os.listdir(directory)
    pkl_files = [file for file in files if file.endswith('.pkl')]
    df = pd.DataFrame(pkl_files, columns=['filename'])
    return df

directory_path = '/content/drive/MyDrive/test'
df = create_df_from_pkl_filenames(directory_path)

# Define AudioDataset for labeled data
class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define AudioDatasetTest for unlabeled data
class AudioDatasetTest(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CNN_reg(nn.Module):
    def __init__(self, architecture, dropout_prob, fc_layer_sizes):
        super(CNN_reg, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 1
        for out_channels, kernel_size, padding in architecture:
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_layer_sizes = fc_layer_sizes
        self.fc_layers = self._create_fc_layers()

    def _create_fc_layers(self):
        layers = []
        in_features = self._get_fc1_in_features()
        for out_features in self.fc_layer_sizes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(self.dropout)
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        return nn.Sequential(*layers).to(device)

    def _get_fc1_in_features(self):
        # Assuming input shape is (batch_size, 1, 16, 342)
        x = torch.zeros(1, 1, 16, 342).to(device)
        self.to(device)
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x)
        return x.numel()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self._get_fc1_in_features())
        x = self.fc_layers(x)
        return x

# Load the dataset objects directly from the pickle files
train_dataset = torch.load('/content/drive/MyDrive/audio_dataset.pth')
test_dataset = torch.load('/content/drive/MyDrive/audio_dataset_test.pth')

# Coarse Hyperparameter Search
# Had to be shortened due to enormous time needed to check all combinations
coarse_params = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32],
    'dropout_prob': [0.3, 0.5, 0.7],
    'weight_decay': [0.001, 0.0001, 0.01],
    'architectures': [
        # [(16, 3, 1), (32, 3, 1), (64, 3, 1)],  # Small architecture
        [(32, 3, 1), (64, 3, 1), (128, 3, 1)],  # Medium architecture
        # [(64, 3, 1), (128, 3, 1), (256, 3, 1)]  # Large architecture
    ],
    'fc_layer_sizes': [
        # [64, 32],
        # [128, 64, 32],
        [256, 128, 64]
    ]
}

# Initialize an empty DataFrame for results
results_df = pd.DataFrame(columns=['learning_rate', 'batch_size', 'dropout_prob', 'architecture', 'fc_layer_sizes', 'epoch', 'loss'])

# Training and evaluation loop
num_epochs = 50

for wd, bs, dp, lr, arch, fc_sizes in product(coarse_params['weight_decay'], coarse_params['batch_size'], coarse_params['dropout_prob'], coarse_params['learning_rate'], coarse_params['architectures'], coarse_params['fc_layer_sizes']):
    # Assuming train_dataset is a TensorDataset or similar
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_dataset.data, train_dataset.labels, test_size=0.15, random_state=seed
    )

    # Create DataLoaders for the new train and validation sets
    train_loader = DataLoader(AudioDataset(train_data, train_labels), batch_size=bs, shuffle=True)
    val_loader = DataLoader(AudioDataset(val_data, val_labels), batch_size=bs, shuffle=False)

    # early stopping parameters
    early_stopping_patience = 5
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Initialize model, criterion, and optimizer
    model = CNN_reg(architecture=arch, dropout_prob=dp, fc_layer_sizes=fc_sizes).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0

      for inputs, labels in train_loader:
          inputs = inputs.unsqueeze(1).to(device)
          labels = labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          labels = labels.view_as(outputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item() * inputs.size(0)

      epoch_loss = running_loss / len(train_loader.dataset)

      # Validate the model
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for inputs, labels in val_loader:
              inputs = inputs.unsqueeze(1).to(device)
              labels = labels.to(device)
              outputs = model(inputs)
              labels = labels.view_as(outputs)
              loss = criterion(outputs, labels)
              val_loss += loss.item() * inputs.size(0)
      val_loss /= len(val_loader.dataset)

      # Early stopping
      if val_loss < best_loss:
          best_loss = val_loss
          epochs_without_improvement = 0
      else:
          epochs_without_improvement += 1
          if epochs_without_improvement >= early_stopping_patience:
              print(f"Early stopping at epoch {epoch+1}")
              break

      # Log the results
      new_row = pd.DataFrame({
          'learning_rate': [lr],
          'batch_size': [bs],
          'dropout_prob': [dp],
          'weight_decay': [wd],
          'architecture': [str(arch)],
          'fc_layer_sizes': [str(fc_sizes)],
          'epoch': [epoch + 1],
          'train_loss': [epoch_loss],
          'val_loss': [val_loss]
      })
      results_df = pd.concat([results_df, new_row], ignore_index=True)

      print(f"LR: {lr}, BS: {bs}, DP: {dp}, WD: {wd}, ARCH: {arch}, FC: {fc_sizes}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save coarse search results to CSV
results_df.to_csv('/content/drive/MyDrive/coarse_hyperparameter_search_results.csv', index=False)

# Results DataFrame for fine search
fine_results_df = pd.DataFrame(columns=['learning_rate', 'batch_size', 'dropout_prob', 'weight_decay', 'architecture', 'fc_layer_sizes', 'epoch', 'loss'])

# Find the best hyperparameters from the coarse search
best_params = results_df.loc[results_df['val_loss'].idxmin()]


# Define fine search ranges based on the best hyperparameters
fine_param_ranges = {
    'learning_rate': [best_params['learning_rate'] * 0.5, best_params['learning_rate'] * 1.5],
    'batch_size': [32, 32],
    'dropout_prob': [best_params['dropout_prob'] * 0.5, best_params['dropout_prob'] * 1.5],
    'weight_decay': [best_params['weight_decay'] * 0.5, best_params['weight_decay'] * 1.5],
    'architecture': [eval(best_params['architecture'])],
    'fc_layer_sizes': [eval(best_params['fc_layer_sizes'])]
}


# Number of fine search iterations
num_fine_search_iterations = 10

# Function to sample random values within a range
def sample_random_value(param_range):
    return np.random.uniform(param_range[0], param_range[1])

# Fine search training and evaluation loop
for _ in range(num_fine_search_iterations):
    lr = sample_random_value(fine_param_ranges['learning_rate'])
    bs = int(sample_random_value(fine_param_ranges['batch_size']))
    dp = sample_random_value(fine_param_ranges['dropout_prob'])
    wd = sample_random_value(fine_param_ranges['weight_decay'])
    arch = fine_param_ranges['architecture'][0]
    fc_sizes = fine_param_ranges['fc_layer_sizes'][0]

    # Split the train dataset into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_dataset.data, train_dataset.labels, test_size=0.15, random_state=seed
    )

    # Create DataLoaders for the new train and validation sets
    train_loader = DataLoader(AudioDataset(train_data, train_labels), batch_size=bs, shuffle=True)
    val_loader = DataLoader(AudioDataset(val_data, val_labels), batch_size=bs, shuffle=False)

    # Add early stopping parameters
    early_stopping_patience = 5
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Initialize model, criterion, and optimizer
    model = CNN_reg(architecture=arch, dropout_prob=dp, fc_layer_sizes=fc_sizes).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.view_as(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1).to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                labels = labels.view_as(outputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Log the results
        new_row = pd.DataFrame({
            'learning_rate': [lr],
            'batch_size': [bs],
            'dropout_prob': [dp],
            'weight_decay': [wd],
            'architecture': [str(arch)],
            'fc_layer_sizes': [str(fc_sizes)],
            'epoch': [epoch + 1],
            'train_loss': [epoch_loss],
            'val_loss': [val_loss]
        })
        fine_results_df = pd.concat([fine_results_df, new_row], ignore_index=True)

        print(f"LR: {lr}, BS: {bs}, DP: {dp}, WD: {wd}, ARCH: {arch}, FC: {fc_sizes}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save fine search results to CSV
fine_results_df.to_csv('/content/drive/MyDrive/fine_hyperparameter_search_results.csv', index=False)

# Find the best hyperparameters from the fine search
best_fine_params = fine_results_df.loc[fine_results_df['val_loss'].idxmin()]


# Extract the best hyperparameters
best_lr = best_fine_params['learning_rate']
best_bs = best_fine_params['batch_size']
best_dp = best_fine_params['dropout_prob']
best_wd = best_fine_params['weight_decay']
best_arch = eval(best_fine_params['architecture'])
best_fc_sizes = eval(best_fine_params['fc_layer_sizes'])

# Initialize model with best hyperparameters
best_model = CNN_reg(architecture=best_arch, dropout_prob=best_dp, fc_layer_sizes=best_fc_sizes).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_wd)

#
test_dataset = torch.load('/content/drive/MyDrive/audio_dataset_test.pth')
test_loader = DataLoader(test_dataset, batch_size=best_bs)

#
num_epochs = 50
early_stopping_patience = 10
best_loss = float('inf')
epochs_without_improvement = 0

# Split the train dataset into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    train_dataset.data, train_dataset.labels, test_size=0.15, random_state=seed
)

# Create DataLoaders for the new train and validation sets
train_loader = DataLoader(AudioDataset(train_data, train_labels), batch_size=best_bs, shuffle=True)
val_loader = DataLoader(AudioDataset(val_data, val_labels), batch_size=best_bs, shuffle=False)

# Train the best_model with the best hyperparameters
num_epochs = 50
early_stopping_patience = 5
best_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    best_model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = best_model(inputs)
        labels = labels.view_as(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validate the model
    best_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            outputs = best_model(inputs)
            labels = labels.view_as(outputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Training Best Model - Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# Prediction loop for test data
if predict:
    best_model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.unsqueeze(1).to(device)
            outputs = best_model(inputs)
            # Extract each prediction from the batch and append to the predictions list
            predictions.extend(outputs.cpu().numpy().flatten())

    df_predictions = pd.DataFrame(predictions, columns=['Label'])
    df_predictions.to_csv('/content/drive/MyDrive/predictions.csv', index=False)