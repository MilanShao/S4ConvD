import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import wandb

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Initialize WandB
wandb.init(project="name", entity="name")

# Set device to CPU
device = torch.device('cpu')
print(f"Using device: {device}")

def load_local_data(name, fraction=0.01):
    path = f'/path/Dataset/{name}.csv'
    df = pd.read_csv(path)
    df = df.sample(frac=fraction, random_state=42)  # Sample 1% of the data
    return df

# Load datasets
train_df = load_local_data('train')
test_df = load_local_data('test')
val_df = load_local_data('val')

# Dataset class
class ASHRAEDataset(Dataset):
    def __init__(self, df, input_length=48):
        super().__init__()
        self.input_length = input_length
        self.df = df[['meter_reading', 'air_temperature', 'cloud_coverage', 'dew_temperature']].dropna()

        # Normalize data
        self.means = self.df.mean()
        self.stds = self.df.std()
        self.data = (self.df - self.means) / self.stds

    def __getitem__(self, index):
        input_start = index
        input_end = input_start + self.input_length
        if input_end >= len(self.data):
            raise IndexError("Index exceeds data limits during sequence setup.")

        input_data = self.data.iloc[input_start:input_end].values
        label_data = self.data.iloc[input_end, 0]

        return {
            "input_sequence": torch.tensor(input_data, dtype=torch.float32),
            "label_sequence": torch.tensor(label_data, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.data) - self.input_length

# Create the datasets
train_dataset = ASHRAEDataset(train_df)
val_dataset = ASHRAEDataset(val_df)
test_dataset = ASHRAEDataset(test_df)

# Prepare the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False)

# Define RMSLE loss function
def rmsle(predictions, targets):
    epsilon = 1e-6
    predictions, targets = predictions.squeeze(), targets.squeeze()
    if predictions.size() != targets.size():
        raise ValueError(f"Size mismatch: predictions {predictions.size()}, targets {targets.size()}")
    predictions = torch.clamp(predictions, min=0)
    targets = torch.clamp(targets, min=0)
    log_predictions = torch.log1p(predictions + epsilon)
    log_targets = torch.log1p(targets + epsilon)
    return torch.sqrt(torch.mean((log_predictions - log_targets) ** 2))

# Define S4Conv model classes
class S4ConvKernel(nn.Module):
    def __init__(self, config):
        super().__init__()
        H = config["model"]["dim"]
        N = config["model"]["block"]["state_dim"]

        self.log_dt = nn.Parameter(torch.rand(H) * (np.log(0.1) - np.log(0.001)) + np.log(0.001))
        self.C_re = nn.Parameter(torch.randn(H, N // 2))
        self.C_im = nn.Parameter(torch.randn(H, N // 2))
        self.log_A_re = nn.Parameter(torch.randn(H, N // 2))
        self.A_im = nn.Parameter(torch.ones(H, N // 2))

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = self.C_re + 1j * self.C_im
        A = -torch.exp(self.log_A_re) + 1j * self.A_im
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.0) / A
        return 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

class S4ConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.D = nn.Parameter(torch.randn(config["model"]["dim"]))
        self.kernel_layer = S4ConvKernel(config)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config["model"]["block"]["dropout"])
        self.output_linear = nn.Sequential(
            nn.Conv1d(config["model"]["dim"], 2 * config["model"]["dim"], kernel_size=1),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        x_orig = x
        x = x.transpose(-1, -2)
        L = x.size(-1)
        k = self.kernel_layer(L).to(x.device)

        # Fourier transform, convolution, and inverse transform
        x_fft = torch.fft.rfft(x, n=2 * L)
        k_fft = torch.fft.rfft(k, n=2 * L)
        y = torch.fft.irfft(x_fft * k_fft, n=2 * L)[..., :L]

        y = y + x * self.D.unsqueeze(-1)
        y = self.activation(self.dropout(y))
        y = self.output_linear(y)
        y = y.transpose(-1, -2) + x_orig
        return y

class Model(nn.Module):
    def __init__(self, state_dim, measurement_dim, input_dim=4, output_dim=1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, measurement_dim)
        config = {
            "model": {
                "dim": measurement_dim,
                "block": {
                    "state_dim": state_dim,
                    "dropout": 0.01
                }
            }
        }
        self.blocks = nn.ModuleList([S4ConvBlock(config) for _ in range(4)])
        self.decoder = nn.Linear(measurement_dim, output_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)

        for block in self.blocks:
            x = block(x)

        x = self.decoder(x)
        return x[:, -1, :]

# Initialize and train the model
model = Model(state_dim=64, measurement_dim=128, input_dim=4, output_dim=1).to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Log model architecture and hyperparameters
wandb.watch(model, log="all")

# Training loop
num_epochs = 5
log_interval = 200  # Log less frequent
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", ncols=100) as pbar:
        for batch_idx, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_sequence = batch_data["input_sequence"].to(device)
            label_sequence = batch_data["label_sequence"].to(device)

            logits = model(input_sequence)
            loss = rmsle(logits, label_sequence)
            if torch.isnan(loss):
                continue

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            pbar.update()

            wandb.log({"Train Loss": loss.item()})

        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}")

    # Validate the model
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_data in val_dataloader:
            input_sequence = batch_data["input_sequence"].to(device)
            label_sequence = batch_data["label_sequence"].to(device)
            logits = model(input_sequence)

            if logits.size(0) != label_sequence.size(0):
                print(f"Size mismatch in validation: predictions {logits.size()}, targets {label_sequence.size()}")
                continue
            
            val_loss = rmsle(logits.squeeze(), label_sequence.squeeze())
            val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

        wandb.log({"Validation Loss": avg_val_loss})

# Test the model
model.eval()
predictions = []
with torch.no_grad():
    for batch_data in test_dataloader:
        input_sequence = batch_data["input_sequence"].to(device)
        logits = model(input_sequence)
        predictions.extend(logits.cpu().numpy().flatten())

# Extract actual values from the test data
actual_values = test_df['meter_reading'].iloc[test_dataset.input_length:].values

# Align lengths of predictions and actual values
min_length = min(len(predictions), len(actual_values))
predictions = predictions[:min_length]
actual_values = actual_values[:min_length]

# Calculate RMSLE
predictions = np.clip(predictions, a_min=0, a_max=None)  # Avoid negative predictions
test_rmsle = np.sqrt(np.mean((np.log1p(actual_values) - np.log1p(predictions)) ** 2))
print(f"Test RMSLE: {test_rmsle:.4f}")

# Log test RMSLE to wandb
wandb.log({"Test RMSLE": test_rmsle})

# Finish the wandb run
wandb.finish()