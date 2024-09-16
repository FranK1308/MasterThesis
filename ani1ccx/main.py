# Import the necessary libraries
import time
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from ase import Atoms
from ase.io import read
from egnn import EGNN_Full
from convert import Convert
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

molecules = read("/home/frank/multif/egnn/data/subtracted_ani1ccx.extxyz", index=':25000')
print('Loaded structure data')

# Define the cutoff distance for neighbors
cutoff = 5.0  # Adjust this value based on your specific needs

# Initialize the dataset with the list of Atoms objects and cutoff distance
dataset = Convert(atoms=molecules, cutoff=cutoff)
print('Formatted the data')

# Iterate through the dataset and print the contents
#for i, data in enumerate(dataset):
#    print(f"Data {i}:")
#    print(f"Positions: {data.pos}")
#    print(f"Edge index: {data.edge_index}")
#    print(f"Atomic numbers: {data.Z}")
#    print(f"One-hot encoded atomic numbers: {data.x}")
#    print(f"Energy: {data.energy}")
#    print(f"Index: {data._idx}")
#    print("----------------------")

# Split the data into training, validation and testing
train_val_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Split the data:")
print(f"  Train data: {len(train_dataset)}")
print(f"  Validation data: {len(val_dataset)}")
print(f"  Test data: {len(test_dataset)} \n")

# Create the EGNN model
model = EGNN_Full(
    depth=3,
    hidden_features=128,
    node_features=5,
    out_features=1,
    activation="relu",
    norm="layer",
    aggr="sum",
    pool="add",
    residual=True,
    RFF_dim=64,
    RFF_sigma=1.0,
)

# Use GPU if available
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Initialize the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Max epochs and stopping criteria
num_epochs = 500
lr_threshold = 1e-6 
best_val_loss = float('inf')

# Determine the width of the epoch number for formatting
epoch_width = len(str(num_epochs))

# Track the previous learning rate
prev_lr = scheduler.optimizer.param_groups[0]['lr']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print model parameters
print(f"Model Parameters:")
print(f"  Depth: {model.depth}")
print(f"  Hidden Features: {model.hidden_features}")
print(f"  Node Features: {model.node_features}")
print(f"  Output Features: {model.out_features}")
print(f"  Activation: {model.activation}")
print(f"  Normalization: {model.norm}")
print(f"  Aggregation: {model.aggr}")
print(f"  Pooling: {model.pool}")
print(f"  Residual: {model.residual}")
print(f"  RFF Dimension: {model.RFF_dim}")
print(f"  RFF Sigma: {model.RFF_sigma} \n")

# Print whether GPU or CPU is being used
print(f'Using device {device}')

# Print the total number of parameters in the model
total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")

print(f'Cutoff radius: {cutoff}')
print(f'Number of epochs: {num_epochs}')
print(f'Initial learning rate {prev_lr:.1e} \n')

for epoch in range(num_epochs):
    # Start time for the epoch
    start_time_epoch = time.time()

    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch).squeeze(1)  # Remove the singleton dimension to match target shape
        loss = criterion(output, batch.energy)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch).squeeze(1)
            loss = criterion(output, batch.energy)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Step the learning rate scheduler
    scheduler.step(val_loss)
    
    # End time for the epoch
    end_time_epoch = time.time()

    # Calculate and print the epoch time
    epoch_time = end_time_epoch - start_time_epoch

    # Print the epoch, train loss, validation loss, and epoch time
    print(f'Epoch {epoch + 1:>{epoch_width}}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Time: {epoch_time:.2f} seconds')

    # Check if the learning rate has changed
    current_lr = scheduler.optimizer.param_groups[0]['lr']
    if current_lr != prev_lr:
        print(f'Learning rate reduced to {current_lr:.1e}')
        prev_lr = current_lr
        
        # Check if the learning rate has dropped below the threshold
        if current_lr < lr_threshold:
            print(f'Learning rate dropped below {lr_threshold:.1e}. Stopping training.')
            break

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Test
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        output = model(batch).squeeze(1)
        loss = criterion(output, batch.energy)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.8f}')

