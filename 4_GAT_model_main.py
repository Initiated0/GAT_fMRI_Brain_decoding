import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
SUB_ID = "UTS01" # Running for one subject first
DATA_DIR = "/work/nayeem/Huth_deepfMRI/processed_data_transformed_atlas_float32/"
SAVE_DIR = os.path.join(DATA_DIR, "GAT_Results")
os.makedirs(SAVE_DIR, exist_ok=True)



####################################################################################################################################################
###############################################################   Classes and Functions   ##########################################################
####################################################################################################################################################
# --- 1. DATA PREPARATION ---
class BrainStateDataset(Dataset):
    def __init__(self, node_features_path, raw_fmri_path, adj_path):
        super().__init__()
        
        # 1. Load Node Features X (Shape: [Total_TRs, Nodes, Features])
        # These are the X_t inputs
        self.X = torch.load(node_features_path).float() 
        
        # 2. Load Raw fMRI for Targets Y (Shape: [Total_TRs, Nodes])
        # These are the ground truth BOLD signals we want to predict
        raw_data = np.load(raw_fmri_path)
        self.Y_raw = torch.tensor(raw_data, dtype=torch.float32)
        
        # 3. Load Adjacency Matrix
        # We convert the static adjacency matrix into Edge Indices (COO format)
        adj_data = np.load(adj_path)
        adjacency = adj_data['adjacency'] # Shape (423, 423)
        
        # Convert binary matrix to edge_index [[sources], [targets]]
        # We find indices where adjacency == 1
        rows, cols = np.where(adjacency == 1)
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long)
        
        # 4. Alignment
        # Input X at index t should predict Target Y at index t+1
        # Length is Total_TRs - 1
        self.num_samples = self.X.shape[0] - 1
        
        print(f"Dataset Loaded. Samples: {self.num_samples}")
        print(f"Input Feature Dim: {self.X.shape[2]}")
        print(f"Edges in Graph: {self.edge_index.shape[1]}")

    def len(self):
        return self.num_samples

    def get(self, idx):
        # Input: Features at time t
        x_t = self.X[idx] 
        
        # Target: Raw BOLD signal at time t+1
        # Shape: (Nodes, 1) -> We predict 1 scalar value per node
        y_next = self.Y_raw[idx + 1].unsqueeze(1) 
        
        # Create PyG Data Object
        # We share the same edge_index for all snapshots (Static Connectivity)
        data = Data(x=x_t, edge_index=self.edge_index, y=y_next)
        return data

# --- 2. THE MODEL (GAT) ---

class BrainGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        
        # Layer 1: Attention Layer
        # Concatenates heads: Output dim = hidden_dim * heads
        self.gat1 = GATv2Conv(in_channels, 64, heads=heads, dropout=dropout)
        
        # Layer 2: Attention Layer
        # Output dim = 32 * heads
        self.gat2 = GATv2Conv(64 * heads, 32, heads=heads, dropout=dropout)
        
        # Layer 3: Regression Head (Linear)
        # We aggregate head outputs and map to 1 scalar (BOLD value)
        self.fc = nn.Linear(32 * heads, out_channels)
        
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GAT Block 1
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT Block 2
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final Prediction
        x = self.fc(x)
        
        return x

# --- 3. METRICS HELPER ---
def compute_metrics(y_pred, y_true):
    """
    Computes MSE, MAE, R2, and Pearson Correlation.
    Expects inputs of shape (Batch_Size * Nodes, 1) or (Total_Nodes, 1)
    """
    y_pred = y_pred.detach().cpu().numpy().flatten()
    y_true = y_true.detach().cpu().numpy().flatten()
    
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R2 Score
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Pearson Correlation
    # Note: This computes correlation across ALL nodes and timepoints in the batch.
    # A more granular approach is correlation per-node over time, 
    # but this gives a good global fit indicator.
    corr, _ = pearsonr(y_true, y_pred)
    
    return {"mse": mse, "mae": mae, "r2": r2, "corr": corr}
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################








# Paths (Update these based on your file system)
x_path = os.path.join(DATA_DIR, f"{SUB_ID}_node_features_X.pt")
raw_path = os.path.join("/content/drive/MyDrive/UofSC/AIISC/Data-fMRI/NL_Deep_fMRI/", f"{SUB_ID}_combined.npy") # UPDATE THIS TO REAL PATH
adj_path = os.path.join(DATA_DIR, "fMRI_FC_results", f"{SUB_ID}_FC_thr0.5.npz")


Load Dataset
dataset = BrainStateDataset(x_path, raw_path, adj_path)

# Temporal Split (Train 80% / Val 10% / Test 10%)
total_len = len(dataset)
train_idx = int(total_len * 0.8)
val_idx = int(total_len * 0.9)

# Note: We cannot use random_split. We must slice indices.
# PyG doesn't support direct slicing of Dataset objects easily without 'Subset'
train_dataset = dataset[:train_idx]
val_dataset = dataset[train_idx:val_idx]
test_dataset = dataset[val_idx:]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) # Shuffle=False preserves time order (optional in GAT but good practice)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize Model
model = BrainGAT(in_channels=332, out_channels=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Tracking
history = {'train_loss': [], 'val_loss': [], 'val_corr': []}





print("--- Starting Training ---")
for epoch in range(EPOCHS):
    
    # TRAIN
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch) # Forward
        loss = criterion(out, batch.y) # Compare vs Ground Truth
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # VALIDATION
    model.eval()
    val_loss = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = criterion(out, batch.y)
            val_loss += loss.item()
            
            all_preds.append(out)
            all_trues.append(batch.y)
            
    avg_val_loss = val_loss / len(val_loader)
    
    # Calculate Val Metrics
    # Stack all validation outputs to compute metrics over the whole val set
    val_preds_full = torch.cat(all_preds, dim=0)
    val_trues_full = torch.cat(all_trues, dim=0)
    metrics = compute_metrics(val_preds_full, val_trues_full)
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_corr'].append(metrics['corr'])
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Corr: {metrics['corr']:.4f}")
    
    # Checkpoint (Save best model based on Correlation)
    if epoch > 0 and metrics['corr'] > max(history['val_corr'][:-1]):
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{SUB_ID}_best_GAT.pth"))

# --- 5. FINAL EVALUATION & SAVING ---

# Load best model
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{SUB_ID}_best_GAT.pth")))
model.eval()

# Run on Test Set
test_preds = []
test_trues = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        test_preds.append(out)
        test_trues.append(batch.y)

test_preds = torch.cat(test_preds, dim=0)
test_trues = torch.cat(test_trues, dim=0)

final_metrics = compute_metrics(test_preds, test_trues)
print("\n--- Final Test Results ---")
print(f"MSE: {final_metrics['mse']:.5f}")
print(f"MAE: {final_metrics['mae']:.5f}")
print(f"R2 Score: {final_metrics['r2']:.4f}")
print(f"Pearson Correlation (Global): {final_metrics['corr']:.4f}")

# Save Metrics to CSV
df_metrics = pd.DataFrame([final_metrics])
df_metrics.to_csv(os.path.join(SAVE_DIR, f"{SUB_ID}_test_metrics.csv"), index=False)

# Save Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title(f'Training Loss - {SUB_ID}')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, f"{SUB_ID}_loss_curve.png"))
print("Saved metrics and plots.")


















































