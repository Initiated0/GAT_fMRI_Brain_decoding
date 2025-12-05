"""
creation date: Dec 4, 2025
Optimized Version
To run this script you need:
module load python3/anaconda/2023.1
module load cuda/12.1
source activate /work/nayeem/ENV/torch_cuda12
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # Regularization to prevent overfitting
EPOCHS = 60
SUB_IDs = ["UTS01"]  # adjust if you want more subjects
TRs_to_remove = [1340, 2138, 2842, 2843, 3688, 8676]
DROPOUT = 0.4 # tried= 0.3, 0.0, 0.15, and 0.45 
DATA_DIR = "/work/nayeem/Huth_deepfMRI/processed_data_transformed_atlas_float32/"
SAVE_DIR = os.path.join(DATA_DIR, "GAT_Results_gpt")
os.makedirs(SAVE_DIR, exist_ok=True)




####################################################################################################################################################
###############################################################   Classes and Functions   ##########################################################
####################################################################################################################################################
# --- CUSTOM LOSS FUNCTION ---
class NegativePearsonLoss(nn.Module):
    """
    Optimizes directly for Pearson Correlation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # Flatten to (N,)
        x = preds.flatten()
        y = targets.flatten()
        
        # Center variables
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        
        # Compute Correlation
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        
        # Return negative correlation (we want to minimize this)
        return -cost

# --- 1. DATA PREPARATION ---
# --- 1. DATA PREPARATION ---
class BrainStateDataset(Dataset):
    def __init__(self, node_features_path, raw_fmri_path, adj_path, trs_to_remove):
        super().__init__()
        
        # 1. Load Node Features X (Shape: [Total_TRs, Nodes, Features])
        # These are the X_t inputs
        # Added weights_only=True to fix the warning
        self.X = torch.load(node_features_path, weights_only=True).float() 
        
        # 2. Load and Normalize Targets Y
        # Load raw numpy data
        raw_data = np.load(raw_fmri_path) # Shape: [Total_TRs, Nodes]
        
        # --- CRITICAL FIX: ALIGN Y WITH X ---
        # We must remove the same TRs from Y that we removed from X
        raw_data = np.delete(raw_data, trs_to_remove, axis=0)
        
        if raw_data.shape[0] != self.X.shape[0]:
            raise ValueError(f"Shape Mismatch! X has {self.X.shape[0]} TRs, but Y has {raw_data.shape[0]} TRs.")

        # --- NEW: Standardization Logic ---
        # We calculate the split index (80%) to ensure we only fit on Training data
        # This prevents "Data Leakage" from the validation/test sets.
        train_split_idx = int(raw_data.shape[0] * 0.8)
        
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        
        # Fit the scaler ONLY on the training section of the data
        self.scaler.fit(raw_data[:train_split_idx])
        
        # Transform the ENTIRE dataset using the statistics from the training set
        raw_data_scaled = self.scaler.transform(raw_data)
        
        # raw_data_scaled = np.clip(raw_data_scaled, -6.0, 6.0)
        
        # Store the SCALED data as self.Y
        self.Y = torch.tensor(raw_data_scaled, dtype=torch.float32)
        
        print(f"Target Normalization stats (Train subset):")
        print(f"  Mean: {raw_data_scaled[:train_split_idx].mean():.4f}")
        print(f"  Std:  {raw_data_scaled[:train_split_idx].std():.4f}")
        # ----------------------------------
        
        # 3. Load Adjacency Matrix
        adj_data = np.load(adj_path)
        adjacency = adj_data['adjacency'] 
        
        # Convert binary matrix to edge_index 
        rows, cols = np.where(adjacency == 1)
        # Added np.array wrapper to fix the UserWarning
        self.edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
        
        # 4. Alignment
        self.num_samples = self.X.shape[0] - 1
        
        print(f"Dataset Loaded. Samples: {self.num_samples}")
        print(f"Input Feature Dim: {self.X.shape[2]}")
        print(f"Edges in Graph: {self.edge_index.shape[1]}")

    def len(self):
        return self.num_samples

    def get(self, idx):
        # Input: Features at time t
        x_t = self.X[idx] 
        
        # Target: SCALED BOLD signal at time t+1
        # UPDATED: Use self.Y (the scaled version) instead of self.Y_raw
        y_next = self.Y[idx + 1].unsqueeze(1) 
        
        # Create PyG Data Object
        data = Data(x=x_t, edge_index=self.edge_index, y=y_next)
        return data

# --- 2. THE MODEL (GAT) ---

class BrainGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=DROPOUT):
        super().__init__()
        
        # Layer 1: Attention Layer
        # Concatenates heads: Output dim = hidden_dim * heads
        self.gat1 = GATv2Conv(in_channels, 64, heads=heads, dropout=dropout)
        self.proj1 = nn.Linear(in_channels, 64 * heads) # Projection for skip connection
        
        # Layer 2: Attention Layer
        # Output dim = 32 * heads
        self.gat2 = GATv2Conv(64 * heads, 32, heads=heads, dropout=dropout)
        self.proj2 = nn.Linear(64 * heads, 32 * heads)  # Projection for skip connection
        
        # Layer 3: Regression Head (Linear)
        # We aggregate head outputs and map to 1 scalar (BOLD value)
        self.fc = nn.Linear(32 * heads, out_channels)
        
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GAT Block 1
        identity = self.proj1(x)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity # Residual concat 
        
        # GAT Block 2
        identity = self.proj2(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity # Residual concat 
        
        # Final Prediction
        x = self.fc(x)
        return x

# --- 3. METRICS HELPER ---
def compute_metrics(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy().flatten()
    y_true = y_true.detach().cpu().numpy().flatten()
    mse = np.mean((y_true - y_pred)**2)
    corr, _ = pearsonr(y_true, y_pred)
    return {"mse": mse, "corr": corr}
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################





bins = {
    "lexical_bin0":      ["gpt2_layer_0"],
    "syntax_bin1":       ["gpt2_layer_1", "gpt2_layer_2", "gpt2_layer_3"],
    "semantic_bin2":     ["gpt2_layer_4", "gpt2_layer_5", "gpt2_layer_6", "gpt2_layer_7", "gpt2_layer_8"],
    "discourse_bin3":    ["gpt2_layer_9", "gpt2_layer_10", "gpt2_layer_11", "gpt2_layer_12"]
}

# --- MAIN LOOP WITH PLOTTING ---
for sub_id in SUB_IDs:
    for bin_name in bins.keys():
        print(f"\n--- Processing {bin_name} (Optimized) ---")
        
        # 1. Setup Data
        x_path = os.path.join(DATA_DIR, f"{sub_id}_node_features_X_FIR_stdPCA_{bin_name}.pt")
        raw_path = os.path.join(DATA_DIR, f"{sub_id}_combined.npy")
        adj_path = os.path.join(DATA_DIR, "fMRI_FC_results", f"{sub_id}_FC_thr0.5.npz")
        
        dataset = BrainStateDataset(x_path, raw_path, adj_path, TRs_to_remove)
        
        total_len = len(dataset)
        train_idx, val_idx = int(total_len * 0.8), int(total_len * 0.9)
        
        train_loader = DataLoader(dataset[:train_idx], batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(dataset[train_idx:val_idx], batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(dataset[val_idx:], batch_size=BATCH_SIZE, shuffle=False)
        
        # 2. Setup Model
        model = BrainGAT(in_channels=dataset.X.shape[2], out_channels=1).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        criterion = NegativePearsonLoss()
        
        # 3. Training Loop
        best_val_corr = -1.0
        model_save_path = os.path.join(SAVE_DIR, f"{sub_id}_{bin_name}_best.pth")
        
        # --- TRACKING HISTORY ---
        history = {'train_loss': [], 'val_loss': [], 'val_corr': []}
        
        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss_accum = 0
            for batch in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
            
            avg_train_loss = train_loss_accum / len(train_loader)
            
            # Validation
            model.eval()
            all_preds, all_trues = [], []
            val_loss_accum = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    out = model(batch)
                    # Calc loss for tracking
                    loss = criterion(out, batch.y)
                    val_loss_accum += loss.item()
                    
                    all_preds.append(out)
                    all_trues.append(batch.y)
            
            avg_val_loss = val_loss_accum / len(val_loader)
            metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_trues))
            
            # Update History
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_corr'].append(metrics['corr'])
            
            # Step Scheduler
            scheduler.step(metrics['corr'])
            
            print(f"Ep {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Corr: {metrics['corr']:.4f}")
            
            if metrics['corr'] > best_val_corr:
                best_val_corr = metrics['corr']
                torch.save(model.state_dict(), model_save_path)
        
        # 4. Final Evaluation
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                test_preds.append(model(batch))
                test_trues.append(batch.y)
        
        # Inverse Transform
        test_preds_np = torch.cat(test_preds).cpu().numpy().reshape(-1, dataset.Y.shape[1])
        test_trues_np = torch.cat(test_trues).cpu().numpy().reshape(-1, dataset.Y.shape[1])
        
        preds_orig = dataset.scaler.inverse_transform(test_preds_np)
        trues_orig = dataset.scaler.inverse_transform(test_trues_np)
        
        final_metrics = compute_metrics(torch.tensor(preds_orig), torch.tensor(trues_orig))
        print(f"--- FINAL {bin_name}: Corr {final_metrics['corr']:.4f} ---")
        
        # 5. --- PLOTTING ---
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss (Neg Pearson)')
        plt.plot(history['val_loss'], label='Val Loss (Neg Pearson)')
        plt.title(f'{bin_name} - Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Correlation
        plt.subplot(1, 2, 2)
        plt.plot(history['val_corr'], color='orange', label='Val Correlation')
        plt.title(f'{bin_name} - Correlation')
        plt.xlabel('Epochs')
        plt.ylabel('Pearson r')
        plt.legend()
        plt.grid(True)
        
        # Save
        plot_path = os.path.join(SAVE_DIR, f"{sub_id}_{bin_name}_training_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to: {plot_path}")
    
