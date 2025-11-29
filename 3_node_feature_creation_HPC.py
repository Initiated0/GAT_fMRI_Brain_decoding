"""
creation date: Nov 29, 2025

To run this script you need

module load python3/anaconda/2023.1
module load cuda/12.1
source activate /work/nayeem/ENV/torch_cuda12

"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


NUM_TRS = 8968
NUM_NODES = 423
WINDOW_SIZE = 20
D_FMRI = 32
D_WORD = 300 
BATCH_SIZE = 32  # Process 32 timepoints at once (Adjust up to 64/128 if VRAM allows)

# "/work/nayeem/Huth_deepfMRI/processed_data_transformed_atlas_float32/"
root_dir = "/work/nayeem/Huth_deepfMRI/"
# Paths
fMRI_BOLD_data_dir = os.path.join(root_dir, "processed_data_transformed_atlas_float32")
SUB_IDs = ["UTS01","UTS03","UTS07"]


for sub_id in SUB_IDs:
    # /content/gdrive/MyDrive/UofSC/AIISC/Data-fMRI/NL_Deep_fMRI/UTS01_combined.npy
    fMRI_data = np.load(os.path.join(fMRI_BOLD_data_dir, f"{sub_id}_combined.npy"))
    print(f"Before removing: {fMRI_data.shape}")
    # I need to remove six TRs
    TRs_to_remove = [1340, 2138, 2842, 2843, 3688, 8676]
    fmri_fixed = np.delete(fMRI_data, TRs_to_remove, axis=0)
    print(f"After removing: {fmri_fixed.shape}")   # (8968, 423)
    fmri_tensor = torch.tensor(fmri_fixed, dtype=torch.float32)
    # PAD the fMRI data at the start so we don't need 'if t < WINDOW' logic inside the loop
    # We add (Window-1) rows of zeros at the top
    padding = torch.zeros(WINDOW_SIZE - 1, NUM_NODES)
    fmri_padded = torch.cat([padding, fmri_tensor], dim=0)
    
    
    
    df = pd.read_csv(os.path.join(root_dir, "derivatives/Huth_fMRI_25_ordered_stories_SCOPEmerged_Spacy_tagged_trimmed_fasttext_df_TR-levelFinal.csv"))
    print(df)
    
    
    
    # Function to fix the stringified embedding column
    def clean_embedding_string(x):
        """
        Parses a string representation of a numpy array back into a numpy array.
        Handles both comma-separated '[0.1, 0.2]' and space-separated '[0.1 0.2]'
        """
        if isinstance(x, str):
            # Remove brackets and newlines
            clean_str = x.replace('[', '').replace(']', '').replace('\n', ' ')
            
            # Determine separator (comma or space)
            if ',' in clean_str:
                return np.fromstring(clean_str, sep=',')
            else:
                return np.fromstring(clean_str, sep=' ')
        return x
    
    # Apply the cleaning
    print("Parsing embedding strings...")
    df['embedding'] = df['embedding'].apply(clean_embedding_string)
    
    # Verify it worked
    print(f"Type after fix: {type(df.at[0, 'embedding'])}") # Should be <class 'numpy.ndarray'>
    print(f"Shape of one embedding: {df.at[0, 'embedding'].shape}") # Should be (300,)
    
    # Convert all embeddings to a single Torch Tensor for speed
    # Shape: (8968, 300)
    word_data_tensor = torch.tensor(np.stack(df['embedding'].values), dtype=torch.float32)
    
    
    final_X = torch.zeros(NUM_TRS, NUM_NODES, D_FMRI + D_WORD, dtype=torch.float32)
    
    
    
    
    # --- 4. MODEL SETUP ---
    class TimeSeriesEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
            self.fc = nn.Linear(16 * (WINDOW_SIZE - 4), D_FMRI)
    
        def forward(self, x):
            # x input: (Batch_Size * Nodes, 1, Window_Size)
            x = self.conv1(x)
            x = torch.relu(x)
            x = x.view(x.size(0), -1) 
            x = self.fc(x)
            return x 
    
    encoder = TimeSeriesEncoder().to(device) # Move model to GPU
    
    
    
    # --- 5. BATCHED LOOP ---
    print("Starting Batch Processing...")
    
    # Iterate in chunks of BATCH_SIZE
    for i in tqdm(range(0, NUM_TRS, BATCH_SIZE), desc="Processing Batches"):
        
        # Define start and end indices for this batch
        start_idx = i
        end_idx = min(i + BATCH_SIZE, NUM_TRS)
        current_batch_size = end_idx - start_idx
        
        # --- A. VECTORIZED WINDOW SLICING ---
        # We need to extract windows for [start_idx ... end_idx]
        # Because we padded fmri_padded, the window for time 't' is simply [t : t+WINDOW_SIZE]
        
        batch_windows = []
        for t in range(start_idx, end_idx):
            # Slice: (Window, Nodes)
            # Note: We slice from fmri_padded
            window = fmri_padded[t : t + WINDOW_SIZE, :] 
            batch_windows.append(window.T) # (Nodes, Window)
        
        # Stack windows: (Batch_Size, Nodes, Window)
        batch_windows = torch.stack(batch_windows)
        
        # Reshape for CNN: (Batch_Size * Nodes, 1, Window)
        # We flatten batch and nodes to feed them all into the CNN in parallel
        cnn_input = batch_windows.view(current_batch_size * NUM_NODES, 1, WINDOW_SIZE).to(device)
        
        # --- B. ENCODE (GPU) ---
        with torch.no_grad():
            # Output: (Batch_Size * Nodes, 32)
            encoded_flat = encoder(cnn_input)
            
            # Reshape back: (Batch_Size, Nodes, 32)
            fmri_batch_emb = encoded_flat.view(current_batch_size, NUM_NODES, D_FMRI)
            
        # --- C. WORDS (GPU) ---
        # Get words for this batch: (Batch_Size, 300)
        word_batch = word_data_tensor[start_idx : end_idx].to(device)
        
        # Expand: (Batch_Size, 1, 300) -> (Batch_Size, 423, 300)
        word_batch_expanded = word_batch.unsqueeze(1).expand(-1, NUM_NODES, -1)
        
        # --- D. CONCATENATE ---
        # (Batch, 423, 32) + (Batch, 423, 300) -> (Batch, 423, 332)
        batch_features = torch.cat([fmri_batch_emb, word_batch_expanded], dim=2)
        
        # --- E. STORE (CPU) ---
        # Move result back to CPU to save GPU memory and store in final tensor
        final_X[start_idx : end_idx] = batch_features.cpu()
    
    print("Done!")
    print("Final Shape:", final_X.shape)
    
    
    # --- 6. SAVING THE DATA ---
    
    # Define your save paths
    # (Using the directory you defined earlier: fMRI_BOLD_data_dir)
    output_path_pt = os.path.join(fMRI_BOLD_data_dir, f"{SUB_IDs[0]}_node_features_X.pt")
    output_path_npy = os.path.join(fMRI_BOLD_data_dir, f"{SUB_IDs[0]}_node_features_X.npy")
    
    print(f"Saving data to: {fMRI_BOLD_data_dir} ...")
    
    # 1. Save as PyTorch Tensor (Recommended)
    # This preserves gradients (if any) and data types perfectly
    torch.save(final_X, output_path_pt) # here is the thing -- these two file combinedly is more than 9.2 GB. I didn't run with the loop. Just got for subject 'UTS01'
    
    # 2. Save as NumPy Array (Optional backup)
    # We convert to numpy first. This is safer for portability.
    np.save(output_path_npy, final_X.numpy())
    
    # 3. Save the Shape metadata explicitly (per your request)
    # Sometimes it's useful to have a tiny text file to check dimensions 
    # without loading the huge 1GB+ file.
    # shape_info = str(final_X.shape)
    # with open(os.path.join(fMRI_BOLD_data_dir, f"{SUB_IDs[0]}_X_shape.txt"), "w") as f:
    #     f.write(shape_info)
    
    print("Save Complete.")
    print(f"Saved Tensor Shape: {final_X.shape}")
    print(f"Files created:\n - {output_path_pt}\n - {output_path_npy}")






