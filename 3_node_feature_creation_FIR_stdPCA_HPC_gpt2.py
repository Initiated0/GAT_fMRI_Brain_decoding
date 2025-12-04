"""
creation date: Dec 4, 2025

To run this script you need

module load python3/anaconda/2023.1
module load cuda/12.1
source activate /work/nayeem/ENV/torch_cuda12
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
# import torch
# import torch.nn as nn
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Running on: {device}")

NUM_TRS = 8968
NUM_NODES = 423
WINDOW_SIZE = 20
D_FMRI = 32
D_WORD = 300
BATCH_SIZE = 32  # Process 32 timepoints at once (Adjust up to 64/128 if VRAM allows)

# FIR lags (in TRs): previous 2–4 TRs -> 4–8 seconds (TR = 2s)
LAGS = [2, 3, 4]
num_lags = len(LAGS)
D_WORD_FIR = D_WORD * num_lags  # 300 * 3 = 900

# "/work/nayeem/Huth_deepfMRI/processed_data_transformed_atlas_float32/"
root_dir = "/work/nayeem/Huth_deepfMRI/"
# Paths
fMRI_BOLD_data_dir = os.path.join(root_dir, "processed_data_transformed_atlas_float32")
SUB_IDs = ["UTS01"]  # adjust if you want more subjects

"""
Pipeline for word embeddings:
  1) Load fastText embeddings per TR (300D)
  2) FIR on lags [2,3,4] -> (NUM_TRS, 900)
  3) Standardize FIR matrix
  4) PCA (90% variance) -> (NUM_TRS, D_PCA)
  5) Use PCA-FIR features as word features per TR (D_PCA)
"""

# =========================
# Load and process word embeddings (same for all subjects)
# =========================
# fasttext_csv = os.path.join(
#     root_dir,
#     "derivatives/Huth_fMRI_25_ordered_stories_SCOPEmerged_Spacy_tagged_trimmed_fasttext_df_TR-levelFinal.csv"
# )

gpt_embed_pkl = os.path.join(root_dir, 'derivatives/GPT2/Huth_fMRI_GPT2_Separated_Layers.pkl')


with open(gpt_embed_pkl, "rb") as f:
    data = pickle.load(f)

print("Loaded word embedding dataframe:")
print(data)
print(data.columns)
print(data.shape)

# ---- Define layer groups (bins) ----
bins = {
    "lexical_bin0":      ["gpt2_layer_0"],                      # Non-contextual lexical
    "syntax_bin1":       ["gpt2_layer_1", "gpt2_layer_2", "gpt2_layer_3"],
    "semantic_bin2":     ["gpt2_layer_4", "gpt2_layer_5", "gpt2_layer_6", "gpt2_layer_7", "gpt2_layer_8"],
    "discourse_bin3":    ["gpt2_layer_9", "gpt2_layer_10", "gpt2_layer_11", "gpt2_layer_12"]
}

# ---- Function to average lists of arrays ----
def average_embeddings(row, cols):
    vectors = [row[col] for col in cols]
    return np.mean(vectors, axis=0)

# ---- Create new columns ----
for bin_name, columns in bins.items():
    data[bin_name] = data.apply(lambda row: average_embeddings(row, columns), axis=1)

print(data.head())
print(data.columns)

# def clean_embedding_string(x):
#     """
#     Parses a string representation of a numpy array back into a numpy array.
#     Handles both comma-separated '[0.1, 0.2]' and space-separated '[0.1 0.2]'.
#     """
#     if isinstance(x, str):
#         clean_str = x.replace('[', '').replace(']', '').replace('\n', ' ')
#         if ',' in clean_str:
#             return np.fromstring(clean_str, sep=',')
#         else:
#             return np.fromstring(clean_str, sep=' ')
#     return x

# print("Parsing embedding strings...")
# df['embedding'] = df['embedding'].apply(clean_embedding_string)

# print(f"Type after fix: {type(df.at[0, 'embedding'])}")  # Should be <class 'numpy.ndarray'>
# print(f"Shape of one embedding: {df.at[0, 'embedding'].shape}")  # Should be (300,)

# # (NUM_TRS, 300)
# word_np = np.stack(df['embedding'].values).astype(np.float32)
# assert word_np.shape == (NUM_TRS, D_WORD), f"Expected ({NUM_TRS}, {D_WORD}), got {word_np.shape}"

# # =========================
# # FIR on word embeddings (before PCA)
# # =========================
# print("Building FIR matrix for word embeddings...")
# word_fir_np = np.zeros((NUM_TRS, D_WORD_FIR), dtype=np.float32)  # (NUM_TRS, 900)

# for i, lag in enumerate(LAGS):
#     pad = np.zeros((lag, D_WORD), dtype=np.float32)
#     shifted = np.concatenate([pad, word_np[:-lag]], axis=0)  # (NUM_TRS, 300)

#     start = i * D_WORD
#     end = (i + 1) * D_WORD
#     word_fir_np[:, start:end] = shifted

# # =========================
# # Standardize + PCA(90%) on FIR matrix
# # =========================
# print("Standardizing FIR features...")
# scaler = StandardScaler(with_mean=True, with_std=True)
# word_fir_std = scaler.fit_transform(word_fir_np)  # (NUM_TRS, 900)

# print("Running PCA on FIR features (keep 90% variance)...")
# pca = PCA(n_components=0.90, svd_solver="full")
# word_fir_pca = pca.fit_transform(word_fir_std)  # (NUM_TRS, D_PCA)

# D_PCA = word_fir_pca.shape[1]
# print(f"FIR PCA reduced from {D_WORD_FIR} -> {D_PCA} dims (90% variance)")

# # Convert to Torch tensor for later
# word_fir_pca_tensor = torch.tensor(word_fir_pca, dtype=torch.float32)  # (NUM_TRS, D_PCA)

# # Total feature dim per node = fMRI(32) + PCA(FIR(word)) (D_PCA)
# D_TOTAL = D_FMRI + D_PCA
# print(f"Final feature dim per node: {D_TOTAL} = {D_FMRI} (fMRI) + {D_PCA} (PCA-FIR word)")

# # =====================================================================
# # Process each subject: fMRI CNN encoding + expand PCA-FIR word features
# # =====================================================================
# for sub_id in SUB_IDs:
#     print(f"\n========== Processing subject: {sub_id} ==========\n")

#     # -----------------------------
#     # 1. Load and fix fMRI data
#     # -----------------------------
#     fMRI_data_path = os.path.join(fMRI_BOLD_data_dir, f"{sub_id}_combined.npy")
#     fMRI_data = np.load(fMRI_data_path)
#     print(f"Before removing TRs: {fMRI_data.shape}")

#     # Remove six TRs
#     TRs_to_remove = [1340, 2138, 2842, 2843, 3688, 8676]
#     fmri_fixed = np.delete(fMRI_data, TRs_to_remove, axis=0)
#     print(f"After removing TRs: {fmri_fixed.shape}")   # should be (8968, 423)
#     assert fmri_fixed.shape == (NUM_TRS, NUM_NODES)

#     fmri_tensor = torch.tensor(fmri_fixed, dtype=torch.float32)

#     # PAD the fMRI data at the start: (WINDOW_SIZE - 1) rows of zeros
#     padding = torch.zeros(WINDOW_SIZE - 1, NUM_NODES)
#     fmri_padded = torch.cat([padding, fmri_tensor], dim=0)  # shape: (NUM_TRS + WINDOW_SIZE - 1, 423)

#     # -----------------------------
#     # 2. Allocate final_X
#     # -----------------------------
#     final_X = torch.zeros(NUM_TRS, NUM_NODES, D_TOTAL, dtype=torch.float32)

#     # -----------------------------
#     # 3. TimeSeriesEncoder model (CNN over fMRI windows)
#     # -----------------------------
#     class TimeSeriesEncoder(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
#             self.fc = nn.Linear(16 * (WINDOW_SIZE - 4), D_FMRI)

#         def forward(self, x):
#             # x input: (Batch_Size * Nodes, 1, Window_Size)
#             x = self.conv1(x)
#             x = torch.relu(x)
#             x = x.view(x.size(0), -1)
#             x = self.fc(x)
#             return x

#     encoder = TimeSeriesEncoder().to(device)  # Move model to GPU

#     # -----------------------------
#     # 4. Batched processing
#     # -----------------------------
#     print("Starting Batch Processing...")

#     for i in tqdm(range(0, NUM_TRS, BATCH_SIZE), desc=f"Processing Batches ({sub_id})"):
#         start_idx = i
#         end_idx = min(i + BATCH_SIZE, NUM_TRS)
#         current_batch_size = end_idx - start_idx

#         # --- A. Build fMRI windows for this batch ---
#         batch_windows = []
#         for t in range(start_idx, end_idx):
#             # Slice: (Window, Nodes) from padded
#             window = fmri_padded[t : t + WINDOW_SIZE, :]  # (Window, 423)
#             batch_windows.append(window.T)                # -> (Nodes, Window)

#         # (Batch_Size, Nodes, Window)
#         batch_windows = torch.stack(batch_windows)

#         # Reshape for CNN: (Batch_Size * Nodes, 1, Window)
#         cnn_input = batch_windows.view(current_batch_size * NUM_NODES, 1, WINDOW_SIZE).to(device)

#         # --- B. Encode fMRI with CNN (no-grad) ---
#         with torch.no_grad():
#             encoded_flat = encoder(cnn_input)  # (Batch_Size * Nodes, 32)
#             fmri_batch_emb = encoded_flat.view(current_batch_size, NUM_NODES, D_FMRI)  # (B, 423, 32)

#         # --- C. Word PCA-FIR features for this batch ---
#         # word_fir_pca_tensor: (NUM_TRS, D_PCA)
#         word_batch = word_fir_pca_tensor[start_idx:end_idx].to(device)  # (B, D_PCA)

#         # Expand across nodes: (B, 1, D_PCA) -> (B, 423, D_PCA)
#         word_batch_expanded = word_batch.unsqueeze(1).expand(-1, NUM_NODES, -1)

#         # --- D. Concatenate fMRI(32) and word(PCA-FIR) ---
#         # (B, 423, 32 + D_PCA) = (B, 423, D_TOTAL)
#         batch_features = torch.cat([fmri_batch_emb, word_batch_expanded], dim=2)

#         # --- E. Store in final_X (on CPU) ---
#         final_X[start_idx:end_idx] = batch_features.cpu()

#     print("Done!")
#     print("Final X shape:", final_X.shape)

#     # -----------------------------
#     # 5. Save final_X
#     # -----------------------------
#     output_path_pt = os.path.join(fMRI_BOLD_data_dir, f"{sub_id}_node_features_X_FIR_stdPCA.pt")
#     output_path_npy = os.path.join(fMRI_BOLD_data_dir, f"{sub_id}_node_features_X_FIR_stdPCA.npy")

#     print(f"Saving data to: {fMRI_BOLD_data_dir} ...")
#     torch.save(final_X, output_path_pt)
#     np.save(output_path_npy, final_X.numpy())

#     print("Save Complete.")
#     print(f"Saved Tensor Shape: {final_X.shape}")
#     print(f"Files created:\n - {output_path_pt}\n - {output_path_npy}")
