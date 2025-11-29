"""
Created: Nov24,2025
RUNS ON:
module load python3/anaconda/2023.1
module load cuda/12.1

source activate /work/nayeem/ENV/torch_cu12

"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model

NL_Dataset_words_dir = "/work/nayeem/Huth_deepfMRI/derivatives/"

df_path = os.path.join(NL_Dataset_words_dir, 'Huth_fMRI_25_ordered_stories_SCOPEmerged_Spacy_tagged_trimmed_df.csv')
dataframe = pd.read_csv(df_path)
print(dataframe)


# --- 1. Setup Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
model.to(device)
model.eval()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# --- 2. Define Extraction Function ---
def get_gpt2_layer_embeddings(word_list, context_len=100, batch_size=8):
    """
    Returns a list of numpy arrays. Each array has shape (13, 768).
    """
    all_embeddings = []
    prompts = []
    current_text_accum = []
    
    # Build prompts with sliding history
    for word in word_list:
        current_text_accum.append(str(word))
        full_prompt = " ".join(current_text_accum)
        prompts.append(full_prompt)

    # Batch processing
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=context_len
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Stack layers -> (13, batch, seq, 768)
        stacked_layers = torch.stack(outputs.hidden_states)
        
        # Get last token -> (13, batch, 768)
        last_token_embeddings = stacked_layers[:, :, -1, :]
        
        # Permute to (batch, 13, 768)
        batch_embeddings = last_token_embeddings.permute(1, 0, 2).cpu().numpy()
        all_embeddings.extend(list(batch_embeddings))
        
    return all_embeddings

# --- 3. Initialize Columns ---
# GPT-2 base has 12 layers + 1 embedding layer = 13 outputs
layer_names = [f"gpt2_layer_{i}" for i in range(13)]

# Initialize columns with None to avoid PerformanceWarnings
for col in layer_names:
    dataframe[col] = None

# --- 4. Main Loop ---
unique_stories = dataframe['story'].unique()

for story in tqdm(unique_stories, desc="Processing Stories"):
    
    # Get indices and words for this story
    story_indices = dataframe[dataframe['story'] == story].index
    words = dataframe.loc[story_indices, 'word'].tolist()
    
    # Extract (returns list of (13, 768) arrays)
    # Adjust context_len as needed (1024 is max, 20-50 is common for sentence-level)
    raw_embeddings = get_gpt2_layer_embeddings(words, context_len=256, batch_size=16)
    
    # --- Reshape for Columns ---
    # Convert list of arrays to a big numpy block: Shape (N_words, 13, 768)
    embeddings_block = np.stack(raw_embeddings)
    
    # Distribute into columns
    for layer_idx in range(13):
        col_name = f"gpt2_layer_{layer_idx}"
        
        # Slice the specific layer: Shape (N_words, 768)
        layer_vectors = embeddings_block[:, layer_idx, :]
        
        # Convert back to list of arrays for DataFrame storage
        # We use list comprehension so each cell holds a (768,) array
        vector_list = [row for row in layer_vectors]
        
        # Assign to dataframe using pd.Series to align indices correctly
        dataframe.loc[story_indices, col_name] = pd.Series(vector_list, index=story_indices)

# --- 5. Review and Save ---
print("\nExtraction Complete.")
print(dataframe)
print(dataframe.columns)

# Check one cell
print(f"Shape of layer 12 embedding for first word: {dataframe.iloc[0]['gpt2_layer_12'].shape}")
# Expected: (768,)

# Save to pickle (Essential for preserving array objects)
save_dir = os.path.join(NL_Dataset_words_dir, 'GPT2')
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'Huth_fMRI_GPT2_Separated_Layers.pkl')
dataframe.to_pickle(save_path)