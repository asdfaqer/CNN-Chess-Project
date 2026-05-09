import os
import random
import torch
from tqdm import tqdm
import config
from data_utils import load_batch, decompress_data, compress_data_mpv, save_batch

def regularize_batches():
    """
    Perform a global shuffle of all games across all batches to ensure
    maximum regularity and diversity in the training data.
    """
    print(f"=== Batch Regularization: Global Shuffle ===")
    
    # 1. Find all relevant batch files
    cache_dir = config.DATA_CACHE_DIR
    all_files = [f for f in os.listdir(cache_dir) 
                 if f.startswith("batch_mpv_") and f.endswith(".zst")]
    
    if not all_files:
        print(f"No batch files found in {cache_dir}.")
        return

    print(f"Found {len(all_files)} batches. Loading all games into memory...")
    all_games = []
    
    # Load all batches
    for filename in tqdm(all_files, desc="Loading"):
        path = os.path.join(cache_dir, filename)
        try:
            packed = load_batch(path)
            games = decompress_data(packed)
            all_games.extend(games)
            # Free memory as we go
            del packed
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    total_games = len(all_games)
    print(f"\nLoaded total of {total_games} games.")
    
    # 2. Global Shuffle
    print("Shuffling games globally...")
    random.shuffle(all_games)
    
    # 3. Redistribute and Save
    print("Redistributing games back to batches and saving...")
    num_batches = len(all_files)
    games_per_batch = total_games // num_batches
    
    for i, filename in enumerate(tqdm(all_files, desc="Saving")):
        start_idx = i * games_per_batch
        # Take everything remaining for the last batch
        end_idx = (i + 1) * games_per_batch if i < num_batches - 1 else total_games
        
        batch_subset = all_games[start_idx:end_idx]
        
        try:
            packed = compress_data_mpv(batch_subset)
            path = os.path.join(cache_dir, filename)
            save_batch(packed, path)
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            
    print("\nRegularization complete! All batches now contain a random mix of games.")

if __name__ == "__main__":
    regularize_batches()
