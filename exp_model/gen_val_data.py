"""Script to generate a fixed global validation set."""
import os
import multiprocessing
import torch
import config
from generate_data import generate_data_batch_parallel
from data_utils import compress_data_mpv

VAL_SET_SIZE = 2000
VAL_FILE_PATH = config.GLOBAL_VAL_FILE

def generate_global_val():
    print(f"--- Generating Global Validation Set ({VAL_SET_SIZE} games) ---")
    
    if os.path.exists(VAL_FILE_PATH):
        print(f"Validation file already exists at {VAL_FILE_PATH}. Overwriting...")

    # Temporary override for generation size
    old_size = config.DATA_BATCH_SIZE
    config.DATA_BATCH_SIZE = VAL_SET_SIZE
    
    try:
        # Generate data (pass dummy cycle_id 0)
        raw_data = generate_data_batch_parallel(0)
        
        # Compress
        print("Compressing validation data...")
        packed_data = compress_data_mpv(raw_data)
        
        # Save
        print(f"Saving to {VAL_FILE_PATH}")
        torch.save(packed_data, VAL_FILE_PATH)
        print("Done!")
    finally:
        config.DATA_BATCH_SIZE = old_size

if __name__ == "__main__":
    multiprocessing.freeze_support()
    generate_global_val()
