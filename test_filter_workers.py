import os
import torch
import numpy as np
import time
from config import (DEVICE, DATA_CHANNELS, USE_HISTORY, DATA_CACHE_DIR, BASE_CHANNELS)
from data_utils import load_batch, decompress_data, PositionDataset, collate_fn
from model import ChessRCCN
from torch.utils.data import DataLoader
import multiprocessing

MODEL_PATH = os.path.join("checkpoints", "epoch_7.pt")
BATCH_SIZE = 4096

def main():
    multiprocessing.set_start_method('spawn', force=True)
    print(f"Loading model on {DEVICE}...")
    model = ChessRCCN(hidden_dim=64, use_lstm=False, input_channels=DATA_CHANNELS).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_files = sorted([f for f in os.listdir(DATA_CACHE_DIR) 
                      if f.startswith("batch_mpv_") and f.endswith(".zst")])
    
    if not all_files:
        print("No batches found.")
        return

    test_file = all_files[0]
    batch_path = os.path.join(DATA_CACHE_DIR, test_file)
    
    print(f"Loading {test_file}...")
    packed_b = load_batch(batch_path)
    
    print("Decompressing...")
    games = decompress_data(packed_b)
    
    ds = PositionDataset(games, use_history=USE_HISTORY)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    print(f"Total positions: {len(ds)} in {len(loader)} batches")
    
    print("Evaluating...")
    start_time = time.time()
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch[0].to(DEVICE)
            outputs, _ = model(inputs.float())
            count += inputs.size(0)
            if i % 10 == 0:
                print(f"  Processed {count}/{len(ds)}...")
            if i > 50: break # Just test a bit
            
    eval_time = time.time() - start_time
    print(f"Evaluated subset in {eval_time:.2f}s ({count / eval_time:.1f} pos/s)")

if __name__ == "__main__":
    main()
