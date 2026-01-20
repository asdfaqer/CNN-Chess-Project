import os
import torch
import numpy as np
import zstandard as zstd
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc

from config import (DEVICE, DATA_CHANNELS, USE_HISTORY, MINIBATCH_SIZE, 
                   DATA_CACHE_DIR, EV_TANH_K, BASE_CHANNELS)
from data_utils import load_batch, decompress_data, PositionDataset, collate_fn, save_batch, compress_data_mpv
from model import ChessRCCN
import multiprocessing

# Configuration for filtering
MODEL_PATH = os.path.join("checkpoints", "epoch_7.pt")
FILTERED_DATA_DIR = "generated_data_filtered"
IMPORTANCE_THRESHOLD = 0.2  # EV gap threshold
BATCH_SIZE = 4096 # Prediction batch size

def scores_to_win_prob(cp, mate, k=EV_TANH_K):
    """Convert raw cp and mate scores to win probability [0, 1]."""
    cp_eff = cp.float()
    mate_mask = mate != 0
    if mate_mask.any():
        mate_scores = torch.where(mate > 0, 10000.0 - mate.float(), -10000.0 - mate.float())
        cp_eff = torch.where(mate_mask, mate_scores, cp_eff)
    return 0.5 + 0.5 * torch.tanh(cp_eff / k)

def compute_importance(logits, mpv_data):
    """Compute EV gap importance."""
    # mpv_data: [cp, mate, depth, moves]
    mpv_cp, mpv_mate, mpv_depth, mpv_moves = mpv_data
    mpv_scores = scores_to_win_prob(mpv_cp, mpv_mate)
    
    # ev_gt: Stockfish's top move EV (-1 to 1)
    ev_gt = 2.0 * mpv_scores[:, 0] - 1.0
    
    # ev_pred: Model's expected EV based on its policy
    probs = torch.softmax(logits, dim=1)
    
    mask = mpv_moves != -1
    safe_moves = mpv_moves.clone()
    safe_moves[~mask] = 0
    
    n_probs = torch.gather(probs, 1, safe_moves)
    n_probs = n_probs * mask.float()
    
    win_prob_pred = torch.sum(n_probs * mpv_scores, dim=1)
    ev_pred = 2.0 * win_prob_pred - 1.0
    
    importance = torch.clamp(ev_gt - ev_pred, min=0.0)
    return importance

def process_batch(batch_path, model, device):
    """Process a single batch, filtering low-importance positions."""
    # Load and decompress
    packed_b = load_batch(batch_path)
    games = decompress_data(packed_b)
    
    # Flatten games into positions for model evaluation
    ds = PositionDataset(games, use_history=USE_HISTORY)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    all_importances = []
    
    model.eval()
    with torch.no_grad():
        total_batches = len(loader)
        for i, batch in enumerate(loader):
            inputs = batch[0].to(device)
            mpv_data = [b.to(device) for b in batch[3:]]
            
            outputs, _ = model(inputs.float())
            logits = outputs.view(-1, 4672)
            
            importance = compute_importance(logits, mpv_data)
            all_importances.append(importance.cpu().numpy())
            
            if (i + 1) % 50 == 0:
                print(f"    - Sub-batch {i+1}/{total_batches} processed...")
            
    importances = np.concatenate(all_importances)
    
    # Reconstruct games but filter positions
    filtered_games = []
    pos_idx = 0
    total_original = 0
    total_kept = 0
    
    for game in games:
        # game: (states, moves, weights, cp, mate, depth, moves_mpv)
        num_moves = len(game[1])
        total_original += num_moves
        
        game_importances = importances[pos_idx : pos_idx + num_moves]
        mask = game_importances >= IMPORTANCE_THRESHOLD
        
        if mask.any():
            # Keep only positions above threshold
            new_states = game[0][mask]
            new_moves = game[1][mask]
            new_weights = game[2][mask]
            
            # Special handling for mpv data which are indices 3+
            new_mpv = [field[mask] for field in game[3:]]
            
            filtered_games.append(tuple([new_states, new_moves, new_weights] + new_mpv))
            total_kept += mask.sum()
            
        pos_idx += num_moves
        
    return filtered_games, total_original, total_kept

def main():
    multiprocessing.set_start_method('spawn', force=True)
    if not os.path.exists(FILTERED_DATA_DIR):
        os.makedirs(FILTERED_DATA_DIR)
        
    print(f"Loading model from {MODEL_PATH}...")
    model = ChessRCCN(hidden_dim=64, use_lstm=False, input_channels=DATA_CHANNELS).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    all_files = sorted([f for f in os.listdir(DATA_CACHE_DIR) 
                      if f.startswith("batch_mpv_") and f.endswith(".zst")])
    
    total_original_all = 0
    total_kept_all = 0
    
    pbar = tqdm(all_files, desc="Filtering Batches")
    for filename in pbar:
        batch_path = os.path.join(DATA_CACHE_DIR, filename)
        filtered_games, n_orig, n_kept = process_batch(batch_path, model, DEVICE)
        
        total_original_all += n_orig
        total_kept_all += n_kept
        
        if filtered_games:
            out_path = os.path.join(FILTERED_DATA_DIR, filename)
            packed_filtered = compress_data_mpv(filtered_games)
            save_batch(packed_filtered, out_path, use_zstd=True)
            pct = (n_kept / n_orig * 100) if n_orig > 0 else 0
            print(f"  ✓ Saved {filename} ({n_kept}/{n_orig} positions, {pct:.1f}%)")
        pbar.set_postfix({"Kept": f"{pct:.1f}%"})
        
    overall_pct = (total_kept_all / total_original_all * 100) if total_original_all > 0 else 0
    print(f"\nFiltering Complete!")
    print(f"Total positions: {total_original_all}")
    print(f"Kept positions:  {total_kept_all}")
    print(f"Overall Kept:    {overall_pct:.2f}%")
    print(f"Filtered out:    {100 - overall_pct:.2f}%")

if __name__ == "__main__":
    main()
