import os
import io
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
from config import BASE_CHANNELS, USE_HISTORY, HISTORY_FILE, PLOT_FILE, USE_ZSTD

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

from utils import stack_history

# --- Zstandard I/O Utilities ---

def save_batch(data: Dict[str, np.ndarray], path: str, use_zstd: bool = USE_ZSTD) -> None:
    """Save a compressed data batch. Uses zstd if enabled and available."""
    if use_zstd and ZSTD_AVAILABLE:
        # Serialize with torch to bytes, then compress
        buffer = io.BytesIO()
        torch.save(data, buffer)
        raw_bytes = buffer.getvalue()
        
        cctx = zstd.ZstdCompressor(level=3)  # Level 3 = good balance speed/ratio
        compressed = cctx.compress(raw_bytes)
        
        # Save with .zst extension (avoid double extension)
        if path.endswith(".zst"):
            zst_path = path
        else:
            zst_path = path.replace(".pt", ".zst") if path.endswith(".pt") else path + ".zst"
        
        with open(zst_path, "wb") as f:
            f.write(compressed)
    else:
        torch.save(data, path)

def load_batch(path: str) -> Dict[str, np.ndarray]:
    """Load a data batch. Auto-detects zstd vs legacy .pt format.
    
    Looks for files in this order:
    1. batch_mpv_X.zst (new multi-PV format)
    2. batch_X.zst (zstd compressed)
    3. batch_X.pt (legacy torch)
    """
    # Extract batch ID and construct possible paths
    basename = os.path.basename(path)
    batch_id = basename.replace("batch_", "").replace("batch_mpv_", "").replace(".pt", "").replace(".zst", "")
    dir_path = os.path.dirname(path) or DATA_CACHE_DIR
    
    mpv_zst = os.path.join(dir_path, f"batch_mpv_{batch_id}.zst")
    zst_path = os.path.join(dir_path, f"batch_{batch_id}.zst")
    pt_path = os.path.join(dir_path, f"batch_{batch_id}.pt")
    
    # Try mpv format first (most space efficient, has multi-PV data)
    if os.path.exists(mpv_zst) and ZSTD_AVAILABLE:
        with open(mpv_zst, "rb") as f:
            compressed = f.read()
        dctx = zstd.ZstdDecompressor()
        raw_bytes = dctx.decompress(compressed)
        buffer = io.BytesIO(raw_bytes)
        return torch.load(buffer, weights_only=False)
    
    # Try regular zst
    if os.path.exists(zst_path) and ZSTD_AVAILABLE:
        with open(zst_path, "rb") as f:
            compressed = f.read()
        dctx = zstd.ZstdDecompressor()
        raw_bytes = dctx.decompress(compressed)
        buffer = io.BytesIO(raw_bytes)
        return torch.load(buffer, weights_only=False)
    
    # Fallback to legacy .pt
    if os.path.exists(pt_path):
        return torch.load(pt_path, weights_only=False)
    
    raise FileNotFoundError(f"No batch file found for ID {batch_id} in {dir_path}")





def compress_data(raw_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    all_states = []
    all_moves = []
    all_weights = []
    game_boundaries = [0]
    
    for item in raw_data:
        if len(item) == 3:
            states, moves, weights = item
        else:
            states, moves = item
            weights = np.ones(len(moves), dtype=np.float32)
        packed = np.packbits(states.astype(bool).reshape(-1))
        all_states.append(packed)
        all_moves.append(moves.astype(np.uint16))
        all_weights.append(weights.astype(np.float16))
        game_boundaries.append(game_boundaries[-1] + len(moves))
        
    return {
        "states": np.concatenate(all_states),
        "moves": np.concatenate(all_moves),
        "weights": np.concatenate(all_weights),
        "boundaries": np.array(game_boundaries, dtype=np.int32)
    }

def compress_data_mpv(raw_data: List[Tuple]) -> Dict[str, np.ndarray]:
    all_states = []
    all_moves = []
    all_weights = []
    all_mpv_cp = []
    all_mpv_mate = []
    all_mpv_depth = []
    all_mpv_moves = []
    game_boundaries = [0]
    
    for item in raw_data:
        # Expected structure: (states, moves, weights, mpv_cp, mpv_mate, mpv_depth, mpv_moves)
        if len(item) == 7:
            states, moves, weights, mpv_cp, mpv_mate, mpv_depth, mpv_moves = item
        elif len(item) == 5:
            # Legacy mpv format: (states, moves, weights, mpv_scores, mpv_moves)
            states, moves, weights, mpv_scores, mpv_moves = item
            mpv_cp = (mpv_scores * 0).astype(np.int16) # Dummy
            mpv_mate = (mpv_scores * 0).astype(np.int16)
            mpv_depth = (mpv_scores * 0).astype(np.uint8)
        else:
            states, moves, weights = item[:3]
            mpv_cp = np.zeros((len(moves), 1), dtype=np.int16)
            mpv_mate = np.zeros((len(moves), 1), dtype=np.int16)
            mpv_depth = np.zeros((len(moves), 1), dtype=np.uint8)
            mpv_moves = np.full((len(moves), 1), -1, dtype=np.int16)

        packed = np.packbits(states.astype(bool).reshape(-1))
        all_states.append(packed)
        all_moves.append(moves.astype(np.uint16))
        all_weights.append(weights.astype(np.float16))
        all_mpv_cp.append(mpv_cp.astype(np.int16))
        all_mpv_mate.append(mpv_mate.astype(np.int16))
        all_mpv_depth.append(mpv_depth.astype(np.uint8))
        all_mpv_moves.append(mpv_moves.astype(np.int16))
        game_boundaries.append(game_boundaries[-1] + len(moves))
        
    return {
        "states": np.concatenate(all_states),
        "moves": np.concatenate(all_moves),
        "weights": np.concatenate(all_weights),
        "mpv_cp": np.concatenate(all_mpv_cp),
        "mpv_mate": np.concatenate(all_mpv_mate),
        "mpv_depth": np.concatenate(all_mpv_depth),
        "mpv_moves": np.concatenate(all_mpv_moves),
        "boundaries": np.array(game_boundaries, dtype=np.int32)
    }

def decompress_data(packed_dict: Dict[str, np.ndarray], target_channels: int = BASE_CHANNELS) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    flat_bits = np.unpackbits(packed_dict["states"])
    num_bits = flat_bits.size
    boundaries = packed_dict["boundaries"]
    
    # Stay in uint8 to save 4x memory
    if num_bits % (target_channels * 64) == 0:
        raw_states = flat_bits.reshape(-1, target_channels, 8, 8)
    elif num_bits % (17 * 64) == 0:
        raw_legacy = flat_bits.reshape(-1, 17, 8, 8)
        raw_states = np.zeros((raw_legacy.shape[0], target_channels, 8, 8), dtype=np.uint8)
        raw_states[:, :17, :, :] = raw_legacy
        
        for g_idx in range(len(boundaries) - 1):
            start, end = boundaries[g_idx], boundaries[g_idx+1]
            clock = 0
            for i in range(start, end):
                if i > start:
                    pieces_now = np.sum(raw_legacy[i, :12])
                    pieces_prev = np.sum(raw_legacy[i-1, :12])
                    
                    prev_us_pawns = raw_legacy[i-1, 0]
                    prev_them_pawns = raw_legacy[i-1, 6]
                    now_us_pawns = raw_legacy[i, 0]
                    now_them_pawns = raw_legacy[i, 6]
                    
                    p_us_mirrored = np.flip(prev_us_pawns, axis=0)
                    p_them_mirrored = np.flip(prev_them_pawns, axis=0)
                    
                    pawn_reset = not (np.array_equal(now_us_pawns, p_them_mirrored) and 
                                    np.array_equal(now_them_pawns, p_us_mirrored))
                    
                    if pieces_now < pieces_prev or pawn_reset:
                        clock = 0
                    else:
                        clock += 1
                if clock >= 98:
                    raw_states[i, 17, :, :] = 1
    else:
        raise ValueError(f"Cannot reshape bits ({num_bits}) into 17 or {target_channels} channels.")

    raw_moves = packed_dict["moves"].astype(np.int64)
    
    if "weights" in packed_dict:
        raw_weights = packed_dict["weights"].astype(np.float32)
    else:
        raise ValueError("Importance weights missing from data batch.")
    
    mpv_data = {}
    if "mpv_cp" in packed_dict:
        mpv_data["cp"] = packed_dict["mpv_cp"].astype(np.int16)
        mpv_data["mate"] = packed_dict["mpv_mate"].astype(np.int16)
        mpv_data["depth"] = packed_dict["mpv_depth"].astype(np.uint8)
        mpv_data["moves"] = packed_dict["mpv_moves"].astype(np.int64)
    elif "mpv_scores" in packed_dict:
        # Legacy fallback
        mpv_data["scores"] = packed_dict["mpv_scores"].astype(np.float32)
        if "mpv_moves" in packed_dict:
            mpv_data["moves"] = packed_dict["mpv_moves"].astype(np.int64)
    
    decompressed = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        chunk = [raw_states[start:end], raw_moves[start:end], raw_weights[start:end]]
        if mpv_data:
            for k in mpv_data:
                chunk.append(mpv_data[k][start:end])
        decompressed.append(tuple(chunk))
        
    return decompressed

class PositionDataset(Dataset):
    """Dataset that treats every move as a single sample (flattened games).
    
    Uses lazy evaluation for history stacking to avoid slow startup times.
    History is stacked on-demand in __getitem__ and parallelized by DataLoader workers.
    """
    def __init__(self, games: List[Tuple], use_history: bool = False):
        self.use_history = use_history
        self.samples = []  # (game_idx, move_idx) tuples for lazy lookup
        self.games = games  # Keep reference for lazy access
        
        # Build index mapping (instant - just stores indices)
        for game_idx, game in enumerate(games):
            moves = game[1]
            for move_idx in range(len(moves)):
                self.samples.append((game_idx, move_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        game_idx, move_idx = self.samples[idx]
        item = self.games[game_idx]
        states, moves, weights = item[0], item[1], item[2]
        
        if self.use_history:
            state = stack_history(states, move_idx)
        else:
            state = states[move_idx]
        
        result = [
            torch.from_numpy(state.astype(np.float32)), 
            torch.as_tensor(moves[move_idx], dtype=torch.long), 
            torch.as_tensor(weights[move_idx], dtype=torch.float32)
        ]
        
        if len(item) > 3:
            # item structure matches decompressed chunk: index 3 onwards are the keys in mpv_data
            for j in range(3, len(item)):
                result.append(torch.as_tensor(item[j][move_idx]))
            
        return tuple(result)


class ChessDataset(Dataset):
    """Legacy dataset that returns full game sequences."""
    def __init__(self, games: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], use_history: bool = False):
        self.games = games
        self.use_history = use_history

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        item = self.games[idx]
        states, moves, weights = item[0], item[1], item[2]
            
        if self.use_history:
            history_states = []
            for i in range(len(moves)):
                history_states.append(stack_history(states, i))
            res = [torch.from_numpy(np.array(history_states)), torch.as_tensor(moves), torch.as_tensor(weights)]
        else:
            res = [torch.from_numpy(states), torch.as_tensor(moves), torch.as_tensor(weights)]
        
        if len(item) > 3:
            res.append(torch.as_tensor(item[3], dtype=torch.float32))
            res.append(torch.as_tensor(item[4], dtype=torch.long))
            
        return tuple(res)

def collate_fn(batch: List[Tuple]) -> Tuple:
    has_mpv = len(batch[0]) > 3
    
    max_len = max(s.size(0) for s in batch[0][0]) if batch[0][0].dim() == 4 else 1
    batch_size = len(batch)
    channels = batch[0][0].size(1) if batch[0][0].dim() > 3 else batch[0][0].size(0)
    
    res_list = []
    
    # 0: states, 1: moves, 2: weights
    if batch[0][0].dim() == 4:
        padded_states = torch.zeros(batch_size, max_len, channels, 8, 8, dtype=torch.float32)
        padded_moves = torch.full((batch_size, max_len), -1, dtype=torch.long)
        padded_weights = torch.zeros(batch_size, max_len, dtype=torch.float32)
    else:
        padded_states = torch.zeros(batch_size, channels, 8, 8, dtype=torch.float32)
        padded_moves = torch.full((batch_size,), -1, dtype=torch.long)
        padded_weights = torch.zeros(batch_size, dtype=torch.float32)
        
    res_list = [padded_states, padded_moves, padded_weights]
    
    if has_mpv:
        # Dynamically create tensors for extra fields
        for j in range(3, len(batch[0])):
            example = batch[0][j]
            if batch[0][0].dim() == 4:
                # [batch, seq, ...]
                shape = (batch_size, max_len) + example.shape[1:]
                fill = -1 if example.dtype in [torch.long, torch.int, torch.int16] else 0
                res_list.append(torch.full(shape, fill, dtype=example.dtype))
            else:
                # [batch, ...]
                shape = (batch_size,) + example.shape
                fill = -1 if example.dtype in [torch.long, torch.int, torch.int16] else 0
                res_list.append(torch.full(shape, fill, dtype=example.dtype))

    for i in range(batch_size):
        item = batch[i]
        s, m, w = item[0], item[1], item[2]
        if s.dim() == 4: # Full game sequence
            l = s.size(0)
            padded_states[i, :l].copy_(s)
            padded_moves[i, :l].copy_(m)
            padded_weights[i, :l].copy_(w)
            if has_mpv:
                for j in range(3, len(item)):
                    res_list[j][i, :l].copy_(item[j])
        else: # Single position
            padded_states[i].copy_(s)
            padded_moves[i].copy_(m)
            padded_weights[i].copy_(w)
            if has_mpv:
                for j in range(3, len(item)):
                    res_list[j][i].copy_(item[j])
        
    return tuple(res_list)

def simple_collate(batch):
    if len(batch[0]) == 3:
        s, m, w = zip(*batch)
        return torch.from_numpy(np.stack(s)), torch.from_numpy(np.stack(m)), torch.from_numpy(np.stack(w))
    else:
        s, m = zip(*batch)
        return torch.from_numpy(np.stack(s)), torch.from_numpy(np.stack(m))

def save_history(history: List[Dict]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_history() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load history: {e}")
    return []

def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Find the latest epoch_*.pt file in save_dir."""
    if not os.path.exists(save_dir):
        return None
    files = [f for f in os.listdir(save_dir) if f.startswith("epoch_") and f.endswith(".pt")]
    if not files:
        return None
    # Sort by epoch number: epoch_10.pt > epoch_2.pt correctly
    files.sort(key=lambda x: int(x.replace("epoch_", "").replace(".pt", "")))
    return os.path.join(save_dir, files[-1])

def update_plot(history: List[Dict], output_path: str = PLOT_FILE):
    if not history: return
    
    epochs = [h.get("epoch", i+1) for i, h in enumerate(history)]
    train_loss = [h.get("train_loss", 0) for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]
    top1 = [h.get("val_acc1", 0) * 100 for h in history]
    top3 = [h.get("val_acc3", 0) * 100 for h in history]
    elo = [h.get("estimated_elo", None) for h in history]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="blue", marker="o", markersize=3)
    ax1.plot(epochs, val_loss, label="Val Loss", color="red", marker="x", markersize=3)
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Progress")
    ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Accuracy
    ax2.plot(epochs, top1, label="Top-1 Acc", color="green", marker="s", markersize=3)
    ax2.plot(epochs, top3, label="Top-3 Acc", color="orange", marker="d", markersize=3)
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.6)

    # Elo
    valid_elo = [(e, v) for e, v in zip(epochs, elo) if v is not None]
    if valid_elo:
        ex, ey = zip(*valid_elo)
        ax3.plot(ex, ey, label="Est. Elo", color="purple", marker="*", markersize=5)
    
    # Add benchmark line (Run 1 Peak - T=0.8 Corrected)
    ax3.axhline(y=1298, color='gray', linestyle='--', label='Run 1 Peak (1298)', alpha=0.8)
    
    ax3.set_ylabel("Elo")
    ax3.set_xlabel("Epoch")
    ax3.legend(); ax3.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
