"""
Verbose Migration script - Adds raw CP, Mate, Depth, and Move Indices to legacy data.
Creates a small test sample first for verification.
"""
import os
import io
import math
import torch
import chess
import chess.engine
import numpy as np
import zstandard as zstd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import argparse

import sys

# Try to import from local config, fall back to defaults for cloud
try:
    from config import DATA_CACHE_DIR, MULTI_PV, BASE_CHANNELS
except ImportError:
    DATA_CACHE_DIR = "generated_data"
    MULTI_PV = 5
    BASE_CHANNELS = 18

try:
    from utils import encode_move, STOCKFISH_PATH
except ImportError:
    # Fallback for cloud environment
    STOCKFISH_PATH = "/usr/local/bin/stockfish"
    def encode_move(move, turn):
        # Basic encoding if utils not available
        import chess
        from_sq = move.from_square
        to_sq = move.to_square
        promo = move.promotion
        if turn == chess.BLACK:
            from_sq = chess.square_mirror(from_sq)
            to_sq = chess.square_mirror(to_sq)
        base = from_sq * 64 + to_sq
        if promo:
            promo_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}.get(promo, 0)
            base = 4096 + from_sq % 8 * 64 + to_sq % 8 * 8 + promo_offset
        return base

# Performance settings
ANALYSIS_TIME = 0.020  # 20ms per position
NUM_WORKERS = max(1, multiprocessing.cpu_count() // 2)
POSITIONS_PER_CHUNK = 500

def load_legacy_batch(path: str) -> dict:
    """Load a batch file, handling zstd compression."""
    if path.endswith('.zst'):
        dctx = zstd.ZstdDecompressor()
        with open(path, 'rb') as f:
            raw = dctx.decompress(f.read())
        return torch.load(io.BytesIO(raw), weights_only=False)
    return torch.load(path, weights_only=False)

def save_batch(packed: dict, path: str):
    """Save batch with zstd compression."""
    buffer = io.BytesIO()
    torch.save(packed, buffer)
    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(buffer.getvalue())
    with open(path, 'wb') as f:
        f.write(compressed)

def unpack_state(packed_bytes: np.ndarray, num_positions: int, channels: int = BASE_CHANNELS) -> np.ndarray:
    """Unpack bitpacked states back to [N, C, 8, 8] tensor."""
    bits = np.unpackbits(packed_bytes)
    total_bits_needed = num_positions * channels * 8 * 8
    bits = bits[:total_bits_needed]
    return bits.reshape(num_positions, channels, 8, 8).astype(np.float32)

def reconstruct_board(state: np.ndarray) -> chess.Board:
    """Reconstruct chess.Board from a single state [C, 8, 8]. Uses only first 12 channels."""
    try:
        board = chess.Board(fen=None)
        board.clear()
        piece_map = {0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING}
        
        # Use only first 12 channels (our pieces + opponent pieces)
        for rank in range(8):
            for file in range(8):
                for i, pt in piece_map.items():
                    if state[i, rank, file] > 0.5:
                        board.set_piece_at(chess.square(file, 7-rank), chess.Piece(pt, chess.WHITE))
                    if state[i+6, rank, file] > 0.5:
                        board.set_piece_at(chess.square(file, 7-rank), chess.Piece(pt, chess.BLACK))
        board.turn = chess.WHITE
        return board
    except Exception as e:
        return None

def analyze_positions(positions: list) -> list:
    """Analyze a list of (index, state) tuples and return verbose MPV data."""
    results = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"Threads": 1, "Hash": 16})
        limit = chess.engine.Limit(time=ANALYSIS_TIME)
    except Exception as e:
        print(f"Engine init failed: {e}")
        return [(idx, [0]*MULTI_PV, [0]*MULTI_PV, [0]*MULTI_PV, [-1]*MULTI_PV) for idx, _ in positions]

    for idx, state in positions:
        board = reconstruct_board(state)
        if board is None or not list(board.legal_moves):
            results.append((idx, [0]*MULTI_PV, [0]*MULTI_PV, [0]*MULTI_PV, [-1]*MULTI_PV))
            continue
            
        try:
            infos = engine.analyse(board, limit, multipv=min(MULTI_PV, len(list(board.legal_moves))))
            cp_s, mate_s, depth_s, move_s = [], [], [], []
            
            for info in infos:
                score = info.get("score")
                pv = info.get("pv")
                if score and pv:
                    s = score.white()
                    cp_s.append(s.score() if s.score() is not None else 0)
                    mate_s.append(s.mate() if s.mate() is not None else 0)
                    depth_s.append(info.get("depth", 0))
                    move_s.append(encode_move(pv[0], chess.WHITE))
            
            # Pad to MULTI_PV
            while len(cp_s) < MULTI_PV:
                cp_s.append(cp_s[-1] if cp_s else 0)
                mate_s.append(mate_s[-1] if mate_s else 0)
                depth_s.append(depth_s[-1] if depth_s else 0)
                move_s.append(-1)
            
            results.append((idx, cp_s[:MULTI_PV], mate_s[:MULTI_PV], depth_s[:MULTI_PV], move_s[:MULTI_PV]))
        except Exception as e:
            results.append((idx, [0]*MULTI_PV, [0]*MULTI_PV, [0]*MULTI_PV, [-1]*MULTI_PV))
    
    engine.quit()
    return results

def migrate_batch(batch_path: str, output_path: str = None, max_positions: int = None):
    """Migrate a single batch file to verbose format."""
    print(f"\n--- Migrating: {os.path.basename(batch_path)} ---")
    
    packed = load_legacy_batch(batch_path)
    
    # Check if already migrated
    if "mpv_cp" in packed:
        print("  Already in verbose format. Skipping.")
        return True
    
    # Get dimensions
    num_positions = len(packed["moves"])
    channels = BASE_CHANNELS
    
    if max_positions:
        num_positions = min(num_positions, max_positions)
        print(f"  Limiting to {num_positions} positions (test mode)")
    
    print(f"  Total positions: {num_positions}")
    
    # Unpack states
    print("  Unpacking states...")
    states = unpack_state(packed["states"], num_positions, channels)
    
    # Create position list for parallel processing
    positions = [(i, states[i]) for i in range(num_positions)]
    chunks = [positions[i:i+POSITIONS_PER_CHUNK] for i in range(0, len(positions), POSITIONS_PER_CHUNK)]
    
    print(f"  Analyzing {len(chunks)} chunks with {NUM_WORKERS} workers...")
    
    # Process in parallel
    all_results = [None] * num_positions
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(analyze_positions, chunk): chunk for chunk in chunks}
        for future in tqdm(as_completed(futures), total=len(chunks), desc="  Analyzing"):
            for idx, cp, mate, depth, moves in future.result():
                all_results[idx] = (cp, mate, depth, moves)
    
    # Build new arrays
    print("  Building output arrays...")
    mpv_cp = np.array([r[0] for r in all_results], dtype=np.int16)
    mpv_mate = np.array([r[1] for r in all_results], dtype=np.int16)
    mpv_depth = np.array([r[2] for r in all_results], dtype=np.uint8)
    mpv_moves = np.array([r[3] for r in all_results], dtype=np.int16)
    
    # Create new packed dict
    new_packed = {
        "states": packed["states"] if not max_positions else np.packbits(states.astype(bool).reshape(-1)),
        "moves": packed["moves"][:num_positions],
        "weights": packed["weights"][:num_positions],
        "mpv_cp": mpv_cp,
        "mpv_mate": mpv_mate,
        "mpv_depth": mpv_depth,
        "mpv_moves": mpv_moves,
        "boundaries": packed["boundaries"] if not max_positions else np.array([0, num_positions], dtype=np.int32)
    }
    
    # Save
    out_path = output_path or batch_path
    print(f"  Saving to {os.path.basename(out_path)}...")
    save_batch(new_packed, out_path)
    print("  Done!")
    return True

def create_test_sample(source_batch: str, output_path: str, num_positions: int = 1000):
    """Create a small test sample from a batch for verification."""
    print(f"\n=== Creating Test Sample ({num_positions} positions) ===")
    migrate_batch(source_batch, output_path, max_positions=num_positions)
    
    # Verify the output
    print("\n  Verifying output...")
    result = load_legacy_batch(output_path)
    print(f"  Keys: {list(result.keys())}")
    print(f"  mpv_cp shape: {result['mpv_cp'].shape}")
    print(f"  mpv_moves shape: {result['mpv_moves'].shape}")
    print(f"  Sample mpv_cp[0]: {result['mpv_cp'][0]}")
    print(f"  Sample mpv_moves[0]: {result['mpv_moves'][0]}")
    print("  Test sample created successfully!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Migrate chess data to verbose MPV format")
    parser.add_argument("--test", action="store_true", help="Create a small test sample only")
    parser.add_argument("--batch", type=str, help="Specific batch file to migrate")
    parser.add_argument("--output", type=str, help="Output path (defaults to same as input)")
    parser.add_argument("--all", action="store_true", help="Migrate all batch files")
    args = parser.parse_args()
    
    if args.test:
        source = os.path.join(DATA_CACHE_DIR, "batch_mpv_3.zst")
        output = os.path.join(DATA_CACHE_DIR, "test_verbose_sample.zst")
        create_test_sample(source, output, num_positions=500)
    elif args.batch:
        migrate_batch(args.batch, output_path=args.output)
    elif args.all:
        files = sorted([f for f in os.listdir(DATA_CACHE_DIR) if f.startswith("batch_mpv_") and f.endswith(".zst")])
        for f in files:
            migrate_batch(os.path.join(DATA_CACHE_DIR, f))
    else:
        print("Usage: python migrate_mpv_moves.py --test | --batch <path> [--output <path>] | --all")
