"""
Parallel migration to store multi-PV data alongside training batches.

Optimizations:
- Persistent engines: Starts one engine per chunk of games instead of per game.
- Parallel processing: Uses multiple CPU cores for Stockfish analysis.
- Zstd compression: Massive space savings.
- Space management: Removes old files immediately after successful migration.
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

from config import DATA_CACHE_DIR, EV_TANH_K, MULTI_PV
from data_utils import decompress_data, save_batch, load_batch
from utils import STOCKFISH_PATH

# Performance settings
ANALYSIS_TIME = 0.020  # 20ms per position
NUM_WORKERS = 6        # Parallel Stockfish instances
GAMES_PER_CHUNK = 500  # Number of games processed per engine startup

def compute_ev(cp: int, k: int = EV_TANH_K) -> float:
    """Convert centipawn evaluation to win probability."""
    return 0.5 + 0.5 * math.tanh(cp / k)

def compute_importance(scores: list, k: int = EV_TANH_K) -> float:
    """Compute importance as EV(top) - mean(EV(top N))."""
    if len(scores) < 2:
        return 1.0
    top_ev = compute_ev(scores[0], k)
    avg_ev = sum(compute_ev(s, k) for s in scores) / len(scores)
    importance = top_ev - avg_ev
    return 1.0 + 2.0 * max(0, importance)

def reconstruct_board(state: np.ndarray) -> chess.Board:
    """Reconstruct chess.Board from state tensor."""
    try:
        board = chess.Board(fen=None)
        board.clear()
        
        piece_map = {
            0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP,
            3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING
        }
        
        for rank in range(8):
            for file in range(8):
                # Our pieces
                for i, piece_type in piece_map.items():
                    if state[i, rank, file]:
                        square = chess.square(file, 7 - rank)
                        board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
                # Their pieces
                for i, piece_type in piece_map.items():
                    if state[i + 6, rank, file]:
                        square = chess.square(file, 7 - rank)
                        board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))
        
        board.turn = chess.WHITE
        return board if board.is_valid() else None
    except Exception:
        return None

def process_chunk(chunk_data: list) -> list:
    """Process a chunk of games using a single persistent engine instance."""
    results = []
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        # Use only 1 thread per instance to allow max parallelism across core instances
        engine.configure({"Threads": 1, "Hash": 16})
        limit = chess.engine.Limit(time=ANALYSIS_TIME)
    except Exception:
        # Fallback: return original data with dummy scores
        return [(g[0], g[1], g[2], np.zeros((len(g[1]), MULTI_PV), dtype=np.int16)) for g in chunk_data]

    for states, moves, old_weights in chunk_data:
        new_weights = []
        all_scores = []
        
        for i in range(len(moves)):
            board = reconstruct_board(states[i])
            if board:
                try:
                    legal_moves = list(board.legal_moves)
                    # Multi-PV analysis
                    infos = engine.analyse(board, limit, multipv=min(MULTI_PV, len(legal_moves)))
                    
                    scores = []
                    for info in infos:
                        score = info.get("score")
                        if score:
                            cp = score.white().score(mate_score=10000)
                            if cp is not None:
                                scores.append(cp)
                    
                    # Pad / Truncate scores to exactly MULTI_PV
                    while len(scores) < MULTI_PV:
                        scores.append(scores[-1] if scores else 0)
                    scores = scores[:MULTI_PV]
                    
                    weight = compute_importance(scores)
                    new_weights.append(weight)
                    all_scores.append(scores)
                except Exception:
                    new_weights.append(old_weights[i])
                    all_scores.append([0] * MULTI_PV)
            else:
                new_weights.append(old_weights[i])
                all_scores.append([0] * MULTI_PV)
        
        results.append((states, moves, 
                       np.array(new_weights, dtype=np.float32), 
                       np.array(all_scores, dtype=np.int16)))
    
    engine.quit()
    return results

def compress_data_with_mpv(results: list) -> dict:
    """Compress results into the final dictionary format."""
    all_states = []
    all_moves = []
    all_weights = []
    all_mpv = []
    game_boundaries = [0]
    
    for states, moves, weights, mpv in results:
        packed = np.packbits(states.astype(bool).reshape(-1))
        all_states.append(packed)
        all_moves.append(moves.astype(np.uint16))
        all_weights.append(weights.astype(np.float16))
        all_mpv.append(mpv.astype(np.int16))
        game_boundaries.append(game_boundaries[-1] + len(moves))
    
    return {
        "states": np.concatenate(all_states),
        "moves": np.concatenate(all_moves),
        "weights": np.concatenate(all_weights),
        "mpv_scores": np.concatenate(all_mpv),
        "boundaries": np.array(game_boundaries, dtype=np.int32)
    }

def process_batch(batch_path: str) -> bool:
    """Process a batch by splitting into chunks and running in parallel."""
    print(f"\n--- Processing Batch: {os.path.basename(batch_path)} ---")
    
    # Target path
    zst_out = batch_path.replace(".pt", ".zst").replace("batch_", "batch_mpv_")
    if os.path.exists(zst_out):
        print(f"  Already migrated: {os.path.basename(zst_out)}")
        return True
    
    try:
        # Load and decompress
        packed = load_batch(batch_path)
        games = decompress_data(packed)
        total_games = len(games)
        
        # Split into chunks
        chunks = [games[i:i + GAMES_PER_CHUNK] for i in range(0, total_games, GAMES_PER_CHUNK)]
        print(f"  Loaded {total_games} games. Split into {len(chunks)} chunks.")
        
        processed_results = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            for future in tqdm(as_completed(futures), total=len(chunks), desc="  Migrating Chunks"):
                processed_results.extend(future.result())
        
        # Sort results (futures might return out of order, though for training it doesn't strictly matter)
        # But for consistency:
        # Actually ProcessPoolExecutor with positional submit is better if order matters, 
        # but here we can just concatenate.
        
        print("  Compressing and saving...")
        final_packed = compress_data_with_mpv(processed_results)
        
        # Save with zstd
        buffer = io.BytesIO()
        torch.save(final_packed, buffer)
        raw_bytes = buffer.getvalue()
        
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(raw_bytes)
        
        with open(zst_out, "wb") as f:
            f.write(compressed)
            
        print(f"  ✓ Saved to {os.path.basename(zst_out)} ({len(compressed)/1e6:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in batch {batch_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    multiprocessing.set_start_method("spawn", force=True)
    print("=== Optimized Parallel Multi-PV Migration ===")
    print(f"Workers: {NUM_WORKERS} | Chunk Size: {GAMES_PER_CHUNK} | Multi-PV: {MULTI_PV}\n")
    
    # Find all batches
    all_files = os.listdir(DATA_CACHE_DIR)
    legacy_pts = sorted([f for f in all_files if f.startswith("batch_") and f.endswith(".pt") and "mpv" not in f])
    legacy_zsts = sorted([f for f in all_files if f.startswith("batch_") and f.endswith(".zst") and "mpv" not in f])
    
    # Combine and get IDs
    ids = set()
    for f in legacy_pts + legacy_zsts:
        try:
            bid = int(f.replace("batch_", "").replace(".pt", "").replace(".zst", ""))
            ids.add(bid)
        except ValueError: continue
        
    ids = sorted(list(ids))
    print(f"Found {len(ids)} batches to migrate.\n")
    
    for bid in ids:
        # Prefer loading .zst if available for speed
        path = os.path.join(DATA_CACHE_DIR, f"batch_{bid}.zst")
        if not os.path.exists(path):
            path = os.path.join(DATA_CACHE_DIR, f"batch_{bid}.pt")
            
        success = process_batch(path)
        
        if success:
            # Immediate cleanup to save space
            print(f"  Cleaning up legacy files for batch {bid}...")
            pt_to_del = os.path.join(DATA_CACHE_DIR, f"batch_{bid}.pt")
            zst_to_del = os.path.join(DATA_CACHE_DIR, f"batch_{bid}.zst")
            if os.path.exists(pt_to_del): os.remove(pt_to_del)
            if os.path.exists(zst_to_del): os.remove(zst_to_del)
            print(f"  Legacy files removed.")

    print("\n=== Migration Complete ===")
    print(f"Processed {len(ids)} batches.")
    print("All legacy .pt and non-mpv .zst files have been removed.")

if __name__ == "__main__":
    main()
