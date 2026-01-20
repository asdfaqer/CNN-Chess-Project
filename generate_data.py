import os
import time
import math
import random
import multiprocessing
from typing import List, Tuple
import chess
import chess.engine
import numpy as np
import torch
from tqdm import tqdm

import config
from utils import encode_move, board_to_tensor, STOCKFISH_PATH, get_dynamic_resource_info
from data_utils import compress_data, compress_data_mpv, save_batch, load_batch, decompress_data

def compute_ev(cp: int, k: int = None) -> float:
    """Convert centipawn evaluation to win probability using tanh."""
    if k is None: k = config.EV_TANH_K
    return 0.5 + 0.5 * math.tanh(cp / k)

def compute_importance(scores: List[int], k: int = None) -> float:
    """Compute importance as EV(top move) - mean(EV(top N moves)).
    
    Higher importance means the top move is significantly better than alternatives.
    Returns a value typically in range [0, 1] where 1 = very important move.
    """
    if k is None: k = config.EV_TANH_K
    if len(scores) < 2:
        return 1.0  # Single legal move is maximally important
    
    top_ev = compute_ev(scores[0], k)
    avg_ev = sum(compute_ev(s, k) for s in scores) / len(scores)
    
    # Importance is the difference, scaled to be positive and reasonable
    # Typical range: 0 (alternatives are equal) to ~0.5 (top move much better)
    importance = top_ev - avg_ev
    
    # Scale to make it a useful weight: base weight of 1.0 + importance bonus
    return 1.0 + 2.0 * max(0, importance)

def worker_generate_games(args: Tuple[int, str, float, int, bool, int]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate games with either activity or importance weighting."""
    from utils import compute_activity_weight
    
    count, sf_path, m_time, m_moves, use_importance, multi_pv = args
    games_chunk = []
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    except Exception as e:
        print(f"[ERROR] Worker failed to start Stockfish at {sf_path}: {e}")
        return []

    limit = chess.engine.Limit(time=m_time)

    for _ in range(count):
        board = chess.Board()
        game_states = []
        game_moves = []
        game_weights = []
        game_mpv_cp = []
        game_mpv_mate = []
        game_mpv_depth = []
        game_mpv_moves = []
        
        while not board.is_game_over() and len(game_states) < m_moves:
            game_states.append(board_to_tensor(board))
            turn_before_move = board.turn
            
            try:
                if use_importance and multi_pv > 1:
                    # Multi-PV analysis for importance calculation
                    infos = engine.analyse(board, limit, multipv=min(multi_pv, len(list(board.legal_moves))))
                    
                    if not infos:
                        break
                    
                    # Extract centipawn scores
                    cp_scores = []
                    mate_scores = []
                    depth_values = []
                    move_indices = []
                    
                    best_move_obj = None
                    for i, info in enumerate(infos):
                        pv = info.get("pv")
                        score = info.get("score")
                        depth = info.get("depth", 0)
                        
                        if pv and score:
                            move_obj = pv[0]
                            if i == 0:
                                best_move_obj = move_obj
                            
                            s_obj = score.white()
                            cp = s_obj.score()
                            mate = s_obj.mate()
                            
                            # Flip score if black to move
                            if turn_before_move == chess.BLACK:
                                if cp is not None: cp = -cp
                                if mate is not None: mate = -mate
                                
                            cp_scores.append(cp if cp is not None else 0)
                            mate_scores.append(mate if mate is not None else 0)
                            depth_values.append(depth)
                            move_indices.append(encode_move(move_obj, turn_before_move))
                    
                    if best_move_obj is None:
                        break
                    
                    # Pad to MULTI_PV
                    while len(cp_scores) < multi_pv:
                        cp_scores.append(cp_scores[-1] if cp_scores else 0)
                        mate_scores.append(mate_scores[-1] if mate_scores else 0)
                        depth_values.append(depth_values[-1] if depth_values else 0)
                        move_indices.append(-1)
                    
                    game_mpv_cp.append(cp_scores[:multi_pv])
                    game_mpv_mate.append(mate_scores[:multi_pv])
                    game_mpv_depth.append(depth_values[:multi_pv])
                    game_mpv_moves.append(move_indices[:multi_pv])
                    
                    # Compute importance weight (static) for backward compat or logging
                    # We can use the first item in cp_scores and mate_scores to estimate importance
                    # For now just use a simple mock or the first cp
                    weight = compute_importance([c for c in cp_scores])
                else:
                    # Fallback to single move + activity weight
                    result = engine.play(board, limit)
                    best_move_obj = result.move
                    weight = compute_activity_weight(board)
                    
                    # Dummy MPV data
                    game_mpv_cp.append([0] * multi_pv)
                    game_mpv_mate.append([0] * multi_pv)
                    game_mpv_depth.append([0] * multi_pv)
                    game_mpv_moves.append([-1] * multi_pv)
                
            except (chess.engine.EngineTerminatedError, Exception):
                break
            
            if best_move_obj is None:
                break
            
            label_idx = encode_move(best_move_obj, turn_before_move)
            game_moves.append(label_idx)
            game_weights.append(weight)
            
            # Epsilon-greedy exploration
            epsilon = config.EPSILON_START * math.exp(-config.MOVE_EPSILON_DECAY * len(game_states))
            
            if random.random() < epsilon:
                move_to_play = random.choice(list(board.legal_moves))
            else:
                move_to_play = best_move_obj
                
            board.push(move_to_play)
            
        if len(game_states) > 10:
            games_chunk.append((
                np.array(game_states), 
                np.array(game_moves), 
                np.array(game_weights, dtype=np.float32),
                np.array(game_mpv_cp, dtype=np.int16),
                np.array(game_mpv_mate, dtype=np.int16),
                np.array(game_mpv_depth, dtype=np.uint8),
                np.array(game_mpv_moves, dtype=np.int16)
            ))

    engine.quit()
    return games_chunk

def generate_data_batch_parallel(cycle_id: int, num_workers: int = None) -> List[Tuple]:
    if num_workers is None: num_workers = config.NUM_WORKERS_GEN
    weight_mode = "Importance (Multi-PV)" if config.USE_IMPORTANCE_WEIGHT else "Activity"
    print(f"--- Starting Parallel Data Generation: {config.DATA_BATCH_SIZE} Games (Cycle {cycle_id}, {weight_mode}, Workers: {num_workers}) ---")
    start_time = time.time()
    
    tasks = []
    games_remaining = config.DATA_BATCH_SIZE
    while games_remaining > 0:
        n = min(config.GAMES_PER_WORKER, games_remaining)
        tasks.append((n, STOCKFISH_PATH, config.MOVE_TIME, config.MAX_MOVES, config.USE_IMPORTANCE_WEIGHT, config.MULTI_PV))
        games_remaining -= n

    all_games = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(worker_generate_games, tasks), 
                           total=len(tasks), 
                           desc="Generating Games",
                           unit="chunk"))
        for res in results:
            all_games.extend(res)

    duration = time.time() - start_time
    print(f"Generation Complete. Generated {len(all_games)} valid games.")
    print(f"Time Taken: {duration:.2f}s (Avg {duration/config.DATA_BATCH_SIZE:.4f}s/game)")
    return all_games

def get_next_batch_id():
    if not os.path.exists(config.DATA_CACHE_DIR):
        os.makedirs(config.DATA_CACHE_DIR)
    # Detect new .zst files with mpv prefix
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) 
             if f.startswith("batch_mpv_") and f.endswith(".zst")]
    if not files: return 0
    try:
        ids = [int(f.replace("batch_mpv_", "").replace(".zst", "")) for f in files]
        return max(ids) + 1
    except ValueError:
        return 0

def main():
    multiprocessing.freeze_support()
    print("=== Standalone Data Generator Mode ===")
    
    while True:
        batch_id = get_next_batch_id()
        
        # Dynamic resource adjustment
        res = get_dynamic_resource_info()
        # Use roughly 75% of available CPU cores to keep system responsive
        safe_workers = max(1, int(res["cpu_count"] * 0.75))
        
        print(f"\nScanning for work... Next Batch ID: {batch_id}")
        raw_data = generate_data_batch_parallel(batch_id, num_workers=safe_workers)
        
        if raw_data:
            # Mix with a random existing batch to increase data regularity
            existing_files = [f for f in os.listdir(config.DATA_CACHE_DIR) 
                            if f.startswith("batch_mpv_") and f.endswith(".zst") and f != f"batch_mpv_{batch_id}.zst"]
            
            if existing_files:
                target_file = random.choice(existing_files)
                target_path = os.path.join(config.DATA_CACHE_DIR, target_file)
                print(f"Mixing new batch with {target_file} to regularize data...")
                try:
                    packed_old = load_batch(target_path)
                    old_games = decompress_data(packed_old)
                    
                    combined = raw_data + old_games
                    random.shuffle(combined)
                    
                    # Split 50/50
                    half = len(combined) // 2
                    raw_data = combined[:half]
                    leftover = combined[half:]
                    
                    # Update the existing batch with mixed games
                    packed_leftover = compress_data_mpv(leftover)
                    save_batch(packed_leftover, target_path)
                    print(f"Successfully mixed current batch with {target_file}.")
                    del old_games, combined, leftover, packed_old, packed_leftover
                except Exception as e:
                    print(f"Mixing skipped due to error: {e}")

            print(f"Compressing and saving batch {batch_id} (Multi-PV format)...")
            packed_data = compress_data_mpv(raw_data)
            final_path = os.path.join(config.DATA_CACHE_DIR, f"batch_mpv_{batch_id}.pt")
            save_batch(packed_data, final_path)
            print(f"Batch {batch_id} saved successfully.")
        
        print("Short rest before next batch...")
        time.sleep(10)

if __name__ == "__main__":
    main()
