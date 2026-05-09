import chess
import chess.engine
import chess.pgn
import numpy as np
import random
from tqdm import tqdm
import os
import sys
import pickle
import multiprocessing
import functools
import time
import itertools
import datetime
import math

# --- CONFIGURATION ---
STOCKFISH_PATH = r"C:\Users\ccbdc\Desktop\stockfish\stockfish-windows-x86-64-avx2.exe"
NUM_GAMES = 10000
GAMES_PER_BATCH = 100

INITIAL_RANDOM_MOVE_PROB = 0.20 
RANDOM_MOVE_DECAY_RATE = 0.985 
MIN_RANDOM_MOVE_PROB = 0.15 
STOCKFISH_TIME_LIMIT_MS = 5
EVAL_SCALE_FACTOR = 410.0
CPU_CORES_TO_USE = os.cpu_count() or 1
SAVE_THREADS = 4  # Number of files to write simultaneously

RAW_DB_FILE = 'chess_positions_db.pkl'
# Base name for the training files
NEW_TRAIN_FILE_BASE = 'chess_training_data_original_'

MY_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
OPP_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# --- HELPER FUNCTIONS ---
def handle_stockfish_eval(info: dict) -> int:
    score = info.get("score")
    if score is None: return 0
    if score.is_mate():
        mate_in = score.relative.moves
        return (30000 - mate_in) if mate_in > 0 else (-30000 - mate_in)
    return score.relative.cp

_FLIP_CACHE = {sq: chess.square(chess.square_file(sq), 7 - chess.square_rank(sq)) for sq in range(64)}

def flip_move_uci(uci_move: str) -> str:
    if uci_move == "GAME_END": return "GAME_END"
    try:
        move = chess.Move.from_uci(uci_move)
        flipped_from = _FLIP_CACHE[move.from_square]
        flipped_to = _FLIP_CACHE[move.to_square]
        return chess.Move(flipped_from, flipped_to, move.promotion).uci()
    except:
        return "GAME_END"

def convert_board_to_tensor(board: chess.Board, repetition_plane_value: float) -> np.ndarray:
    tensor = np.zeros((19, 8, 8), dtype=np.float16)
    for i, piece_type in enumerate(MY_PIECES):
        for sq in board.pieces(piece_type, chess.WHITE):
            rank, file = chess.square_rank(sq), chess.square_file(sq)
            tensor[i, rank, file] = 1.0
    for i, piece_type in enumerate(OPP_PIECES):
        for sq in board.pieces(piece_type, chess.BLACK):
            rank, file = chess.square_rank(sq), chess.square_file(sq)
            tensor[i + 6, rank, file] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE): tensor[12] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[13] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[14] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[15] = 1.0
    ep_sq = board.ep_square
    if ep_sq is not None:
        rank, file = chess.square_rank(ep_sq), chess.square_file(ep_sq)
        tensor[16, rank, file] = 1.0
    tensor[17] = repetition_plane_value
    tensor[18] = board.halfmove_clock / 100.0
    return tensor

def calculate_value(eval_cp: int) -> float:
    scaled = eval_cp / EVAL_SCALE_FACTOR
    return 1.0 / (1.0 + np.exp(-scaled))

# --- WORKER FUNCTIONS ---
def play_game_batch_worker(num_games_to_play, initial_random_prob, decay_rate, min_random_prob, time_limit_ms, stockfish_path):
    engine = None
    batch_new_positions = {}
    batch_total_ply = 0
    first_game_pgn_moves = None
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 32})
        time_limit_sec = time_limit_ms / 1000.0

        for _ in range(num_games_to_play):
            board = chess.Board()
            game_fen_history = {} 
            game_moves = [] 
            
            while True:
                if board.is_game_over(claim_draw=True):
                    batch_total_ply += board.ply()
                    break

                fen_full = board.fen()
                base_fen = fen_full[:fen_full.rfind(' ', 0, fen_full.rfind(' '))]
                repetition_count = game_fen_history.get(base_fen, 0) + 1
                game_fen_history[base_fen] = repetition_count
                repetition_plane_value = 1.0 if repetition_count > 1 else 0.0

                best_move_uci = "GAME_END"
                eval_cp = 0
                
                try:
                    result = engine.analyse(board, chess.engine.Limit(time=time_limit_sec))
                    eval_cp = handle_stockfish_eval(result)
                    pv = result.get("pv")
                    if pv: best_move_uci = pv[0].uci()
                    if repetition_count >= 2: eval_cp = 0
                except:
                    break

                if board.turn == chess.WHITE:
                    canonical_base_fen = base_fen
                    canonical_move_uci = best_move_uci
                    canonical_eval_cp = eval_cp
                else:
                    mirrored_fen = board.mirror().fen()
                    canonical_base_fen = mirrored_fen[:mirrored_fen.rfind(' ', 0, mirrored_fen.rfind(' '))]
                    canonical_move_uci = flip_move_uci(best_move_uci)
                    canonical_eval_cp = eval_cp
                
                position_key = (canonical_base_fen, repetition_plane_value)
                if position_key not in batch_new_positions:
                    batch_new_positions[position_key] = (canonical_move_uci, canonical_eval_cp)

                current_random_prob = max(initial_random_prob * (decay_rate ** board.ply()), min_random_prob)
                
                move_to_play = None
                if random.random() <= current_random_prob:
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        batch_total_ply += board.ply()
                        break
                    move_to_play = random.choice(legal_moves)
                else:
                    if best_move_uci != "GAME_END":
                        move_to_play = chess.Move.from_uci(best_move_uci)

                board.push(move_to_play)
                game_moves.append(move_to_play)

            if first_game_pgn_moves is None and game_moves:
                first_game_pgn_moves = game_moves
    except Exception: pass
    finally:
        if engine: engine.quit()
    return batch_new_positions, batch_total_ply, first_game_pgn_moves

def run_playouts(num_games: int, initial_random_prob: float, decay_rate: float, min_random_prob: float, num_workers: int) -> tuple:
    num_batches = (num_games + GAMES_PER_BATCH - 1) // GAMES_PER_BATCH
    tasks = [GAMES_PER_BATCH] * (num_games // GAMES_PER_BATCH)
    remainder = num_games % GAMES_PER_BATCH
    if remainder > 0: tasks.append(remainder)

    print(f"Running {num_games} games across {len(tasks)} batches on {num_workers} cores...")
    
    worker_func = functools.partial(play_game_batch_worker,
                                    initial_random_prob=initial_random_prob,
                                    decay_rate=decay_rate,
                                    min_random_prob=min_random_prob,
                                    time_limit_ms=STOCKFISH_TIME_LIMIT_MS,
                                    stockfish_path=STOCKFISH_PATH)
    
    new_raw_data_dict = {}
    total_ply_played = 0
    first_game_moves_global = None
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iterator = tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Playing Batches")
        for batch_positions, batch_ply, batch_pgn in results_iterator:
            new_raw_data_dict.update(batch_positions)
            total_ply_played += batch_ply
            if first_game_moves_global is None and batch_pgn:
                first_game_moves_global = batch_pgn
    return new_raw_data_dict, len(new_raw_data_dict), total_ply_played, first_game_moves_global

def process_position_worker(item):
    try:
        position_key, (canonical_move_uci, eval_cp) = item
        base_fen, repetition_plane_value = position_key
        fen_for_board = f"{base_fen} 0 1"
        board = chess.Board(fen_for_board)
        board_tensor = convert_board_to_tensor(board, repetition_plane_value)
        value_target = calculate_value(eval_cp)
        return (board_tensor, value_target, canonical_move_uci)
    except: return None

def process_data(raw_positions_dict: dict, num_workers: int) -> list:
    total_items = len(raw_positions_dict)
    print(f"Processing {total_items} raw positions in parallel...")
    if total_items == 0: return []
    
    BATCH_SIZE = 50000 
    iterator = iter(raw_positions_dict.items())
    all_results = []
    worker_chunksize = 2500 
    
    pbar = tqdm(total=total_items, desc="Processing")
    with multiprocessing.Pool(processes=num_workers) as pool:
        while True:
            batch_chunk = list(itertools.islice(iterator, BATCH_SIZE))
            if not batch_chunk: break
            batch_results_iter = pool.imap_unordered(process_position_worker, batch_chunk, chunksize=worker_chunksize)
            for res in batch_results_iter:
                if res is not None: all_results.append(res)
                pbar.update(1)
            del batch_chunk
    pbar.close()
    return all_results

# --- NEW PARALLEL SAVE LOGIC ---

def save_shard_worker(args):
    """Saves a chunk of data to a specific filename."""
    data_chunk, filename = args
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        return filename
    except Exception as e:
        return f"Error: {e}"

def save_data_multiprocess(data_list, base_filename, num_files=4):
    """Splits data_list into num_files and saves them in parallel."""
    if not data_list:
        return
    
    total_len = len(data_list)
    print(f"Splitting {total_len} items into {num_files} files for parallel saving...")
    
    # Split list into N chunks
    chunk_size = math.ceil(total_len / num_files)
    chunks = []
    
    for i in range(num_files):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_len)
        if start_idx >= total_len: break
        
        chunk = data_list[start_idx:end_idx]
        filename = f"{base_filename}_part_{i}.pkl"
        chunks.append((chunk, filename))
    
    # Run the save operation in parallel
    with multiprocessing.Pool(processes=num_files) as pool:
        # Use tqdm to show a progress bar for the FILES being saved
        for _ in tqdm(pool.imap_unordered(save_shard_worker, chunks), total=len(chunks), desc="Saving Files"):
            pass
            
    print(f"Parallel save complete. Created {len(chunks)} files.")

def load_raw_db(f): 
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        if size_mb > 1000:
            print(f"WARNING: raw_db is {size_mb:.0f}MB. Loading might take time.")
    return pickle.load(open(f, 'rb')) if os.path.exists(f) else {}

def save_raw_db(f, d): 
    # Note: We can't easily parallelize saving a SINGLE dict, 
    # but we use protocol 5 for speed.
    print("Saving Raw DB (Single File)...")
    pickle.dump(d, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def main():
    if not STOCKFISH_PATH or not os.path.exists(STOCKFISH_PATH): sys.exit("Stockfish not found")

    try:
        all_positions_data = load_raw_db(RAW_DB_FILE)
        print(f"Loaded {len(all_positions_data)} unique positions.")
        existing_fens_set = set(all_positions_data.keys())

        # --- PLAY GAMES ---
        start_time = time.time()
        new_raw_data, total_found, total_ply, pgn_moves = run_playouts(
            NUM_GAMES, INITIAL_RANDOM_MOVE_PROB, RANDOM_MOVE_DECAY_RATE,
            MIN_RANDOM_MOVE_PROB, CPU_CORES_TO_USE
        )
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*40)
        print(f" PERFORMANCE STATS")
        print("="*40)
        print(f" Total Time       : {elapsed_time:.2f} seconds")
        print("="*40 + "\n")

        # --- PROCESS DATA ---
        truly_new_raw_data = {k: v for k, v in new_raw_data.items() if k not in existing_fens_set}
        print(f"New unique positions added: {len(truly_new_raw_data)}")
        
        if not truly_new_raw_data: return

        # Process tensor conversion
        new_processed_data = process_data(truly_new_raw_data, CPU_CORES_TO_USE)
        
        # Shuffle
        np.random.shuffle(new_processed_data)
        
        # --- PARALLEL SAVE ---
        # Instead of saving one huge file, we use the multiprocessing save function
        save_data_multiprocess(new_processed_data, NEW_TRAIN_FILE_BASE, num_files=SAVE_THREADS)
        
        # Update Raw DB (Still single threaded, but critical for uniqueness check)
        all_positions_data.update(truly_new_raw_data)
        save_raw_db(RAW_DB_FILE, all_positions_data)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()