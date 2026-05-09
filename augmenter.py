# augmenter.py
# This is the refactored augmentation logic, designed to be imported.

import chess
import chess.engine
import numpy as np
import pickle
import os
import sys
import random
import collections
import functools
import itertools
from tqdm import tqdm
import multiprocessing

# --- Constants ---
SQUARES = chess.SQUARES
SQUARE_NAMES = chess.SQUARE_NAMES
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# --- Copied Helper ---
# This is needed for the worker to have access
def handle_stockfish_eval(info: dict) -> int:
    score = info.get("score")
    if score is None: return 0
    if score.is_mate():
        mate_in = score.relative.moves
        if mate_in > 0: return 30000 - mate_in
        else: return -30000 - mate_in
    else: return score.relative.cp

# --- 3. HEURISTIC & MODIFICATION CORE ---

def calculate_score(mod_key, context_key, db, local_boost, exploration_bonus):
    """Calculates the final score for a mod using your two-level system."""
    local_score = db.get((mod_key, context_key), 0)
    global_score = db.get((mod_key, 'GLOBAL'), 0)
    
    # This is your logic: (local * boost) + global + exploration
    return (local_score * local_boost) + global_score + exploration_bonus

def get_legal_and_scored_mods(board, context_key, db, local_boost, exploration_bonus):
    """
    This is the "filter" you designed. It finds all legal micro-modifications
    for the current board and scores them using the heuristic database.
    """
    legal_mods = []
    scores = []
    
    # 1. Castling Rights (4 mods)
    castling_fen = board.castling_xfen() 
    
    if 'K' in castling_fen:
        key = 'mod_remove_K_castle_white'
        legal_mods.append(key)
        scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))
    if 'Q' in castling_fen:
        key = 'mod_remove_Q_castle_white'
        legal_mods.append(key)
        scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))
    if 'k' in castling_fen:
        key = 'mod_remove_k_castle_black'
        legal_mods.append(key)
        scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))
    if 'q' in castling_fen:
        key = 'mod_remove_q_castle_black'
        legal_mods.append(key)
        scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))

    # 2. Pawn Shifts (Forward: 64 mods, Backward: 56 mods)
    for sq in SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            sq_name = SQUARE_NAMES[sq]
            color = piece.color
            
            # Forward
            fwd_sq = sq + 8 if color == chess.WHITE else sq - 8
            if 0 <= fwd_sq <= 63 and board.piece_at(fwd_sq) is None:
                key = f'mod_pawn_fwd_{sq_name}'
                legal_mods.append(key)
                scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))

            # Backward
            back_rank = 1 if color == chess.WHITE else 6
            if chess.square_rank(sq) != back_rank:
                back_sq = sq - 8 if color == chess.WHITE else sq + 8
                if 0 <= back_sq <= 63 and board.piece_at(back_sq) is None:
                    key = f'mod_pawn_back_{sq_name}'
                    legal_mods.append(key)
                    scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))

    # 3. Swap Friendly Minors (12 mods)
    for color in [chess.WHITE, chess.BLACK]:
        knights = board.pieces(chess.KNIGHT, color)
        bishops = board.pieces(chess.BISHOP, color)
        # Find all (Knight, Bishop) pairs to swap
        for n_sq, b_sq in itertools.product(knights, bishops):
            key = f'mod_swap_{SQUARE_NAMES[n_sq]}_{SQUARE_NAMES[b_sq]}'
            legal_mods.append(key)
            scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))

    # 4. En Passant Square (16 mods)
    for file_idx in range(8):
        # White can set EP on rank 6 (e.g., d6) if a black pawn is on d5
        ep_sq_white = chess.square(file_idx, 5) # e.g., d6
        pawn_sq_white = chess.square(file_idx, 4) # e.g., d5
        if board.piece_at(pawn_sq_white) == chess.Piece(chess.PAWN, chess.BLACK) and \
           board.piece_at(ep_sq_white) is None:
            key = f'mod_ep_file_{SQUARE_NAMES[ep_sq_white][0]}' # mod_ep_file_d
            legal_mods.append(key)
            scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))

        # Black can set EP on rank 3 (e.g., d3) if a white pawn is on d4
        ep_sq_black = chess.square(file_idx, 2) # e.g., d3
        pawn_sq_black = chess.square(file_idx, 3) # e.g., d4
        if board.piece_at(pawn_sq_black) == chess.Piece(chess.PAWN, chess.WHITE) and \
           board.piece_at(ep_sq_black) is None:
            key = f'mod_ep_file_{SQUARE_NAMES[ep_sq_black][0]}' # mod_ep_file_d
            legal_mods.append(key)
            scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))

    # 5. Add Pawn (Safety Net Mod)
    safe_squares_white = [chess.square(f, 1) for f in range(8)] # a2-h2
    safe_squares_black = [chess.square(f, 6) for f in range(8)] # a7-h7
    
    for sq in safe_squares_white:
        if board.piece_at(sq) is None:
            key = f'mod_add_pawn_{SQUARE_NAMES[sq]}'
            legal_mods.append(key)
            scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))
            
    for sq in safe_squares_black:
        if board.piece_at(sq) is None:
            key = f'mod_add_pawn_{SQUARE_NAMES[sq]}'
            legal_mods.append(key)
            scores.append(calculate_score(key, context_key, db, local_boost, exploration_bonus))
            
    return legal_mods, scores

def apply_modification(board: chess.Board, mod_key: str) -> chess.Board:
    """
    Applies the chosen micro-modification. This function assumes
    the modification was already checked for legality.
    """
    try:
        # 1. Castling
        if mod_key.startswith('mod_remove_'):
            new_fen_part = list(board.castling_xfen())
            if mod_key == 'mod_remove_K_castle_white': new_fen_part.remove('K')
            elif mod_key == 'mod_remove_Q_castle_white': new_fen_part.remove('Q')
            elif mod_key == 'mod_remove_k_castle_black': new_fen_part.remove('k')
            elif mod_key == 'mod_remove_q_castle_black': new_fen_part.remove('q')
            board.set_castling_fen("".join(new_fen_part) or "-")

        # 2. Pawn Forward
        elif mod_key.startswith('mod_pawn_fwd_'):
            sq_name = mod_key.split('_')[-1]
            sq = chess.parse_square(sq_name)
            piece = board.piece_at(sq)
            fwd_sq = sq + 8 if piece.color == chess.WHITE else sq - 8
            board.set_piece_at(fwd_sq, piece)
            board.set_piece_at(sq, None)

        # 3. Pawn Backward
        elif mod_key.startswith('mod_pawn_back_'):
            sq_name = mod_key.split('_')[-1]
            sq = chess.parse_square(sq_name)
            piece = board.piece_at(sq)
            back_sq = sq - 8 if piece.color == chess.WHITE else sq + 8
            board.set_piece_at(back_sq, piece)
            board.set_piece_at(sq, None)

        # 4. Swap Minors
        elif mod_key.startswith('mod_swap_'):
            _, _, sq_name_1, sq_name_2 = mod_key.split('_')
            sq1, sq2 = chess.parse_square(sq_name_1), chess.parse_square(sq_name_2)
            piece1, piece2 = board.piece_at(sq1), board.piece_at(sq2)
            board.set_piece_at(sq1, piece2)
            board.set_piece_at(sq2, piece1)

        # 5. En Passant
        elif mod_key.startswith('mod_ep_file_'):
            file_name = mod_key.split('_')[-1]
            pawn_sq = chess.parse_square(f"{file_name}4")
            if board.piece_at(pawn_sq) == chess.Piece(chess.PAWN, chess.WHITE):
                ep_sq = chess.parse_square(f"{file_name}3")
            else:
                ep_sq = chess.parse_square(f"{file_name}6")
            board.ep_square = ep_sq
            
        # 6. Add Pawn
        elif mod_key.startswith('mod_add_pawn_'):
            sq_name = mod_key.split('_')[-1]
            sq = chess.parse_square(sq_name)
            color = chess.WHITE if chess.square_rank(sq) == 1 else chess.BLACK
            piece = chess.Piece(chess.PAWN, color)
            board.set_piece_at(sq, piece)

        # Final check
        if not board.is_valid() or board.is_check():
            return None
            
        return board
    
    except Exception:
        return None

# --- 4. THE PARALLEL WORKER ---

def augment_worker(work_item, global_heuristic_db, stockfish_path, time_limit_ms, max_attempts_per_sample, local_boost, exploration_bonus, existing_fens_set):
    """
    This function runs in a separate process to generate data for one move.
    All config is passed in.
    """
    target_move_uci, num_needed, base_positions = work_item
    
    local_heuristic_updates = collections.defaultdict(int)
    newly_generated_data = {} # {(fen, rep_val): (move, eval, legal_list)} # <-- MODIFIED
    
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1})
        
        attempts = 0
        max_attempts = num_needed * max_attempts_per_sample

        pbar = tqdm(total=num_needed, desc=f"Augmenting {target_move_uci}", leave=False, position=multiprocessing.current_process()._identity[0])

        while len(newly_generated_data) < num_needed and attempts < max_attempts:
            attempts += 1
            
            # --- CHANGE 2: Fix worker input unpacking ---
            # (base_move, base_eval) = random.choice(base_positions) # <-- OLD
            (base_fen, rep_val), value_tuple = random.choice(base_positions) # <-- MODIFIED
            # value_tuple is (base_move, base_eval, base_legal_list)
            # We only need the base_fen, so this is fine.
            # --- END CHANGE 2 ---
            
            base_board = chess.Board(base_fen)
            
            try:
                move_obj = chess.Move.from_uci(target_move_uci)
                from_sq = move_obj.from_square
                piece = base_board.piece_at(from_sq)
                if piece is None: continue 
                context_key = (from_sq, piece.piece_type)
            except Exception:
                continue 

            legal_mods, scores = get_legal_and_scored_mods(base_board, context_key, global_heuristic_db, local_boost, exploration_bonus)
            
            if not legal_mods:
                continue 
                
            chosen_mod_key = random.choices(legal_mods, weights=scores, k=1)[0]
            
            modified_board = apply_modification(base_board.copy(), chosen_mod_key)
            
            if modified_board is None:
                continue 
                
            new_key = (modified_board.fen(), rep_val)
            
            # Check for duplicates against the *entire* known dataset
            if new_key in existing_fens_set or new_key in newly_generated_data:
                continue

            info = engine.analyse(modified_board, chess.engine.Limit(time=time_limit_ms / 1000.0))
            best_move_obj = info.get("pv", [None])[0]
            
            if best_move_obj is None:
                continue

            if best_move_obj.uci() == target_move_uci:
                
                # --- CHANGE 3: Fix worker output format ---
                new_eval_cp = handle_stockfish_eval(info)
                
                # The base_fen was canonical, so modified_board is also canonical.
                # We can just grab its legal moves directly.
                new_legal_moves_list = []
                for move in modified_board.legal_moves:
                    new_legal_moves_list.append(move.uci())

                # Save the new 3-item tuple
                newly_generated_data[new_key] = (target_move_uci, new_eval_cp, new_legal_moves_list) # <-- MODIFIED
                # --- END CHANGE 3 ---
                
                local_heuristic_updates[(chosen_mod_key, context_key)] += 1
                local_heuristic_updates[(chosen_mod_key, 'GLOBAL')] += 1
                
                pbar.update(1)

        pbar.close()
        
    except Exception as e:
        print(f"Error in worker for {target_move_uci}: {e}")
    finally:
        if engine:
            engine.quit()
            
    return (newly_generated_data, local_heuristic_updates)

# --- 5. THE MAIN ORCHESTRATOR (Importable Function) ---

def run_augmentation(raw_position_db, heuristic_db, stockfish_path, 
                     target_samples, max_attempts_per_sample, stockfish_time_ms, 
                     local_boost_factor, exploration_bonus, cpu_cores):
    """
    Main entry point for the augmentation module.
    """
    
    # 1. Analyze and group data by move
    print("Analyzing base dataset and grouping by move...")
    base_position_bank = collections.defaultdict(list)
    
    # --- CHANGE 1: Fix input DB read loop ---
    for key, value_tuple in raw_position_db.items(): # <-- MODIFIED
        if not isinstance(value_tuple, (tuple, list)) or len(value_tuple) < 2:
            continue # Skip malformed data
            
        move_uci = value_tuple[0] # <-- MODIFIED (Get from index 0)
        # eval_cp = value_tuple[1] # We don't need this here
        
        if move_uci != "GAME_END":
            base_position_bank[move_uci].append((key, value_tuple)) # <-- MODIFIED (Append full tuple)
    # --- END CHANGE 1 ---
            
    print(f"Found {len(base_position_bank)} unique moves in the database.")
    
    # 2. Create the work list (find moves that are *under-represented*)
    #    We find the *minimum* count and try to bring all others up to that.
    #    Or, for now, just augment everything by N.
    work_list = []
    for move_uci, positions in base_position_bank.items():
        # TODO: Add logic to only augment moves *under* a certain threshold
        # For now, we follow the original script's plan: augment *all* moves by N
        work_list.append((move_uci, target_samples, positions))
        
    if not work_list:
        print("No work to do.")
        return {}, heuristic_db
        
    print(f"Starting augmentation for {len(work_list)} moves, target of {target_samples} new positions each...")

    # 3. Run the parallel workers
    final_augmented_data = {}
    
    # We must pass the full FEN set so workers don't create duplicates
    existing_fens_set = set(raw_position_db.keys())
    
    worker_func = functools.partial(augment_worker, 
                                     global_heuristic_db=heuristic_db,
                                     stockfish_path=stockfish_path,
                                     time_limit_ms=stockfish_time_ms,
                                     max_attempts_per_sample=max_attempts_per_sample,
                                     local_boost=local_boost_factor,
                                     exploration_bonus=exploration_bonus,
                                     existing_fens_set=existing_fens_set)
    
    with multiprocessing.Pool(processes=cpu_cores) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, work_list), total=len(work_list), desc="Overall Augmentation"))
    
    # 4. Merge all results
    print("\nMerging results from all workers...")
    updated_heuristic_db = heuristic_db.copy()
    for new_data_chunk, heuristic_updates in results:
        
        final_augmented_data.update(new_data_chunk)
        
        for key, update_value in heuristic_updates.items():
            updated_heuristic_db[key] = updated_heuristic_db.get(key, 0) + update_value

    return final_augmented_data, updated_heuristic_db