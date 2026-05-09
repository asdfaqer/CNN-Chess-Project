import chess
import chess.engine
import numpy as np
import pickle
import os
import sys
import random
from tqdm import tqdm

# --- CONFIGURATION ---
ENGINE_PATH = r"C:\Users\ccbdc\OneDrive\Desktop\leela\lc0.exe" 
RAW_DB_FILE = 'chess_positions_db.pkl'
EVAL_SCALE_FACTOR = 410.0
NUM_SAMPLES_TO_CHECK = 500
DATA_GEN_NODES = 1000 
CHECK_NODE_LIMIT = 5000 

# --- Copied Helper Functions ---
def handle_engine_eval(info: dict) -> int:
    """Handles eval from any engine that outputs CP or Mate."""
    score = info.get("score")
    if score is None: return 0
    if score.is_mate():
        mate_in = score.relative.moves
        if mate_in > 0: return 30000 - mate_in
        else: return -30000 - mate_in
    else:
        cp = score.relative.cp
        if cp is None: return 0
        return cp

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def calculate_value(eval_cp: int) -> float:
    scaled_eval = eval_cp / EVAL_SCALE_FACTOR 
    return sigmoid(scaled_eval)

def load_raw_db(filename: str) -> dict:
    if not os.path.exists(filename):
        print(f"Error: Raw DB file not found at {filename}")
        return {}
    print(f"Loading existing position database from {filename}...")
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}. File might be empty or corrupt.")
        return {}

def main():
    print("--- 🕵️‍♂️ Data Label Verifier (Leela Edition) ---")
    
    if not os.path.exists(ENGINE_PATH):
        print(f"Error: Leela (LC0) not found at {ENGINE_PATH}"); sys.exit(1)
        
    all_positions_data = load_raw_db(RAW_DB_FILE)
    if not all_positions_data:
        print("No data to check.")
        return

    all_items = list(all_positions_data.items())
    
    if len(all_items) < NUM_SAMPLES_TO_CHECK:
        print(f"Warning: Only {len(all_items)} positions available. Checking all.")
        sampled_items = all_items
        num_to_check = len(all_items)
    else:
        sampled_items = random.sample(all_items, NUM_SAMPLES_TO_CHECK)
        num_to_check = NUM_SAMPLES_TO_CHECK
    
    if num_to_check == 0:
        print("No samples to check.")
        return
        
    print(f"Checking {num_to_check} random positions against Leela ({CHECK_NODE_LIMIT} nodes)...")
    
    engine = None
    total_squared_error = 0.0
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        engine.configure({"Threads": 1})
        
        for key, value in tqdm(sampled_items, desc="Verifying Labels"):
            (canonical_fen, _) = key
            
            try:
                stored_eval_cp = value[0]
            except (IndexError, TypeError):
                print(f"Skipping malformed data for key {canonical_fen}")
                continue
            
            board = chess.Board(canonical_fen)
            
            # --- THIS IS THE FIX ---
            if board.is_game_over(claim_draw=True):
                # Position is terminal, do not analyze.
                # Manually get the eval.
                result = board.result(claim_draw=True)
                if result == '1-0':
                    fresh_eval_cp = 30000
                elif result == '0-1':
                    fresh_eval_cp = -30000
                else: # Draw
                    fresh_eval_cp = 0
            
            else:
                # Position is not terminal, safe to analyze.
                info = engine.analyse(board, chess.engine.Limit(nodes=CHECK_NODE_LIMIT))
                fresh_eval_cp = handle_engine_eval(info)
            # --- END OF FIX ---
            
            # 4. Calculate Squared Error
            stored_value = calculate_value(stored_eval_cp)
            fresh_value = calculate_value(fresh_eval_cp)
            
            total_squared_error += (stored_value - fresh_value) ** 2
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if engine:
            engine.quit()

    # 5. Report Results
    print("\n--- Verification Complete ---")
    print(f"Samples Checked: {num_to_check}")
    
    mean_squared_error = total_squared_error / num_to_check
    rmse = np.sqrt(mean_squared_error)
    
    print(f"Mean Squared Error (MSE):       {mean_squared_error:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 30)
    
    if rmse > 0.15: 
        print("❌ FAILED: High RMSE detected (> 0.15).")
        print("This indicates a significant discrepancy between your stored")
        print(f"labels (Leela @ {DATA_GEN_NODES} nodes) and the ground truth (Leela @ {CHECK_NODE_LIMIT} nodes).")
    else:
        print("✅ PASSED: Data labels appear reasonably consistent.")
        print(f"The RMSE ({rmse:.4f}) is within an acceptable tolerance.")
        print(f"Noise between {DATA_GEN_NODES} and {CHECK_NODE_LIMIT} node analysis is expected.")

if __name__ == "__main__":
    main()