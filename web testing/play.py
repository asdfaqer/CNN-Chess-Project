import os
import sys
import random
import time
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import chess
import chess.engine
import chess.pgn

# --- DEPENDENCIES ---
try:
    from chess_cnn import ChessCNN
    from alphazero_utils import build_alphazero_map
except ImportError:
    print("Error: Missing dependencies.")
    print("Ensure 'chess_cnn.py' and 'alphazero_utils.py' are in this directory.")
    sys.exit(1)


# --- CONFIGURATION ---
# UPDATE THIS PATH TO YOUR STOCKFISH EXECUTABLE
STOCKFISH_PATH = r"C:\Users\ccbdc\Desktop\stockfish\stockfish-windows-x86-64-avx2.exe"

MODEL_PATH_CURRENT = 'chess_cnn.pth'
MODEL_PATH_PREVIOUS = 'chess_cnn_small.pth' 

# --- CONSTANTS ---
MY_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
OPP_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
EVAL_SCALE_FACTOR = 410.0
NUM_POLICY_OUTPUTS = 4672 # Fixed AlphaZero Output Size


# ###########################################################################
## 1. HELPER FUNCTIONS
# ###########################################################################

def flip_move_uci(uci_move: str) -> str:
    """Flips a UCI move string (e.g., "e7e5" -> "e2e4")."""
    if uci_move == "GAME_END": return "GAME_END"
    try:
        move = chess.Move.from_uci(uci_move)
        from_sq, to_sq = move.from_square, move.to_square
        flipped_from = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
        flipped_to = chess.square(chess.square_file(to_sq), 7 - chess.square_rank(to_sq))
        return chess.Move(flipped_from, flipped_to, move.promotion).uci()
    except Exception:
        return "GAME_END"


def convert_board_to_tensor(board: chess.Board, repetition_plane_value: float) -> np.ndarray:
    """
    Converts a chess.Board object into the (19, 8, 8) symmetric tensor.
    ASSUMES: The 'board' is *always* from the perspective of chess.WHITE.
    """
    tensor = np.zeros((19, 8, 8), dtype=np.float32)

    for i, piece_type in enumerate(MY_PIECES):
        for sq in board.pieces(piece_type, chess.WHITE):
            tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    for i, piece_type in enumerate(OPP_PIECES):
        for sq in board.pieces(piece_type, chess.BLACK):
            tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0

    ep_sq = board.ep_square
    if ep_sq:
        tensor[16, chess.square_rank(ep_sq), chess.square_file(ep_sq)] = 1.0

    tensor[17, :, :] = repetition_plane_value
    tensor[18, :, :] = board.halfmove_clock / 100.0

    return tensor


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calculate_value(eval_cp: int) -> float:
    scaled_eval = eval_cp / EVAL_SCALE_FACTOR
    return sigmoid(scaled_eval)


def handle_stockfish_eval(info: dict) -> int:
    score = info.get("score")
    if score is None: return 0
    if score.is_mate():
        mate_in = score.relative.moves
        return 30000 - mate_in if mate_in > 0 else -30000 - mate_in
    else:
        return score.relative.cp


# ###########################################################################
## 2. CORE INFERENCE FUNCTIONS
# ###########################################################################

def load_model_and_map(model_path, device, num_res_blocks, num_filters):
    """
    Generates the AlphaZero map in-memory and loads the model.
    """
    # 1. Generate Map (No pickle needed!)
    print(f"Generating AlphaZero Map for {model_path}...")
    MOVE_TO_INDEX = build_alphazero_map()
    INDEX_TO_MOVE = {v: k for k, v in MOVE_TO_INDEX.items()}
    
    # 2. Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found.")
        sys.exit(1)

    model = ChessCNN(num_policy_outputs=NUM_POLICY_OUTPUTS, num_res_blocks=num_res_blocks, num_filters=num_filters)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError:
        print("!! Error: Model Architecture Mismatch.")
        print("The saved model does not match the AlphaZero architecture (4672 outputs).")
        print("Did you retrain using the new train.py?")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    print(f"Successfully loaded model.")
    
    return model, MOVE_TO_INDEX, INDEX_TO_MOVE


def create_legal_mask(board, MOVE_TO_INDEX):
    """Creates a mask of legal moves, respecting the canonical (White) perspective."""
    _mask = torch.zeros(NUM_POLICY_OUTPUTS, dtype=torch.float32)
    is_white_turn = (board.turn == chess.WHITE)

    for move in board.legal_moves:
        move_uci = move.uci()
        if not is_white_turn:
            move_uci = flip_move_uci(move_uci)

        if move_uci in MOVE_TO_INDEX:
            _mask[MOVE_TO_INDEX[move_uci]] = 1.0

    return _mask


def run_model_inference(model, board, game_history, device):
    """Runs the model on the current board state."""
    base_fen = board.fen().rsplit(' ', 2)[0]
    rep_count = game_history.get(base_fen, 0) + 1
    repetition_plane_value = 1.0 if rep_count > 1 else 0.0

    board_to_convert = board.mirror() if board.turn == chess.BLACK else board

    tensor_np = convert_board_to_tensor(board_to_convert, repetition_plane_value)
    tensor = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred_value, pred_policy_logits = model(tensor)
        pred_value = pred_value.item()
        pred_policy_probs = F.softmax(pred_policy_logits, dim=1).squeeze(0).cpu()

    return pred_value, pred_policy_probs, repetition_plane_value

def get_model_move(model, board, game_history, MOVE_TO_INDEX, INDEX_TO_MOVE, device, temperature=0.1):
    """
    Decodes the model output, handling the collision between standard moves 
    and Queen promotions (e.g., 'e2e1' vs 'e2e1q').
    """
    # 1. Run inference
    pred_value, pred_probs, _ = run_model_inference(model, board, game_history, device)

    # 2. Create legal move mask
    mask = create_legal_mask(board, MOVE_TO_INDEX)

    # 3. Mask probabilities
    masked_probs = pred_probs * mask
    legal_prob_sum = masked_probs.sum().item()

    # --- FAILSAFE: Model is confused or masked out completely ---
    if legal_prob_sum <= 1e-6:
        print("--- FAILSAFE: Model is confused or masked out completely ---")
        return random.choice(list(board.legal_moves)), pred_value, 0.0

    # Re-normalize
    masked_probs /= legal_prob_sum

    # 4. Selection (Deterministic or Stochastic)
    if random.random() >= temperature:
        chosen_index = torch.argmax(masked_probs).item()
    else:
        weighted_probs = torch.pow(masked_probs, 1.0 / temperature)
        weighted_probs /= weighted_probs.sum()
        chosen_index = torch.multinomial(weighted_probs, 1).item()

    # 5. DECODE MOVE (Fixing the Collision)
    canonical_move_uci = INDEX_TO_MOVE[chosen_index]

    # Handle Perspective Flip
    if board.turn == chess.WHITE:
        real_move_uci = canonical_move_uci
    else:
        real_move_uci = flip_move_uci(canonical_move_uci)

    # --- COLLISION RESOLUTION LOGIC ---
    # The map might return 'e2d1q' even if the piece is a King moving 'e2d1'.
    # We must check which version is actually legal on the board.
    
    candidate_move = None
    
    # Attempt 1: Try the exact string from the map (e.g. 'e2d1q')
    try:
        m = chess.Move.from_uci(real_move_uci)
        if m in board.legal_moves:
            candidate_move = m
    except:
        pass

    # Attempt 2: If that failed, and it's a promotion, try stripping the 'q' (e.g. 'e2d1')
    if candidate_move is None and len(real_move_uci) == 5 and real_move_uci.endswith('q'):
        stripped_uci = real_move_uci[:-1] # Remove 'q'
        try:
            m = chess.Move.from_uci(stripped_uci)
            if m in board.legal_moves:
                candidate_move = m
        except:
            pass

    # Attempt 3: Failsafe if both failed (should not happen if mask is correct)
    if candidate_move is None:
        print(f"Warning: Model move {real_move_uci} was masked as legal but failed verification.")
        candidate_move = random.choice(list(board.legal_moves))

    return candidate_move, pred_value, legal_prob_sum

# ###########################################################################
## 3. MONTE CARLO TREE SEARCH (For Human Play)
# ###########################################################################
class MCTSNode:
    def __init__(self, parent=None, move=None, prior_p=0.0):
        self.parent = parent
        self.move = move
        self.children = {} 
        self.N = 0 
        self.W = 0.0 
        self.P = prior_p 
        self.is_expanded = False

    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0
    
    def UCT(self, c_puct=1.5): 
        if self.parent is None: return 0.0
        q_score = -self.Q() 
        u_score = c_puct * self.P * (math.sqrt(self.parent.N) / (1 + self.N))
        return q_score + u_score
    
def get_nn_output_for_mcts(model, board, game_history, MOVE_TO_INDEX, INDEX_TO_MOVE, device):
    value, policy_probs, _ = run_model_inference(model, board, game_history, device)
    mask = create_legal_mask(board, MOVE_TO_INDEX)
    
    masked_probs = policy_probs * mask
    sum_masked = masked_probs.sum().item()

    if sum_masked == 0.0:
        legal_moves = list(board.legal_moves)
        if not legal_moves: return value, {}
        uniform = 1.0 / len(legal_moves)
        return value, {m: uniform for m in legal_moves}
    
    masked_probs /= sum_masked

    policy_dict = {}
    is_white_turn = (board.turn == chess.WHITE)
    
    for move in board.legal_moves:
        move_uci = move.uci()
        canonical_uci = move_uci if is_white_turn else flip_move_uci(move_uci)
        
        if canonical_uci in MOVE_TO_INDEX:
            idx = MOVE_TO_INDEX[canonical_uci]
            prob = masked_probs[idx].item()
            if prob > 0:
                policy_dict[move] = prob
    
    return value, policy_dict

def get_model_move_monte_carlo(model, board, game_history, MOVE_TO_INDEX, INDEX_TO_MOVE, device, time_limit_seconds):
    # 1. Instant Mate Check
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            print(f"[MCTS] Found immediate mate: {move}")
            return move
        board.pop()

    # 2. Setup
    root_board = board.copy()
    root = MCTSNode(parent=None, move=None, prior_p=1.0)
    
    if root_board.is_game_over(claim_draw=True):
        return random.choice(list(root_board.legal_moves))

    val, pol = get_nn_output_for_mcts(model, root_board, game_history, MOVE_TO_INDEX, INDEX_TO_MOVE, device)
    val = (val * 2.0) - 1.0 
    
    root.is_expanded = True
    root.N = 1
    root.W = val
    
    noise = np.random.dirichlet([0.3] * len(pol))
    for i, (m, p) in enumerate(pol.items()):
        root.children[m] = MCTSNode(parent=root, move=m, prior_p=(0.9 * p + 0.1 * noise[i]))

    start_time = time.time()
    sims = 0
    
    # 3. Search Loop
    while time.time() - start_time < time_limit_seconds:
        node = root
        temp_board = root_board.copy()
        
        while node.is_expanded and not temp_board.is_game_over():
             best_move = max(node.children.keys(), key=lambda m: node.children[m].UCT())
             node = node.children[best_move]
             temp_board.push(node.move)
        
        if not temp_board.is_game_over():
            v_raw, p_dict = get_nn_output_for_mcts(model, temp_board, game_history, MOVE_TO_INDEX, INDEX_TO_MOVE, device)
            v_leaf = (v_raw * 2.0) - 1.0
            
            node.is_expanded = True
            for m, p in p_dict.items():
                node.children[m] = MCTSNode(parent=node, move=m, prior_p=p)
        else:
            if temp_board.is_checkmate(): v_leaf = -1.0 
            else: v_leaf = 0.0

        curr = node
        while curr is not None:
            curr.N += 1
            curr.W += v_leaf
            v_leaf = -v_leaf
            curr = curr.parent
        
        sims += 1

    best_move = max(root.children.keys(), key=lambda m: root.children[m].N)
    chosen = root.children[best_move]
    print(f"\n[MCTS] {sims} sims | Best: {best_move} | Eval: {-chosen.Q():.3f}")
    return best_move


# ###########################################################################
## 4. AUTOMATED PITTING
# ###########################################################################

def run_pitting_game(model_current, model_prev, device, MOVE_TO_INDEX, INDEX_TO_MOVE, stockfish_engine):
    """Plays one game, returns result and metrics."""
    board = chess.Board()
    game_history = {} 
    models = {chess.WHITE: model_current, chess.BLACK: model_prev}
    game_metrics = {
        'total_moves': 0,
        'legal_prob_sum': 0.0,
        'policy_matches_sum': 0.0,
        'value_mse_sum': 0.0
    }

    while not board.is_game_over(claim_draw=True):
        model_to_play = models[board.turn]

        # Get Stockfish analysis for metrics
        try:
            info = stockfish_engine.analyse(board, chess.engine.Limit(time=0.1))
            stockfish_eval_cp = handle_stockfish_eval(info)
            stockfish_value = calculate_value(stockfish_eval_cp) # [0, 1]
            stockfish_move = info.get("pv", [None])[0]
        except Exception:
            stockfish_value = 0.5
            stockfish_move = None

        # Get model move (Using Fast/Direct Inference for Pitting)
        chosen_move, pred_value, legal_prob = get_model_move(
            model_to_play, board, game_history, MOVE_TO_INDEX, INDEX_TO_MOVE, device
        )

        # Update metrics
        game_metrics['total_moves'] += 1
        game_metrics['legal_prob_sum'] += legal_prob
        game_metrics['value_mse_sum'] += (pred_value - stockfish_value) ** 2
        if stockfish_move and chosen_move == stockfish_move:
            game_metrics['policy_matches_sum'] += 1

        base_fen = board.fen().rsplit(' ', 2)[0]
        game_history[base_fen] = game_history.get(base_fen, 0) + 1
        board.push(chosen_move)

    result = board.result(claim_draw=True)
    return result, game_metrics, board


def run_pitting(model_current, model_prev, device, MOVE_TO_INDEX, INDEX_TO_MOVE):
    print("\n--- 🤖 Starting Automated Pitting ---")
    print("Model (White) vs. Previous Checkpoint (Black)")
    print("Playing 100 games...")

    stockfish_engine = None
    if os.path.exists(STOCKFISH_PATH):
        try:
            stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            stockfish_engine.configure({"Threads": 1})
        except Exception as e:
            print(f"Could not start Stockfish engine: {e}")
    else:
        print(f"Warning: Stockfish not found at {STOCKFISH_PATH}. Metrics will be empty.")

    win_counts = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    total_metrics = {'moves': 0, 'legal_prob': 0.0, 'policy_match': 0.0, 'value_mse': 0.0}
    NUM_GAMES = 10

    for i in tqdm(range(NUM_GAMES)):
        result, game_metrics, final_board = run_pitting_game(
            model_current, model_prev, device, MOVE_TO_INDEX, INDEX_TO_MOVE, stockfish_engine
        )

        # --- SAVE PGN (FIRST 5 ONLY) ---
        if i < 5:
            pgn_game = chess.pgn.Game.from_board(final_board)
            pgn_game.headers["Event"] = "Model Pitting"
            pgn_game.headers["Round"] = str(i + 1)
            pgn_game.headers["White"] = "Model_Current"
            pgn_game.headers["Black"] = "Model_Previous"
            pgn_game.headers["Result"] = result
            
            pgn_filename = f"pitting_game_{i+1:02d}.pgn"
            try:
                with open(pgn_filename, "w", encoding="utf-8") as f:
                    print(pgn_game, file=f, end="\n\n")
            except Exception as e:
                print(f"Error saving PGN: {e}")

        win_counts[result] += 1
        total_metrics['moves'] += game_metrics['total_moves']
        total_metrics['legal_prob'] += game_metrics['legal_prob_sum']
        total_metrics['policy_match'] += game_metrics['policy_matches_sum']
        total_metrics['value_mse'] += game_metrics['value_mse_sum']

    if stockfish_engine: stockfish_engine.quit()

    print("\n--- Pitting Results ---")
    print(f"Games: {NUM_GAMES}")
    print(f" Model Wins (White): {win_counts['1-0']}")
    print(f" Prev. Wins (Black): {win_counts['0-1']}")
    print(f" Draws: {win_counts['1/2-1/2']}")

    if total_metrics['moves'] > 0:
        print(f"\n--- Metrics ---")
        print(f" Avg Legal Probability: {(total_metrics['legal_prob'] / total_metrics['moves']) * 100:.2f}%")
        print(f" Stockfish Match Rate: {(total_metrics['policy_match'] / total_metrics['moves']) * 100:.2f}%")
        print(f" Value MSE: {(total_metrics['value_mse'] / total_metrics['moves']):.4f}")


# ###########################################################################
## 5. MAIN
# ###########################################################################

def play_human_game(model, device, MOVE_TO_INDEX, INDEX_TO_MOVE):
    board = chess.Board()
    game_history = {}
    human_color = None
    
    while human_color is None:
        c = input("Play as (w)hite or (b)lack? ").lower()
        if c == 'w': human_color = chess.WHITE
        elif c == 'b': human_color = chess.BLACK

    print("\n--- 🧑 vs 🤖 ---")
    while not board.is_game_over(claim_draw=True):
        print("\n" + str(board))
        
        if board.turn == human_color:
            # Human
            while True:
                try:
                    uci = input("\nYour move: ")
                    m = chess.Move.from_uci(uci)
                    if m in board.legal_moves:
                        move = m
                        break
                    else: print("Illegal move.")
                except: print("Invalid UCI format (e.g., e2e4).")
        else:
            # AI
            print("\nAI Thinking (MCTS)...")
            move = get_model_move_monte_carlo(
                model, board, game_history, MOVE_TO_INDEX, INDEX_TO_MOVE, device, time_limit_seconds=5.0
            )
        
        base_fen = board.fen().rsplit(' ', 2)[0]
        game_history[base_fen] = game_history.get(base_fen, 0) + 1
        board.push(move)

    print(f"Game Over: {board.result()}")


def main():
    print("Initializing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Current Model
    model_curr, M2I, I2M = load_model_and_map(MODEL_PATH_CURRENT, device, 10, 64)
    
    # Load Previous Model (for pitting)
    # If the file is the same, it just loads the same weights again
    model_prev, _, _ = load_model_and_map(MODEL_PATH_PREVIOUS, device, 6, 64)

    while True:
        print("\n--- Chess AI Menu ---")
        print("1. Play Human vs AI (MCTS)")
        print("2. Run Pitting (AI vs Previous Ver.)")
        print("3. Quit")
        c = input("> ")

        if c == '1': 
            play_human_game(model_curr, device, M2I, I2M)
        elif c == '2': 
            run_pitting(model_curr, model_prev, device, M2I, I2M)
        elif c == '3': 
            break

if __name__ == "__main__":
    main()