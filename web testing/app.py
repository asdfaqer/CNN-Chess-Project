import chess
import chess.pgn
import io  # Needed to read PGN string
import torch
import torch.nn.functional as F
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. IMPORT FROM YOUR LOGIC SCRIPT (play.py) ---
try:
    from play import (
        load_model_and_map,
        get_model_move,              # The "fast" function
        get_model_move_monte_carlo,  # The "MCTS" function
        MODEL_PATH_CURRENT
    )
except ImportError as e:
    print("---! ERROR !---")
    print("Could not import from 'play.py'.")
    print("Ensure 'play.py', 'chess_cnn.py', and 'alphazero_utils.py' are in this folder.")
    print(f"Details: {e}")
    exit()

# --- 2. DEFINE GLOBALS & LOAD MODEL ON STARTUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MOVE_TO_INDEX = None
INDEX_TO_MOVE = None

def init_model():
    """Loads the model and maps into the global variables."""
    global MODEL, MOVE_TO_INDEX, INDEX_TO_MOVE
    print(f"Loading model from {MODEL_PATH_CURRENT}...")
    
    # Updated to match new signature in play.py (No map path needed)
    MODEL, MOVE_TO_INDEX, INDEX_TO_MOVE = load_model_and_map(MODEL_PATH_CURRENT, DEVICE, 10,64)
    
    if MODEL is None:
        print("---! FATAL ERROR !---")
        print("Model failed to load.")
        exit()
        
    print(f"Model loaded. Map size: {len(MOVE_TO_INDEX)}")

# --- 3. HELPER TO REBUILD GAME HISTORY ---
def build_game_history_from_pgn(pgn_string: str):
    """
    Replays a PGN string to create the game_history dict
    needed for the repetition plane calculation.
    """
    game_history = {}
    if not pgn_string:
        return game_history, chess.Board() # Return empty history and new board

    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    
    if game is None:
        return game_history, chess.Board()
        
    board = game.board()
    
    # Replay all moves to build the history
    for move in game.mainline_moves():
        # This creates the history dict your bot expects
        base_fen = board.fen().rsplit(' ', 2)[0]
        game_history[base_fen] = game_history.get(base_fen, 0) + 1
        board.push(move)
        
    # Now 'board' is at the current state, and 'game_history' is populated
    return game_history, board

# --- 4. FLASK SERVER SETUP ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/get_move", methods=["POST"])
def handle_get_move():
    data = request.json
    
    # Get all the new data from game.js
    fen = data.get("fen")
    pgn = data.get("pgn")
    ai_mode = data.get("ai_mode", "fast")
    time_limit = float(data.get("time_limit", 3.0))

    if not fen:
        return jsonify({"error": "Missing FEN string"}), 400

    # --- Rebuild history and board from PGN ---
    game_history, board = build_game_history_from_pgn(pgn)
    
    # Safety check: if PGN fails or mismatches, trust the FEN
    if board.fen() != fen:
        # print("Warning: PGN/FEN mismatch. Using FEN only (repetition detection disabled).")
        board = chess.Board(fen)
        game_history = {} 

    if board.is_game_over():
        return jsonify({"message": "Game over"})

    # --- 5. CHOOSE THE CORRECT AI FUNCTION ---
    ai_move_object = None
    try:
        if ai_mode == "mcts":
            print(f"Calling MCTS ({time_limit}s)...")
            ai_move_object = get_model_move_monte_carlo(
                MODEL, board, game_history, MOVE_TO_INDEX, INDEX_TO_MOVE, DEVICE, time_limit
            )
        else: # Default to "fast"
            # print("Calling Fast AI...")
            # Your fast function returns (move, value, prob)
            ai_move_object, _, _ = get_model_move(
                MODEL, board, game_history, MOVE_TO_INDEX, INDEX_TO_MOVE, DEVICE
            )
    except Exception as e:
        print(f"---! ERROR DURING AI MOVE CALCULATION !---")
        print(f"Mode: {ai_mode}, FEN: {fen}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "AI encountered an error"}), 500

    
    if ai_move_object:
        return jsonify({"move": ai_move_object.uci()})
    else:
        return jsonify({"error": "AI failed to find a move"}), 500

if __name__ == '__main__':
    init_model()
    app.run(debug=True, port=5000)