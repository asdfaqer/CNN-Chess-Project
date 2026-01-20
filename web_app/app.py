import os
import sys
import chess
import chess.pgn
import io
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path so we can import model and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import ChessRCCN
from utils import encode_move, board_to_tensor, stack_history

# --- CONFIGURATION ---
DEFAULT_MODEL_CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints_v1', 'epoch_7.pt'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HISTORY = True
DATA_CHANNELS = 102 if USE_HISTORY else 18

app = Flask(__name__)
CORS(app)

# Global model instance
MODEL = None

def init_model(checkpoint_path=DEFAULT_MODEL_CHECKPOINT):
    global MODEL
    print(f"Loading model from {checkpoint_path}...")
    MODEL = ChessRCCN(hidden_dim=64, use_lstm=False, input_channels=DATA_CHANNELS).to(DEVICE)
    if not os.path.exists(checkpoint_path):
        print(f"FAILED: Checkpoint not found at {checkpoint_path}")
        return False
    
    MODEL.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
    MODEL.eval()
    print("Model loaded successfully.")
    return True

def get_history_tensors(pgn_string: str):
    """Replays PGN to get a sequence of board tensors."""
    tensors = []
    if not pgn_string:
        # Just the starting board
        return [board_to_tensor(chess.Board())]

    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        return [board_to_tensor(chess.Board())]

    board = game.board()
    tensors.append(board_to_tensor(board))
    
    for move in game.mainline_moves():
        board.push(move)
        tensors.append(board_to_tensor(board))
    
    return tensors

@app.route("/get_move", methods=["POST"])
def handle_get_move():
    data = request.json
    fen = data.get("fen")
    pgn = data.get("pgn", "")
    
    if not fen:
        return jsonify({"error": "Missing FEN"}), 400

    try:
        # Rebuild history
        history_tensors = get_history_tensors(pgn)
        # We only need up to the last 8 states for stack_history
        if len(history_tensors) > 8:
            history_tensors = history_tensors[-8:]
        
        history_seq = np.array(history_tensors)
        input_np = stack_history(history_seq, len(history_seq) - 1)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(DEVICE)
        
        board = chess.Board(fen)
        
        with torch.no_grad():
            output, _ = MODEL(input_tensor)
        
        logits = output.view(-1).cpu().numpy()
        
        legal_moves = list(board.legal_moves)
        best_move = None
        best_score = -float('inf')
        
        # We need to collect probabilities for the UI to show "top moves"
        move_scores = []
        
        for move in legal_moves:
            idx = encode_move(move, board.turn)
            score = float(logits[idx])
            move_scores.append({"move": move.uci(), "score": score})
            if score > best_score:
                best_score = score
                best_move = move
        
        # Sort moves by score
        move_scores.sort(key=lambda x: x["score"], reverse=True)
        
        if best_move:
            return jsonify({
                "move": best_move.uci(),
                "top_moves": move_scores[:5]
            })
        else:
            return jsonify({"error": "No legal moves found"}), 500

    except Exception as e:
        print(f"Error in move calculation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "ready" if MODEL else "loading",
        "device": str(DEVICE),
        "checkpoint": DEFAULT_MODEL_CHECKPOINT
    })

if __name__ == "__main__":
    if init_model():
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        print("Initialization failed.")
