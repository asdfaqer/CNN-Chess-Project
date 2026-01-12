import os
import sys
import datetime
import argparse
from typing import Tuple, Optional

import chess
import chess.engine
import chess.pgn
import torch
import numpy as np

from model import ChessRCCN
from utils import encode_move, board_to_tensor, STOCKFISH_PATH

# --- CONFIGURATION ---
DEFAULT_MODEL_CHECKPOINT = "checkpoints/epoch_7.pt" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HISTORY = True
DATA_CHANNELS = 102 if USE_HISTORY else 18

# --- BOT CLASS ---

class RCCNBot:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model = ChessRCCN(hidden_dim=64, use_lstm=False, input_channels=DATA_CHANNELS).to(device)
        if not os.path.exists(model_path):
            print(f"Check your path: {model_path} not found.")
            sys.exit(1)
        
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        self.model.eval()
        self.history = []

    def new_game(self):
        self.history = []

    def get_best_move(self, board: chess.Board) -> chess.Move:
        from utils import stack_history
        
        current_tensor = board_to_tensor(board)
        self.history.append(current_tensor)
        if len(self.history) > 8:
            self.history.pop(0)
            
        # stack_history expects a sequence and an index
        # We can just create a temporary sequence of the current history
        history_seq = np.array(self.history)
        input_np = stack_history(history_seq, len(self.history) - 1)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(input_tensor)
        
        logits = output.view(-1).cpu().numpy()
        
        legal_moves = list(board.legal_moves)
        best_move = None
        best_score = -float('inf')
        
        for move in legal_moves:
            idx = encode_move(move, board.turn)
            score = logits[idx]
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def observe_move(self, board: chess.Board):
        """Updates history without predicting."""
        current_tensor = board_to_tensor(board)
        self.history.append(current_tensor)
        if len(self.history) > 8:
            self.history.pop(0)

# --- MATCH LOGIC ---

def play_game(bot: RCCNBot, engine: chess.engine.SimpleEngine, engine_elo: int, bot_color: chess.Color) -> Tuple[float, chess.pgn.Game]:
    board = chess.Board()
    bot.new_game()
    
    # Setup PGN
    pgn = chess.pgn.Game()
    pgn.headers["Event"] = f"Gauntlet Match vs {engine_elo}"
    pgn.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    pgn.headers["White"] = "RCCN Bot" if bot_color == chess.WHITE else f"Stockfish {engine_elo}"
    pgn.headers["Black"] = f"Stockfish {engine_elo}" if bot_color == chess.WHITE else "RCCN Bot"
    node = pgn
    
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": engine_elo})
    
    while not board.is_game_over():
        if board.turn == bot_color:
            move = bot.get_best_move(board)
        else:
            bot.observe_move(board) 
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
            
        board.push(move)
        node = node.add_variation(move)
            
    res = board.result()
    pgn.headers["Result"] = res
    
    points = 0.5
    if res == "1-0": points = 1.0 if bot_color == chess.WHITE else 0.0
    if res == "0-1": points = 1.0 if bot_color == chess.BLACK else 0.0
    
    return points, pgn

def play_match(bot: RCCNBot, sf_path: str, elo: int, num_games: int = 5) -> float:
    print(f"\n--- MATCH VS STOCKFISH {elo} ({num_games} Games) ---")
    score = 0
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    
    try:
        for i in range(num_games):
            bot_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            points, pgn = play_game(bot, engine, elo, bot_color)
            score += points
            print(f"Game {i+1}: Result: {points}")
            
            if i == 0:
                print("\n[PGN of Game 1]")
                print(pgn)
                print("-" * 30 + "\n")
    finally:
        engine.quit()
        
    print(f"Match Result: {score}/{num_games}")
    return score

def main():
    parser = argparse.ArgumentParser(description="Play Chess RCCN Bot against Stockfish")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_MODEL_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--engine", type=str, default=STOCKFISH_PATH, help="Path to Stockfish engine")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    bot = RCCNBot(args.checkpoint, DEVICE)
    
    # Simple Gauntlet
    levels = [1320, 1500, 2000]
    total_score = 0
    for elo in levels:
        score = play_match(bot, args.engine, elo, 5)
        losses = 5 - score
        if score <= losses:
            est_elo = elo + (score - losses) * 40 # Simple heuristic
            print(f"Stopped at {elo}. Est Elo: ~{est_elo:.0f}")
            return
        total_score += score

    # If completed all
    rating_diff = score - (5 - score)
    est_elo = 2000 + (rating_diff * 400 / 5)
    print(f"\nCompleted Gauntlet! Est Elo: {est_elo:.0f}")

if __name__ == "__main__":
    main()
