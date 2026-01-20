import os
import chess
import chess.engine
import torch
import numpy as np
import math
import threading
from typing import Tuple, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_CHANNELS, DEVICE, TRAIN_MODE
from model import ChessRCCN
from utils import board_to_tensor, stack_history, encode_move, STOCKFISH_PATH

class EloTracker:
    def __init__(self, stockfish_path: str = STOCKFISH_PATH, num_workers: int = 4):
        self.stockfish_path = stockfish_path
        self.num_workers = num_workers
        self._thread_local = threading.local()
    
    def _get_engine(self):
        """Get or create a thread-local Stockfish engine."""
        if not hasattr(self._thread_local, "engine") or self._thread_local.engine is None:
            if not os.path.exists(self.stockfish_path) and not self.stockfish_path.endswith(".exe"):
                 # If it's just 'stockfish', it might be in PATH, so we can't os.path.exists it easily
                 pass 
            
            try:
                self._thread_local.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except Exception as e:
                print(f"[ERROR] Could not start Stockfish engine at {self.stockfish_path}: {e}")
                print(f"Please check your STOCKFISH_PATH in utils.py")
                raise FileNotFoundError(f"Stockfish not found at {self.stockfish_path}")
        return self._thread_local.engine

    def __del__(self):
        # Note: Thread-local engines might not be cleaned up perfectly here 
        # but popen_uci processes usually exit when the parent dies or we can quit them explicitly.
        pass

    def get_best_move(self, model: ChessRCCN, board: chess.Board, history: list, temperature: float = 1.0) -> chess.Move:
        current_tensor = board_to_tensor(board)
        history.append(current_tensor)
        if len(history) > 8:
            history.pop(0)
            
        history_seq = np.array(history)
        input_np = stack_history(history_seq, len(history) - 1)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output, _ = model(input_tensor)
            
        logits = output.view(-1).cpu().numpy()
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None

        move_logits = []
        for move in legal_moves:
            idx = encode_move(move, board.turn)
            move_logits.append(logits[idx])
        
        move_logits = np.array(move_logits)

        if temperature <= 1e-6:
            best_idx = np.argmax(move_logits)
            return legal_moves[best_idx]
        else:
            move_logits = move_logits / temperature
            exp_logits = np.exp(move_logits - np.max(move_logits))
            probs = exp_logits / np.sum(exp_logits)
            move_idx = np.random.choice(len(legal_moves), p=probs)
            return legal_moves[move_idx]

    def play_single_game(self, m1: ChessRCCN, m2: Optional[ChessRCCN], sf_elo: Optional[int], 
                         is_m1_white: bool, temperature: float = 1.0) -> float:
        """
        Play a single game. If m2 is None, play against Stockfish at sf_elo.
        Returns score for m1 (1.0 win, 0.5 draw, 0.0 loss).
        """
        board = chess.Board()
        history = []
        m1_color = chess.WHITE if is_m1_white else chess.BLACK
        
        try:
            engine = None
            if m2 is None:
                engine = self._get_engine()
                engine.configure({"UCI_Elo": sf_elo, "UCI_LimitStrength": True})

            while not board.is_game_over(claim_draw=True) and board.fullmove_number < 150:
                if board.turn == m1_color:
                    move = self.get_best_move(m1, board, history, temperature=temperature)
                else:
                    if m2 is not None:
                        # Copy of history for opponent model
                        # Note: In a fully correct implementation history needs careful handling per player
                        # but simple history list usually works for short sequences.
                        move = self.get_best_move(m2, board, history.copy(), temperature=temperature)
                    else:
                        result = engine.play(board, chess.engine.Limit(time=0.01))
                        move = result.move
                
                if move is None: break
                board.push(move)
            
            result = board.result(claim_draw=True)
            if result == "1-0": return 1.0 if m1_color == chess.WHITE else 0.0
            elif result == "0-1": return 1.0 if m1_color == chess.BLACK else 0.0
            else: return 0.5
        except Exception as e:
            print(f"Game error: {e}")
            raise e # Raise to fail properly if engine config fails

    def play_parallel_match(self, m1: ChessRCCN, m2: Optional[ChessRCCN], sf_elo: Optional[int], 
                           num_games: int, temperature: float = 1.0) -> float:
        """Play a full match in parallel."""
        total_score = 0.0
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(num_games):
                is_m1_white = (i % 2 == 0)
                futures.append(executor.submit(self.play_single_game, m1, m2, sf_elo, is_m1_white, temperature))
            
            for future in as_completed(futures):
                total_score += future.result()
        return total_score

    def calculate_mle_elo(self, results: List[Tuple[float, float, int]]) -> Tuple[int, float]:
        if not results: return 1200, 400.0
        total_score = sum(r[1] for r in results)
        total_games = sum(r[2] for r in results)
        low, high = 200, 3000
        for _ in range(30):
            mid = (low + high) / 2
            expected_score = 0
            variance = 0
            for opp_elo, score, n_games in results:
                p = 1.0 / (1.0 + 10**((opp_elo - mid) / 400.0))
                expected_score += n_games * p
                variance += n_games * p * (1.0 - p)
            if expected_score < total_score:
                low = mid
            else:
                high = mid
        rating = int(mid)
        se = 173.7 / math.sqrt(variance) if variance > 0 else 400.0
        return rating, se

    def estimate_elo_uniform(self, model_path: str, pool: List[Tuple[str, float]], 
                            stockfish_pool: List[int] = None,
                            games_per_opponent: int = 20, temperature: float = 0.8) -> int:
        use_lstm = (TRAIN_MODE == "FULL")
        m1 = ChessRCCN(hidden_dim=64, use_lstm=use_lstm, input_channels=DATA_CHANNELS).to(DEVICE)
        m1.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False), strict=False)
        m1.eval()

        all_results = []
        print(f"\n[Elo] Parallel Uniform Pool Evaluation (T={temperature})...")
        
        # 1. Model Pool
        for opp_path, opp_elo in pool:
            if not os.path.exists(opp_path): continue
                
            m2 = ChessRCCN(hidden_dim=64, use_lstm=use_lstm, input_channels=DATA_CHANNELS).to(DEVICE)
            m2.load_state_dict(torch.load(opp_path, map_location=DEVICE, weights_only=False), strict=False)
            m2.eval()
            
            score = self.play_parallel_match(m1, m2, None, games_per_opponent, temperature)
            all_results.append((opp_elo, score, games_per_opponent))
            print(f"    vs {os.path.basename(opp_path)} ({opp_elo}): {score}/{games_per_opponent}")
            del m2
            torch.cuda.empty_cache()

        # 2. Stockfish
        if stockfish_pool:
            for sf_elo in stockfish_pool:
                score = self.play_parallel_match(m1, None, sf_elo, games_per_opponent, temperature)
                all_results.append((sf_elo, score, games_per_opponent))
                print(f"    vs Stockfish ({sf_elo}): {score}/{games_per_opponent}")

        if not all_results: return 1200
        final_elo, sigma = self.calculate_mle_elo(all_results)
        print(f"  -> Final Estimated Elo: {final_elo} (sigma={sigma:.1f})\n")
        
        # Attempt to close thread-local engines (imperfect but helps)
        # In a long-running app you'd want better lifecycle management.
        return final_elo
