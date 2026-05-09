import os
import torch
import json
import numpy as np
import multiprocessing
import math
from elo_tracker import EloTracker
from config import DATA_CHANNELS, DEVICE, TRAIN_MODE
from utils import STOCKFISH_PATH
from model import ChessRCCN

def solve_tournament_elo(matrix, names, anchor_name, anchor_elo):
    """
    matrix[i][j] = score of names[i] against names[j]
    games[i][j] = total games between names[i] and names[j]
    """
    n = len(names)
    elos = np.full(n, 1000.0)
    
    # We also include 'Stockfish 1350' in the pool
    # but we will keep its Elo fixed if we use it as an anchor.
    
    for _ in range(100): # Iterative solver
        new_elos = elos.copy()
        for i in range(n):
            if names[i] == anchor_name:
                new_elos[i] = anchor_elo
                continue
                
            total_score = 0
            expected_score = 0
            for j in range(n):
                if i == j or matrix[i][j] + matrix[j][i] == 0:
                    continue
                
                n_games = matrix[i][j] + matrix[j][i]
                actual_score = matrix[i][j]
                
                # Bradley-Terry / Elo expectation
                p_i = 1.0 / (1.0 + 10**((elos[j] - elos[i]) / 400.0))
                
                total_score += actual_score
                expected_score += n_games * p_i
            
            # MLE update (approximate)
            if expected_score > 0:
                # Basic update rule: elo += 400 * log10(actual/expected)
                # We'll use a safer damped update
                ratio = (total_score + 0.5) / (expected_score + 0.5)
                new_elos[i] += 40 * math.log10(ratio)
        
        elos = new_elos
        # Re-anchor every step to be safe
        anchor_idx = names.index(anchor_name)
        offset = anchor_elo - elos[anchor_idx]
        elos += offset

    return elos

def load_model(path):
    use_lstm = (TRAIN_MODE == "FULL")
    model = ChessRCCN(hidden_dim=64, use_lstm=use_lstm, input_channels=DATA_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False), strict=False)
    model.eval()
    return model

def run_round_robin():
    v1_dir = "checkpoints_v1"
    epochs = [1, 2, 3, 4, 5, 6, 7]
    names = [f"Epoch {e}" for e in epochs]
    paths = [os.path.join(v1_dir, f"epoch_{e}.pt") for e in epochs]
    
    # Add Stockfish 1350 to the tournament pool
    names.append("Stockfish 1350")
    n = len(names)
    
    # matrix[i][j] = total score of i against j
    score_matrix = np.zeros((n, n))
    game_matrix = np.zeros((n, n))
    
    tracker = EloTracker(STOCKFISH_PATH, num_workers=min(multiprocessing.cpu_count(), 16))
    games_per_match = 40
    
    print(f"=== Round Robin Tournament: {len(epochs)} Models + Stockfish 1350 ===")
    print(f"Total Matches: {(n*(n-1))//2}")
    
    # 1. Pre-load models (they are small, 4MB each)
    models = {}
    for i, path in enumerate(paths):
        if os.path.exists(path):
            models[i] = load_model(path)
    
    # 2. Play all pairs
    for i in range(n):
        for j in range(i + 1, n):
            name_i, name_j = names[i], names[j]
            print(f"\nMatch: {name_i} vs {name_j}")
            
            m1 = models.get(i)
            m2 = models.get(j)
            
            if name_j == "Stockfish 1350":
                # Model vs Stockfish
                score = tracker.play_parallel_match(m1, None, 1350, games_per_match, 0.8)
                score_matrix[i][j] = score
                score_matrix[j][i] = games_per_match - score
            elif name_i == "Stockfish 1350":
                # Should not happen with i < j, but for safety:
                score = tracker.play_parallel_match(m2, None, 1350, games_per_match, 0.8)
                score_matrix[j][i] = score
                score_matrix[i][j] = games_per_match - score
            else:
                # Model vs Model
                score = tracker.play_parallel_match(m1, m2, None, games_per_match, 0.8)
                score_matrix[i][j] = score
                score_matrix[j][i] = games_per_match - score
            
            game_matrix[i][j] = games_per_match
            game_matrix[j][i] = games_per_match
            print(f"  Result: {score_matrix[i][j]} - {score_matrix[j][i]}")

    # 3. Solve for Elo
    print("\nSolving for MLE Elo (Anchored to Stockfish 1350)...")
    final_elos = solve_tournament_elo(score_matrix, names, "Stockfish 1350", 1350.0)
    
    results = {}
    for i in range(n):
        results[names[i]] = {
            "elo": round(float(final_elos[i]), 1),
            "raw_scores": score_matrix[i].tolist()
        }
        print(f"  {names[i]}: {results[names[i]]['elo']} Elo")

    with open("run1_round_robin_results.json", "w") as f:
        json.dump({
            "names": names,
            "elos": results,
            "matrix": score_matrix.tolist()
        }, f, indent=4)
    
    print("\nTournament Complete. Results saved to run1_round_robin_results.json")

if __name__ == "__main__":
    run_round_robin()
