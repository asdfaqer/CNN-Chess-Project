import os
import torch
import json
import numpy as np
import multiprocessing
from elo_tracker import EloTracker
from config import DATA_CHANNELS, DEVICE
from utils import STOCKFISH_PATH

def run1_tournament():
    print("=== Run 1 Epoch Evaluation Tournament ===")
    
    # Run 1 checkpoints are in checkpoints_v1
    v1_dir = "checkpoints_v1"
    epochs = [1, 2, 3, 4, 5, 6, 7]
    checkpoints = [os.path.join(v1_dir, f"epoch_{e}.pt") for e in epochs]
    
    # We'll use Stockfish at various Elo levels as benchmarks
    # Run 1 models are likely in the 800-1400 range based on previous evaluations
    sf_levels = [800, 1000, 1200, 1400]
    games_per_level = 40  # Total 160 games per epoch against SF
    
    tracker = EloTracker(STOCKFISH_PATH, num_workers=min(multiprocessing.cpu_count(), 16))
    
    results = {}
    
    for epoch, cp_path in zip(epochs, checkpoints):
        if not os.path.exists(cp_path):
            print(f"Skipping epoch {epoch}, checkpoint not found.")
            continue
            
        print(f"\nEvaluating Epoch {epoch}...")
        
        match_results = []
        for sf_elo in sf_levels:
            score = tracker.play_parallel_match(
                m1_path_or_model=cp_path, 
                m2=None, 
                sf_elo=sf_elo, 
                num_games=games_per_level, 
                temperature=0.8
            )
            match_results.append((sf_elo, score, games_per_level))
            print(f"  vs Stockfish {sf_elo}: {score}/{games_per_level}")
        
        elo, sigma = tracker.calculate_mle_elo(match_results)
        results[epoch] = {
            "elo": elo,
            "sigma": round(sigma, 2),
            "match_results": match_results
        }
        print(f"  Result: {elo} Elo (sigma: {sigma:.2f})")

    # Save results
    output_path = "run1_elo_refinement.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nTournament complete. Results saved to {output_path}")

# Patch EloTracker to accept model path directly for easier evaluation if needed
# Actually play_parallel_match takes ChessRCCN, so we need a wrapper
def evaluate_path(cp_path, sf_levels, games_per_level):
    from model import ChessRCCN
    from config import TRAIN_MODE
    use_lstm = (TRAIN_MODE == "FULL")
    model = ChessRCCN(hidden_dim=64, use_lstm=use_lstm, input_channels=DATA_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(cp_path, map_location=DEVICE, weights_only=False), strict=False)
    model.eval()
    
    tracker = EloTracker(STOCKFISH_PATH, num_workers=min(multiprocessing.cpu_count(), 16))
    match_results = []
    for sf_elo in sf_levels:
        score = tracker.play_parallel_match(model, None, sf_elo, games_per_level, 0.8)
        match_results.append((sf_elo, score, games_per_level))
    
    elo, sigma = tracker.calculate_mle_elo(match_results)
    return {
        "elo": elo,
        "sigma": round(sigma, 2),
        "match_results": match_results
    }

if __name__ == "__main__":
    v1_dir = "checkpoints_v1"
    epochs = [1, 2, 3, 4, 5, 6, 7]
    sf_levels = [1350, 1400, 1450, 1500]
    games_per_level = 40
    
    final_results = {}
    for e in epochs:
        path = os.path.join(v1_dir, f"epoch_{e}.pt")
        if os.path.exists(path):
            print(f"Tournament Round: Epoch {e}")
            try:
                res = evaluate_path(path, sf_levels, games_per_level)
                final_results[e] = res
                print(f"  -> {res['elo']} Elo +/- {res['sigma']}")
            except Exception as e:
                print(f"  [ERROR] Evaluation failed for epoch {e}: {e}")
            
    with open("run1_elo_refinement.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("\n Tournament Complete. Results saved to run1_elo_refinement.json")
