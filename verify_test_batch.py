import torch
from data_utils import load_batch, decompress_data

def verify():
    path = "generated_data/batch_mpv_test.zst"
    print(f"Loading {path}...")
    
    # load_batch returns the packed dictionary
    packed_dict = load_batch(path)
    print(f"Packed dict keys: {packed_dict.keys()}")
    
    # Decompress into games
    games = decompress_data(packed_dict)
    print(f"Loaded {len(games)} games.")
    
    if len(games) > 0:
        game = games[0]
        # game tuple: (states, moves, weights, [mpv_data...])
        print(f"Game 0 structure length: {len(game)}")
        
        # In verbose mode, game should have length 7:
        # 0: states, 1: moves, 2: weights, 3: cp, 4: mate, 5: depth, 6: move_indices
        if len(game) >= 7:
            print("[OK] Game contains all 7 verbose fields.")
            cp = game[3]
            mate = game[4]
            indices = game[6]
            print(f"Sample CP scores (first 5 moves): {cp[:5]}")
            print(f"Sample Mate scores (first 5 moves): {mate[:5]}")
            print(f"Sample Move indices (first 5 moves): {indices[:5]}")
        else:
            print(f"[FAIL] Game only has {len(game)} fields. Missing verbose data?")
            
if __name__ == "__main__":
    verify()
