import pickle
import glob
import numpy as np
from alphazero_utils import build_alphazero_map

# Config
INPUT_PATTERN = 'chess_training_data_original_7_part_*.pkl'
OUTPUT_PREFIX = 'train_ready_7_part_'

def convert_dataset():
    # 1. Build the map once
    move_map = build_alphazero_map()
    print(f"Map built. {len(move_map)} moves.")

    files = glob.glob(INPUT_PATTERN)
    
    for fpath in files:
        print(f"Converting {fpath}...")
        
        with open(fpath, 'rb') as f:
            raw_data = pickle.load(f)
            
        boards_raw, values_raw, policies_raw = zip(*raw_data)
        
        valid_boards = []
        valid_values = []
        valid_policies = [] # These will be INTEGERS now
        
        for i, move_str in enumerate(policies_raw):
            if move_str in move_map:
                valid_boards.append(boards_raw[i])
                valid_values.append(values_raw[i])
                # CRITICAL: We save the Integer, not the String
                valid_policies.append(move_map[move_str])
        
        # Save as optimized numpy arrays to save space and load faster
        data_pack = {
            'boards': np.array(valid_boards, dtype=np.int8),
            'values': np.array(valid_values, dtype=np.float32),
            'policies': np.array(valid_policies, dtype=np.int16) # Int16 is enough for 4672
        }
        
        new_name = fpath.replace('chess_training_data_original_', OUTPUT_PREFIX)
        with open(new_name, 'wb') as f:
            pickle.dump(data_pack, f)
        print(f" -> Saved to {new_name}")

if __name__ == "__main__":
    convert_dataset()