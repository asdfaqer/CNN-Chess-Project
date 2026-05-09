import pickle
import numpy as np
import os
from tqdm import tqdm

# --- Configuration ---
# Point these to the files you want to check
ORIGINAL_TRAIN_FILE = 'chess_training_data_original.pkl'
UNIFORM_TRAIN_FILE = 'chess_training_data_uniform.pkl'
VALIDATION_FILE = 'validation.pkl'
# ---------------------

def load_data(filename: str) -> list:
    """Loads a processed data file and returns it as a list."""
    print(f"Loading data from {filename}...")
    if not os.path.exists(filename):
        print(f"Error: File not found at {filename}")
        print("Please make sure the file exists.")
        return []
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} samples.")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def create_sample_set(data: list) -> set:
    """
    Converts a list of data samples into a set of hashable tuples.
    This is used to find unique samples.
    """
    sample_set = set()
    if not data:
        return sample_set
        
    print("Creating set of unique samples (this may take a moment)...")
    # Data format: (board_tensor, value_target, policy_target_str)
    for sample in tqdm(data, desc="Hashing samples"):
        board_tensor, value, policy_str = sample
        
        # Convert the numpy array to hashable bytes
        hashable_tensor = board_tensor.tobytes()
        
        # Create a unique key for this sample
        sample_key = (hashable_tensor, value, policy_str)
        sample_set.add(sample_key)
        
    return sample_set

def main():
    print("--- 🕵️‍♂️ Data Leakage Checker ---")
    
    # 1. Load all three datasets
    val_data = load_data(VALIDATION_FILE)
    original_data = load_data(ORIGINAL_TRAIN_FILE)
    uniform_data = load_data(UNIFORM_TRAIN_FILE)
    
    if not val_data:
        print("Validation data is empty. Cannot check for leaks.")
        return
        
    # 2. Create a set of unique validation samples
    print("\n--- Processing Validation Data ---")
    validation_set = create_sample_set(val_data)
    print(f"Found {len(validation_set)} unique samples in {VALIDATION_FILE}")
    
    # 3. Create a set of unique original training samples
    if original_data:
        print("\n--- Processing Original Training Data ---")
        original_train_set = create_sample_set(original_data)
        print(f"Found {len(original_train_set)} unique samples in {ORIGINAL_TRAIN_FILE}")
    else:
        original_train_set = set()

    # 4. Create a set of unique uniform training samples
    if uniform_data:
        print("\n--- Processing Uniform Training Data ---")
        uniform_train_set = create_sample_set(uniform_data)
        print(f"Found {len(uniform_train_set)} unique samples in {UNIFORM_TRAIN_FILE}")
    else:
        uniform_train_set = set()

    # 5. Perform the overlap checks
    print("\n--- 📊 Results ---")
    
    # Check overlap between ORIGINAL and VALIDATION
    overlap_original = validation_set.intersection(original_train_set)
    print(f"Overlap between '{ORIGINAL_TRAIN_FILE}' and '{VALIDATION_FILE}': {len(overlap_original)} samples")
    
    # Check overlap between UNIFORM and VALIDATION
    overlap_uniform = validation_set.intersection(uniform_train_set)
    print(f"Overlap between '{UNIFORM_TRAIN_FILE}' and '{VALIDATION_FILE}': {len(overlap_uniform)} samples")
    
    print("-" * 20)
    if len(overlap_original) == 0 and len(overlap_uniform) == 0:
        print("✅ PASSED: No data leakage detected. Your datasets are clean!")
    else:
        print("❌ FAILED: Leakage detected!")
        print("This means samples from your validation set are also present in your training set.")

if __name__ == "__main__":
    main()