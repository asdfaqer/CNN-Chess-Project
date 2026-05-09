import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import sys
import multiprocessing
import glob
import gc
from tqdm import tqdm

# --- IMPORTS ---
try:
    from chess_model_large import ChessCNN
except ImportError:
    print("Error: 'chess_cnn.py' not found.")
    sys.exit(1)

# --- CONFIGURATION ---
# Must match the output prefix from convert_data.py
TRAIN_FILE_PATTERN = 'train_ready_4.pkl' 
VAL_DATA_FILE = 'validation.pkl' # Ensure this is also pre-converted!
MODEL_SAVE_PATH = 'chess_cnn.pth'

# --- HYPERPARAMETERS ---
BATCH_SIZE = 4096
# Fixed AlphaZero output size (8x8x73)
NUM_POLICY_OUTPUTS = 4672 
LEARNING_RATE = 0.001
NUM_EPOCHS_PER_FILE = 1
WEIGHT_DECAY = 1e-4

class ChessDataset(Dataset):
    def __init__(self, boards, values, policies):
        self.boards = boards
        self.values = values
        self.policies = policies

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.boards[idx]).float(),
            self.values[idx],
            self.policies[idx]
        )

def load_preconverted_data(filepath):
    """
    Loads data that has ALREADY been converted to integers/tensors.
    Zero processing logic here = Maximum Speed.
    """
    print(f"Loading {filepath}...")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return None, None, None

    # Verify format
    if not isinstance(data, dict):
        print(f"SKIP: {filepath} is not a dictionary. Did you run convert_data.py?")
        return None, None, None

    # Extract Numpy arrays (saved as int8/int16 to save space)
    boards_np = data['boards'] 
    values_np = data['values']
    policies_np = data['policies']

    # Check if file is empty
    if len(boards_np) == 0:
        print(f"SKIP: {filepath} contains 0 samples.")
        return None, None, None

    # Wrap in Tensors
    # values need to be (N, 1)
    values_tensor = torch.from_numpy(values_np).float().unsqueeze(1)
    # policies need to be Long for CrossEntropy
    policy_tensor = torch.from_numpy(policies_np).long()
    
    return boards_np, values_tensor, policy_tensor

def main():
    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True 
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        use_amp = True 
    else:
        device = torch.device("cpu")
        use_amp = False
        print("Using CPU")

    # 2. Locate Files
    training_files = sorted(glob.glob(TRAIN_FILE_PATTERN))
    if not training_files:
        print(f"No files found matching '{TRAIN_FILE_PATTERN}'")
        print("Please run 'convert_data.py' first to generate these files.")
        sys.exit(1)

    # 3. Initialize Model
    # We don't need to build the map here anymore, we know the size is 4672.
    model = ChessCNN(num_policy_outputs=NUM_POLICY_OUTPUTS).to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model weights...")
        try:
            state = torch.load(MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(state)
        except Exception as e:
            print(f"Warning: Could not load weights: {e}. Starting fresh.")

    # 4. Optimizer
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 5. Validation Loader (Optional)
    val_loader = None
    if os.path.exists(VAL_DATA_FILE):
        vb, vv, vp = load_preconverted_data(VAL_DATA_FILE)
        if vb is not None:
            print("Validation set loaded.")
            val_dataset = ChessDataset(vb, vv, vp)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 6. TRAINING LOOP
    print(f"Starting training on {len(training_files)} files...")
    
    for file_idx, file_path in enumerate(training_files):
        print(f"\nProcessing {file_idx + 1}/{len(training_files)}: {file_path}")

        train_boards, train_values, train_policies = load_preconverted_data(file_path)
        
        if train_boards is None: continue

        train_dataset = ChessDataset(train_boards, train_values, train_policies)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        for epoch in range(NUM_EPOCHS_PER_FILE):
            model.train()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", mininterval=1.0)
            
            for boards, values, policies in pbar:
                boards = boards.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True)
                policies = policies.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred_values, pred_policy_logits = model(boards)
                    loss_v = value_loss_fn(pred_values, values)
                    loss_p = policy_loss_fn(pred_policy_logits, policies)
                    total_loss = loss_v + loss_p

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.set_postfix(v_loss=loss_v.item(), p_loss=loss_p.item())

            # Validation
            if val_loader:
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for boards, values, policies in val_loader:
                        boards = boards.to(device)
                        policies = policies.to(device)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            _, pp = model(boards)
                            _, predicted = torch.max(pp, 1)
                            val_correct += (predicted == policies).sum().item()
                            val_total += boards.size(0)
                print(f" -> Val Acc: {100 * val_correct/val_total:.2f}%")

            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Force Cleanup
        del train_dataset, train_loader, train_boards, train_values, train_policies
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("All files processed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()