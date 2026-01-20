import os
import multiprocessing
import torch

# --- DATA GENERATION CONFIG ---
DATA_BATCH_SIZE = 20000       # Games to generate per cycle
GAMES_PER_WORKER = 50        # Amortize engine startup cost
VAL_SET_SIZE = 2000
GLOBAL_VAL_FILE = os.path.join("generated_data", "global_val.pt")
MOVE_TIME = 0.020            # 20ms per move
MAX_MOVES = 90               # Truncate long games
EPSILON_START = 0.5          # Probability of random move at start of game
MOVE_EPSILON_DECAY = 0.1     # Decay rate per move

# --- TRAINING CONFIG ---
TRAIN_EPOCHS_PER_BATCH = 10  # This now refers to total epochs to run
PATIENCE = 3                 # Stop if no improvement (though unlikely in continuous mode)
MINIBATCH_SIZE = 2048         # Reverted to a safer but high value
LEARNING_RATE = 1e-3
LR_WARMUP_STEPS = 1000        # Steps to reach peak LR
MIN_LR = 1e-6                 # Floor for scheduler
SAVE_DIR = "checkpoints_v3"
NUM_WORKERS_GEN = 4 

TRAIN_MODE = "CNN_ONLY"      # "FULL" or "CNN_ONLY"
INITIAL_CHECKPOINT = os.path.join("checkpoints", "epoch_7.pt")
CONTINUOUS_TRAINING = True   # Enable cycle-less execution
SEED = 42
USE_AMP = False
USE_COMPILE = False
DATA_CACHE_DIR = "generated_data_filtered"
HISTORY_FILE = os.path.join(SAVE_DIR, "training_history.json")
PLOT_FILE = os.path.join(SAVE_DIR, "training_progress.png")
USE_ZSTD = True  # Enable Zstandard compression for data batches

# --- LOSS CONFIG ---

# EV-based Importance Weighting (replaces activity weight)
USE_IMPORTANCE_WEIGHT = True
EV_TANH_K = 400  # Scaling factor: tanh(cp / k) maps centipawns to win probability
MULTI_PV = 5  # Number of top moves to consider for importance calculation

# Dynamic Importance (scaling loss during training based on EV gap)
USE_DYNAMIC_IMPORTANCE = True
USE_MARGIN_LOSS = False         # Use Margin Ranking Loss instead of CE or EV expectation
MARGIN_SCALE = 1.0             # Scale for the EV-regret margin (higher = more separation)
VERBOSE_DATA = True  # Store raw scores and depths for all PVs

# Simultaneous Gen/Train
BACKGROUND_GEN = False        # If True, starts next cycle generation while current one trains

# --- ARCHITECTURE ---
USE_HISTORY = True
BASE_CHANNELS = 18
DATA_CHANNELS = 102 if USE_HISTORY else BASE_CHANNELS

# --- HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
