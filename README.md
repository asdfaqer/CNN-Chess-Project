# Chess CNN: AI Bot & Training Pipeline

Welcome to the **Chess CNN** project. This repository contains a full-stack implementation of a Convolutional Neural Network (CNN) trained to play high-level chess. Using a custom training pipeline and Stockfish-based data generation, this model achieves significant performance improvements over standard CNN architectures.

## Quick Start: Play Against the Bot

We have developed a premium web interface for you to test the model's capabilities in real-time.

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the Backend**:
    ```bash
    cd web_app
    python app.py
    ```
3.  **Open the UI**:
    Open `web_app/index.html` in your favorite web browser.
4.  **Play**:
    Make your moves on the board. The model will analyze the position (including move history) and respond with its best move and evaluation.

---

## Model Performance (Checkpoint v1)

The v1 checkpoint represents our first major training run using the CNN architecture. Below is the training progress demonstrated through loss and accuracy curves.

![Training Progress](checkpoints_v1/training_progress.png)

### Key Metrics:
- **Architecture**: 102-channel input (8-ply history), 64 hidden dimensions.
- **Training Duration**: 7 Epochs.
- **Peak Accuracy**: Achieved high policy alignment with Stockfish 16.1.
- **Estimated Elo**: ~1500-1800 depending on the gauntlet level.

---

## Testing Walkthrough

To rigorously test the model and reproduce our results, follow these steps:

### 1. Verification of Environment
Ensure your Stockfish path is correctly set in `utils.py`. You can verify your setup by running:
```bash
python verify_test_batch.py
```

### 2. Running a Gauntlet Match
To see how the bot performs against Stockfish at different Elo levels, use the `play.py` script:
```bash
python play.py --checkpoint checkpoints_v1/epoch_7.pt
```
This will run a series of matches against Stockfish at Elos 1320, 1500, and 2000.

### 3. Elo Refinement & Tournament
For a more detailed evaluation, we use a round-robin tournament format:
```bash
python run1_tournament_rr.py
```
Results are saved to `run1_elo_refinement.json`, which tracks the Elo progression across training epochs.

### 4. Direct Move Analysis
You can use `play.py` in interactive mode (if modified) or simply observe the `web_app` evaluation logs to see how the model ranks candidate moves. The policy head outputs logits that correlate with move quality.

---

## Architecture & Pipeline

- **Data Generation**: `generate_data.py` uses multi-PV analysis to find the "Ground Truth" and alternative moves.
- **Importance Weighting**: We use a dynamic loss weight based on the EV gap, focusing the model's learning on "critical" mistakes.
- **Zstandard Pipeline**: Datasets are compressed using `zstd` to allow for massive training sets with minimal disk footprint.
- **Orchestration**: `orchestrator.py` manages the seamless transition between data generation and GPU training.

---

## Project Structure

- `/web_app`: Premium Flask-based web interface.
- `/checkpoints_v1`: Logs, history, and weights for the first stable run.
- `model.py`: The RCCN PyTorch definition.
- `trainer.py`: Custom training loop with linear warmup and milestones.
- `utils.py`: Bitboard encoding and board canonicalization logic.
