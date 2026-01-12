# Chess RCCN (Recurrent Convolutional Neural Network)

A high-performance chess engine training pipeline using PyTorch. This project implements a Recurrent Convolutional Neural Network (RCCN) with support for simultaneous data generation and training.

## Features

- **RCCN Architecture**: Hybrid model combining CNN feature extraction with optional LSTM history processing.
- **Simultaneous Pipeline**: Background data generation with Stockfish multi-PV analysis while the model trains on previous batches.
- **Dynamic Importance Weighting**: Loss function adjusted based on the EV gap between model predictions and Stockfish's top moves.
- **Advanced LR Scheduling**: Linear warmup (1000 steps) with 1/8th-epoch validation milestones using `ReduceLROnPlateau`.
- **Zstandard Compression**: Efficient storage of massive move-level datasets.
- **Automated Elo Tracking**: Integrated round-robin tournament evaluation anchored against Stockfish.

## Project Structure

- `model.py`: Neural network architecture.
- `trainer.py`: Core training loop, scheduler, and validation logic.
- `orchestrator.py`: Manages the generation-training cycle.
- `generate_data.py`: Multi-threaded data generation using Stockfish.
- `config.py`: Centralized hyperparameters and path settings.
- `elo_tracker.py`: Tools for model vs. model and model vs. engine matches.

## Getting Started

1. Set up your Stockfish path in `utils.py` or via environment variables.
2. Generate validation data: `python gen_val_data.py`
3. Start the training pipeline: `python orchestrator.py`

## Configuration

Edit `config.py` to adjust:
- Learning rate and warmup steps.
- Dataset sizes and batching.
- Model architecture (CNN_ONLY vs FULL).
- Hardware utilization (AMP, Compile, Workers).
