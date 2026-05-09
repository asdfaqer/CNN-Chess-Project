# Chess CNN - Web Testing Interface

This directory contains a web-based interface for testing and playing against your Chess CNN models.

## 🚀 Quick Start

1. **Install Dependencies**:
   Ensure you have the required Python packages installed:
   ```bash
   pip install flask flask-cors torch python-chess
   ```

2. **Launch the Server**:
   Run the Flask backend:
   ```bash
   python app.py
   ```
   The server will initialize the model and start on `http://127.0.0.1:5000`.

3. **Open the UI**:
   - Open `index.html` directly in your browser.
   - Or navigate to `http://127.0.0.1:5000` in your browser.

## 📂 Required Files

The web app depends on the following files being present in this folder:
- **`app.py`**: The web server handling move requests.
- **`play.py`**: The core AI logic and MCTS implementation.
- **`chess_cnn.py` & `alphazero_utils.py`**: Core model architecture and encoding utilities.
- **`chess_cnn.pth`**: The primary model weights (Residual Blocks: 10, Filters: 64).
- **`chess_cnn_small.pth`**: Secondary model weights (Residual Blocks: 6, Filters: 64).

## 🎮 AI Modes

- **Fast Mode**: The model selects a move directly based on the highest probability from its policy head.
- **MCTS Mode**: Uses **Monte Carlo Tree Search** to simulate future games and find a stronger move. You can adjust the "Thinking Time" (default 3s) in the UI.

## 🛠️ Configuration

- **Port**: The default port is `5000`. You can change this at the bottom of `app.py`.
- **Hardware**: The server will automatically use a **CUDA-enabled GPU** if available; otherwise, it defaults to the CPU.
- **Model Architecture**: The model parameters (layers/filters) are configured in the `init_model()` call within `app.py`.

## 🐛 Troubleshooting

- **Import Errors**: If the server fails to start, verify that `chess_cnn.py` and `alphazero_utils.py` were copied from the root project directory into this folder.
- **Model Mismatch**: If you see a "RuntimeError: Error(s) in loading state_dict", ensure the `num_res_blocks` and `num_filters` in `app.py` match the architecture used during training.
