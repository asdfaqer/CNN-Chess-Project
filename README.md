# CNN Chess Project

A Convolutional Neural Network (CNN) trained to play chess, inspired by AlphaZero.

## Project Structure

- **`chess_model.py`**: Primary neural network architecture.
- **`train.py`**: Main training pipeline.
- **`data_gen.py`**: Data generation and preprocessing scripts.
- **`web testing/`**: A web-based interface to play against the model.
- **`img/`**: Assets for the web interface.

## Web Interface

For instructions on how to run the web-based testing UI, see the [web testing/README.md](web%20testing/README.md).

## Getting Started

1. Install dependencies:
   ```bash
   pip install torch python-chess numpy flask flask-cors
   ```
2. Train the model or use a provided checkpoint (`chess_cnn.pth`).
3. Run the web app:
   ```bash
   cd "web testing"
   python app.py
   ```
