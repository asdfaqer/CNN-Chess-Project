import torch
import torch.nn as nn
import chess_model
import pickle
import sys
import argparse  # <-- 1. Import argparse

# --- 2. Set up Argument Parser ---
parser = argparse.ArgumentParser(description="Quantize a PyTorch Chess CNN model.")
parser.add_argument(
    '--precision', 
    type=str, 
    choices=['fp16', 'fp8'], 
    default='fp16', 
    help='Target precision for quantization (default: fp16)'
)
args = parser.parse_args()

# --- 3. Define File Paths and Settings ---
move_map_PATH = "chess_cnn_move_map.pkl"
MODEL_PATH = 'chess_cnn.pth' 

# Set target dtype and save path based on arguments
if args.precision == 'fp16':
    TARGET_DTYPE = torch.float16
    SAVE_PATH = 'chess_cnn_fp16.pth'
elif args.precision == 'fp8':
    # NOTE: FP8 (e.g., e4m3fn) requires NVIDIA H100/RTX 40-series hardware
    TARGET_DTYPE = torch.float8_e4m3fn 
    SAVE_PATH = 'chess_cnn_fp8.pth'

print(f"--- Running Quantizer ---")
print(f"Target Precision: {args.precision.upper()}")
print(f"Model Input:      {MODEL_PATH}")
print(f"Model Output:     {SAVE_PATH}")
print("-------------------------")

try:
    # --- 4. Un-pickle the move_map file ---
    with open(move_map_PATH, 'rb') as f:
        move_map = pickle.load(f)
    
    if 'INDEX_TO_MOVE' not in move_map:
        print(f"Error: 'INDEX_TO_MOVE' key not found in {move_map_PATH}.")
        sys.exit()
        
    move_list = move_map['INDEX_TO_MOVE']
    print(f"Successfully loaded move map with {len(move_list)} moves.")

    # --- 5. Create the model structure ---
    model_fp32 = chess_model.ChessCNN(move_list)
    print("Model structure created successfully.")

    # --- 6. Load the saved weights (state_dict) ---
    model_fp32.load_state_dict(torch.load(MODEL_PATH))
    print(f"Successfully loaded model weights from {MODEL_PATH}")
    
    # --- 7. Set model to evaluation mode ---
    model_fp32.eval()

    # --- 8. Convert the model to the target precision ---
    if args.precision == 'fp16':
        model_quantized = model_fp32.half()
    else: # This handles 'fp8'
        model_quantized = model_fp32.to(TARGET_DTYPE)
        
    print(f"Converted model to {args.precision.upper()}. Dtype: {next(model_quantized.parameters()).dtype}")

    # --- 9. Move model to GPU if available ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.precision == 'fp8':
        print("WARNING: FP8 is not supported on CPU and requires specific hardware. This will likely fail.")
        
    model_quantized = model_quantized.to(device)
    print(f"Model moved to {device.upper()}.")

    # --- 10. Run Inference ---
    input_data = torch.randn(4, 19, 8, 8) 
    print(f"\nUsing DUMMY input data with shape: {input_data.shape}")

    # Convert input data to the target precision
    if args.precision == 'fp16':
        input_data_quantized = input_data.half().to(device)
    else: # This handles 'fp8'
        input_data_quantized = input_data.to(TARGET_DTYPE).to(device)

    print(f"Input tensor dtype: {input_data_quantized.dtype}")

    with torch.no_grad(): 
        policy_output, value_output = model_quantized(input_data_quantized)
        
        print("--- Inference Successful ---")
        print(f"Policy output shape: {policy_output.shape}") 
        print(f"Value output shape: {value_output.shape}")

    # --- 11. SAVE THE QUANTIZED MODEL ---
    torch.save(model_quantized.state_dict(), SAVE_PATH)
    print(f"\nSuccessfully saved {args.precision.upper()} model weights to {SAVE_PATH}")

except FileNotFoundError as e:
    print(f"Error: File not found.")
    print(f"Details: {e}")
    print(f"Please check your paths: \n- {MODEL_PATH}\n- {move_map_PATH}")
except AttributeError as e:
    print(f"Error: An AttributeError occurred.")
    print(f"Details: {e}")
except RuntimeError as e:
    print(f"An error occurred during loading or inference: {e}")
    if "size mismatch" in str(e):
        print("\n*** This is a size mismatch error. ***")
    if "not implemented" in str(e) or "CPU" in str(e):
         print("\n*** This error likely means your hardware or PyTorch version does not support FP8. ***")
except Exception as e:
    print(f"An unknown error occurred: {e}")