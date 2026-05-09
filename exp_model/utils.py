import chess
import numpy as np
import torch
import os
import platform

# --- CONFIGURATION ---
def _find_stockfish():
    if platform.system() == "Windows":
        # Check specific hardcoded path
        primary = r"D:\stockfish\stockfish-windows-x86-64-avx2.exe"
        if os.path.exists(primary):
            return primary
        
        secondary = r"C:\Users\ccbdc\Desktop\stockfish\stockfish-windows-x86-64-avx2.exe"
        if os.path.exists(secondary):
            return secondary
        
        # Check standard desktop folder (assuming user moved it or folder name changed)
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        search_dirs = [
            desktop,
            os.path.join(desktop, "stockfish"),
            "C:\\stockfish",
            "."
        ]
        
        for d in search_dirs:
            if not os.path.exists(d): continue
            for f in os.listdir(d):
                if "stockfish" in f.lower() and f.lower().endswith(".exe"):
                    return os.path.abspath(os.path.join(d, f))
        
        return "stockfish.exe" # Fallback to PATH
    else:
        # Linux/macOS
        for p in ["/usr/local/bin/stockfish", "/usr/bin/stockfish", "stockfish"]:
            if os.path.exists(p): return p
        return "stockfish"

STOCKFISH_PATH = _find_stockfish()
print(f"[*] Using Stockfish at: {STOCKFISH_PATH}")

def encode_move(move: chess.Move, turn: chess.Color) -> int:
    """
    Encodes a chess.Move into an AlphaZero index (0-4671).
    Handles vertical flipping if it's Black's turn.
    """
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(move.from_square)
        to_sq = chess.square_mirror(move.to_square)
    else:
        from_sq = move.from_square
        to_sq = move.to_square

    from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
    to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
    
    dx = to_file - from_file
    dy = to_rank - from_rank

    compass = [ (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1) ]
    plane_idx = -1

    is_underpromotion = (move.promotion is not None) and (move.promotion != chess.QUEEN)

    if not is_underpromotion:
        for dir_idx, (dn_x, dn_y) in enumerate(compass):
            for dist in range(1, 8):
                if dx == dn_x * dist and dy == dn_y * dist:
                    plane_idx = dir_idx * 7 + (dist - 1)
                    break
            if plane_idx != -1: break

        if plane_idx == -1:
            knight_moves = [ (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2) ]
            for k_idx, (kn_x, kn_y) in enumerate(knight_moves):
                if dx == kn_x and dy == kn_y:
                    plane_idx = 56 + k_idx
                    break
    else:
        promo_map = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
        promo_code = promo_map.get(move.promotion, 0)
        direction_code = 1 
        if dx == -1: direction_code = 0
        elif dx == 1: direction_code = 2
        plane_idx = 64 + (direction_code * 3) + promo_code

    if plane_idx == -1: return 0 
    return (from_sq * 73) + plane_idx

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Returns [18, 8, 8] tensor optimally using bitboards.
    Canoncialized: Always presented as White to move (Black flipped).
    Plane 17 is 50-move rule proximity (>= 98 half-moves).
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    us = board.turn
    them = not us
    
    # Mirror if Black to move so we always have a "White" perspective
    mirror = (us == chess.BLACK)
    
    # Piece order for planes
    # US: P, N, B, R, Q, K
    # THEM: P, N, B, R, Q, K
    for p_idx, piece_type in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]):
        # US pieces
        bb = board.pieces(piece_type, us)
        for sq in chess.SquareSet(bb):
            if mirror: sq = chess.square_mirror(sq)
            tensor[p_idx, sq // 8, sq % 8] = 1.0
            
        # THEM pieces
        bb = board.pieces(piece_type, them)
        for sq in chess.SquareSet(bb):
            if mirror: sq = chess.square_mirror(sq)
            tensor[p_idx + 6, sq // 8, sq % 8] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(us): tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(us): tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(them): tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(them): tensor[15, :, :] = 1.0
    
    if board.ep_square is not None:
        sq = board.ep_square
        if mirror: sq = chess.square_mirror(sq)
        tensor[16, sq // 8, sq % 8] = 1.0

    if board.halfmove_clock >= 98:
        tensor[17, :, :] = 1.0

    return tensor

def stack_history(states: np.ndarray, move_idx: int) -> np.ndarray:
    """
    Assembles a [102, 8, 8] tensor from a sequence of [T, 18, 8, 8] states.
    - move_idx: The index of the current move in the sequence.
    - Includes 18 current planes + 7 * 12 history piece planes = 18 + 84 = 102.
    - Odd history indices are flipped on both axes to maintain consistent piece positions.
    """
    stacked = np.zeros((102, 8, 8), dtype=states.dtype)
    
    # current_state: [18, 8, 8]
    current_state = states[move_idx]
    stacked[:18] = current_state
    
    for h_idx in range(1, 8):
        prev_idx = move_idx - h_idx
        if prev_idx >= 0:
            hist_state = states[prev_idx]
            # Piece planes (0-11)
            piece_planes = hist_state[:12].copy()
            
            # Flip both spatial axes for odd history indices to keep pieces
            # on the same squares (compensating for White/Black perspective swap)
            if h_idx % 2 != 0:
                piece_planes = np.flip(piece_planes, axis=(1, 2))
                
            start_p = 18 + (h_idx - 1) * 12
            stacked[start_p : start_p + 12] = piece_planes
            
    return stacked

def compute_activity_weight(board: chess.Board) -> float:
    """
    Computes a normalized activity score based on bitboard popcounts.
    Much faster than square-by-square iteration.
    """
    # Get bitboards of all attacked squares
    # This approximates mobility/activity
    w_attackers = 0
    b_attackers = 0
    
    # Iterate pieces and get their attack bitboards
    for piece_type in range(1, 7): # PAWN to KING
        for sq in board.pieces(piece_type, chess.WHITE):
            w_attackers += int(board.attacks(sq)).bit_count()
        for sq in board.pieces(piece_type, chess.BLACK):
            b_attackers += int(board.attacks(sq)).bit_count()
            
    attacked_total = w_attackers + b_attackers
    # Normalize: typical range for total attacks is 40-120
    return max(0.5, min(attacked_total / 60.0, 2.0))

def get_dynamic_resource_info():
    """
    Returns system resource info for dynamic scaling.
    """
    import multiprocessing
    info = {
        "cpu_count": multiprocessing.cpu_count(),
        "gpu_mem_free": 0.0,
        "gpu_mem_total": 0.0,
        "device": "cpu"
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["gpu_mem_free"] = free / (1024**3)
        info["gpu_mem_total"] = total / (1024**3)
        info["device"] = "cuda"
    return info
