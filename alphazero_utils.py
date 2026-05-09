import numpy as np

def build_alphazero_map():
    """
    Generates a dictionary: { 'e2e4': index } 
    where index is between 0 and 4671 (8*8*73).
    
    Total expected keys: 1,968
    """
    move_map = {}
    
    files = 'abcdefgh'
    ranks = '12345678'
    
    # 8 Directions for Queen moves (N, NE, E, SE, S, SW, W, NW)
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1), 
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    
    # 8 Knight jumps
    knight_moves = [
        (1, 2), (2, 1), (2, -1), (1, -2), 
        (-1, -2), (-2, -1), (-2, 1), (-1, 2)
    ]
    
    underpro_pieces = ['n', 'b', 'r'] 

    for f_idx, f in enumerate(files):
        for r_idx, r in enumerate(ranks):
            from_sq_str = f + r
            from_sq_idx = r_idx * 8 + f_idx 
            
            for t_f_idx, t_f in enumerate(files):
                for t_r_idx, t_r in enumerate(ranks):
                    to_sq_str = t_f + t_r
                    if from_sq_str == to_sq_str: continue

                    df = t_f_idx - f_idx
                    dr = t_r_idx - r_idx
                    
                    # --- 1. Queen/Sliding Moves (Planes 0-55) ---
                    # (Also covers Pawn pushes/captures geometrically)
                    is_queen_geometry = False
                    for dir_idx, (ddf, ddr) in enumerate(directions):
                        for dist in range(1, 8):
                            if df == ddf * dist and dr == ddr * dist:
                                plane = (dir_idx * 7) + (dist - 1)
                                idx = (plane * 64) + from_sq_idx
                                
                                # Add standard move (e.g. "e2e4" or "a7a8")
                                move_map[from_sq_str + to_sq_str] = idx
                                
                                # Add Queen Promotion suffix (e.g. "a7a8q")
                                if (r == '7' and t_r == '8') or (r == '2' and t_r == '1'):
                                    move_map[from_sq_str + to_sq_str + 'q'] = idx
                                
                                is_queen_geometry = True
                                break
                        if is_queen_geometry: break
                    
                    # --- 2. Knight Moves (Planes 56-63) ---
                    if not is_queen_geometry:
                        for k_idx, (kdf, kdr) in enumerate(knight_moves):
                            if df == kdf and dr == kdr:
                                plane = 56 + k_idx
                                idx = (plane * 64) + from_sq_idx
                                move_map[from_sq_str + to_sq_str] = idx
                                break

                    # --- 3. Underpromotions (Planes 64-72) ---
                    # CRITICAL FIX: This is now an independent check.
                    # Even if it was a "Queen geometry" (like a pawn push), 
                    # we must ALSO add the underpromotion keys if it's on the promo rank.
                    if (r == '7' and t_r == '8') or (r == '2' and t_r == '1'):
                        # Calculate direction relative to "forward"
                        # 0: Forward, 1: Right, 2: Left
                        up_dir_idx = -1
                        
                        # Logic to determine which of the 3 underpromotion clusters to use
                        # Note: We are normalizing so that left/right is relative to the board, 
                        # consistent with the AlphaZero paper's "left/right capture" concept.
                        if df == 0: 
                            up_dir_idx = 0 # Forward
                        elif df == 1: 
                            up_dir_idx = 1 # Right Diagonal
                        elif df == -1: 
                            up_dir_idx = 2 # Left Diagonal
                        
                        if up_dir_idx != -1:
                            for p_idx, p_char in enumerate(underpro_pieces):
                                # Plane Offset = 64 + (Direction * 3) + PieceType
                                plane = 64 + (up_dir_idx * 3) + p_idx
                                idx = (plane * 64) + from_sq_idx
                                move_map[from_sq_str + to_sq_str + p_char] = idx

    return move_map