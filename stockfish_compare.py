import chess
import chess.engine
import os
import statistics

# ---------------------------------------------------------------------------
# IMPORTANT: UPDATE THIS PATH
# You MUST change this to the full path of your Stockfish executable
#
# Examples:
# STOCKFISH_PATH = "C:\\Users\\YourName\\Downloads\\stockfish\\stockfish.exe"
# STOCKFISH_PATH = "/usr/local/bin/stockfish"
# ---------------------------------------------------------------------------
STOCKFISH_PATH = r"C:\Users\ccbdc\Desktop\stockfish\stockfish-windows-x86-64-avx2.exe"


def get_centipawns(score: chess.engine.PovScore) -> int:
    """
    Converts a PovScore object to a numerical centipawn value.
    Assigns a very large value for mates.
    """
    if score.is_mate():
        # A 'mate in x' is given a very high (or low) score
        # We cap it to make it a large but finite number
        mate_score = 100000
        if score.mate() > 0:
            return mate_score  # White has a mate in X
        else:
            return -mate_score # Black has a mate in X
    else:
        # This is the centipawn score, relative to the current player
        return score.relative.score(mate_score=100000)

def compare_engine_evals(num_games: int = 100):
    """
    Plays N games, comparing evals from a 5ms and 50ms engine at each ply.
    """
    
    # --- 1. Validation and Setup ---
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Error: Stockfish executable not found at '{STOCKFISH_PATH}'")
        print("Please download Stockfish and update the STOCKFISH_PATH variable.")
        return

    print("Initializing Stockfish engines...")
    try:
        # Initialize two separate engine processes
        engine_5ms = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine_5ms_var_test = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine_10ms = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine_20ms = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine_50ms = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine_100ms = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except (OSError, chess.engine.EngineError) as e:
        print(f"Error initializing engine: {e}")
        print("Please check your STOCKFISH_PATH and engine permissions.")
        return

    # Define the time limits
    limit_5ms = chess.engine.Limit(time=0.001)
    limit_10ms = chess.engine.Limit(time=0.01)
    limit_20ms = chess.engine.Limit(time=0.02)
    limit_50ms = chess.engine.Limit(time=0.05)
    limit_100ms = chess.engine.Limit(time=0.1)

    print(f"Starting comparison for {num_games} games...")

    # --- 2. Game Loop ---
    for i in range(num_games):
        board = chess.Board()

        try:
            matchs_1 = 0
            matchs_2 = 0
            matchs_3 = 0
            matchs_4 = 0
            matchs_1_var = 0
            total_moves = 0
            while not board.is_game_over(claim_draw=True):
                
                # --- A. Play a move ---
                # We use the 50ms engine to play a "real" move to advance the game
                # We use 'play' here, which just returns the best move
                result = engine_100ms.play(board, limit_50ms)
                if result.move is None:
                    break # Game ended unexpectedly
                
                board.push(result.move)

                # --- B. Analyze the new position ---
                # Now, we ask both engines to 'analyse' (evaluate) this *same* position
                result_1 = engine_5ms.play(board, limit_5ms)

                result_2 = engine_10ms.play(board, limit_10ms)

                result_3 = engine_20ms.play(board, limit_20ms)

                result_4 = engine_50ms.play(board, limit_50ms)

                result_5 = engine_100ms.play(board, limit_100ms)

                result_1_var = engine_5ms_var_test.play(board, limit_5ms)

                # --- C. Compare Evals ---
                move_1 = result_1.move
                move_2 = result_2.move
                move_3 = result_3.move
                move_4 = result_4.move
                move_5 = result_5.move
                
                move_1_var = result_1_var.move
                # move_5 is the most reliable move
                if move_1 == move_5:
                    matchs_1 += 1
                if move_2 == move_5:
                    matchs_2 += 1
                if move_3 == move_5:
                    matchs_3 += 1
                if move_4 == move_5:
                    matchs_4 += 1
                if move_1 == move_1_var:
                    matchs_1_var += 1
                total_moves += 1
                    

        

        except (chess.engine.EngineError, BrokenPipeError) as e:
            print(f"\nEngine crashed on game {i+1}: {e}")
            print("Restarting engines...")
            # Quit and restart the engines if they fail
            engine_5ms.quit()
            engine_50ms.quit()
            engine_5ms = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            engine_50ms = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # --- 3. Final Report ---
    print(f"Game {i+1}/{num_games} completed.")
    print(f"5ms matches 100ms: {matchs_1}/{total_moves} ({(matchs_1/total_moves)*100:.2f}%)")
    print(f"10ms matches 100ms: {matchs_2}/{total_moves} ({(matchs_2/total_moves)*100:.2f}%)")
    print(f"20ms matches 100ms: {matchs_3}/{total_moves} ({(matchs_3/total_moves)*100:.2f}%)")
    print(f"50ms matches 100ms: {matchs_4}/{total_moves} ({(matchs_4/total_moves)*100:.2f}%)")
    print(f"5ms matches 5ms (var test): {matchs_1_var}/{total_moves} ({(matchs_1_var/total_moves)*100:.2f}%)")
    print("-" * 40)
    engine_5ms.quit()
    engine_10ms.quit()
    engine_20ms.quit()
    engine_50ms.quit()
    engine_100ms.quit()
    engine_5ms_var_test.quit()



# --- Run the comparison ---
if __name__ == "__main__":
    compare_engine_evals(1)