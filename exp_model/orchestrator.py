import os
import time
import torch
import multiprocessing
import random
import numpy as np
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm


from config import (DATA_CACHE_DIR, GLOBAL_VAL_FILE, DEVICE, TRAIN_MODE, 
                   DATA_CHANNELS, INITIAL_CHECKPOINT, USE_COMPILE, 
                   PATIENCE, MINIBATCH_SIZE, USE_HISTORY, SAVE_DIR, SEED,
                   AMP_DTYPE, USE_AMP, TRAIN_EPOCHS_PER_BATCH, BACKGROUND_GEN)
from data_utils import decompress_data, ChessDataset, PositionDataset, collate_fn, simple_collate, load_history, load_batch, get_latest_checkpoint
from trainer import Trainer
from model import ChessRCCN
from utils import get_dynamic_resource_info
from elo_tracker import EloTracker
import generate_data

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def background_generator_process():
    """Eternal process to keep generating next data batches."""
    import sys
    # Redirect output to a log file to prevent console corruption with TQDM bars
    log_file = open("datagen.log", "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(f"\n--- Starting Background Generator Log: {time.ctime()} ---")
    try:
        generate_data.main()
    except KeyboardInterrupt:
        print("[PRODUCER] Background Generator search stopped.")
    finally:
        log_file.close()

def main():
    multiprocessing.freeze_support()
    set_seed(SEED)
    
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    print("[*] Background generator output redirected to 'datagen.log'")
    # 1. Start Background Generator (Producer)
    # MUST be non-daemonic because it spawns its own multiprocessing Pool
    p_gen = None
    if BACKGROUND_GEN:
        p_gen = multiprocessing.Process(target=background_generator_process, daemon=False)
        p_gen.start()
    else:
        print("[*] Background generator is DISABLED in config.")
    
    try:
        # 2. Initialize Model & Trainer
        use_lstm = (TRAIN_MODE == "FULL")
        model = ChessRCCN(hidden_dim=64, use_lstm=use_lstm, input_channels=DATA_CHANNELS).to(DEVICE)
        
        # Determine initial checkpoint (Manual Override > Auto Resume)
        ckpt_to_load = INITIAL_CHECKPOINT
        if not ckpt_to_load:
            ckpt_to_load = get_latest_checkpoint(SAVE_DIR)
            
        if ckpt_to_load and os.path.exists(ckpt_to_load):
            print(f"[*] Resuming from checkpoint: {ckpt_to_load}")
            checkpoint = torch.load(ckpt_to_load, map_location=DEVICE, weights_only=False)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
        else:
            print("[*] Starting training from scratch.")
        
        if USE_COMPILE and hasattr(torch, 'compile'):
            try: model = torch.compile(model)
            except Exception as e: print(f"Compilation failed: {e}")

        trainer = Trainer(model)
        
        # 3. Load Validation Set
        if not os.path.exists(GLOBAL_VAL_FILE):
            print(f"[!] {GLOBAL_VAL_FILE} not found. Run gen_val_data.py first.")
            return
        
        print("Loading global validation set...")
        packed_val = torch.load(GLOBAL_VAL_FILE, weights_only=False)
        raw_val_data = decompress_data(packed_val)
        
        if TRAIN_MODE == "CNN_ONLY":
            # Use PositionDataset for flat evaluation
            val_loader = DataLoader(
                PositionDataset(raw_val_data, use_history=USE_HISTORY), 
                batch_size=MINIBATCH_SIZE, 
                shuffle=False, 
                pin_memory=torch.cuda.is_available()
            )
        else:
            val_loader = DataLoader(ChessDataset(raw_val_data, use_history=USE_HISTORY), batch_size=MINIBATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

        # 4. Continuous Training Loop (Learner)
        history = load_history()
        current_epoch = len(history) + 1
        
        print(f"=== Eternal Orchestrator Mode (Architecture: {TRAIN_MODE}) ===")
        
        while True:
            # Detect all available batches (both .pt and .zst)
            all_files = os.listdir(DATA_CACHE_DIR)
            batch_ids = []
            for f in all_files:
                if f.startswith("batch_mpv_") and f.endswith(".zst"):
                    part = f.replace("batch_mpv_", "").replace(".zst", "")
                    if part.isdigit():
                        batch_ids.append(int(part))
            batch_ids = sorted(list(set(batch_ids)))
            
            if not batch_ids:
                print("[CONSUMER] Waiting for first batch of data...")
                time.sleep(30)
                continue

            # Global shuffle of batches to ensure diverse training order
            random.shuffle(batch_ids)
            print(f"\n--- EPOCH {current_epoch} (Files to process: {len(batch_ids)}) ---")
            
            # Lazy Loading Loop: Load one file, train, discard, repeat.
            res = get_dynamic_resource_info()
            num_workers = max(1, int(res["cpu_count"] * 0.25))
            
            epoch_train_loss = 0
            epoch_train_samples = 0

            num_batches = len(batch_ids)
            validation_milestones = [max(1, (num_batches * k) // 8) for k in range(1, 8)]
            
            for i, b_id in enumerate(tqdm(batch_ids, desc=f"  Epoch {current_epoch} Batch Progress")):
                b_path = os.path.join(DATA_CACHE_DIR, f"batch_mpv_{b_id}.zst")
                try:
                    packed_b = load_batch(b_path)  # Auto-detects .zst or .pt
                    batch_game_data = decompress_data(packed_b)
                    # Build temporary loader
                    if TRAIN_MODE == "CNN_ONLY":
                        # Use PositionDataset to flatten games into individual moves
                        ds = PositionDataset(batch_game_data, use_history=USE_HISTORY)
                        loader = DataLoader(ds, batch_size=MINIBATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=num_workers)
                    else:
                        ds = ChessDataset(batch_game_data, use_history=USE_HISTORY)
                        loader = DataLoader(ds, batch_size=MINIBATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), num_workers=num_workers)

                    metrics = trainer.train_epoch([(b_id, loader)], val_loader, cycle_id=0, epoch=current_epoch, log_history=False)
                    epoch_train_loss += metrics["train_loss"]
                    epoch_train_samples += 1
                    
                    # Quarter-epoch validation
                    if (i + 1) in validation_milestones:
                        progress_pct = round(((i + 1) / num_batches) * 100)
                        print(f"\n[Validation {progress_pct}%] {i+1}/{num_batches} batches processed.")
                        v_loss, v_acc1, v_acc3 = trainer.evaluate_model(val_loader)
                        trainer.step_scheduler(v_loss)
                        print(f"  Val Loss: {v_loss:.4f} | Top-1: {v_acc1*100:.2f}% | LR: {trainer.optimizer.param_groups[0]['lr']}")
                    
                    del batch_game_data, ds, loader, packed_b
                    gc.collect()
                except Exception as e:
                    print(f"Error processing batch {b_id}: {e}")

            # End of full pass through all batches
            avg_loss = epoch_train_loss / epoch_train_samples if epoch_train_samples > 0 else 0
            val_loss, val_acc1, val_acc3 = trainer.evaluate_model(val_loader)
            
            # 1. Compile base metrics
            epoch_metrics = {
                "epoch": current_epoch, "lr": trainer.optimizer.param_groups[0]['lr'],
                "train_loss": avg_loss, "val_loss": val_loss, "val_acc1": val_acc1, "val_acc3": val_acc3
            }
            
            # 2. Save epoch checkpoint
            ckpt_path = os.path.join(SAVE_DIR, f"epoch_{current_epoch}.pt")
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(save_model.state_dict(), ckpt_path)
            
            # 3. Estimate Elo against Run 1 Pool (Uniform Round-Robin)
            try:
                print(f"\n[Elo] Evaluating epoch_{current_epoch}...")
                tracker = EloTracker()
                
                # Build pool from Run 1 (T=0.8 Corrected)
                run1_ratings = {1: 936, 2: 1100, 3: 1150, 4: 1195, 5: 1298, 6: 1255, 7: 1277}
                pool = []
                for ep, elo in run1_ratings.items():
                    p = os.path.join("checkpoints_v1", f"epoch_{ep}.pt")
                    if os.path.exists(p): pool.append((p, elo))
                
                # Use Uniform Comparison against the whole pool (140 games total)
                est_elo = tracker.estimate_elo_uniform(ckpt_path, pool, stockfish_pool=[1320], games_per_opponent=20)
                epoch_metrics["estimated_elo"] = est_elo
            except Exception as e:
                print(f"[Elo] Evaluation failed: {e}")

            # 4. Finalize and Plot
            trainer.log_epoch_end(epoch_metrics)
            trainer.step_scheduler(val_loss)
            
            print(f"Epoch {current_epoch} Complete | Loss: {avg_loss:.4f} | Val: {v_loss:.4f} | Top-1: {v_acc1*100:.2f}%")
            
            current_epoch += 1
            trainer.clear_cache()

    finally:
        print("[ORCHESTRATOR] Shutting down background generator...")
        if p_gen:
            p_gen.terminate()
            p_gen.join()

if __name__ == "__main__":
    main()
