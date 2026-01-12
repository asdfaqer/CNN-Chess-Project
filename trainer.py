import os
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from tqdm import tqdm

from config import (DEVICE, AMP_DTYPE, USE_AMP, TRAIN_MODE, LEARNING_RATE, 
                   LR_WARMUP_STEPS, MIN_LR,
                   SAVE_DIR, DATA_CHANNELS, USE_HISTORY,
                   USE_DYNAMIC_IMPORTANCE, USE_IMPORTANCE_WEIGHT, EV_TANH_K)
from data_utils import ChessDataset, PositionDataset, collate_fn, load_history, save_history, update_plot
from elo_tracker import EloTracker

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def scores_to_win_prob(self, cp, mate, k=EV_TANH_K):
        """Convert raw cp and mate scores to win probability [0, 1]."""
        cp_eff = cp.float()
        mate_mask = mate != 0
        if mate_mask.any():
            # For mate, use a very high centipawn value. 
            # Mate in 1 is better than mate in 5.
            # 10000 - mate_score
            mate_scores = torch.where(mate > 0, 10000.0 - mate.float(), -10000.0 - mate.float())
            cp_eff = torch.where(mate_mask, mate_scores, cp_eff)
        return 0.5 + 0.5 * torch.tanh(cp_eff / k)

    def forward(self, inputs, targets, weights=None, mpv_data=None):
        loss = self.ce(inputs, targets)
        
        if USE_DYNAMIC_IMPORTANCE and mpv_data:
            # mpv_data can be [scores, moves] (legacy) or [cp, mate, depth, moves] (verbose)
            if len(mpv_data) == 2:
                mpv_scores, mpv_moves = mpv_data
            elif len(mpv_data) == 4:
                mpv_cp, mpv_mate, mpv_depth, mpv_moves = mpv_data
                mpv_scores = self.scores_to_win_prob(mpv_cp, mpv_mate)
            else:
                if USE_IMPORTANCE_WEIGHT and weights is not None:
                    return (loss * weights).mean()
                return loss.mean()

            # 1. Compute EV of ground truth (Stockfish top move)
            ev_gt = 2.0 * mpv_scores[:, 0] - 1.0
            
            # 2. Compute EV of prediction
            probs = torch.softmax(inputs, dim=1)
            
            mask = mpv_moves != -1
            safe_moves = mpv_moves.clone()
            safe_moves[~mask] = 0
            
            n_probs = torch.gather(probs, 1, safe_moves)
            n_probs = n_probs * mask.float()
            
            win_prob_pred = torch.sum(n_probs * mpv_scores, dim=1)
            ev_pred = 2.0 * win_prob_pred - 1.0
            
            # 3. Dynamic Importance = Gap
            dyn_weight = torch.clamp(ev_gt - ev_pred, min=0.0)
            
            # 4. Global Adaptive Weighting: Normalize so batch average is 1.0
            # This prevents loss from decaying as the model's EV gap narrows.
            avg_dyn = dyn_weight.mean()
            if avg_dyn > 1e-7:
                dyn_weight = dyn_weight / avg_dyn
            
            loss = loss * dyn_weight
            
        elif USE_IMPORTANCE_WEIGHT and weights is not None:
            loss = loss * weights
            
        return loss.mean()

def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set and return loss, top-1, and top-3 accuracy."""
    model.eval()
    val_loss, acc1, acc3, samples = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels, weights = batch[0], batch[1], batch[2] if len(batch) == 3 else None
            
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if weights is not None: 
                weights = weights.to(device, non_blocking=True)
            
            outputs, _ = model(inputs.float())
            flat_outputs = outputs.float().view(-1, 4672)
            flat_labels = labels.view(-1)
            mask = flat_labels != -1
            
            if mask.any():
                if isinstance(criterion, WeightedCrossEntropyLoss):
                    mpv_data = []
                    if len(batch) > 3:
                        for j in range(3, len(batch)):
                            mpv_data.append(batch[j].to(device, non_blocking=True).view(-1, batch[j].size(-1)))
                    loss = criterion(flat_outputs, flat_labels, weights.view(-1) if weights is not None else None, mpv_data)
                else:
                    loss = criterion(flat_outputs, flat_labels)
                
                val_loss += loss.item() * mask.sum().item()
                
                _, pred = flat_outputs[mask].topk(3, 1, True, True)
                correct = pred.eq(flat_labels[mask].view(-1, 1).expand_as(pred))
                acc1 += correct[:, :1].sum().item()
                acc3 += correct[:, :3].sum().item()
                samples += mask.sum().item()

    return (val_loss / samples if samples > 0 else 0, 
            acc1 / samples if samples > 0 else 0, 
            acc3 / samples if samples > 0 else 0)

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        # Smoother drops: 0.7 factor, floor at MIN_LR, threshold for meaningful improvement
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=1, min_lr=MIN_LR, threshold=0.005
        )
        
        self.criterion = WeightedCrossEntropyLoss(ignore_index=-1)
            
        self.scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
        self.history = load_history()
        self.warmup_steps = LR_WARMUP_STEPS
        # If history exists, we assume we've completed the warmup phase
        self.total_steps = len(self.history) * 5000 if len(self.history) > 0 else 0 

    def train_epoch(self, loaders: List[Tuple[int, DataLoader]], val_loader: DataLoader, 
                    cycle_id: int, epoch: int, log_history: bool = True):
        """Train for one epoch across all provided data loaders."""
        self.model.train()
        train_loss, train_samples = 0.0, 0
        
        for c_id, loader_b in tqdm(loaders, desc=f"  Processing Batches", unit="batch", leave=False):
            pbar = tqdm(loader_b, desc=f"    Batch {c_id}", unit="step", leave=False)
            for batch in pbar:
                inputs, labels, weights = batch[0], batch[1], batch[2] if len(batch) == 3 else None
                
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                if weights is not None: 
                    weights = weights.to(DEVICE, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                    outputs, _ = self.model(inputs.float())
                    target_outputs = outputs.float().view(-1, 4672)
                    target_labels = labels.view(-1)
                    target_weights = weights.view(-1) if weights is not None else None
                    
                    if isinstance(self.criterion, WeightedCrossEntropyLoss):
                        mpv_data = []
                        if len(batch) > 3:
                            for j in range(3, len(batch)):
                                mpv_data.append(batch[j].to(DEVICE, non_blocking=True).view(-1, batch[j].size(-1)))
                        loss = self.criterion(target_outputs, target_labels, target_weights, mpv_data)
                    else:
                        loss = self.criterion(target_outputs, target_labels)
                        if USE_DYNAMIC_IMPORTANCE and target_mpv_scores is not None:
                            # Simple scaling logic could go here if criterion was not FocalLoss
                            pass
                
                if USE_AMP and AMP_DTYPE == torch.float16:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    
                    # Apply Linear Warmup if within warmup phase
                    if self.total_steps < self.warmup_steps:
                        lr = LEARNING_RATE * (max(1, self.total_steps) / self.warmup_steps)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    
                    # Apply Linear Warmup if within warmup phase
                    if self.total_steps < self.warmup_steps:
                        lr = LEARNING_RATE * (max(1, self.total_steps) / self.warmup_steps)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                            
                    self.optimizer.step()
                
                self.total_steps += 1
                
                num_valid = (target_labels != -1).sum().item()
                if num_valid > 0:
                    train_loss += loss.item() * num_valid
                    train_samples += num_valid
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
        
        if log_history:
            val_loss, val_acc1, val_acc3 = evaluate(self.model, val_loader, self.criterion, DEVICE)
            epoch_metrics = {
                "epoch": epoch, "lr": self.optimizer.param_groups[0]['lr'],
                "train_loss": avg_train_loss, "val_loss": val_loss, "val_acc1": val_acc1, "val_acc3": val_acc3
            }
            self.log_epoch_end(epoch_metrics)
            self.scheduler.step(val_loss)
            return epoch_metrics
        
        return {"train_loss": avg_train_loss}

    def evaluate_model(self, val_loader):
        return evaluate(self.model, val_loader, self.criterion, DEVICE)

    def step_scheduler(self, val_loss):
        """Manually step the learning rate scheduler."""
        self.scheduler.step(val_loss)

    def log_epoch_end(self, metrics: Dict):
        self.history.append(metrics)
        save_history(self.history)
        update_plot(self.history)

    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================
# STANDALONE TRAINER MODE
# ============================================
def main():
    import random
    import numpy as np
    from torch.utils.data import DataLoader
    from config import (SEED, TRAIN_MODE, DATA_CHANNELS, DEVICE, INITIAL_CHECKPOINT, 
                       USE_COMPILE, GLOBAL_VAL_FILE, MINIBATCH_SIZE, USE_HISTORY, 
                       DATA_CACHE_DIR, SAVE_DIR)
    from model import ChessRCCN
    from data_utils import decompress_data, load_history, load_batch, ChessDataset, PositionDataset, collate_fn, get_latest_checkpoint
    from utils import get_dynamic_resource_info

    def set_seed(s: int):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    set_seed(SEED)
    
    # Initialize Model
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
        try: 
            model = torch.compile(model)
        except Exception as e: 
            print(f"Compilation failed: {e}")

    trainer = Trainer(model)
    
    # Load Global Validation Set
    if not os.path.exists(GLOBAL_VAL_FILE):
        print(f"Error: {GLOBAL_VAL_FILE} not found. Run gen_val_data.py first.")
        return
    
    print("Loading global validation set...")
    packed_val = torch.load(GLOBAL_VAL_FILE, weights_only=False)
    raw_val_data = decompress_data(packed_val)
    
    # Setup Validation DataLoader
    if TRAIN_MODE == "CNN_ONLY":
        val_loader = DataLoader(
            PositionDataset(raw_val_data, use_history=USE_HISTORY), 
            batch_size=MINIBATCH_SIZE, 
            shuffle=False, 
            pin_memory=torch.cuda.is_available()
        )
    else:
        val_loader = DataLoader(
            ChessDataset(raw_val_data, use_history=USE_HISTORY), 
            batch_size=MINIBATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )

    print(f"=== Standalone Trainer Mode (Architecture: {TRAIN_MODE}) ===")
    
    # History tracking
    existing_history = load_history()
    current_epoch = len(existing_history) + 1

    # Training Loop
    while True:
        all_files = [f for f in os.listdir(DATA_CACHE_DIR) 
                     if f.startswith("batch_mpv_") and f.endswith(".zst")]
        if not all_files:
            print("[TRAINER] No 'batch_mpv_' data batches found. Waiting 30s...")
            time.sleep(30)
            continue

        batch_ids = []
        for f in all_files:
            part = f.replace("batch_mpv_", "").replace(".zst", "")
            if part.isdigit():
                batch_ids.append(int(part))
        batch_ids = sorted(list(set(batch_ids)))
        random.shuffle(batch_ids)
        print(f"\n--- Standalone EPOCH {current_epoch} (Files: {len(batch_ids)}) ---")
        
        res = get_dynamic_resource_info()
        num_workers = max(1, int(res["cpu_count"] * 0.25))
        
        epoch_train_loss = 0
        epoch_train_samples = 0

        num_batches = len(batch_ids)
        validation_milestones = [max(1, (num_batches * k) // 8) for k in range(1, 8)]

        for i, b_id in enumerate(tqdm(batch_ids, desc=f"  Epoch {current_epoch} Progress")):
            b_path = os.path.join(DATA_CACHE_DIR, f"batch_mpv_{b_id}.zst")
            try:
                packed_b = load_batch(b_path)  # Auto-detects .zst or .pt
                batch_data = decompress_data(packed_b)
                
                if TRAIN_MODE == "CNN_ONLY":
                    ds = PositionDataset(batch_data, use_history=USE_HISTORY)
                    loader = DataLoader(ds, batch_size=MINIBATCH_SIZE, shuffle=True, 
                                       pin_memory=torch.cuda.is_available(), num_workers=num_workers)
                else:
                    ds = ChessDataset(batch_data, use_history=USE_HISTORY)
                    loader = DataLoader(ds, batch_size=MINIBATCH_SIZE, shuffle=True, 
                                       collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), 
                                       num_workers=num_workers)

                metrics = trainer.train_epoch([(b_id, loader)], val_loader, cycle_id=0, 
                                             epoch=current_epoch, log_history=False)
                epoch_train_loss += metrics["train_loss"]
                epoch_train_samples += 1

                # Quarter-epoch validation
                if (i + 1) in validation_milestones:
                    progress_pct = round(((i + 1) / num_batches) * 100)
                    print(f"\n[Validation {progress_pct}%] {i+1}/{num_batches} batches processed.")
                    v_loss, v_acc1, v_acc3 = trainer.evaluate_model(val_loader)
                    trainer.step_scheduler(v_loss)
                    print(f"  Val Loss: {v_loss:.4f} | Top-1: {v_acc1*100:.2f}% | LR: {trainer.optimizer.param_groups[0]['lr']}")

                del batch_data, ds, loader, packed_b
                gc.collect()
            except Exception as e:
                print(f"Error on batch {b_id}: {e}")

        # Finalize Epoch metrics
        avg_loss = epoch_train_loss / epoch_train_samples if epoch_train_samples > 0 else 0
        v_loss, v_acc1, v_acc3 = trainer.evaluate_model(val_loader)
        
        epoch_metrics = {
            "epoch": current_epoch, "lr": trainer.optimizer.param_groups[0]['lr'],
            "train_loss": avg_loss, "val_loss": v_loss, "val_acc1": v_acc1, "val_acc3": v_acc3
        }
        
        # Save epoch checkpoint (required for EloTracker)
        ckpt_path = os.path.join(SAVE_DIR, f"epoch_{current_epoch}.pt")
        save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(save_model.state_dict(), ckpt_path)
        
        # Estimate Elo against Run 1 Pool (Uniform Round-Robin)
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

        # Finalize and Plot
        trainer.log_epoch_end(epoch_metrics)
        trainer.step_scheduler(v_loss)
        
        print(f"Epoch {current_epoch} Complete | Loss: {avg_loss:.4f} | Val: {v_loss:.4f} | Top-1: {v_acc1*100:.2f}%")
        
        current_epoch += 1
        trainer.clear_cache()
        time.sleep(5)

if __name__ == "__main__":
    main()
