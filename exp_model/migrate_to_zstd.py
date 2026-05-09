"""
Migrate existing .pt data batches to zstd-compressed .zst format.
Preserves originals until verification passes.
"""
import os
import io
import torch
import zstandard as zstd
from tqdm import tqdm
from data_utils import decompress_data
from config import DATA_CACHE_DIR

def migrate_batch(pt_path: str, level: int = 3) -> tuple:
    """Migrate a single .pt file to .zst format.
    Returns (success, original_size, compressed_size).
    """
    zst_path = pt_path.replace(".pt", ".zst")
    
    # Skip if already migrated
    if os.path.exists(zst_path):
        return True, 0, 0
    
    # Load original
    original_data = torch.load(pt_path, weights_only=False)
    original_size = os.path.getsize(pt_path)
    
    # Verify it's valid data before migration
    try:
        games = decompress_data(original_data)
        if not games:
            print(f"  [SKIP] {pt_path}: No valid games found")
            return False, original_size, 0
    except Exception as e:
        print(f"  [SKIP] {pt_path}: Invalid data - {e}")
        return False, original_size, 0
    
    # Compress
    buffer = io.BytesIO()
    torch.save(original_data, buffer)
    raw_bytes = buffer.getvalue()
    
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(raw_bytes)
    compressed_size = len(compressed)
    
    # Save compressed version
    with open(zst_path, "wb") as f:
        f.write(compressed)
    
    # Verify the compressed file loads correctly
    try:
        with open(zst_path, "rb") as f:
            verify_compressed = f.read()
        dctx = zstd.ZstdDecompressor()
        verify_raw = dctx.decompress(verify_compressed)
        verify_buffer = io.BytesIO(verify_raw)
        verify_data = torch.load(verify_buffer, weights_only=False)
        verify_games = decompress_data(verify_data)
        
        if len(verify_games) != len(games):
            raise ValueError(f"Game count mismatch: {len(verify_games)} vs {len(games)}")
    except Exception as e:
        print(f"  [FAIL] {pt_path}: Verification failed - {e}")
        os.remove(zst_path)  # Remove corrupted file
        return False, original_size, 0
    
    return True, original_size, compressed_size

def main():
    print("=== Migrating .pt files to .zst format ===\n")
    
    # Find all .pt batch files
    pt_files = [f for f in os.listdir(DATA_CACHE_DIR) 
                if f.startswith("batch_") and f.endswith(".pt")]
    
    if not pt_files:
        print("No .pt files found to migrate.")
        return
    
    print(f"Found {len(pt_files)} batch files to process.\n")
    
    total_original = 0
    total_compressed = 0
    migrated = 0
    skipped = 0
    
    for filename in tqdm(pt_files, desc="Migrating"):
        pt_path = os.path.join(DATA_CACHE_DIR, filename)
        success, orig_size, comp_size = migrate_batch(pt_path)
        
        if success and comp_size > 0:
            total_original += orig_size
            total_compressed += comp_size
            migrated += 1
        elif success:
            skipped += 1  # Already migrated
        else:
            skipped += 1
    
    print(f"\n=== Migration Complete ===")
    print(f"Files migrated: {migrated}")
    print(f"Files skipped: {skipped}")
    
    if total_original > 0:
        ratio = total_original / total_compressed
        print(f"\nOriginal size: {total_original / 1e9:.2f} GB")
        print(f"Compressed size: {total_compressed / 1e9:.2f} GB")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Space saved: {(total_original - total_compressed) / 1e9:.2f} GB ({(1 - total_compressed/total_original)*100:.1f}%)")
    
    print("\n[!] Original .pt files preserved. Delete them manually after verifying training works.")
    print("    Command: del generated_data\\*.pt")

if __name__ == "__main__":
    main()
